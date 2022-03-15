# ************
# File: main_verification_process.py
# Top contributors (to current version): 
# 	Panagiotis Kouvaros (panagiotis.kouvaros@gmail.com)
# This file is part of the Venus project.
# Copyright: 2019-2021 by the authors listed in the AUTHORS file in the
# top-level directory.
# License: BSD 2-Clause (see the file LICENSE in the top-level directory).
# Description: Main verification process.
# ************

from timeit import default_timer as timer
from venus.verification.verification_problem import VerificationProblem
from venus.verification.verification_process import VerificationProcess
from venus.verification.verification_report import VerificationReport
from venus.split.split_process import SplitProcess
from venus.split.input_splitter import InputSplitter
from venus.split.split_strategy import SplitStrategy
from venus.split.split_report import SplitReport
from venus.solver.solve_result import SolveResult
from venus.solver.solve_report import SolveReport
from venus.common.logger import get_logger
import multiprocessing as mp
import queue

class Verifier:
    
    logger = None

    def __init__(self, nn, spec, config):
        """
        Arguments:

            nn:
                NeuralNetwork.

            spec:
                Specification.

            config:
                Configuration.
        """
        self.prob = VerificationProblem(nn, spec, 0, config)
        self.config = config
        self.reporting_queue = mp.Queue()
        self.jobs_queue = mp.Queue()
        if Verifier.logger is None:
            Verifier.logger = get_logger(__name__, config.LOGGER.LOGFILE)
        self.split_procs = []
        self.ver_procs = []


    def verify(self):
        """
        Verifies a neural network against a specification.

        Returns:

            VerificationReport
        """
        Verifier.logger.info(f'Verifying {self.prob.to_string()}')
        if self.config.VERIFIER.COMPLETE == True:
            ver_report = self.verify_complete()
        else:
            ver_report = self.verify_incomplete()
        Verifier.logger.info('Verification completed')
        Verifier.logger.info(ver_report.to_string())

        return ver_report



    def verify_incomplete(self):
        """
        Attempts to solve a verification problem using symbolic interval propagation.

        Returns:

            VerificationReport
        """
        ver_report = VerificationReport(self.config.LOGGER.LOGFILE)
        start = timer()
        self.prob.bound_analysis()
        if self.prob.satisfies_spec():
            ver_report.result = SolveResult.SAFE
            Verifier.logger.info('Verification problem was solved via bound analysis')
        else:
            ver_report.result = SolveResult.UNDECIDED
            Verifier.logger.info('Verification problem could not be solved via bound analysis.')

        ver_report.runtime = timer() - start

        return ver_report
    
       
    def verify_complete(self):
        """
        Solves a verification problem by solving its MILP representation.

        Arguments:
            
            prob: VerificationProblem

        Returns:

            VerificationReport
        """
        start = timer()
        # try to solve the problem using the bounds and lp
        report = self.verify_incomplete()
        if report.result == SolveResult.SAFE:
            return report
    
        # start the splitting and worker processes
        self.generate_procs()
        # read results
        ver_report = self.process_report_queue()
        self.terminate_procs()

        ver_report.runtime = timer() - start



        return ver_report

    def process_report_queue(self):
        """ 
        Reads results from the reporting queue until encountered an UNSATISFIED
        result, or until all the splits have completed

        Returns

            VerificationReport
        """
        start = timer()
        ver_report = VerificationReport(self.config.LOGGER.LOGFILE)
        while True:
            try:
                time_elapsed = timer() - start
                tmo = self.config.SOLVER.TIME_LIMIT - time_elapsed
                report = self.reporting_queue.get(timeout=tmo)
                if isinstance(report, SplitReport):
                    ver_report.process_split_report(report)
                elif isinstance(report, SolveReport):
                    ver_report.process_solve_report(report)
                    if report.result == SolveResult.BRANCH_THRESHOLD:
                        Verifier.logger.info('Threshold of MIP nodes reached. Turned off monitor split.')
                        self.config.SOLVER.MONITOR_SPLIT = False
                        self.generate_procs()
                    elif report.result == SolveResult.UNSAFE:
                        Verifier.logger.info('Read UNSATisfied result. Terminating ...')
                        break
                else:
                        raise Exception(f'Unexpected report read from reporting queue {type(report)}')

                # termination conditions
                if ver_report.finished_split_procs_count == len(self.split_procs) \
                and ver_report.finished_jobs_count >= ver_report.jobs_count:
                    Verifier.logger.info("All subproblems have finished. Terminating...")
                    if ver_report.timedout_jobs_count == 0:
                        ver_report.result = SolveResult.SAFE
                    else:
                        ver_report.result = SolveResult.TIMEOUT
                    break
            except queue.Empty:
                # Timeout occurred
                ver_report.result = SolveResult.TIMEOUT
                break
            except KeyboardInterrupt:
                # Received terminating signal
                ver_report.result = SolveResult.INTERRUPTED
                break
                    
        return ver_report


    def generate_procs(self):
        """
        Creates splitting and verification processes.

        Returns:
            
            None
        """

        self.generate_split_procs()
        self.generate_ver_procs()

    def generate_split_procs(self):
        """
        Creates splitting  processes.

        Returns:
            
            None
        """
        if self.config.SPLITTER.SPLIT_STRATEGY != SplitStrategy.NONE \
        and self.config.SOLVER.MONITOR_SPLIT is False:
            if self.config.SPLITTER.SPLIT_PROC_NUM > 0:
                isplitter = InputSplitter(
                    self.prob,
                    self.prob.stability_ratio,
                    self.config
                )
                splits = isplitter.split_up_to_depth(self.config.SPLITTER.SPLIT_PROC_NUM)
            else:
                splits = [self.prob]
            self.split_procs = [
                SplitProcess(
                    i+1,
                    splits[i],
                    self.jobs_queue,
                    self.reporting_queue,
                    self.config
                )
                for i in range(len(splits))]
            for proc in self.split_procs:
                proc.start()
                Verifier.logger.info(f'Generated {len(self.split_procs)} split processes')
        else:
            self.jobs_queue.put(self.prob)
            Verifier.logger.info('Added original verification problem to job queue.')

    def generate_ver_procs(self):
        """
        Creates verification  processes.

        Returns:
            
        None
        """
        if self.config.SOLVER.MONITOR_SPLIT == True \
        or self.config.SPLITTER.SPLIT_STRATEGY == SplitStrategy.NONE:
            procs_to_gen = range(1)
        else:
            procs_to_gen = range(len(self.ver_procs), self.config.VERIFIER.VER_PROC_NUM)
           
        ver_procs = [
            VerificationProcess(
                i+1,
                self.jobs_queue,
                self.reporting_queue,
                self.config
                
            )
            for i in procs_to_gen
        ]
        for proc in ver_procs:
            proc.start()
        self.ver_procs = self.ver_procs + ver_procs
        Verifier.logger.info(f'Generated {len(procs_to_gen)} verification processes.')

    def terminate_procs(self):
        """
        Terminates all splitting and verification processes.

        Returns:

            None
        """
        self.reporting_queue.cancel_join_thread()
        self.jobs_queue.cancel_join_thread()
        self.terminate_split_procs()
        self.terminate_ver_procs()

    def terminate_split_procs(self):
        """
        Terminates all splitting processes.

        Returns:

            None
        """
        try:
            for proc in self.split_procs:
                proc.terminate()
        except:
            raise Exception("Could not terminate splitting processes.")

    def terminate_ver_procs(self):
        """
        Terminates all verification processes.

        Returns:

            None
        """
        try:
            for proc in self.ver_procs:
                proc.terminate()
        except:
            raise Exception("Could not terminate verification processes.")
