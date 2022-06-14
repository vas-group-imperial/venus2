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

import torch.multiprocessing as mp
import queue
from timeit import default_timer as timer

from venus.verification.verification_problem import VerificationProblem
from venus.verification.subverifier import SubVerifier
from venus.verification.verification_report import VerificationReport
from venus.verification.pgd import ProjectedGradientDescent
from venus.split.splitter import Splitter
from venus.split.input_splitter import InputSplitter
from venus.split.split_strategy import SplitStrategy
from venus.split.split_report import SplitReport
from venus.solver.solve_result import SolveResult
from venus.solver.solve_report import SolveReport
from venus.common.logger import get_logger
from venus.common.configuration import Config

import torch
from venus.solver.milp_solver import MILPSolver
from venus.solver.milp_encoder import MILPEncoder


def run_split_process(
        prob: VerificationProblem,
        jobs_queue: mp.Queue,
        reporting_queue: mp.Queue,
        config: Config
):
    proc = Splitter(
        prob, jobs_queue, reporting_queue, config
    )
    proc.run()

def run_subverifier( 
        jobs_queue: mp.Queue,
        reporting_queue: mp.Queue,
        config: Config
):

    proc = SubVerifier(
        config, jobs_queue=jobs_queue, reporting_queue=reporting_queue
    )
    proc.run()

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

        self._mp_context = mp.get_context('spawn')
        self._manager = self._mp_context.Manager()
        self.reporting_queue = self._manager.Queue()
        self.jobs_queue = self._manager.Queue()

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

        start = timer()

        slv_report = SubVerifier(self.config).verify_incomplete(self.prob)

        if slv_report.result != SolveResult.UNDECIDED:
            ver_report = VerificationReport(
                slv_report.result, slv_report.cex, slv_report.runtime
            )

        elif self.config.VERIFIER.COMPLETE is True:
            if self.config.SOLVER.MONITOR_SPLIT is True:
                slv_report = SubVerifier(self.config).verify_complete(self.prob)
                if slv_report.result == SolveResult.BRANCH_THRESHOLD:
                    # turn off monitor split
                    self.config.SOLVER.MONITOR_SPLIT = False
                    # start the splitting and worker processes
                    self.generate_procs()
                    # read results
                    ver_report = self.process_report_queue()
                    # terminate procs
                    self.terminate_procs()
                else:
                    ver_report = VerificationReport(
                        slv_report.result, slv_report.cex, slv_report.runtime
                    )
            else:
                ver_report = VerificationReport(
                    slv_report.result, slv_report.cex, slv_report.runtime
                )

            ver_report.runtime = timer() - start

        Verifier.logger.info('Verification completed')
        Verifier.logger.info(ver_report.to_string())

        return ver_report
    
    def process_report_queue(self):
        """ 
        Reads results from the reporting queue until encountered an UNSATISFIED
        result, or until all the splits have completed

        Returns:
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
                    self._process_split_report(ver_report, report)

                elif isinstance(report, SolveReport):
                    self._process_solve_report(ver_report, report)

                    if report.result == SolveResult.UNSAFE:
                        Verifier.logger.info('Read UNSATisfied result. Terminating ...')
                        break

                else:
                        raise Exception(
                            f'Unexpected report read from queue {type(report)}'
                        )

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

    def _process_split_report(self, ver_report: VerificationReport, split_report: SplitReport):
        """ 
        Updates the verification state with a split report. 

        Returns:
            VerificationReport
        """
        ver_report.process_split_report(split_report)

    def _process_solve_report(self, ver_report: VerificationReport, solve_report: SolveReport):
        """ 
        Updates the verification state  with a solve report. 

        Returns:
            VerificationReport
        """
        ver_report.process_solve_report(solve_report)
        if solve_report.result == SolveResult.BRANCH_THRESHOLD:
            Verifier.logger.info(
                'Threshold of MIP nodes reached. Turned off monitor split.'
            )
            self.config.SOLVER.MONITOR_SPLIT = False
            self.generate_procs()
            

    def generate_procs(self):
        """
        Creates splitting and verification processes.
        """

        self.generate_split_procs()
        self.generate_ver_procs()

    def generate_split_procs(self):
        """
        Creates splitting  processes.
        """
        if self.config.SPLITTER.SPLIT_STRATEGY != SplitStrategy.NONE \
        and self.config.SOLVER.MONITOR_SPLIT is False:
            if self.config.SPLITTER.SPLIT_PROC_NUM > 0 and \
            ( \
                self.config.SPLITTER.SPLIT_STRATEGY == SplitStrategy.INPUT or \
                self.config.SPLITTER.SPLIT_STRATEGY == SplitStrategy.INPUT_NODE \
            ):
                isplitter = InputSplitter(
                    self.prob,
                    self.prob.stability_ratio,
                    self.config
                )
                splits = isplitter.split_up_to_depth(
                    self.config.SPLITTER.SPLIT_PROC_NUM
                )
            else:
                splits = [self.prob]

            self.split_procs = [
                self._mp_context.Process(
                    target=run_split_process,
                    args=(
                        splits[i],
                        self.jobs_queue,
                        self.reporting_queue,
                        self.config
                    )
                )
                for i in range(len(splits))
            ]
            
            for proc in self.split_procs:
                proc.start()
            
            Verifier.logger.info(f'Generated {len(self.split_procs)} split processes')

        else:
            self.jobs_queue.put(self.prob)
            Verifier.logger.info('Added original verification problem to job queue.')

    def generate_ver_procs(self):
        """
        Creates verification  processes.
        """
        if self.config.SOLVER.MONITOR_SPLIT is True or \
        self.config.SPLITTER.SPLIT_STRATEGY == SplitStrategy.NONE:
            procs_to_gen = range(1)
        else:
            procs_to_gen = range(len(self.ver_procs), self.config.VERIFIER.VER_PROC_NUM)
          
        ver_procs = [
            self._mp_context.Process(
                    target=run_subverifier,
                    args=(
                        self.jobs_queue,
                        self.reporting_queue,
                        self.config
                    )
            )
            for _ in procs_to_gen
        ]

        for proc in ver_procs:
            proc.start()

        self.ver_procs = self.ver_procs + ver_procs
        Verifier.logger.info(f'Generated {len(self.ver_procs)} verification processes')


    def terminate_procs(self):
        """
        Terminates all splitting and verification processes.
        """
        # self.reporting_queue.cancel_join_thread()
        # self.jobs_queue.cancel_join_thread()
        self.terminate_split_procs()
        self.terminate_ver_procs()

    def terminate_split_procs(self):
        """
        Terminates all splitting processes.
        """
        try:
            for proc in self.split_procs:
                proc.terminate()
        except:
            raise Exception("Could not terminate splitting processes.")

    def terminate_ver_procs(self):
        """
        Terminates all verification processes.
        """
        try:
            for proc in self.ver_procs:
                proc.terminate()
        except:
            raise Exception("Could not terminate verification processes.")
