# ************
# File: subverifier.py
# Top contributors (to current version): 
# 	Panagiotis Kouvaros (panagiotis.kouvaros@gmail.com)
# This file is part of the Venus  project.
# Copyright: 2019-2021 by the authors listed in the AUTHORS file in the
# top-level directory.
# License: BSD 2-Clause (see the file LICENSE in the top-level directory).
# Description: SubVerifier for solving a verification problem.
# ************

import itertools
import traceback
import torch
from timeit import default_timer as timer
import torch.multiprocessing as mp
from torch.multiprocessing import Queue

from venus.common.logger import get_logger
from venus.solver.milp_solver import MILPSolver
from venus.solver.solve_report import SolveReport
from venus.solver.solve_result import SolveResult
from venus.verification.pgd import ProjectedGradientDescent
from venus.verification.verification_problem import VerificationProblem

class SubVerifier:

    logger = None
    TIMEOUT = 3600
    id_iter = itertools.count()

    def __init__(self, config, jobs_queue=None, reporting_queue=None):
        """
        Arguments:
            config: 
                Configuration.
            jobs_queue:
                Queue of verification problems to read from.
            reporting_queue:
                Queue of solve reports to enqueue the verification results.
        """
        super(SubVerifier, self).__init__()
        self.id = next(SubVerifier.id_iter)
        self.jobs_queue = jobs_queue
        self.reporting_queue = reporting_queue
        self.config = config
        if SubVerifier.logger is None:
            SubVerifier.logger = get_logger(__name__ + str(self.id), config.LOGGER.LOGFILE)

    def run(self):
        if self.jobs_queue is None:
            raise ValueError('jobs_queue shoud be Queue.')
        if self.reporting_queue is None:
            raise ValueError('reporting_queue shoud be Queue.')

        while True:
            try:
                prob = self.jobs_queue.get(timeout=self.TIMEOUT)
                SubVerifier.logger.info(
                    'Verification SubVerifier {} started job {}, '.format(
                        self.id, prob.id
                    )
                )
                slv_report = self.verify(prob)
                SubVerifier.logger.info(
                    'SubVerifier {} finished job {}, result: {}, time: {:.2f}.'.format(
                        self.id, 
                        prob.id, 
                        slv_report.result.value,
                        slv_report.runtime
                    )
                )
                print(prob.id, self.config.SOLVER.MONITOR_SPLIT)
                self.reporting_queue.put(slv_report)

            except Exception as error:
                print(traceback.format_exc())
                SubVerifier.logger.info(
                    f"Subprocess {self.id} terminated because of {str(error)}."
                )
                break


    def verify(self, prob):
        start = timer() 

        if prob.inc_ver_done is not True:
            slv_report = self.verify_incomplete(prob)
            if slv_report.result != SolveResult.UNDECIDED:
                return slv_report

        slv_report = self.verify_complete(prob)
        slv_report.runtime = timer() - start
        return slv_report
        

    def verify_incomplete(self, prob: VerificationProblem):
        """
        Attempts to solve a verification problem using projected gradient
        descent and symbolic interval propagation.

        Returns:
            SolveReport
        """
        start = timer()
        
        prob.inc_ver_done = True
        slv_report = SolveReport(SolveResult.UNDECIDED, 0, None)

        # try pgd
        if self.config.VERIFIER.PGD is True and prob.pgd_ver_done is not True:
            subreport = self.verify_pgd(prob)
            slv_report.result = subreport.result
            slv_report.cex = subreport.cex

        # try bound analysis
        if slv_report.result == SolveResult.UNDECIDED and \
        prob.bounds_ver_done is not True:
            subreport = self.verify_bounds(prob)
            slv_report.result = subreport.result

        # try LP analysis
        if slv_report.result == SolveResult.UNDECIDED and \
        self.config.VERIFIER.LP is True and \
        prob.lp_ver_done is not True:
            subreport = self.verify_lp(prob)
            slv_report.result = subreport.result
            slv_report.cex = subreport.cex

        prob.detach()
        prob.clean_vars()
        prob.spec.input_node.bounds.detach()
        slv_report.runtime = timer() - start
        return slv_report
       
    def verify_pgd(self, prob: VerificationProblem):
        """
        Attempts to solve a verification problem using projected gradient
        descent.

        Returns:
            SolveReport
        """
        start = timer()

        prob.pgd_ver_done = True

        pgd = ProjectedGradientDescent(self.config)
        cex = pgd.start(prob)
        if cex is not None:
            SubVerifier.logger.info(
                f'Verification problem {prob.id} was solved via PGD'
            )
            return SolveReport(SolveResult.UNSAFE, timer() - start, cex)

        SubVerifier.logger.info(
            f'PGD done. Verification problem {prob.id} could not be solved'
        )

        return SolveReport(SolveResult.UNDECIDED, timer() - start, cex)

    def verify_bounds(self, prob: VerificationProblem):
        """
        Attempts to solve a verification problem using bounds.

        Returns:
            SolveReport
        """
        start = timer()

        prob.bound_ver_done = True

        prob.bound_analysis()

        if prob.satisfies_spec():
            SubVerifier.logger.info(
                f'Verification problem {prob.id} was solved via bound analysis')
     
            return SolveReport(SolveResult.SAFE, timer() - start, None)

        SubVerifier.logger.info(
            f'Bound analysis done. Verification problem {prob.id} could not be solved'
        )

        return SolveReport(SolveResult.UNDECIDED, timer() - start, None)

    def verify_lp(self, prob: VerificationProblem):
        """
        Attempts to solve a verification problem using linear relaxation.

        Returns:
            SolveReport
        """
        start = timer()

        prob.lp_ver_done = True

        solver = MILPSolver(prob, self.config, lp=True)
        slv_report =  solver.solve()

        if slv_report.result == SolveResult.SAFE:
            SubVerifier.logger.info(
                f'Verification problem {prob.id} was solved via LP')
            return slv_report

        elif slv_report.result == SolveResult.UNSAFE and self.config.VERIFIER.PGD_ON_LP is True:
            cex = torch.tensor(
                slv_report.cex, dtype=self.config.PRECISION
            )
            pgd = ProjectedGradientDescent(self.config)
            cex = pgd.start(prob, init_adv=cex)
            if cex is not None:
                SubVerifier.logger.info(
                    f'Verification problem {prob.id} was solved via PGD on LP'
                )
                return SolveReport(SolverReport.UNSAFE, timer() - start, cex)
     
        SubVerifier.logger.info(
            'LP analysis done. Verification problem could not be solved'
        )

        return SolveReport(SolveResult.UNDECIDED, timer() - start, cex)

    def verify_complete(self, prob: VerificationProblem):
        """
        Attempts to solve a verification problem using MILP.

        Returns:
            SolveReport
        """
        solver = MILPSolver(prob, self.config)
        slv_report = solver.solve()
        prob.detach()
        prob.clean_vars()
        return slv_report

