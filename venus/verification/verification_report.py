# ************
# File: verification_report.py
# Top contributors (to current version): 
# 	Panagiotis Kouvaros (panagiotis.kouvaros@gmail.com)
# This file is part of the Venus project.
# Copyright: 2019-2021 by the authors listed in the AUTHORS file in the
# top-level directory.
# License: BSD 2-Clause (see the file LICENSE in the top-level directory).
# Description: Data regarding a verification result.
# ************

from venus.solver.solve_result import SolveResult

class VerificationReport:
    def __init__(self, result=SolveResult.UNDECIDED, cex=None, runtime=0):
        """
        Arguments:
            
            config: 
                Configuration 

            result: 
                SolveResult (result from MILPSolver)

            cex: 
                np.array of a counter-example

            runtime: 
                double (total verification time)

        """
        self.result = result
        self.cex = cex
        self.runtime = runtime
        self.jobs_count = 0
        self.finished_jobs_count = 0
        self.timedout_jobs_count = 0
        self.input_split_count = 0
        self.node_split_count = 0
        self.node_splitter_initiated = False
        self.finished_split_procs_count = 0

    def process_split_report(self, report):
        """
        Integrates the data from a split report.

        Arguments:

            report: SplitReport

        Returns:

            None
        """
        self.jobs_count += report.jobs_count
        self.node_split_count += report.node_split_count
        self.input_split_count += report.input_split_count
        self.finished_split_procs_count += 1

    def process_solve_report(self, report):
        """
        Integrates the data from a solve report.

        Arguments:

            report: SolveReport

        Returns:

            None
        """
        if report.result == SolveResult.BRANCH_THRESHOLD:
            self.node_splitter_initiated =  True
        elif report.result == SolveResult.UNSAFE:
            self.finished_jobs_count += 1
            self.result = report.result
            self.cex = report.cex
        elif report.result == SolveResult.SAFE:
            self.finished_jobs_count += 1
        elif report.result == SolveResult.TIMEOUT:
            self.timedout_jobs_count += 1
        else:
            raise Exception(f'Unexpected result read from solve report {report.result}')

    def to_string(self):
        """
        Returns:

            str describing the report.
        """
        string = 'Result: ' + self.result.value + '\n'
        string += '' if self.cex is None else 'Counter-example: ' + str(list(self.cex)) + '\n'
        string += 'Time: ' + str(self.runtime) + '\n' + \
            'Total sub-problems solved: ' + str(self.finished_jobs_count) + '\n' +  \
            'Total sub-problems timed out: ' + str(self.timedout_jobs_count) + '\n' + \
            'Total input splits: ' + str(self.input_split_count) + '\n' + \
            'Total node splits: ' + str(self.node_split_count) + '\n'

        return string
