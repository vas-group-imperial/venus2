# ************
# File: satisfaction_reader.py
# Top contributors (to current version): 
# 	Panagiotis Kouvaros (panagiotis.kouvaros@gmail.com)
# This file is part of the Venus project.
# Copyright: 2019-2021 by the authors listed in the AUTHORS file in the
# top-level directory.
# License: BSD 2-Clause (see the file LICENSE in the top-level directory).
# Description: Reads satisfaction statuses of verification queries.
# ************

from venus.solver.solve_result import SolveResult
import csv

class SatisfactionReader:

    def __init__(self, path):
        """
        Arguments:

            path: csv filepath
        """
        self.path = path
    
    def read_results(self):
        """
        Reads satisfaction results.

        Returns:
                
            A map of networks, specifications to satisfaction statuses
        """
        results = {}

        with open(self.path, 'r') as f:
            reader = csv.reader(f) 
            for row in reader:
                if row[2] == 'Safe':
                    results[(row[0], row[1])] = SolveResult.SAFE
                else:
                    results[(row[0], row[1])] = SolveResult.UNSAFE

        return results





