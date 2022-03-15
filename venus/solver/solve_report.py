# ************
# File: solve_report.py
# Top contributors (to current version): 
# 	Panagiotis Kouvaros (panagiotis.kouvaros@gmail.com)
# This file is part of the Venus project.
# Copyright: 2019-2021 by the authors listed in the AUTHORS file in the
# top-level directory.
# License: BSD 2-Clause (see the file LICENSE in the top-level directory).
# Description: Data regarding a milp result.
# ************

class SolveReport:
    def __init__(self, result, runtime, cex=None):
        self.result = result
        self.runtime = runtime
        self.cex = cex
