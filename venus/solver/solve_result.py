# ************
# File: solve_result.py
# Top contributors (to current version): 
# 	Panagiotis Kouvaros (panagiotis.kouvaros@gmail.com)
# This file is part of the Venus  project.
# Copyright: 2019-2021 by the authors listed in the AUTHORS file in the
# top-level directory.
# License: BSD 2-Clause (see the file LICENSE in the top-level directory).
# Description: Enum class for MILP solving outcomes.
# ************

from enum import Enum

class SolveResult(Enum):
    TIMEOUT = 'timeout'
    INTERRUPTED = 'interrupted'
    BRANCH_THRESHOLD = 'branch-threshold'    
    SAFE = 'safe'
    UNSAFE = 'unsafe'
    UNDECIDED = 'undecided'
