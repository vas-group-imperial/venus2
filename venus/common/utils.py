"""
# File: utils.py
# Top contributors (to current version): 
# 	Panagiotis Kouvaros (panagiotis.kouvaros@gmail.com)
# This file is part of the Venus  project.
# Copyright: 2019-2021 by the authors listed in the AUTHORS file in the
# top-level directory.
# License: BSD 2-Clause (see the file LICENSE in the top-level directory).
# Description: Auxiliary class definitions and methods.
"""

from enum import Enum

class DFSState(Enum):
    """
    States during DFS
    """
    UNVISITED = 'unvisited'
    VISITING = 'visiting'
    VISITED = 'visited'

