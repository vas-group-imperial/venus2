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

import sys
import linecache
from enum import Enum

class DFSState(Enum):
    """
    States during DFS
    """
    UNVISITED = 'unvisited'
    VISITING = 'visiting'
    VISITED = 'visited'

class ReluState(Enum):

    ACTIVE = 0
    INACTIVE = 1
    UNSTABLE = 2

    @staticmethod
    def inverse(s):
        """
        Inverts a given relu state.

        Arguments:

            s:
                ReluState Item

        Returns

            ReluState Item
        """
        if s == ReluState.INACTIVE:
            return ReluState.ACTIVE

        elif s == ReluState.ACTIVE:
            return ReluState.INACTIVE

        return None

class ReluApproximation(Enum):
    ZERO = 0
    IDENTITY = 1
    PARALLEL = 2
    MIN_AREA = 3
    VENUS = 4

class OSIPMode(Enum):
    """
    Modes of operation.
    """
    OFF = 0
    ON = 1
    SPLIT = 2
