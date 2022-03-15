# ************
# File: activations.py
# Top contributors (to current version): 
# 	Panagiotis Kouvaros (panagiotis.kouvaros@gmail.com)
# This file is part of the Venus project.
# Copyright: 2019-2021 by the authors listed in the AUTHORS file in the
# top-level directory.
# License: BSD 2-Clause (see the file LICENSE in the top-level directory).
# Description: Enum classes for activation functions
# ************

from enum import Enum

class Activations(Enum):
    relu = 0
    linear = 1

class ReluApproximation(Enum):
    ZERO = 0
    IDENTITY = 1
    PARALLEL = 2
    MIN_AREA = 3
    VENUS_HEURISTIC = 4
    OPT_HEURISTIC = 5

class ReluState(Enum):

    ACTIVE = 0
    INACTIVE = 1
    UNSTABLE = 2

    @staticmethod
    def inverse(s):
        """
        Inverts a given relu state.

        Arguments:

            s: ReluState Item

        Returns

            ReluState Item
        """
        if s == ReluState.INACTIVE:
            return ReluState.ACTIVE
        elif s == ReluState.ACTIVE:
            return ReluState.INACTIVE
        else:
            return None
