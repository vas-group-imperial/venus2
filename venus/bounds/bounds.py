# ************
# File: bounds.py
# Top contributors (to current version): 
# 	Panagiotis Kouvaros (panagiotis.kouvaros@gmail.com)
# This file is part of the Venus  project.
# Copyright: 2019-2021 by the authors listed in the AUTHORS file in the
# top-level directory.
# License: BSD 2-Clause (see the file LICENSE in the top-level directory).
# Description: class for bounds of nodes.
# ************

import numpy as np

class Bounds:
    def __init__(self, lower=None, upper=None):
        self.lower = lower
        self.upper = upper

    def normalise(self, mean, std):
        """
        Normalises the bounds

        Arguments:
            mean:
                normalisation mean
            std:
                normalisation standard deviation
        Returns
            None
        """
        self.lower = ( self.lower - mean ) / std
        self.upper = ( self.upper - mean ) / std

    def clip(self, min_value, max_value):
        """
        Clips the  bounds

        Arguments:
            min_value:
                valid lower bound
            max_value:
                valid upper bound
        Returns:
            None
        """
        self.lower = np.clip(self.lower, min_value, max_value) 
        self.upper = np.clip(self.upper, min_value, max_value) 

    def get_range(self):
        """
        Returns the range of the bounds.
        """
        return self.upper - self.lower

    def copy(self):
        """
        Returns:
            a copy of the calling bounds object
        """
        return Bounds(self.lower.copy(), self.upper.copy())



