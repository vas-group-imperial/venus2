# ************
# File: cuts.py
# Top contributors (to current version): 
# 	Panagiotis Kouvaros (panagiotis.kouvaros@gmail.com)
# This file is part of the Venus project.
# Copyright: 2019-2021 by the authors listed in the AUTHORS file in the
# top-level directory.
# License: BSD 2-Clause (see the file LICENSE in the top-level directory).
# Description: Base class for IdealFormulation and DepCuts.
# ************

from gurobipy import *
import numpy as np
import math

class Cuts:

    def __init__(self, prob, gmodel, freq, config):
        """
        Arguments:

            prob:
                VerificationProblem.
            gmodel:
                Gurobi model.
            freq:
                float, cuts are added every 1 in pow(milp_nodes_solved, freq).
            config:
                Configuration.
        """
        self.prob = prob
        self.gmodel = gmodel
        self.freq = freq
        self.config = config

    def freq_check(self, depth=1):
        """
        Cuts are only added as per a required frequency constant. Given the
        number nodcnt of MIP nodes solved so far the probability of adding a
        cut is 1 in nodcnt*freq_const*depth where freq_const if the frequency
        constant. Adding the cuts at every callback call slows down the solver.
        
        Arguments:
            
            depth: int of the depth of the layer for which cuts are to be added.

        Returns:
            
            bool expressing whether or not to add cuts.
        """
        freq = math.ceil(pow(GRB.Callback.MIPNODE_NODCNT + 1, self.freq)) * depth
        rnd = np.random.randint(0, freq, 1)

        return True if rnd == 0 else False
