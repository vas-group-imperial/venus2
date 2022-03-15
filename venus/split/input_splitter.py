"""
# File: input_splitter.py
# Top contributors (to current version):
# 	Panagiotis Kouvaros (panagiotis.kouvaros@gmail.com)
# This file is part of the Venus project.
# Copyright: 2019-2021 by the authors listed in the AUTHORS file in the
# top-level directory.
# License: BSD 2-Clause (see the file LICENSE in the top-level directory).
# Description: Splits a verification problem into subproblems resulting from
# bisecting an input node.
"""

import random

from venus.verification.verification_problem import VerificationProblem
from venus.network.layers import Input

class InputSplitter:
    """
    Bisects the input domain of a verification problem.
    """

    def __init__(self, prob, init_st_ratio, config):
        """
        Arguments:

            prob: VerificationProblem to split.

            init_st_ratio:
                double of the stability ratio of the original
                VerificationProblem.
            config:
                configuration.
        """

        self.prob = prob
        self.init_st_ratio = init_st_ratio
        self.config = config

    def split(self):
        """
        Splits the verification problem  into a pair of subproblems. Splitting
        is via bisection of the input dimension leading to the highest
        stability ratio for the subproblems. The worthness of the split is
        evaluated.

        Returns:

            list of VerificationProblem
        """
        if self.prob.stability_ratio < self.config.SPLITTER.STABILITY_RATIO_CUTOFF:
            subprobs = self.soft_split()
            ws = self.prob.worth_split(subprobs, self.init_st_ratio)
            if ws == True:
                return subprobs
        return []


    def soft_split(self, prob=None):
        """
        Splits the verification problem  into a pair of subproblems. Splitting
        is via bisection of the input dimension leading to the highest
        stability ratio for the subproblems. The worthness of the split is NOT
        evaluated.

        Returns:

            list of VerificationProblem
        """
        if prob is None: prob = self.prob
        size = prob.spec.input_layer.input_size
        best_ratio = 0
        best_prob1 = None
        best_prob2 = None

        if size < self.config.SPLITTER.SMALL_N_INPUT_DIMENSIONS:
            #If the number of input dimensions is not big, choose the best
            #dimension to split
            for dim in range(size):
                prob1, prob2 = self.split_dimension(prob, dim)
                ratio = (prob1.stability_ratio + prob2.stability_ratio)/2
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_prob1 = prob1
                    best_prob2 = prob2
        else:
            #Otherwise split randomly
            dim = random.randint(0, size - 1)
            best_prob1, best_prob2 =  self.split_dimension(prob, dim)

        return [best_prob1, best_prob2]


    def split_dimension(self, prob, dim):
        """
        Bisects the given input dimension to generate two subproblems of the
        given verification problem.

        Arguments:

            prob:
                VerificationProblem to split.
            dim:
                int of the input dimension to bisect

        Returns:

            pair of VerificationProblem
        """
        l = prob.spec.input_layer.pre_bounds.lower
        u = prob.spec.input_layer.pre_bounds.upper
        split_point = (l[dim] + u[dim]) / 2
        l1 = l.copy()
        l1[dim] = split_point
        prob1 = VerificationProblem(
            prob.nn.copy(),
            prob.spec.copy(Input(l1,u)),
            prob.depth + 1,
            self.config
        )
        prob1.bound_analysis()
        u2 = u.copy()
        u2[dim] = split_point
        prob2 = VerificationProblem(
            prob.nn.copy(),
            prob.spec.copy(Input(l,u2)),
            prob.depth + 1,
            self.config
        )
        prob2.bound_analysis()

        return prob1, prob2


    def split_up_to_depth(self, split_depth_cutoff):
        """
        Recursively splits the verification problem until a given splitting
        depth is reached. Splitting is via bisection of the input dimension
        leading to the highest stability ratio for the subproblems. The
        worthness of each split is NOT evaluated.

        Arguments:

            split_depth_cutoff:
                int of the splitting depth up to which the verification problem
                will be split.

        Returns:

            list of VerificationProblem
        """
        split_depth = 0
        probs = [self.prob]
        while split_depth < split_depth_cutoff:
            subprobs = []
            for p in probs:
                subprobs.extend(self.soft_split(p))
            split_depth += 1
            probs = subprobs

        return probs

