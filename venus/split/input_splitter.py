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
from venus.network.node import Input
from venus.bounds.bounds import Bounds


class InputSplitter:
    """
    Bisects the input domain of a verification problem.
    """

    def __init__(self, prob, init_st_ratio, config):
        """
        Arguments:

            prob:
                The VerificationProblem.
            init_st_ratio:
                Dtability ratio of the original VerificationProblem.
            config:
                Configuration.
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
        if self.prob.stability_ratio <= self.config.SPLITTER.STABILITY_RATIO_CUTOFF:
            try:
                subprobs = self.soft_split()
            except Exception as error:
                return []

            worth = self.prob.worth_split(subprobs, self.init_st_ratio)
            if worth is True:
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
        if prob is None:
            prob = self.prob

        best_ratio = 0
        best_prob1 = None
        best_prob2 = None

        if prob.spec.input_node.output_size < self.config.SPLITTER.SMALL_N_INPUT_DIMENSIONS:
            # If the number of input dimensions is not big, choose the best
            # dimension to split
            for dim in prob.spec.input_node.get_outputs():
                try:
                    prob1, prob2 = self.split_dimension(prob, dim)
                except Exception as error:
                    raise error
                ratio = (prob1.stability_ratio + prob2.stability_ratio) / 2
                if ratio >= best_ratio:
                    best_ratio = ratio
                    best_prob1 = prob1
                    best_prob2 = prob2
        else:
            # Otherwise split randomly
            dim = random.choice(prob.spec.input_node.get_outputs())
            try:
                best_prob1, best_prob2 =  self.split_dimension(prob, dim)
            except Exception as error:
                raise error

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
        l = prob.spec.input_node.bounds.lower.detach().clone()
        u = prob.spec.input_node.bounds.upper.detach().clone()
        split_point = (l[dim].item() + u[dim].item()) / 2

        l[dim] = split_point
        prob1 = VerificationProblem(
            prob.nn.copy(),
            prob.spec.copy(
                Input(Bounds(l, u), self.config)
            ),
            prob.depth + 1,
            self.config
        )
        try:
            prob1.bound_analysis()
            prob1.detach()
        except Exception as error:
            return None, None

        u[dim] = split_point
        prob2 = VerificationProblem(
            prob.nn.copy(),
            prob.spec.copy(
                Input(Bounds(l, u), self.config)
            ),
            prob.depth + 1,
            self.config
        )
        try:
            prob2.bound_analysis()
            prob2.detach()
        except Exception as error:
            return None, None

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
                splits = []
                try:
                    splits = self.soft_split(p)
                except Exception as error:
                    splits = []
                subprobs.extend(splits)
            split_depth += 1
            probs = subprobs

        return probs

    def worth_split(self, subprobs):
        pscore0 = self.score(self.init_st_ratio)
        pscore1 = subprobs[0].score()
        pscore2 = subprobs[1].score()

        _max = max(pscore1, pscore2)
        _min = min(pscore1, pscore2) 

        if pscore0 >= _max:
            return False
        elif _min > pscore0:
            return True
        elif  (pscore1 + pscore2)/2 > pscore0:
            return True
        else:
            return False
 
    def score(self, prob):
        
        return (prob.stability_ratio - self.init_st_ratio)
 
