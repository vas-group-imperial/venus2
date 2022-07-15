"""
# File: sip.py
# Top contributors (to current version):
# 	Panagiotis Kouvaros (panagiotis.kouvaros@gmail.com)
# This file is part of the Venus project.
# Copyright: 2019-2021 by the authors listed in the AUTHORS file in the
# top-level directory.
# License: BSD 2-Clause (see the file LICENSE in the top-level directory).
# Description: Symbolic Interval Propagation.
,"""

import math
import numpy as np
import torch
from timeit import default_timer as timer
from venus.network.node import *
from venus.bounds.bounds import Bounds
from venus.bounds.equation import Equation
from venus.bounds.ibp import IBP
from venus.bounds.os_sip import OSSIP
from venus.bounds.bs_sip import BSSIP
from venus.common.logger import get_logger
from venus.common.configuration import Config

torch.set_num_threads(1)

class SIP:

    logger = None

    NON_SYMBOLIC_NODES = [Mul, Div, ReduceSum]

    def __init__(self, prob, config: Config, delta_flags=None, batch=False):
        """
        Arguments:

            prob:
                The verification problem.
            config:
                Configuration.
        """
        self.prob = prob
        self.config = config
        self.delta_flags = delta_flags
        if SIP.logger is None and config.LOGGER.LOGFILE is not None:
            SIP.logger = get_logger(__name__, config.LOGGER.LOGFILE)

        self.ibp = IBP(self.prob, self.config)
        self.os_sip = OSSIP(self.prob, self.config)
        self.bs_sip = BSSIP(self.prob, self.config)
       
    def init(self):
        self.ibp = IBP(self.prob, self.config)
        self.os_sip = OSSIP(self.prob, self.config)
        self.bs_sip = BSSIP(self.prob, self.config)

    def set_bounds(self):
        """
        Sets the bounds.
        """
        start = timer()
        if self.config.DEVICE == torch.device('cuda'):
            self.prob.cuda()

        # get relaxation slopes
        if self.prob.nn.has_custom_relaxation_slope() is True:
            slopes = self.prob.nn.get_lower_relaxation_slopes(gradient=False)
        else:
            slopes = None

        # set bounds using area approximations
        self._set_bounds(slopes, depth=1)
 
        # optimise the relaxation slopes using pgd
        if self.config.SIP.SLOPE_OPTIMISATION is True and \
        self.prob.nn.has_custom_relaxation_slope() is not True \
        and self.prob.spec.is_satisfied(
            self.prob.nn.tail.bounds.lower,
            self.prob.nn.tail.bounds.upper
        ) is not True:
            bs_sip = BSSIP(self.prob, self.config)
            slopes = bs_sip.optimise(self.prob.nn.tail)
            self.init()
            self._set_bounds(slopes=slopes, depth=1)
            # this should be after _set_bounds as the unstable nodes may change
            # - to refactor
            self.prob.nn.set_lower_relaxation_slopes(slopes[0], slopes[1])

        # print(self.prob.nn.tail.bounds.lower)
        # print(self.prob.nn.tail.bounds.upper)

        if self.config.SIP.SIMPLIFY_FORMULA is True \
        and self.config.SIP.SYMBOLIC is True:
            self.prob.spec.output_formula = self.simplify_formula(
                self.prob.spec.output_formula
            )

        if self.config.DEVICE == torch.device('cuda'):
            self.prob.cpu()

        if self.logger is not None:
            SIP.logger.info(
                'Bounds computed, time: {:.3f}, range: {:.3f}, '.format(
                    timer()-start,
                    torch.mean(
                        self.prob.nn.tail.bounds.upper - self.prob.nn.tail.bounds.lower
                    )
                )
            )

    def _set_bounds(
        self, slopes: tuple=None, delta_flags: torch.Tensor=None, depth=1
    ):
        """
        Sets the bounds.
        """
        for i in range(depth, self.prob.nn.tail.depth + 1):
            nodes = self.prob.nn.get_node_by_depth(i)
            for j in nodes:
                delta = self._get_delta_for_node(j, delta_flags)
                self._set_bounds_for_node(j, slopes, delta)
 
    def _set_bounds_for_node(
        self, node: Node, slopes: tuple=None, delta_flag: torch.Tensor=None
    ):
        # print(node, node.id, node.input_shape, node.output_shape)

        # set interval propagation bounds
        bounds = self.ibp.calc_bounds(node)
        slopes = self._update_bounds(node, bounds, slopes, delta_flag)
        if node.has_relu_activation():
            ia_count = node.to_node[0].get_unstable_count()
            
        # check eligibility for symbolic equations
        symb_elg = self.is_symb_eq_eligible(node)

        # set one step symbolic bounds
        if self.config.SIP.ONE_STEP_SYMBOLIC is True and \
        node.is_non_symbolically_connected() is not True:
            self.os_sip.forward(node, slopes)
            if symb_elg is True:
                bounds =  self.os_sip.calc_bounds(node)
                slopes = self._update_bounds(node, bounds, slopes, delta_flag)
                if node.has_fwd_relu_activation():
                    os_count = node.to_node[0].get_unstable_count()
 
        # recheck eligibility for symbolic equations
        non_linear_depth = self.prob.nn.get_non_linear_starting_depth()
        symb_elg = self.is_symb_eq_eligible(node) and node.depth >= non_linear_depth
 
        # set back substitution symbolic bounds
        if self.config.SIP.SYMBOLIC is True and symb_elg is True:
            if self.config.SIP.ONE_STEP_SYMBOLIC is True and \
            self.config.SIP.EQ_CONCRETISATION is True and \
            node.depth > 2:
                if node.has_fwd_relu_activation() and os_count * 5 > ia_count:
                    concretisations = self.os_sip
                else:
                    self.os_sip.clear_equations()
                    concretisations = None
            else:
                concretisations = None

            bounds, flag = self.bs_sip.calc_bounds(
                node, slopes, concretisations
            )
            slopes = self._update_bounds(node, bounds, slopes=slopes, out_flag=flag)


    def _update_bounds(
        self,
        node: Node,
        bounds: Bounds,
        slopes: tuple=None,
        delta_flags: torch.Tensor=None,
        out_flag: torch.Tensor=None
    ):
        if node.has_relu_activation():
            if slopes is not None:
                # relu node with custom slopes - leave slopes as are but remove
                # slopes from newly stable nodes.
                old_fl = node.get_next_relu().get_unstable_flag()
           
                if out_flag is None:
                    bounds = Bounds(
                        torch.max(node.bounds.lower, bounds.lower),
                        torch.min(node.bounds.upper, bounds.upper)
                    )

                    if delta_flags is not None:
                        bounds.lower[delta_flags[0]] = 0.0
                        bounds.upper[delta_flags[0]] = 0.0
        
                        bounds.lower[delta_flags[1]] = torch.clamp(
                            bounds.lower[delta_flags[1]], 0.0, math.inf
                        )
                        bounds.upper[delta_flags[1]] = torch.clamp(
                            bounds.upper[delta_flags[1]], 0.0, math.inf
                        )

                    new_fl = torch.logical_and(
                        bounds.lower[old_fl]  < 0, bounds.upper[old_fl] > 0
                    )

                    slopes[0][node.get_next_relu().id] = \
                        slopes[0][node.get_next_relu().id][new_fl]
                    slopes[1][node.get_next_relu().id] = \
                        slopes[1][node.get_next_relu().id][new_fl]

                else:
                    bounds = Bounds(
                        torch.max(node.bounds.lower[out_flag], bounds.lower),
                        torch.min(node.bounds.upper[out_flag], bounds.upper)
                    )

                    if delta_flags is not None:
                        bounds.lower[delta_flags[0][out_flag]] = 0.0
                        bounds.upper[delta_flags[0][out_flag]] = 0.0
        
                        bounds.lower[delta_flags[1][out_flag]] = torch.clamp(
                            bounds.lower[delta_flags[1][out_flag]], 0.0, math.inf
                        )
                        bounds.upper[delta_flags[1][out_flag]] = torch.clamp(
                            bounds.upper[delta_flags[1][out_flag]], 0.0, math.inf
                        )

                    new_fl = torch.logical_and(
                        bounds.lower  < 0, bounds.upper > 0
                    )
                    idxs = torch.zeros_like(old_fl, dtype=torch.bool)
                    idxs[out_flag] = new_fl
                    idxs = idxs[old_fl]

                    slopes[0][node.get_next_relu().id] = \
                        slopes[0][node.get_next_relu().id][idxs]
                    slopes[1][node.get_next_relu().id] = \
                        slopes[1][node.get_next_relu().id][idxs]
            
        if slopes is not None and node.has_relu_activation():
            node_slopes = [
                slopes[0][node.get_next_relu().id], slopes[1][node.get_next_relu().id]
            ]
        else:
            node_slopes = None

        node.update_bounds(bounds, node_slopes, out_flag)

        return slopes

    def _get_delta_for_node(self, node: Node, delta_flags: torch.Tensor) -> torch.Tensor:
        if delta_flags is not None and node.has_relu_activation() is True:
            return delta_flags[node.to_node[0].id]
            
        return None

    def _get_slopes_for_node(self, node: Node, slopes: tuple[dict]) -> torch.Tensor:
        if slopes is None or node.has_relu_activation() is not True:
            return None, None

        return slopes[0][node.to_node[0].id], slopes[1][node.to_node[0].id]

    def is_symb_eq_eligible(self, node: None) -> bool:
        """
        Determines whether the node implements function requiring a symbolic
        equation for bound calculation.
        """ 
        if node.depth <= 1 or node.is_non_symbolically_connected() is True:
            return False

        if node is self.prob.nn.tail:
            return True

        non_eligible = [
            Input, Relu, Reshape, Flatten, Slice
        ]
        if type(node) in non_eligible or np.any([node is i for i in self.prob.nn.head]):
            return False

        if node.has_relu_activation() and node.to_node[0].get_unstable_count() > 0:
            return True
    
        return False


    def runtime_bounds(self, eqlow_r, equp_r, eq, lb_r, ub_r, lb, ub, relu_states):
        """
        Modifies the given bound equations and bounds so that they are in line
        with the MILP relu states of the nodes.

        Arguments:

            eqlow: lower bound equations of a layer's output.

            equp: upper bound equations of a layer's output.

            lbounds: the concrete lower bounds of eqlow.

            ubounds: the concrete upper bounds of equp.

            relu_states: a list of floats, one for each node. 0: inactive,
            1:active, anything else: unstable.

        Returns:

            a four-typle of the lower and upper bound equations and the
            concrete lower and upper bounds resulting from stablising nodes as
            per relu_states.
        """
        if relu_states is None:
            return [eqlow_r, equp_r], [lb_r, ub_r]
        eqlow_r.coeffs[(relu_states==0),:] = 0
        eqlow_r.const[(relu_states==0)] = 0
        equp_r.coeffs[(relu_states==0),:] = 0
        equp_r.const[(relu_states==0)] = 0
        lb_r[(relu_states == 0)] = 0
        ub_r[(relu_states == 0)] = 0
        eqlow_r.coeffs[(relu_states==1),:] = eq.coeffs[(relu_states==1),:]
        eqlow_r.const[(relu_states==1)] = eq.const[(relu_states==1)]
        equp_r.coeffs[(relu_states==1),:] = eq.coeffs[(relu_states==1),:]
        equp_r.const[(relu_states==1)] = eq.const[(relu_states==1)]
        lb_r[(relu_states == 0)] = lb[(relu_states == 0)]
        ub_r[(relu_states == 0)] = ub[(relu_states == 0)]

        return [eqlow_r, equp_r], [lb_r, ub_r]


    def branching_bounds(self, relu_states, bounds):
        """
        Modifies the given  bounds so that they are in line with the relu
        states of the nodes as per the branching procedure.

        Arguments:
            
            relu_states:
                Relu states of a layer
            bounds:
                Concrete bounds of the layer
            
        Returns:
            
            a pair of the lower and upper bounds resulting from stablising
            nodes as per the relu states.
        """
        shape = relu_states.shape
        relu_states = relu_states.reshape(-1)
        lbounds = bounds[0]
        ubounds = bounds[1]
        _max = np.clip(lbounds, 0, math.inf)
        _min = np.clip(ubounds, -math.inf, 0)
        lbounds[(relu_states == ReluState.ACTIVE)] = _max[(relu_states == ReluState.ACTIVE)]
        ubounds[(relu_states == ReluState.INACTIVE)] = _min[(relu_states == ReluState.INACTIVE)]

        return lbounds, ubounds

    def simplify_formula(self, formula):
        if self.config.SIP.SYMBOLIC is True:
            simp_formula = self.bs_sip.simplify_formula(formula)
        else:
            simp_formula = formula

        return formula
