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
from venus.specification.formula import  * 

torch.set_num_threads(1)

class SIP:

    logger = None

    def __init__(self, prob, config: Config, delta_flags=None):
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

        self._lower_eq, self._upper_eq = None, None
        
    def set_bounds(self):
        """
        Sets the bounds.
        """
        start = timer()

        # get relaxation slopes
        if self.prob.nn.has_custom_relaxation_slope() is True:
            slopes = self.get_lower_relaxation_slopes(gradient=False)
        else:
            slopes = None

        # set bounds using area approximations
        self._set_bounds(slopes, depth=1)

        # optimise the relaxation slopes using pgd
        if self.config.SIP.SLOPE_OPTIMISATION is True and \
        self.prob.nn.has_custom_relaxation_slope() is not True:
            slopes = self.optimise_slopes(self.prob.nn.tail)
            self.prob.nn.relu_relaxation_slopes = slopes
            starting_depth = self.prob.nn.get_non_linear_starting_depth()
            self._set_bounds(slopes, depth=starting_depth)

        print('\n*', torch.mean(self.prob.nn.tail.bounds.upper), torch.mean(self.prob.nn.tail.bounds.lower), timer() - start)
        print(self.prob.nn.tail.bounds.lower)
        print(self.prob.nn.tail.bounds.upper)

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
        self, slopes: dict=None, delta_flags: torch.Tensor=None, depth=1
    ):
        """
        Sets the bounds.
        """
        ibp = IBP(self.prob, self.config)
        os_sip = OSSIP(self.prob, self.config)
        bs_sip = BSSIP(self.prob, self.config)
        print(self.prob.spec.input_node.bounds.lower)
        for i in range(depth, self.prob.nn.tail.depth + 1):
            nodes = self.prob.nn.get_node_by_depth(i)
            for j in nodes:
                delta = self._get_delta_for_node(j, delta_flags)
                lower_slopes, upper_slopes = self._get_slopes_for_node(j, slopes)

                # set interval propagation bounds
                ia_count = ibp.set_bounds(j, lower_slopes, upper_slopes, delta)

                # check eligibility for symbolic equations
                symb_elg = self.is_symb_eq_eligible(j)


                # set one step symbolic bounds
                if self.config.SIP.ONE_STEP_SYMBOLIC is True:
                    os_sip.forward(j, lower_slopes, upper_slopes)

                    if symb_elg is True:
                        os_count = os_sip.set_bounds(j)
 
                # recheck eligibility for symbolic equations
                symb_elg = self.is_symb_eq_eligible(j)
 
                # set back substitution symbolic bounds
                if self.config.SIP.SYMBOLIC is True and symb_elg is True:
                    if self.config.SIP.ONE_STEP_SYMBOLIC is True and \
                    self.config.SIP.EQ_CONCRETISATION is True and \
                    i > 2:
                        if os_count * 4 > ia_count:
                            concretisations = os_sip
                        else:
                            os_sip.clear_equations()
                            concretisations = None
                    else:
                        concretisations = None

                    bs_count = bs_sip.set_bounds(
                        j, lower_slopes, upper_slopes, concretisations
                    )


    def _get_delta_for_node(self, node: Node, delta_flags: torch.Tensor) -> torch.Tensor:
        if delta_flags is not None and node.has_relu_activation() is True:
            return delta_flags[j.to_node[0].id]
            
        return None

    def _get_slopes_for_node(self, node: Node, slopes: tuple[dict]) -> torch.Tensor:
        if slopes is None:
            return None, None
                    
        return slopes[0][node.id], slopes[1][node.id]

    def is_symb_eq_eligible(self, node: None) -> bool:
        """
        Determines whether the node implements function requiring a symbolic
        equation for bound calculation.
        """ 
        if node is self.prob.nn.tail:
            return True

        if node is self.prob.nn.head:
            return False

        non_eligible = [
            BatchNormalization, Input, Relu, MaxPool, Reshape, Flatten, Slice, Concat
        ]
        if type(node) in non_eligible:
            return False

        if node.has_relu_activation() and node.to_node[0].get_unstable_count() > 0:
            return True

        if node.has_max_pool():
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
        if isinstance(formula, VarVarConstraint):
            coeffs = torch.zeros(
                (1, self.prob.nn.tail.output_size),
                dtype=self.config.PRECISION,
                device=self.config.DEVICE
            )
            const = torch.tensor(
                [0], dtype=self.config.PRECISION, device=self.config.DEVICE
            )
            if formula.sense == Formula.Sense.GT:
                coeffs[0, formula.op1.i], coeffs[0, formula.op2.i] = 1, -1
            elif formula.sense == Formula.Sense.LT:
                coeffs[0, formula.op1.i], coeffs[0, formula.op2.i] = -1, 1
            else:
                raise ValueError('Formula sense {formula.sense} not expected')
            equation = Equation(coeffs, const, self.config)
            diff = self.back_substitution(
                equation, self.prob.nn.tail, 'lower'
            )

            return formula if diff <= 0 else TrueFormula()

        if isinstance(formula, DisjFormula):
            fleft = self.simplify_formula(formula.left)
            fright = self.simplify_formula(formula.right)

            return DisjFormula(fleft, fright)

        if isinstance(formula, ConjFormula):
            fleft = self.simplify_formula(formula.left)
            fright = self.simplify_formula(formula.right)

            return ConjFormula(fleft, fright)

        if isinstance(formula, NAryDisjFormula):
            clauses = [self.simplify_formula(f) for f in formula.clauses]

            return NAryDisjFormula(clauses)

        if isinstance(formula, NAryConjFormula):
            clauses = [self.simplify_formula(f) for f in formula.clauses]

            return NAryConjFormula(clauses)

        return formula
    
    def optimise_slopes(self, node: Node):
        lower_slopes, upper_slopes = self.get_lower_relaxation_slopes(gradient=True)
        lower_slopes =  self._optimise_slopes(node, 'lower', lower_slopes)
        upper_slopes =  self._optimise_slopes(node, 'upper', upper_slopes)

        return (lower_slopes, upper_slopes)


    def _optimise_slopes(self, node: Node, bound: str, slopes: torch.tensor):
        in_flag = self._get_stability_flag(node.from_node[0])
        in_flag = None if in_flag is None else in_flag.flatten()
        out_flag = self._get_out_prop_flag(node)
        if out_flag is None:
            out_flag = torch.ones(
                node.output_size, dtype=torch.bool, device=self.config.DEVICE
            )
        else:
            out_flag = out_flag.flatten(),
        equation = Equation.derive(
            node, self.config, out_flag, in_flag
        )

        best_slopes = slopes
        if bound == 'lower':
            lr = -self.config.SIP.GD_LR
            current_mean, best_mean = -math.inf, torch.mean(node.bounds.lower)
        else:
            lr = self.config.SIP.GD_LR
            current_mean, best_mean = math.inf, torch.mean(node.bounds.upper)
        
        for i in range(self.config.SIP.GD_STEPS):
            bounds = self.back_substitution(
                equation, node.from_node[0], bound, slopes=slopes
            ) 
            bounds = torch.mean(bounds)
            if bound == 'lower' and bounds.item() > current_mean and bounds.item() - current_mean < 10e-3:
                return slopes
            elif bound == 'upper' and bounds.item() < current_mean and current_mean - bounds.item() < 10e-3:
                return slopes
            current_mean = bounds.item()

            bounds.backward()
 
            if (bound == 'lower' and bounds.item() >= best_mean) or \
            (bound == 'upper' and bounds.item() <= best_mean):
                best_mean = bounds.item()
                best_slopes = {i: slopes[i].detach().clone() for i in slopes}

            for j in slopes:
                if slopes[j].grad is not None:
                    step = lr * (slopes[j].grad.data / torch.mean(slopes[j].grad.data))

                    if bound == 'upper':
                        slopes[j].data -= step + torch.mean(slopes[j].data)
                    else:
                        slopes[j].data += step - torch.mean(slopes[j].data)

                    slopes[j].data = torch.clamp(slopes[j].data, 0, 1)

            if (bound == 'lower' and bounds.item() >= best_mean) or \
            (bound == 'upper' and bounds.item() <= best_mean):
                best_mean = bounds.item()
                best_slopes = {i: slopes[i].detach().clone() for i in slopes}

        self.prob.nn.detach()
                
        return slopes

    def get_lower_relaxation_slopes(self, gradient=False):
        lower, upper = {}, {}
        for _, i in self.prob.nn.node.items():
            if isinstance(i, Relu):
                slope = i.get_lower_relaxation_slope()
                if slope[0] is not None:
                    lower[i.id] = slope[0].detach().clone().requires_grad_(gradient)
                if slope[1] is not None:
                    upper[i.id] = slope[1].detach().clone().requires_grad_(gradient)

        return lower, upper
