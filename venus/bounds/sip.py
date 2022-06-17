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
import venus
from venus.network.node import *
from venus.bounds.bounds import Bounds
from venus.bounds.equation import Equation
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
        # sets bounds using minimum area approximations
        self._set_bounds(slopes=slopes)
        # optimises the relaxation slopes using gd
        if self.config.SIP.SLOPE_OPTIMISATION is True and \
        self.prob.nn.has_custom_relaxation_slope() is not True:
            slopes = self.optimise_slopes(self.prob.nn.tail)
            self.prob.nn.relu_relaxation_slopes = slopes
            # sets bounds using optimised slopes
            self._set_bounds(slopes)

        # print('\n*', torch.mean(self.prob.nn.tail.bounds.upper), torch.mean(self.prob.nn.tail.bounds.lower))

        if self.logger is not None:
            SIP.logger.info(
                'Bounds computed, time: {:.3f}, range: {:.3f}, '.format(
                    timer()-start,
                    torch.mean(
                        self.prob.nn.tail.bounds.upper - self.prob.nn.tail.bounds.lower
                    )
                )
            )

    def _set_bounds(self, slopes: tuple[dict]=None, delta_flags: torch.tensor=None):
        """
        Sets the bounds.
        """
        saw_symb_eq_node = False
        for i in range(self.prob.nn.tail.depth):
            nodes = self.prob.nn.get_node_by_depth(i)
            for j in nodes:
                if j.bounds.size() > 0 and saw_symb_eq_node is not True:
                    continue

                delta = delta_flags[j.to_node[0]] if delta_flags is not None and j.has_relu_activation() else None

                self.set_ia_bounds(j, slopes=slopes, delta_flags=delta)
 
                if self.is_symb_eq_eligible(j) is not True:
                    continue

                if saw_symb_eq_node is not True:
                    saw_symb_eq_node = True
                    continue

                self.set_symb_concr_bounds(j, slopes=slopes) 

                # print(j.id, torch.mean(j.bounds.lower), torch.mean(j.bounds.upper))

        # self.set_ia_bounds(self.prob.nn.tail, slopes=slopes)
        self.prob.nn.tail.bounds = Bounds(
            torch.ones(self.prob.nn.tail.output_shape) * -math.inf,
            torch.ones(self.prob.nn.tail.output_shape) * math.inf,
        )
        self.set_symb_concr_bounds(self.prob.nn.tail, slopes)

        # print(
            # self.prob.nn.tail.id,
            # torch.mean(self.prob.nn.tail.bounds.lower[self._get_out_prop_flag(self.prob.nn.tail)]),
            # torch.mean(self.prob.nn.tail.bounds.upper[self._get_out_prop_flag(self.prob.nn.tail)])
        # )
                                                      

    def is_symb_eq_eligible(self, node: None) -> bool:
        """
        Determines whether the node implements function requiring a symbolic
        equation for bound calculation.
        """ 
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


    def set_ia_bounds(self, node: Node, slopes: tuple[dict]=None, delta_flags: torch.tensor=None) -> list:
        inp = node.from_node[0].bounds

        if type(node) in [Relu, BatchNormalization, MaxPool, Slice, Unsqueeze]:
            lower, upper = node.forward(inp.lower), node.forward(inp.upper)

        elif isinstance(node, Flatten):
            lower, upper = inp.lower, inp.upper

        elif isinstance(node, Concat):
            inp_lower = [i.bounds.lower for i in node.from_node]
            inp_upper = [i.bounds.upper for i in node.from_node]
            lower, upper = node.forward(inp_lower), node.forward(inp_upper)

        elif isinstance(node, Sub):
            const_lower = node.const if node.const is not None else node.from_node[1].bounds.lower
            const_upper = node.const if node.const is not None else node.from_node[1].bounds.upper
            lower, upper = inp.lower - const_upper, inp.upper - const_lower

        elif isinstance(node, Add):
            const_lower = node.const if node.const is not None else node.from_node[1].bounds.lower
            const_upper = node.const if node.const is not None else node.from_node[1].bounds.upper
            lower, upper = inp.lower + const_lower, inp.upper + const_upper

        elif type(node) in [Gemm, Conv, ConvTranspose]:
            lower = node.forward(inp.lower, clip='+', add_bias=False) + \
                node.forward(inp.upper, clip='-', add_bias=True)
            upper = node.forward(inp.lower, clip='-', add_bias=False) + \
                node.forward(inp.upper, clip='+', add_bias=True)
 
        elif isinstance(node, MatMul):
            lower = node.forward(inp.lower, clip='+') + node.forward(inp.upper, clip='-')
            upper = node.forward(inp.lower, clip='-') + node.forward(inp.upper, clip='+')

        else:
            raise TypeError(f"IA Bounds computation for {type(node)} is not supported")
        
        if self.delta_flags is not None and node.has_relu_activation() is True:
            lower[self.delta_flags[node.to_node[0].id][0]] = 0
            upper[self.delta_flags[node.to_node[0].id][0]] = 0
            lower[self.delta_flags[node.to_node[0].id][1]] = np.clip(
                lower[self.delta_flags[node.to_node[0].id][1]],
                0,
                math.inf
            )
            upper[self.delta_flags[node.to_node[0].id][1]] = np.clip(
                upper[self.delta_flags[node.to_node[0].id][1]],
                0,
                math.inf
            )

        ia_bounds = Bounds(
            lower.reshape(node.output_shape), upper.reshape(node.output_shape)
        )

        self.update_node_ia_bounds(node, ia_bounds, slopes=slopes, delta_flags=delta_flags)


    def update_node_ia_bounds(
        self,
        node: None,
        bounds: Bounds,
        slopes: tuple[dict]=None,
        out_flag: torch.tensor=None,
        delta_flags: torch.tensor=None
    ):
        if node.has_relu_activation() and slopes is not None and node.bounds.size() > 0:
            # relu node with custom slope - leave slope as is but remove from
            # it newly stable nodes.
            old_un_flag = node.to_node[0].get_unstable_flag()
            
            bounds = Bounds(
                torch.max(node.bounds.lower, bounds.lower),
                torch.min(node.bounds.upper, bounds.upper)
            )
            if delta_flags is not None:
                bounds.lower[delta_flags[0]] = 0.0
                bounds.upper[delta_flags[0]] = 0.0

            new_un_flag = torch.logical_and(
                bounds.lower[old_un_flag]  < 0, bounds.upper[old_un_flag] > 0
            )

            slopes[0][node.to_node[0].id] = slopes[0][node.to_node[0].id][new_un_flag]
            slopes[1][node.to_node[0].id] = slopes[1][node.to_node[0].id][new_un_flag]

        node.update_bounds(bounds, out_flag)


    def set_symb_concr_bounds(self, node: Node, slopes: tuple) -> Bounds:
        in_flag, out_flag = self._get_in_prop_flag(node), self._get_out_prop_flag(node)
        symb_eq = Equation.derive(
            node, 
            None if out_flag is None else out_flag.flatten(), 
            None if in_flag is None else in_flag.flatten(), 
            self.config
        )

        lower_bounds = self.back_substitution(
            symb_eq,
            node.from_node[0],
            'lower',
            None if slopes is None else slopes[0]
        )
        lower_bounds = torch.max(node.bounds.lower[out_flag], lower_bounds)
        upper_bounds = self.back_substitution(
            symb_eq,
            node.from_node[0],
            'upper', 
            None if slopes is None else slopes[1]
        )
        upper_bounds = torch.min(node.bounds.upper[out_flag], upper_bounds)

        symb_concr_bounds = Bounds(lower_bounds, upper_bounds)
        self.update_node_symb_concrete_bounds(node, symb_concr_bounds, slopes, out_flag)


    def update_node_symb_concrete_bounds(
        self,
        node: None,
        bounds: Bounds,
        slopes: tuple[dict]=None,
        out_flag: torch.tensor=None
    ):
        if node.has_relu_activation() and slopes is not None and node.bounds is not None:
            # relu node with custom slope - leave slope as is but remove from
            # it newly stable nodes.
            old_un_flag = node.to_node[0].get_unstable_flag() 
            new_un_flag = out_flag[old_un_flag]

            slopes[0][node.to_node[0].id] = slopes[0][node.to_node[0].id][new_un_flag]
            slopes[1][node.to_node[0].id] = slopes[1][node.to_node[0].id][new_un_flag]


        node.update_bounds(bounds, out_flag)

    def _get_out_prop_flag(self, node: Node):
        if node.has_relu_activation():
            return node.to_node[0].get_unstable_flag()

        elif len(node.to_node) == 0:
            return self.prob.spec.get_output_flag(node.output_shape)
        
        return None 

    def _get_in_prop_flag(self, node: Node):
        stability = node.from_node[0].get_propagation_count()
        if stability / node.input_size >= self.config.SIP.STABILITY_FLAG_THRESHOLD:
            return  None
            
        return node.from_node[0].get_propagation_flag()

    def back_substitution(self, eq: Equation, node: None, bound: str, slopes: dict=None):
        eqs = self._back_substitution(eq, node, bound, slopes=slopes)
        sum_eq = eqs[0]
        for i in eqs[1:]:
            sum_eq = sum_eq.add(i)

        return sum_eq.concrete_values(
            self.prob.spec.input_node.bounds.lower.flatten(),
            self.prob.spec.input_node.bounds.upper.flatten(),
            bound
        )

    def _back_substitution(self, eq, node, bound, slopes=None):
        if bound not in ['lower', 'upper']:
            raise ValueError("Bound type {bound} not recognised.")

        if isinstance(node, Input):
            return  [eq]

        elif node.has_relu_activation() or isinstance(node, MaxPool):
            if slopes is not None and node.to_node[0].id in slopes:
                node_slopes = slopes[node.to_node[0].id]
            else:
                node_slopes = None

            eq = eq.interval_transpose(node, bound, node_slopes)

        elif type(node) in [Relu, Flatten, Unsqueeze]:
            eq = [eq]

        else:
            eq = eq.transpose(node)

        eqs = []

        for i in eq:
            eqs.extend(
                self._back_substitution(i, node.from_node[0], bound, slopes=slopes)
            )

        return eqs

    # def back_substitution(self, eq, node):
        # return Bounds(
            # self._back_substitution(eq, node.from_node[0], 'lower'),
            # self._back_substitution(eq, node.from_node[0], 'upper'),
        # )

    # def _back_substitution(self, eq, node, bound):
        # if bound not in ['lower', 'upper']:
            # raise ValueError("Bound type {bound} not recognised.")

        # if isinstance(node, Input):
            # return eq.concrete_values(
                # node.bounds.lower.flatten(), node.bounds.upper.flatten(), bound
            # )

        # elif node.has_non_linear_op() is True:
            # if node.has_relu_activation() is True and \
            # node.to_node[0].get_unstable_count() == 0  and  \
            # node.to_node[0].get_active_count() == 0:
                # return eq.const
            # else:
                # eq = eq.interval_transpose(node, bound)

        # elif type(node) in [Relu, MaxPool, Flatten]:
            # pass

        # else:
            # eq = eq.transpose(node)

        # return self._back_substitution(eq, node.from_node[0], bound)

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
        in_flag, out_flag = self._get_in_prop_flag(node), self._get_out_prop_flag(node)
        equation = Equation.derive(
            node,
            None if out_flag is None else out_flag.flatten(),
            None if in_flag is None else in_flag.flatten(),
            self.config
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
