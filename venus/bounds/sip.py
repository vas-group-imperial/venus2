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
from venus.specification.formula import Formula, TrueFormula, VarVarConstraint, DisjFormula, \
    NAryDisjFormula, ConjFormula, NAryConjFormula

torch.set_num_threads(1)

class SIP:

    logger = None

    def __init__(self, prob, config: Config, delta_flags=None):
        """
        Arguments:

            nodes:
                A list [input,layer_1,....,layer_k] of the networks nodes
                including an input layer defining the input.
            config:
                Configuration.
            logfile:
                The logfile.
        """
        self.prob = prob
        self.nn = prob.nn
        self.config = config
        self.delta_flags = delta_flags
        if SIP.logger is None and config.LOGGER.LOGFILE is not None:
            SIP.logger = get_logger(__name__, config.LOGGER.LOGFILE)

    def set_bounds(self):
        """
        Sets pre-activation and activation bounds.
        """
        start = timer()
        processed_nodes = {i : False for i in self.nn.node}
        saw_symb_eq_node = False
        for i in range(self.nn.tail.depth):
            nodes = self.nn.get_node_by_depth(i)
            for j in nodes:
                print(j.id, j)
                processed_nodes[j.id] = True
                self.set_ia_bounds(j)

                if self.config.MEMORY_OPTIMISATION is True:
                    for k in j.from_node:
                        cond = np.array([processed_nodes[l.id] for l in k.to_node])
                        if cond.all() is True:
                            k.clear_bounds()

                if self.is_symb_eq_eligible(j) is not True:
                    continue

                if saw_symb_eq_node is not True:
                    saw_symb_eq_node = True
                    continue

                self.set_symb_concr_bounds(j)

        self.nn.tail.bounds = Bounds(
            torch.ones(self.nn.tail.output_shape) * - math.inf,
            torch.ones(self.nn.tail.output_shape) * math.inf,
        )
        self.set_symb_concr_bounds(self.nn.tail)

        # print(self.nn.tail.bounds.lower)
        print('sip done', timer() - start)

        if self.logger is not None:
            SIP.logger.info('Bounds computed, time: {:.3f}, '.format(timer()-start))


    def is_symb_eq_eligible(self, node: None) -> bool:
        """
        Determines whether the node implements function requiring a symbolic
        equation for bound calculation.
        """
        
        if node.has_relu_activation() and node.to_node[0].get_unstable_count() > 0:
            return True

        if node.has_max_pool():
            return True
    
        return False


    def set_ia_bounds(self, node: Node) -> list:
        inp = node.from_node[0].bounds

        if type(node) in [Relu, BatchNormalization, MaxPool, Slice]:
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
        node.update_bounds(ia_bounds)

    def set_symb_concr_bounds(self, node: Node) -> Bounds:
        if node.has_relu_activation():
            out_flag =  node.to_node[0].get_unstable_flag().flatten()
        elif len(node.to_node) == 0:
            out_flag = self.prob.spec.get_output_flag(node.output_shape)
        else:
            out_flag = torch.ones(node.output_size, dtype=torch.bool)

        stability = node.from_node[0].get_propagation_count()
        if stability / node.input_size >= self.config.SIP.FLAG_THRESHOLD:
            in_flag = None
        else:
            in_flag = node.from_node[0].get_propagation_flag()

        symb_eq = Equation.derive(node, out_flag, in_flag, self.config)

        flag = False if in_flag is None else True
        symb_concr_bounds = Bounds(
            self.back_substitution(symb_eq, node.from_node[0], flag, 'lower'),
            self.back_substitution(symb_eq, node.from_node[0], flag, 'upper')
        )
        node.update_bounds(symb_concr_bounds, out_flag)

    def osip_eligibility(self, layer):
        if layer.depth == len(self.nodes) - 1:
            return False
        if self.config.SIP.OSIP_CONV != OSIPMode.ON:
            if isinstance(self.nodes[layer.depth+1], Conv2D) \
            or isinstance(layer, Conv2D):
                return False
        if self.config.SIP.OSIP_FC != OSIPMode.ON:
            if isinstance(self.nodes[layer.depth+1], FullyConnected) \
            or isinstance(layer, FullyConnected):
                return False
        
        return True

    def back_substitution(self, eq, node, flag, bound):
        eqs = self._back_substitution(eq, node, flag, bound)
        sum_eq = eqs[0]
        for i in eqs[1:]:
            sum_eq = sum_eq.add(i)

        return sum_eq.concrete_values(
            self.prob.spec.input_node.bounds.lower.flatten(),
            self.prob.spec.input_node.bounds.upper.flatten(),
            bound
        )

    def _back_substitution(self, eq, node, flag, bound):
        print('*', node)
        if bound not in ['lower', 'upper']:
            raise ValueError("Bound type {bound} not recognised.")

        if isinstance(node, Input):
            return  [eq]

        elif node.has_relu_activation() or isinstance(node, MaxPool):
            eq = [eq.interval_transpose(node, bound)]

        elif type(node) in [Relu, Flatten]:
            eq = [eq]

        else:
            eq = eq.transpose(node, flag)

        eqs = []

        for i in eq:
            eqs.extend(self._back_substitution(i, node.from_node[0], flag, bound))

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
                (1, self.nn.tail.output_size),
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
                raise ValueError('Formula sense {formula.sense} not expeted')
            equation = Equation(coeffs, const, self.config)
            diff = self.back_substitution(
                equation, self.nn.tail, 'lower'
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
