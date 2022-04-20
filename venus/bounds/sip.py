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
        saw_non_linear_node = False
        for i in range(self.nn.tail.depth):
            nodes = self.nn.get_node_by_depth(i)
            for j in nodes:
                print(j)
                j.bounds = self.compute_ia_bounds(j)

                if j.has_relu_activation() is not True:
                    continue

                j.to_node[0].reset_state_flags()

                if saw_non_linear_node is not True:
                    saw_non_linear_node = True
                    continue

                if j.to_node[0].get_unstable_count() == 0:
                    continue

                symb_concr_bounds = self.compute_symb_concr_bounds(j)

                j.bounds.lower[j.to_node[0].get_unstable_flag().reshape(j.output_shape)] = symb_concr_bounds.lower
                j.bounds.upper[j.to_node[0].get_unstable_flag().reshape(j.output_shape)] = symb_concr_bounds.upper
                j.to_node[0].reset_state_flags()

        self.nn.tail.bounds = self.compute_symb_concr_bounds(self.nn.tail)
        print(self.nn.tail.bounds.lower)

        if self.logger is not None:
            SIP.logger.info('Bounds computed, time: {:.3f}, '.format(timer()-start))


    def compute_ia_bounds(self, node: Node) -> list:
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

        return Bounds(
            torch.reshape(lower, node.output_shape),
            torch.reshape(upper, node.output_shape)
        ) 


    def compute_symb_concr_bounds(self, node: Node) -> Bounds:
        symb_eq = self.compute_symb_eq(node)

        return self.back_substitution(symb_eq, node)

    def compute_symb_eq(self, node: Node) -> Equation:
        if len(node.to_node) == 0 or node.has_relu_activation() is not True:
            flag = self.prob.spec.get_output_flag(node.output_shape).flatten()
        else:
            flag = node.to_node[0].get_unstable_flag()
      
        return Equation.derive(node, flag, self.config)

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

    def back_substitution(self, eq, node):
        return Bounds(
            self._back_substitution(eq, node.from_node[0], 'lower'),
            self._back_substitution(eq, node.from_node[0], 'upper'),
        )

    def _back_substitution(self, eq, node, bound):
        if bound not in ['lower', 'upper']:
            raise ValueError("Bound type {bound} not recognised.")

        if isinstance(node, Input):
            return eq.concrete_values(node.bounds.lower.flatten(), node.bounds.upper.flatten(), bound)

        elif type(node) in [Relu, Flatten]:
            pass

        elif node.has_relu_activation() is True:
            if node.to_node[0].get_unstable_count() == 0  and  \
            node.to_node[0].get_active_count() == 0:
                return eq.const
            else:
                eq = eq.interval_transpose(node, bound)

        else:
            eq = eq.transpose(node)

        return self._back_substitution(eq, node.from_node[0], bound)


    def max_pooling(self, layer, eqs, layer_bounds, input_bounds):
        if layer.pool_size == (2,2):
            return self._max_pooling_2x2(
                layer,
                eqs,
                layer_bounds,
                input_bounds
            )
        else:
            return self._max_pooling_general(
                eqs,
                layer_bounds,
                input_bounds
            )

    def _max_pooling_general(self, layer, eqs, layer_bounds, input_bounds):
        so, sho = layer.output_size, layer.output_shape
        si, shi = layer.input_size, layer.input_shape
        #get maxpool indices
        coeffs = np.identity(si,dtype='float64')
        const = np.zeros(si, dtype='float64')
        indices = Equations(coeffs,const).maxpool(shi,sho,layer.pool_size)
        # set low equation to the input equation with the highest lower bound
        coeffs_low = np.zeros(shape=(so,si),dtype='float64')
        coeffs_up = coeffs_low.copy()
        const_low = np.zeros(so, dtype='float64')
        const_up = const_low.copy()
        m_coeffs_low = []
        m_coeffs_up = []
        m_const_low = np.zeros(so,dtype='float32')
        m_const_up = np.zeros(so,dtype='float32')
        lb = np.zeros(so, dtype='float64')
        ub = np.zeros(so, dtype='float64')
        for i in range(so):
            lbounds = [np.take(layer_bounds[0],x[i]) for x in indices]
            lb[i] =  max(lbounds)
            index = lbounds.index(lb[i])
            coeffs_low[i,:] = coeffs[indices[index][i],:]
            m_coeffs_low.append({indices[index][i] : 1})
            ubounds = [np.take(layer_bounds[1],x[i]) for x in indices]
            ub[i] = max(ubounds)
            del ubounds[index]
            if (lb[i] > np.array(ubounds)).all():
                coeffs_up[i,:] = coeffs[indices[index][i],:]
                m_coeffs_up.append({indices[index][i] : 1})
            else:
                const_up[i] = ub[i]
                m_const_up[i] = ub[i]
                m_coeffs_up.append({indices[index][i] : 0})
        q_low = Equations(coeffs_low, const_low)
        q_up = Equations(coeffs_up, const_up)
        m_q_low = Equations(m_coeffs_low, m_const_low)
        m_q_up = Equations(m_coeffs_up, m_const_up)

        return [q_low, q_up], [m_q_low, m_q_up], [lb, ub]


    def global_average_pool(self, layer, eqs, input_bounds):
        """
        set pre-activation and activation bounds of a GlobalAveragePooling layer.
        [it is assumed that the previous layer is a convolutional layer]

        layer: a GlobalAveragePooling layer
        eqs: list of linear equations of the outputs of the preceding nodes in terms of their input variables
        input_bounds: the lower and upper bounds of the input layer 

        returns: linear equations of the outputs of the layer in terms of its input variables
        """ 
        kernels = layer.input_shape[-1]
        size = reduce(lambda i,j : i*j, layer.input_shape[:-1])
        weight = np.float64(1)/size
        const = np.zeros(kernels,dtype='float64')
        coeffs = np.zeros(shape=(kernels,layer.input_size),dtype='float64')
        for k in range(kernels):
            indices = list(range(k,k+size*kernels,kernels))
            coeffs[k,:][indices] = weight
        eq = Equations(coeffs, const)
        l,u = np.empty(kernels,dtype='float64'), np.empty(kernels,dtype='float64')
        ib_l, ib_u = (input_bounds[i].reshape(layer.input_shape) for i in [0,1])
        for k in range(kernels):
            l[k] = np.average(ib_l[:,:,k])
            u[k] = np.average(ib_u[:,:,k]) 

        return eq, l, u


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
                1, self.nn.tail.output_size, 
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
            diff = self._back_substitution(
                equation,
                self.nn.tail,
                'lower'
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
