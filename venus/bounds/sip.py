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
from timeit import default_timer as timer

from venus.network.layers import Layer, FullyConnected, Conv2D, MaxPooling
from venus.bounds.bounds import Bounds
from venus.bounds.equation import Equation
from venus.network.activations import ReluApproximation
from venus.network.activations import ReluState
from venus.network.activations import Activations
from venus.common.logger import get_logger
from venus.bounds.osip import OSIP, OSIPMode
from venus.common.configuration import Config
from venus.specification.formula import Formula, TrueFormula, VarVarConstraint, DisjFormula, \
    NAryDisjFormula, ConjFormula, NAryConjFormula

import tensorflow as tf

class SIP:

    logger = None

    def __init__(self, layers: list, config: Config, delta_flags=None):
        """
        Arguments:

            layers:
                A list [input,layer_1,....,layer_k] of the networks layers
                including an input layer defining the input.
            config:
                Configuration.
            logfile:
                The logfile.
        """

        self.layers = layers
        self.config = config
        self.delta_flags = delta_flags
        if SIP.logger is None and config.LOGGER.LOGFILE is not None:
            SIP.logger = get_logger(__name__, config.LOGGER.LOGFILE)

    def set_bounds(self):
        """
        Sets pre-activation and activation bounds.

        Arguments:

            relu_states:
                A list of floats, one for each node. 0: inactive, 1:active,
                anything else: unstable; use this for runtime computation of
                bounds.

        Returns:

                None
        """
        start = timer()
        for i in self.layers[1:-1]:
            i.reset_state_flags()
            if i.activation == Activations.linear:
                continue
            i.pre_bounds = self.get_pre_ia_bounds(i)
            if i.get_unstable_count() == 0:
                continue
            if i.depth > 1:
                pre_symb_concr_bounds = self.get_pre_symb_concr_bounds(i)
                i.pre_bounds.lower[i.get_unstable_flag().reshape(i.output_shape)] = pre_symb_concr_bounds.lower
                i.pre_bounds.upper[i.get_unstable_flag().reshape(i.output_shape)] = pre_symb_concr_bounds.upper
                i.reset_state_flags()
            
            i.post_bounds = self.get_post_bounds(i)

        self.layers[-1].pre_bounds = self.get_pre_symb_concr_bounds(self.layers[-1])
        self.layers[-1].post_bounds = self.get_post_bounds(self.layers[-1])

        if self.logger is not None:
            SIP.logger.info('Bounds computed, time: {:.3f}, '.format(timer()-start))


    def get_pre_ia_bounds(self, layer: Layer) -> list:
        input_bounds = self.layers[layer.depth - 1].post_bounds
        lower = layer.forward(input_bounds.lower, clip='+', add_bias=False) + \
             layer.forward(input_bounds.upper, clip='-', add_bias=True)
        upper = layer.forward(input_bounds.lower, clip='-', add_bias=False) + \
             layer.forward(input_bounds.upper, clip='+', add_bias=True)
        if self.delta_flags is not None:
            lower[self.delta_flags[layer.depth - 1][0]] = 0
            upper[self.delta_flags[layer.depth - 1][0]] = 0
            lower[self.delta_flags[layer.depth - 1][1]] = np.clip(
                lower[self.delta_flags[layer.depth - 1][1]],
                0,
                math.inf
            )
            upper[self.delta_flags[layer.depth - 1][1]] = np.clip(
                upper[self.delta_flags[layer.depth - 1][1]],
                0,
                math.inf
            )

        return Bounds(
            lower.reshape(layer.output_shape),
            upper.reshape(layer.output_shape)
        )

    def get_pre_symb_concr_bounds(self, layer: Layer) -> Bounds:
        pre_symb_eq = self.get_pre_symb_eq(layer)
        return self.back_substitution(pre_symb_eq, layer.depth)

    def get_pre_symb_eq(self, layer: Layer) -> Equation:
        flag = np.ones(layer.output_size, bool) if layer.depth == self.layers[-1].depth else layer.get_unstable_flag()
        
        return Equation.derive(
                layer,
                self.layers[layer.depth - 1],
                flag
        )


    def get_post_bounds(self, layer):
        if layer.activation == Activations.relu:
            bounds = Bounds(
                np.clip(layer.pre_bounds.lower, 0, math.inf),
                np.clip(layer.pre_bounds.upper, 0, math.inf)
            )

            return bounds

        return layer.pre_bounds


    def __get_post_bounds(self, layer, pre_eqs, pre_b, relu_states):

        if layer.activation == Activations.linear:
            post_eqs = [(i, i) for i in pre_eqs]
            post_b = pre_b

        elif layer.activation == Activations.relu:
            # if self.osip_eligibility(layer):
                # approx = ReluApproximation.IDENTITY
            # else:
                # approx = self.params.RELU_APPROXIMATION
            approx = self.config.SIP.RELU_APPROXIMATION

            if relu_states is None:
                post_eqs, post_b = self.relu(
                    pre_eqs,
                    pre_b,
                    layer.get_inactive_flag(),
                    layer.get_unstable_flag(),
                    approx
                )
            else:
                post_eq_r, post_b_r = self.relu(
                    eq,
                    pre_b[0],
                    pre_b[1],
                    approx
                )
                post_eq, post_b = self.runtime_bounds(
                    post_eq_r[0],
                    post_eq_r[1],
                    eq,
                    post_b_r[0],
                    post_b_r[1],
                    pre_b[0],
                    pre_b[1],
                    relu_states
                )

        return post_eqs,  post_b

    def osip_eligibility(self, layer):
        if layer.depth == len(self.layers) - 1:
            return False
        if self.config.SIP.OSIP_CONV != OSIPMode.ON:
            if isinstance(self.layers[layer.depth+1], Conv2D) \
            or isinstance(layer, Conv2D):
                return False
        if self.config.SIP.OSIP_FC != OSIPMode.ON:
            if isinstance(self.layers[layer.depth+1], FullyConnected) \
            or isinstance(layer, FullyConnected):
                return False
        
        return True

    def back_substitution(self, eq, depth):
        return Bounds(
            self._back_substitution(eq, depth, 'lower'),
            self._back_substitution(eq, depth, 'upper'),
        )

    def _back_substitution(self, eq, depth, bound):
        if bound not in ['lower', 'upper']:
            raise ValueError("Bound type {bound} not recognised.")

        if self.layers[depth - 1].get_unstable_count() == 0  and  \
        self.layers[depth - 1].get_active_count() == 0:
            return eq.const

        for i in range(depth - 1, 0, -1):
            if self.layers[i].activation == Activations.relu:
                eq = eq.interval_transpose(self.layers[i], bound)
            elif self.layers[i].activation == Activations.linear:
                eq = eq.transpose(self.layers[i])
            else:
                raise ValueError("Activation {self.layers[i].activation} is not supported")

        return eq.concrete_values(
            self.layers[0].pre_bounds.lower,
            self.layers[0].pre_bounds.upper,
            bound
        )

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
        eqs: list of linear equations of the outputs of the preceding layers in terms of their input variables
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
            coeffs = np.zeros(shape=(1, self.layers[-1].output_size), dtype=self.config.PRECISION)
            const = np.array([0], dtype=self.config.PRECISION)
            if formula.sense == Formula.Sense.GT:
                coeffs[0, formula.op1.i], coeffs[0, formula.op2.i] = 1, -1
            elif formula.sense == Formula.Sense.LT:
                coeffs[0, formula.op1.i], coeffs[0, formula.op2.i] = -1, 1
            else:
                raise ValueError('Formula sense {formula.sense} not expeted')
            equation = Equation(coeffs, const)
            diff = self._back_substitution(
                equation,
                self.layers[-1].depth + 1,
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
