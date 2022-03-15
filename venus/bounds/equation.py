"""
# File: equation.py
# Top contributors (to current version):
# 	Panagiotis Kouvaros (panagiotis.kouvaros@gmail.com)
# This file is part of the Venus  project.
# Copyright: 2019-2021 by the authors listed in the AUTHORS file in the
# top-level directory.
# License: BSD 2-Clause (see the file LICENSE in the top-level directory).
# Description:  Defines the bound equation (abstract) class.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Union
import math
import numpy as np

from venus.network.activations import Activations, ReluApproximation
from venus.network.layers import Layer, FullyConnected, Conv2D


class Equation():
    """
    The Equation class.
    """

    def __init__(self, matrix: np.array, const: np.array):
        """
        Arguments:

            matrix:
                2D matrix (of size nxm). Each row i represents node's i
                equation coefficients.
            const:
                Vector (of size n). Each row i represents node's i equation
                constant term.
        """
        self.matrix = matrix
        self.const = const
        self.size = matrix.shape[0]
        self.plus_matrix = None
        self.minus_matrix = None
        self.lower_bounds = None
        self.upper_bounds = None

    @abstractmethod
    def copy(self, matrix=None, const=None) -> Equation:
        pass
 
    def _get_plus_matrix(self) -> np.array:
        """
        Clips the coeffs to be only positive.
        """

        return np.clip(self.matrix, 0, math.inf)


    def _get_minus_matrix(self, keep_in_memory=True) -> np.array:
        """
        Clips the coeffs to be only negative.
        """
        
        return np.clip(self.matrix, -math.inf, 0)


    def concrete_values(self, lower: np.array, upper:np.array, bound: str) -> np.array:
        if bound == 'lower':
            return self.min_values(lower, upper)
        
        elif bound == 'upper':
            return self.max_values(lower, upper)
       
        else:
            raise ValueError(f'Bound {bound} is not recognised.')
        

    def max_values(self, lower: np.array, upper: np.array) -> np.array:
        """
        Computes the upper bounds of the equations.
    
        Arguments:

            lower:
                The lower bounds for the variables in the equation. 
            upper:
                The upper bounds for the variables in the equation.
        
        Returns: 

            The upper bounds of the equation.
        """
        return  self.interval_dot('upper', lower, upper)


    def min_values(self, lower, upper):
        """
        Computes the lower bounds of the equations.

        Arguments:

            lower:
                The lower bounds for the variables in the equation. 
            upper:
                The upper bounds for the variables in the equation.

        Returns:

            The lower bounds of the equations.
        """

        return self.interval_dot('lower', lower, upper)


    def interval_dot(self, bound: str, lower: np.array, upper: np.array) -> np.array:
        """
        Computes the interval dot product with either a matrix or an Equation.
        """
        if bound == 'upper':
            return self._get_plus_matrix().dot(upper) + self._get_minus_matrix().dot(lower) + self.const

        elif bound == 'lower':
            return  self._get_plus_matrix().dot(lower) +  self._get_minus_matrix().dot(upper) + self.const

        else: 
            raise ValueError(f'Bound {bound} is not recognised.')


    def pool(self, in_shape, out_shape, pooling):
        """
        Derives the pooling indices of the equations.

        Arguments:
            
            in_shape: tuple of the shape of the equations.

            out_shape: tuple the shape of the equations after pooling.

            pooling: tuple of the pooling size.
        
        Returns:

            List where each item i in the list is a list of indices of the i-th
            pooling neighbourhood.
        """
            
        m,n,_ = in_shape
        M,N,K =  out_shape
        p = pooling[0]
        m_row_extent = M*N*K
        # get Starting block indices
        start_idx = np.arange(p)[:,None]*n*K + np.arange(0,p*K,K)
        # get offsetted indices across the height and width of input 
        offset_idx = np.arange(M)[:,None]*n*K*p + np.array([i*p*K + np.arange(K) for i in  range(N)]).flatten()
        # get all actual indices 
        idx = start_idx.ravel()[:,None] + offset_idx.ravel()
        return [idx[i,:] for i in idx.shape[0]]

    def split_dot(self, a, b):
        size = a.shape[0] 
        c = np.empty(shape=(size, b.shape[1]))
        # est = sys.getsizeof(m) + sys.getsizeof(o) + 2*sys.getsizeof(self.matrix) + sys.getsizeof(eqlow.matrix)
        # est *= 1.2
        # avl = psutil.virtual_memory().available 
        # dec  =  max(int(math.ceil(est/avl)),1)
        # dec = 256
        # if X < dec:
            # ch = X
        # else:
            # ch = int(X / dec)
            
        ch = int(size / 16)

        for i in range(0, size - (size % ch), ch):
            c[range(i, i+ch), :] = np.dot(a[range(i, i+ch), :], b)

        left = size - (size % ch)
        c[range(left, size), :] = np.dot(a[range(left, size), :], b)

        return c


    def transpose(self, layer):
        return Equation(
            layer.transpose(self.matrix),
            self.matrix.dot(Equation._get_const(layer, np.ones(layer.output_size, bool)))
        )


    def interval_transpose(self, layer, bound):
        assert layer.activation == Activations.relu, "Interval transpose is not supported for {layer.activation}"
        lower_slope = layer.get_lower_relu_slope()
        lower_const = Equation._get_const(layer, np.ones(layer.output_size, bool))
        upper_const = lower_const.copy()
        upper_slope = layer.get_upper_relu_slope()
        lower_const *= lower_slope
        upper_const *= upper_slope
        upper_const[layer.get_unstable_flag()]  -= \
            upper_slope[layer.get_unstable_flag()] * \
            layer.pre_bounds.lower.flatten()[layer.get_unstable_flag()]

        if bound == 'lower':
            plus = self._get_plus_matrix() * lower_slope
            minus = self._get_minus_matrix() * upper_slope
            const = self._get_plus_matrix().dot(lower_const) + self._get_minus_matrix().dot(upper_const)

        elif bound == 'upper':
            plus = self._get_plus_matrix()  * upper_slope
            minus = self._get_minus_matrix() * lower_slope
            const = self._get_plus_matrix().dot(upper_const) + self._get_minus_matrix().dot(lower_const)

        else:
            raise ValueError(f'Bound {bound} is not recognised.')

        matrix = layer.transpose(plus) + layer.transpose(minus)
        const += self.const

        return Equation(matrix, const)


    @staticmethod
    def derive(layer: Layer, previous_layer: Layer, flag: np.array) -> Equation:

        if previous_layer.get_unstable_count() == 0  and previous_layer.get_active_count() == 0:
            return Equation(
                np.zeros((np.sum(flag), 0)),
                Equation._get_const(layer, flag)
            )

        return Equation(
            Equation._get_matrix(layer, flag),
            Equation._get_const(layer, flag)
        )
        

    @staticmethod
    def _get_matrix(layer: Conv2D, flag: np.array) -> np.array:
        if isinstance(layer, Conv2D):
            height, width, _, filters = layer.kernels.shape
            flag_size = np.sum(flag)
            prop_flag = np.zeros(layer.get_input_padded_size(), bool)
            prop_flag[layer.get_non_pad_idxs()] = True
            pad = np.ones(
                layer.get_input_padded_size(),
                dtype=np.uint
            ) * layer.input_size
            pad[prop_flag] = np.arange(layer.input_size)
            conv_indices = np.repeat(
                Conv2D.im2col(
                    pad.reshape(layer.get_input_padded_shape()),
                    (height, width),
                    layer.strides
                ),
                filters,
                axis=1
            )[:, flag]
            matrix = np.zeros((flag_size, layer.input_size + 1), dtype=layer.kernels.dtype)
            matrix[range(flag_size), conv_indices] = np.array(
                [
                    layer.kernels[..., i].flatten()
                    for j in range(int(layer.output_size / filters)) for i in range(filters)
                ]
            ).T[:, flag]

            return matrix[:, :layer.input_size]

        elif isinstance(layer, FullyConnected):
            return layer.weights[flag, :]

        else:
            raise  TypeError(f'Layer {layer} is not supported')

    @staticmethod
    def _get_const(layer: Layer, flag: np.array) -> np.array:
        if isinstance(layer, Conv2D):
            filters = layer.kernels.shape[-1]
            out_ch_size = int(layer.output_size / filters)

            return np.array(
                [layer.bias for i in range(out_ch_size)]
            ).flatten()[flag]

        elif isinstance(layer, FullyConnected):
            return layer.bias[flag]

        else:
            raise  TypeError(f'Layer {layer} is not supported')


    # def set_lower_slope(self, lbound, ubound):
        # """ 
        # Sets the lower slopes for the equations, one for computing the lower
        # bounds during back-substitution and one for computing the upper bound.

        # Arguments:
            
            # lbound: vector of the lower slope for the lower bound.
            
            # ubound: vector of the lower slope for the upper bound.

        # Returns:

            # None
        # """
        # self.lower_slope_l_bound = lbound
        # self.lower_slope_u_bound = ubound
        # self.is_slope_optimised = True

    # def get_lower_slope(self, l, u, approx=ReluApproximation.MIN_AREA):
        # """
        # Derives the slope of the lower linear relaxation of the equations.

        # Arguments:

            # lower: vector of lower bounds of the equations
            
            # upper: vector of upper bounds of the equations

            # approx: ReluApproximation

        # Returns:

            # vector of the slopes.
        # """

        # slope = np.zeros(self.size)
        # l, u = l.flatten(), u.flatten()

        # for i in range(self.size):
            # if  l[i] >= 0:
                # slope[i] = 1
            # elif u[i] <= 0: 
                # pass
            # else:
                # if approx == ReluApproximation.ZERO:
                    # pass
                # elif approx == ReluApproximation.IDENTITY:
                    # slope[i] = 1
                # elif approx == ReluApproximation.PARALLEL:
                    # slope[i] = u[i] / (u[i] - l[i])
                # elif approx == ReluApproximation.MIN_AREA:
                    # if abs(l[i]) < u[i]: slope[i] = 1
                # elif approx == ReluApproximation.VENUS_HEURISTIC:
                    # if abs(l[i]) < u[i]: slope[i] = u[i] / (u[i] - l[i])
                # else:
                    # pass

        # return slope 
