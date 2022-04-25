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

import math
import torch
import numpy as np

from venus.network.node import Node, Gemm, Conv, Relu, MatMul, Add, Sub, Constant
from venus.common.configuration import Config

torch.set_num_threads(1)

class Equation():
    """
    The Equation class.
    """

    def __init__(self, matrix: torch.Tensor, const: torch.Tensor, config: Config):
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
        self.config = config
        self.size = matrix.shape[0]

    def copy(self, matrix=None, const=None) -> Equation:
        return Equation(
            self.matrix.detach().clone(),
            self.const.detach().clone(),
            self.size
        )
 
    def _get_plus_matrix(self) -> torch.Tensor:
        """
        Clips the coeffs to be only positive.
        """

        return torch.clamp(self.matrix, 0, math.inf)


    def _get_minus_matrix(self, keep_in_memory=True) -> torch.Tensor:
        """
        Clips the coeffs to be only negative.
        """
        
        return torch.clamp(self.matrix, -math.inf, 0)


    def concrete_values(self, lower: torch.Tensor, upper:torch.Tensor, bound: str) -> torch.Tensor:
        if bound == 'lower':
            return self.min_values(lower, upper)
        
        elif bound == 'upper':
            return self.max_values(lower, upper)
       
        else:
            raise ValueError(f'Bound {bound} is not recognised.')
        

    def max_values(self, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
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


    def interval_dot(self, bound: str, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
        """
        Computes the interval dot product with either a matrix or an Equation.
        """
        if bound == 'upper':
            return self._get_plus_matrix() @ upper + \
                self._get_minus_matrix() @ lower + \
                self.const

        elif bound == 'lower':
            return  self._get_plus_matrix() @ lower + \
                self._get_minus_matrix() @ upper + \
                self.const

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
        offset_idx = np.arange(M)[:,None]*n*K*p + torch.Tensor([i*p*K + np.arange(K) for i in  range(N)]).flatten()
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

    def transpose(self, node: Node):
        matrix = node.transpose(self.matrix)
        const = Equation._get_const(
            node,
            torch.ones(node.output_size, dtype=torch.bool, device=self.config.DEVICE)
        )
        const = (self.matrix @ const) + self.const
        
        return Equation(matrix, const, self.config)


    def interval_transpose(self, node, bound):
        assert node.has_non_linear_op() is True, "Interval transpose is only supported for nodes connected to a non-linear operation."

        (lower_slope, upper_slope), (lower_const, upper_const) = self.get_relaxation_slope(node)

        if bound == 'lower':
            plus = self._get_plus_matrix() * lower_slope
            minus = self._get_minus_matrix() * upper_slope
            const = self._get_plus_matrix() @ lower_const + \
                self._get_minus_matrix() @ upper_const


        elif bound == 'upper':
            plus = self._get_plus_matrix()  * upper_slope
            minus = self._get_minus_matrix() * lower_slope
            const = self._get_plus_matrix() @ upper_const + \
                self._get_minus_matrix() @ lower_const

        else:
            raise ValueError(f'Bound {bound} is not recognised.')

        matrix = node.transpose(plus) + node.transpose(minus)

        const += self.const

        return Equation(matrix, const, self.config)

    
    def get_relaxation_slope(self, node: Node) -> tuple:
        if node.has_relu_activation():
            return self.get_relu_relaxation(node)

        elif node.has_max_pool():
            return self.get_max_pool_relaxation(node)

        else:
            raise NotImplementedError(f'type(node.to_node[0]')

    def get_relu_relaxation(self, node: Node) -> tuple:
        lower_slope = torch.ones(
            node.output_size, dtype=self.config.PRECISION, device=self.config.DEVICE
        )
        upper = node.bounds.upper.flatten()
        lower = node.bounds.lower.flatten()
        idxs = abs(lower) >=  upper
        lower_slope[idxs] = 0.0
        lower_slope[node.to_node[0].get_inactive_flag()] = 0.0
        lower_slope[node.to_node[0].get_active_flag()] = 1.0


        upper_slope = torch.zeros(
            node.output_size, dtype=self.config.PRECISION, device=self.config.DEVICE
        )
        upper = upper[node.to_node[0].get_unstable_flag()]
        lower = lower[node.to_node[0].get_unstable_flag()]
        upper_slope[node.to_node[0].get_unstable_flag()] = upper /  (upper - lower)
        upper_slope[node.to_node[0].get_active_flag()] = 1.0


        lower_const = Equation._get_const(node, np.ones(node.output_size, bool))
        upper_const = lower_const.detach().clone()
        lower_const *= lower_slope
        upper_const *= upper_slope
        upper_const[node.to_node[0].get_unstable_flag()]  -= \
            upper_slope[node.to_node[0].get_unstable_flag()] * \
            node.bounds.lower.flatten()[node.to_node[0].get_unstable_flag()]

        return (lower_slope, upper_slope), (lower_const, upper_const)

    def get_max_pool_relaxation(self, node: Node) -> tuple:
        indices = torch.arange(node.input_size).reshape(node.input_shape)
        lower_indices = node.to_node[0].forward(node.bounds.lower).flatten()
        upper_indices = node.to_node[0].forward(node.bounds.upper).flatten()

        lower_max  = lower_indices > upper_indices
        not_lower_max = torch.logical_not(lower_max)


        lower_slope = torch.zeros(
            node.input_size, dtype=self.config.PRECISION, device=self.config.DEVICE
        )
        lower_slope[lower_indices] = 1.0
        lower_const = torch.zeros(
            node.input_size, dtype=self.config.PRECISION, device=self.config.DEVICE
        )
        lower_const[lower_indices] = Equation._get_const(node, lower_indices)

    
        upper_slope = torch.zeros(
            node.input_size, dtype=self.config.PRECISION, device=self.config.DEVICE
        )
        upper_slope[lower_max] = 1.0
        upper_const = torch.zeros(
            node.input_size, dtype=self.config.PRECISION, device=self.config.DEVICE
        )
        upper_const[lower_max] = Equation._get_const(node, lower_max)
        upper_const[not_lower_max] = node.bounds.upper.flatten()[not_lower_max]

        return (lower_slope, upper_slope), (lower_const, upper_const)


    @staticmethod
    def derive(node: Node, flag: torch.Tensor, config: Config) -> Equation:
        zero_eq = Equation._zero_eq(node, flag)
        
        if zero_eq is not None:
            return zero_eq
      
        else:
            return Equation(
                Equation._derive_matrix(node, flag),
                Equation._derive_const(node, flag),
                config
            )

    def _derive_matrix(node: Node, flag: torch.Tensor):
        if isinstance(node, Conv):
            return Equation._derive_conv_matrix(node, flag)

        elif isinstance(node, Gemm):
            return Equation._derive_gemm_matrix(node, flag)

        elif isinstance(node, MatMul):
            return Equation._derive_matmul_matrix(node, flag)
        
        elif isinstance(node, Add):
            return Equation._derive_add_matrix(node, flag)

        else:
            raise NotImplementedError(f'type(node) equations')

    def _derive_const(node: Node, flag: torch.Tensor):
        if isinstance(node, Conv):
            return Equation._derive_conv_const(node, flag)

        elif isinstance(node, Gemm):
            return Equation._derive_gemm_const(node, flag)

        elif isinstance(node, MatMul):
            return Equation._derive_matmul_const(node, flag)
        
        elif isinstance(node, Add):
            return Equation._derive_add_const(node, flag)

        else:
            raise NotImplementedError(f'type(node) equations')

    @staticmethod 
    def _zero_eq(node: None, flag: torch.Tensor) -> Equation:
        if isinstance(node.from_node[0], Relu) and \
        node.from_node[0].get_unstable_count() == 0  and \
        node.from_node[0].get_active_count() == 0:      
            return Equation(
                np.zeros((torch.sum(flag), 0)), Equation._get_const(node, flag), node.config
            )

        return None

    @staticmethod 
    def _derive_conv_matrix(node: Node, flag: torch.Tensor):
        flag_size = torch.sum(flag).item()

        prop_flag = torch.zeros(node.get_input_padded_size(), dtype=torch.bool)
        prop_flag[node.get_non_pad_idxs()] = True

        pad = torch.ones(
            node.get_input_padded_size(),
            dtype=torch.long
        ) * node.input_size
        pad[prop_flag] = torch.arange(node.input_size)

        im2col = Conv.im2col(
            pad.reshape(node.get_input_padded_shape()),
            (node.krn_height, node.krn_width),
            node.strides
        )
        indices = torch.arange(node.out_ch_sz).repeat(node.out_ch)[flag]
        conv_indices = im2col[:, indices]

        indices = torch.repeat_interleave(torch.arange(node.out_ch), node.out_ch_sz, dim=0)[flag]
        conv_weights = node.kernels.permute(1, 2, 3, 0).reshape(-1, node.out_ch)[:, indices]
            
        matrix = torch.zeros((flag_size, node.input_size + 1), dtype=node.config.PRECISION)
        matrix[torch.arange(flag_size), conv_indices] = conv_weights
        matrix = matrix[:, :node.input_size]

        return matrix

    @staticmethod 
    def _derive_conv_const(node: Node, flag: torch.Tensor):
        return torch.tile(node.bias, (node.out_ch_sz, 1)).T.flatten()[flag]


    @staticmethod 
    def _derive_gemm_matrix(node: Node, flag: torch.Tensor):
        return  node.weights[flag, :]

    @staticmethod 
    def _derive_gemm_const(node: Node, flag: torch.Tensor):
        return node.bias[flag]

    @staticmethod 
    def _derive_matmul_matrix(node: Node, flag: torch.Tensor):
        return  node.weights[flag, :]

    @staticmethod 
    def _derive_matmul_const(node: Node, flag: torch.Tensor):
        return np.zeros(node.output_size, dtype=node.weights.dtype)

    @staticmethod 
    def _derive_add_matrix(node: Node, flag: torch.Tensor):
        if node.const is not None:
            matrix =  np.identity(node.input_size, dtype=node.config.PRECISION)[:, flag] 
        else:
            matrix = np.zeros((node.output_size, 2 * node.output_size), dtype=node.config.PRECISION)
            matrix[range(node.output_size), range(node.output_size)] = 1
            matrix[range(node.output_size), range(node.output_size, 2 * node.output_size)] = 1

        return matrix

    @staticmethod 
    def _derive_add_const(node: Node, flag: torch.Tensor):
        return torch.zeros(
            node.output_size, dtype=node.const.dtype, device=node.const.device
        )


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
