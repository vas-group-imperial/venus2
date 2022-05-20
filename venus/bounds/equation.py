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

from venus.network.node import *
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


    def add(self, eq: Equation) -> Equation:
        """
        Adds the equation to another.

        Arguments:
            eq:
                The equation to add.
        Returns.
            An equation representing the sum.
        """
        return Equation(
            self.matrix + eq.matrix,
            self.const + eq.const,
            self.config
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

        in_flag, out_flag = self._get_flags(node)

        if type(node) in [Gemm, Conv, ConvTranspose]:
            tr_eq =  self._transpose_affine(node, in_flag, out_flag)

        elif isinstance(node, MatMul):
            tr_eq = self._transpose_linear(node, in_flag, out_flag)

        elif isinstance(node, BatchNormalization):
            tr_eq = self._transpose_batch_normalization(node, in_flag, out_flag)

        elif isinstance(node, Slice):
            tr_eq = self._transpose_slice(node)

        elif isinstance(node, Concat):
            tr_eq = self._transpose_concat(node)

        elif isinstance(node, Sub):
            tr_eq = self._transpose_sub(node)

        else:
            raise NotImplementedError(f'Equation transpose for {type(node)}')


        return tr_eq if isinstance(tr_eq, list) else [tr_eq]

    def _get_flags(self, node: Node):
        stab = node.from_node[0].get_propagation_count()
        if stab / node.input_size >= self.config.SIP.STABILITY_FLAG_THRESHOLD:
            in_flag = None
        else:
            in_flag = node.from_node[0].get_propagation_flag()

        stab = node.get_propagation_count()
        if stab / node.output_size >= self.config.SIP.STABILITY_FLAG_THRESHOLD:
            out_flag = None
        else:
            out_flag = node.get_propagation_flag()

        return in_flag, out_flag

    def _transpose_affine(self, node: Node, in_flag: torch.tensor, out_flag: torch.tensor):
        if in_flag is None:
            shape = (self.size,) + node.output_shape_no_batch()
        else:
            shape = self.matrix.shape
 
        matrix =  node.transpose(self.matrix.reshape(shape), in_flag, out_flag)
        matrix = matrix.reshape(self.size, -1)

        const = Equation._derive_const(node, out_flag)
        const = (self.matrix @ const) + self.const

        return Equation(matrix, const, self.config)

    def _transpose_linear(self, node: Node, in_flag: torch.tensor, out_flag:torch.tensor):
        if in_flag is None:
            matrix =  node.transpose(self.matrix)

        else:
            matrix =  node.transpose(self.matrix, in_flag, out_flag)

        return Equation(matrix, self.const, self.config)
        
    def _transpose_batch_normalization(self, node: Node, in_flag: torch.tensor, out_flag: torch.tensor):
        out_ch_sz = node.out_ch_sz()

        scale = torch.tile(node.scale, (out_ch_sz, 1)).T.flatten()
        bias = torch.tile(node.bias, (out_ch_sz, 1)).T.flatten()
        input_mean = torch.tile(node.input_mean, (out_ch_sz, 1)).T.flatten()
        mean_var = torch.sqrt(node.input_mean + node.epsilon)
        mean_var = torch.tile(mean_var, (out_ch_sz, 1)).T.flatten()

        if in_flag is None:
            matrix = (self.matrix * scale) / mean_var
        else:
            prop_flag = in_flag.flatten()
            matrix = (self.matrix[:, prop_flag] * scale[prop_flag]) / mean_var[prop_flag]

        batch_const = - input_mean / mean_var * scale + bias
        const = self.matrix @ batch_const + self.const

        return Equation(matrix, const, self.config)

    def _transpose_slice(self, node: Node):
        matrix = torch.zeros(
            (self.matrix.size, node.input_size),
            dtype=node.config.PRECISION,
            device=node.config.DEVICE
        )
        matrix[node.slices] = self.matrix

        return Equation(matrix, self.const, self.config)

    def _transpose_concat(self, node:Node):
        eqs, idx = [], 0
        for i in node.from_node:
            eqs.append(Equation(
                self.matrix[:, torch.arange(idx, idx + i.output_size)],
                self.const,
                self.config
            ))

        return eqs

    def _transpose_sub(self, node:Node):
        if node.const is None:
            return Equation(
                torch.hstack([self.matrix, -self.matrix], self.const, self.config)
            )
            
        return Equation(
            self.matrix - node.const.flatten(), self.const, self.config
        )
  
 
    def interval_transpose(self, node, bound):

        in_flag, out_flag = self._get_flags(node)

        if node.has_relu_activation():
            return [self.interval_relu_transpose(node, bound, in_flag, out_flag)]

        elif isinstance(node, MaxPool):
            return [self.interval_maxpool_transpose(node, bound)]

        else:
            raise TypeError("Expected either relu or maxpool subsequent layer")


    def interval_relu_transpose(self, node: None, bound: str, in_flag: torch.tensor, out_flag:torch.tensor):
        (lower_slope, upper_slope), (lower_const, upper_const) = self.get_relu_relaxation(node, out_flag)
        _plus, _minus = self._get_plus_matrix(), self._get_minus_matrix()

        if bound == 'lower':
            plus, minus = _plus * lower_slope, _minus * upper_slope
            const = _plus @ lower_const + _minus @ upper_const

        elif bound == 'upper':
            plus, minus = _plus  * upper_slope, _minus * lower_slope
            const = _plus @ upper_const + _minus @ lower_const

        else:
            raise ValueError(f'Bound {bound} is not recognised.')

        if out_flag is None:
            shape = (self.size,) + node.output_shape_no_batch()
        else:
            shape = self.matrix.shape

        matrix = node.transpose(plus.reshape(shape), in_flag, out_flag) + \
            node.transpose(minus.reshape(shape), in_flag, out_flag)

        const += self.const

        return Equation(matrix, const, self.config)


    def get_relu_relaxation(self, node: Node, out_flag: torch.tensor) -> tuple:
        if out_flag is None:
            lower_slope = torch.ones(
                node.output_size, dtype=self.config.PRECISION, device=self.config.DEVICE
            )
        
            lower, upper = node.bounds.lower.flatten(), node.bounds.upper.flatten()
            idxs = abs(lower) >=  upper
            lower_slope[idxs] = 0.0

            upper_slope = torch.zeros(
                node.output_size, dtype=self.config.PRECISION, device=self.config.DEVICE
            )
            idxs = node.to_node[0].get_unstable_flag().flatten()
            lower, upper = lower[idxs], upper[idxs]
            upper_slope[idxs] =  upper / (upper - lower)
            upper_slope[node.to_node[0].get_active_flag().flatten()] = 1


            # upper_slope = torch.ones(
                # node.output_size, dtype=self.config.PRECISION, device=self.config.DEVICE
            # )
            # idxs = node.to_node[0].get_unstable_flag().flatten()
            # upper_slope[idxs] = upper[idxs] /  (upper[idxs] - lower[idxs])
            # upper_slope[node.to_node[0].get_inactive_flag().flatten()] = 0

            lower_const = Equation._derive_const(node)
            upper_const = lower_const.detach().clone()
            lower_const *= lower_slope
            upper_const *= upper_slope
            upper_const[idxs]  -= upper_slope[idxs] *  lower

        else:
            lower_slope = torch.ones(
                node.to_node[0].get_propagation_count(),
                dtype=self.config.PRECISION,
                device=self.config.DEVICE
            )
        
            upper = node.bounds.upper[out_flag].flatten()
            lower = node.bounds.lower[out_flag].flatten()
            idxs = abs(lower) >=  upper
            lower_slope[idxs] = 0.0

            upper_slope = torch.ones(
                node.to_node[0].get_propagation_count(),
                dtype=self.config.PRECISION,
                device=self.config.DEVICE
            )
            idxs = lower < 0
            upper_slope[idxs] = upper[idxs] /  (upper[idxs] - lower[idxs])

            lower_const = Equation._derive_const(
                node, out_flag.flatten()
            )
            upper_const = lower_const.detach().clone()
            lower_const *= lower_slope
            upper_const *= upper_slope
            upper_const[idxs]  -= upper_slope[idxs] *  lower[idxs]

        # print('oo', torch.mean(upper_slope))
        return (lower_slope, upper_slope), (lower_const, upper_const)

    def __get_relu_relaxation(self, node: Node) -> tuple:
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

        lower_const = Equation._derive_const(node)
        upper_const = lower_const.detach().clone()
        lower_const *= lower_slope
        upper_const *= upper_slope
        upper_const[node.to_node[0].get_unstable_flag()]  -= \
            upper_slope[node.to_node[0].get_unstable_flag()] * \
            node.bounds.lower.flatten()[node.to_node[0].get_unstable_flag()]

        return (lower_slope, upper_slope), (lower_const, upper_const)

    def interval_maxpool_transpose(self, node: Node, bound:str):
        lower, indices = node.forward(node.from_node[0].bounds.lower, return_indices=True)
        
        idx_correction = torch.tensor(
            [i * node.from_node[0].out_ch_sz() for i in range(node.from_node[0].out_ch())]
        ).reshape((node.from_node[0].out_ch(), 1, 1))
        if node.has_batch_dimension():
            idx_correction = idx_correction[None, :]
        indices = indices + idx_correction

        lower, indices = lower.flatten(), indices.flatten()
        upper = node.forward(node.from_node[0].bounds.upper).flatten()
        lower_max  = lower > upper
        not_lower_max = torch.logical_not(lower_max)

        _plus = self._get_plus_matrix()
        _minus = self._get_minus_matrix()

        upper_const = torch.zeros(
            node.output_size, dtype=self.config.PRECISION
        )
        upper_const[not_lower_max] = node.from_node[0].bounds.upper.flatten()[indices][not_lower_max]

        if bound == 'lower':
            plus_lower = torch.zeros(
                (self.size, node.input_size), dtype=self.config.PRECISION
            )
            plus_lower[:, indices] = _plus
            minus_upper = torch.zeros(
                (self.size, node.input_size), dtype=self.config.PRECISION
            )
            temp = minus_upper[:, indices]
            temp[:, lower_max] = _minus[:, lower_max]
            minus_upper[:, indices] = temp
            del temp
            matrix = plus_lower + minus_upper

            const = _minus @ upper_const + self.const

        elif bound == 'upper':
            minus_lower = torch.zeros(
                (self.size, node.input_size), dtype=self.config.PRECISION
            )
            minus_lower[:, indices] = _minus
            plus_upper = torch.zeros(
                (self.size, node.input_size), dtype=self.config.PRECISION
            )
            temp = plus_upper[:, indices]
            temp[:, lower_max] = _plus[:, lower_max]
            plus_upper[:, indices] = temp
            del temp
            matrix = minus_lower + plus_upper

            const = _plus @ upper_const + self.const

        else:
            raise ValueError(f'Bound {bound} is not recognised.')

        return Equation(matrix, const, self.config)


        # matrix = torch.zeros(
            # (node.output_size, node.input_size), dtype=self.config.PRECISION, device=self.config.DEVICE
        # )
        # matrix[indices] = 1.0
        # const = torch.zeros(
            # node.output_size, dtype=self.config.PRECISION, device=self.config.DEVICE
        # )
        # lower = Equation(matrix, const, self.config)
            
        # matrix = torch.zeros(
            # (self.output_size, node.input_size), dtype=self.config.PRECISION, device=self.config.DEVICE
        # )
        # matrix[indices][lower_max] = 1.0
        # const = torch.zeros(
            # node.output_size, dtype=self.config.PRECISION, device=self.config.DEVICE
        # ) 
        # const[indices][not_lower_max] = node.bounds.upper.flatten()[indices][not_lower_max]
        # upper = Equation(matrix, const, self.config)

        # return lower, upper

    def get_max_pool_relaxation(self, node: Node) -> tuple:
        lower, indices = node.to_node[0].forward(node.bounds.lower, return_indices=True)
        
        idx_correction = torch.tensor(
            [i * node.out_ch_sz() for i in range(node.out_ch())]
        ).reshape((node.out_ch(), 1, 1))
        if node.has_batch_dimension():
            idx_correction = idx_correction[None, :]
        indices = indices + idx_correction

        lower, indices = lower.flatten(), indices.flatten()
        upper = node.to_node[0].forward(node.bounds.upper).flatten()
        lower_max  = lower > upper
        not_lower_max = torch.logical_not(lower_max)

        lower_slope = torch.zeros(
            node.input_size, dtype=self.config.PRECISION, device=self.config.DEVICE
        )
        lower_slope[indices] = 1.0
        lower_const = torch.zeros(
            node.input_size, dtype=self.config.PRECISION, device=self.config.DEVICE
        )
        lower_const[indices] = Equation._derive_const(node)[indices]

        upper_slope = torch.zeros(
            node.input_size, dtype=self.config.PRECISION, device=self.config.DEVICE
        )
        upper_slope[indices][lower_max] = 1.0
        upper_const = torch.zeros(
            node.input_size, dtype=self.config.PRECISION, device=self.config.DEVICE
        )
        upper_const[indices][lower_max] = Equation._derive_const(node)[indices][lower_max]
        upper_const[indices][not_lower_max] = node.bounds.upper.flatten()[indices][not_lower_max]

        return (lower_slope, upper_slope), (lower_const, upper_const)


    @staticmethod
    def derive(node: Node, out_flag: torch.Tensor, in_flag: torch.Tensor, config: Config) -> Equation:
        zero_eq = Equation._zero_eq(node, out_flag)
       
        if zero_eq is not None:
            return zero_eq
      
        else:
            return Equation(
                Equation._derive_matrix(node, out_flag, in_flag),
                Equation._derive_const(node, out_flag),
                config
            )


    @staticmethod 
    def _zero_eq(node: Node, flag: torch.Tensor) -> Equation:
        if isinstance(node.from_node[0], Relu) and \
        node.from_node[0].get_propagation_count() == 0:
            return Equation(
                np.zeros((torch.sum(flag), 0)),
                Equation._derive_const(node, flag),
                node.config
            )

        return None

    @staticmethod
    def _derive_matrix(node: Node, out_flag: torch.Tensor=None, in_flag: torch.Tensor=None):
        out_flag = torch.ones(node.output_size, dtype=torch.bool) if out_flag is None else out_flag
        
        if isinstance(node, Conv):
            return Equation._derive_conv_matrix(node, out_flag, in_flag)

        elif isinstance(node, Gemm):
            return Equation._derive_gemm_matrix(node, out_flag, in_flag)

        elif isinstance(node, MatMul):
            return Equation._derive_matmul_matrix(node, out_flag, in_flag)
        
        elif isinstance(node, Add):
            return Equation._derive_add_matrix(node, out_flag)

        else:
            raise NotImplementedError(f'{type(node)} equations')

    @staticmethod
    def _derive_const(node: Node, flag: torch.Tensor=None):
        if isinstance(node, Conv):
            return Equation._derive_conv_const(node, flag)

        if isinstance(node, ConvTranspose):
            return Equation._derive_convtranspose_const(node, flag)

        if isinstance(node, Gemm):
            return Equation._derive_gemm_const(node, flag)

        if isinstance(node, MatMul):
            return Equation._derive_matmul_const(node, flag)

        if isinstance(node, BatchNormalization):
            return Equation._derive_batchnormalization_const(node, flag)
        
        if isinstance(node, Add):
            return Equation._derive_add_const(node, flag)

        raise NotImplementedError(f'{type(node)} equations')

    @staticmethod 
    def _derive_conv_matrix(node: Node, out_flag: torch.Tensor, in_flag: torch.Tensor=None):
        flag_size = torch.sum(out_flag).item()

        prop_flag = torch.zeros(node.get_input_padded_size(), dtype=torch.bool)
        if in_flag is None:
            prop_flag[node.get_non_pad_idxs()] = True
            max_index = node.input_size
        else:
            prop_flag[node.get_non_pad_idxs()] = in_flag.flatten()
            max_index = node.from_node[0].get_propagation_count()

        pad = torch.ones(
            node.get_input_padded_size(),
            dtype=torch.long
        ) * max_index
        pad[prop_flag] = torch.arange(max_index)

        im2col = Conv.im2col(
            pad.reshape(node.get_input_padded_shape()),
            (node.krn_height, node.krn_width),
            node.strides
        )
        indices = torch.arange(node.out_ch_sz).repeat(node.out_ch)[out_flag]
        conv_indices = im2col[:, indices]

        indices = torch.repeat_interleave(
            torch.arange(node.out_ch), node.out_ch_sz, dim=0
        )[out_flag]
        conv_weights = node.kernels.permute(1, 2, 3, 0).reshape(-1, node.out_ch)[:, indices]
            
        matrix = torch.zeros(
            (flag_size, max_index + 1), dtype=node.config.PRECISION
        )
        matrix[torch.arange(flag_size), conv_indices] = conv_weights
        matrix = matrix[:, :max_index]

        return matrix

    @staticmethod 
    def _derive_conv_const(node: Node, flag: torch.Tensor):
        if flag is None:
            return torch.tile(node.bias, (node.out_ch_sz, 1)).T.flatten()

        return torch.tile(node.bias, (node.out_ch_sz, 1)).T.flatten()[flag]

    @staticmethod 
    def _derive_convtranspose_const(node: Node, flag: torch.Tensor):
        if flag is None:
            return torch.tile(node.bias, (node.out_ch_sz, 1)).T.flatten()

        return torch.tile(node.bias, (node.out_ch_sz, 1)).T.flatten()[flag]

    @staticmethod 
    def _derive_gemm_matrix(node: Node, out_flag: torch.Tensor, in_flag: torch.Tensor=None):
        if in_flag is None:
            return  node.weights[out_flag, :]

        return  node.weights[out_flag, :][:, in_flag]

    @staticmethod 
    def _derive_gemm_const(node: Node, flag: torch.Tensor):
        if flag is None:
            return node.bias

        return node.bias[flag]

    @staticmethod 
    def _derive_matmul_matrix(node: Node, out_flag: torch.Tensor, in_flag:torch.Tensor=None):
        if in_flag is None:
            return  node.weights[out_flag, :]
            
        return  node.weights[out_flag, :][:, in_flag]

    @staticmethod 
    def _derive_matmul_const(node: Node, flag: torch.Tensor):
        if flag is None:
            return torch.zeros(node.output_size, dtype=node.weights.dtype)

        return torch.zeros(torch.sum(flag), dtype=node.weights.dtype )

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
        if flag is None:
            return torch.zeros(
                node.output_size, dtype=node.config.PRECISION, device=node.config.DEVICE
            )

        return torch.zeros(
            torch.sum(flag).item(), dtype=node.config.PRECISION, device=node.config.DEVICE
        )

    @staticmethod
    def _derive_batchnormalization_const(node: Node, flag: torch.Tensor):
        if flag is None:
            return torch.zeros(
                node.output_size, dtype=node.config.PRECISION, device=node.config.DEVICE
            )

        return torch.zeros(
            torch.sum(flag).item(), dtype=node.config.PRECISION, device=node.config.DEVICE
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
