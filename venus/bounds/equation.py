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
        self.size, self.coeffs_size = matrix.shape


    def copy(self) -> Equation:
        return Equation(
            self.matrix.detach().clone(),
            self.const.detach().clone(),
            self.config
        )


    def zero(self) -> Equation:
        """
        Returns whether the equation is the zero function.
        """
        return self.size == 0

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
 
    def get_plus_matrix(self) -> torch.Tensor:
        """
        Clips the coeffs to be only positive.
        """

        return torch.clamp(self.matrix, 0, math.inf)


    def get_minus_matrix(self, keep_in_memory=True) -> torch.Tensor:
        """
        Clips the coeffs to be only negative.
        """     
        return torch.clamp(self.matrix, -math.inf, 0)


    def concrete_values(
        self, lower: torch.Tensor, upper:torch.Tensor, bound: str
    ) -> torch.Tensor:
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


    def interval_dot(
        self, bound: str, lower: torch.Tensor, upper: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the interval dot product with either a matrix or an Equation.
        """
        if isinstance(lower, Equation) and isinstance(upper, Equation):
            return self._interval_dot_eq(bound, lower, upper)

        elif isinstance(lower, torch.Tensor) and isinstance(upper, torch.Tensor):
            return self._interval_dot_tensor(bound, lower, upper)

        else: 
            raise TypeError(f'Got {type(lower)} and {type(upper)} but expected either trensor or Equation')

    def _interval_dot_eq(
        self, bound: str, lower: torch.Tensor, upper: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the interval dot product with an Equation.
        """
        plus, minus = self.get_plus_matrix(), self.get_minus_matrix()

        if bound == 'upper':
            matrix =  plus @ upper.matrix + minus @ lower.matrix
            const = plus @ upper.const + minus @ lower.const + self.const

        elif bound == 'lower':
            matrix = plus @ lower.matrix + minus @ upper.matrix
            const = plus @ lower.const + minus @ upper.const + self.const

        else: 
            raise ValueError(f'Bound {bound} is not recognised.')

        return Equation(matrix, const, self.config)


    def _interval_dot_tensor(
        self, bound: str, lower: torch.Tensor, upper: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the interval dot product with either a tensor
        """
        if bound == 'upper':
            return self.get_plus_matrix() @ upper + \
                self.get_minus_matrix() @ lower + \
                self.const

        elif bound == 'lower':
            return  self.get_plus_matrix() @ lower + \
                self.get_minus_matrix() @ upper + \
                self.const

        else: 
            raise ValueError(f'Bound {bound} is not recognised.')

    def get_relu_slope(
        self, 
        node: Node,
        slope_type: str,
        bound: str,
        out_flag: torch.tensor=None,
        slopes: torch.tensor=None
    ) -> torch.Tensor:
        if out_flag is None:
            if slope_type  == 'lower':
                sl = torch.ones(
                    node.output_size,
                    dtype=self.config.PRECISION,
                    device=self.config.DEVICE
                )
                sl[node.to_node[0].get_inactive_flag().flatten()] = 0.0
                idxs = node.to_node[0].get_unstable_flag().flatten()
                if slopes is None:
                    slopes = node.to_node[0].get_lower_relaxation_slope()
                    sl[idxs] = slopes[0] if bound == 'lower' else slopes[1]
                else:
                    sl[idxs] = slopes

            elif slope_type == 'upper':
                sl = torch.zeros(
                    node.output_size,
                    dtype=self.config.PRECISION,
                    device=self.config.DEVICE
                )
                idxs = node.to_node[0].get_unstable_flag().flatten()
                lower = node.bounds.lower.flatten()[idxs] 
                upper = node.bounds.upper.flatten()[idxs]
                sl[idxs] =  upper / (upper - lower)
                sl[node.to_node[0].get_active_flag().flatten()] = 1

            else:
                raise Exception(f"Slope type {slope_type} is not recognised.")

        else:
            if slope_type == 'lower':
                sl = torch.ones(
                    node.to_node[0].get_propagation_count(),
                    dtype=self.config.PRECISION,
                    device=self.config.DEVICE
                ) 
                upper = node.bounds.upper[out_flag].flatten()
                lower = node.bounds.lower[out_flag].flatten()
                if slopes is None:
                    idxs = lower < 0 
                    slopes = node.to_node[0].get_lower_relaxation_slope()
                    sl[idxs] = slopes[0] if bound == 'lower' else slopes[1]
                else:
                    idxs = abs(lower) >=  upper
                    sl[idxs] = 0.0

            elif slope_type == 'upper':
                lower = node.bounds.lower[out_flag].flatten()
                upper = node.bounds.upper[out_flag].flatten()
                
                sl = torch.ones(
                    node.to_node[0].get_propagation_count(),
                    dtype=self.config.PRECISION,
                    device=self.config.DEVICE
                )
                idxs = lower < 0
                sl[idxs] = upper[idxs] /  (upper[idxs] - lower[idxs])

            else:
                raise Exception(f"Slope type {slope_type} is not recognised.")

        return sl

    def get_relu_const(
        self, 
        node: Node, 
        const: torch.tensor,
        const_type :str,  
        relu_slope: torch.Tensor,
        out_flag: torch.Tensor=None
    ) -> torch.Tensor:
        """
        Derives the constant tensor of a relu relaxation.

        Arguments:
            node:
                Node with relu activation.
            const:
                The constant tensor of the node.
            const_type:
                Either 'lower' relu const or 'upper' relu const.
            relu_slope:
                The slope tensor of the relu relaxation.
            out_flag:
                Binary flag of output indices to return.

        Returns:
            The constant tensor of a relu relaxation.
        """
        if out_flag is None:
            if const_type == 'lower':
                relu_const = const * relu_slope

            elif const_type == 'upper':
                idxs = node.to_node[0].get_unstable_flag().flatten()
                lower  = node.bounds.lower.flatten()[idxs]
                relu_const = const * relu_slope
                relu_const[idxs]  -= relu_slope[idxs] * lower

            else:
                raise Exception(f"Const type {const_type} is not recognised.")

        else:
            if const_type == 'lower':
                relu_const = const * relu_slope
            
            elif const_type == 'upper':
                lower = node.bounds.lower.flatten()[out_flag]
                idxs = lower < 0
                relu_const = const * relu_slope
                relu_const[idxs]  -= relu_slope[idxs] *  lower[idxs]

        return relu_const

    def get_max_pool_relaxation(self, node: Node) -> tuple:
        lower, indices = node.to_node[0].forward(node.bounds.lower, return_indices=True)
        
        idx_correction = torch.tensor(
            [i * node.in_ch_sz() for i in range(node.in_ch())],
            dtype=torch.long, 
            device=self.config.DEVICE
        ).reshape((node.in_ch(), 1, 1))
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
        lower_const[indices] = Equation.derive_const(node)[indices]

        upper_slope = torch.zeros(
            node.input_size, dtype=self.config.PRECISION, device=self.config.DEVICE
        )
        upper_slope[indices][lower_max] = 1.0
        upper_const = torch.zeros(
            node.input_size, dtype=self.config.PRECISION, device=self.config.DEVICE
        )
        upper_const[indices][lower_max] = Equation.derive_const(node)[indices][lower_max]
        upper_const[indices][not_lower_max] = node.bounds.upper.flatten()[indices][not_lower_max]

        return (lower_slope, upper_slope), (lower_const, upper_const)


    @staticmethod
    def derive(
        node: Node,
        config: Config,
        out_flag: torch.Tensor=None,
        in_flag: torch.Tensor=None,
        sparse: bool=False
    ) -> Equation:

        if out_flag is None:
            out_flag = torch.ones(
                node.output_size, dtype=torch.bool, device=node.config.DEVICE
            )
        else:
            out_flag = out_flag.flatten()

        zero_eq = Equation._zero_eq(node, out_flag)
        if zero_eq is not None:
            return zero_eq
      
        else:
            return Equation(
                Equation._derive_matrix(node, out_flag, in_flag, sparse),
                Equation.derive_const(node, out_flag),
                config
            )

    @staticmethod 
    def _zero_eq(node: Node, flag: torch.Tensor) -> Equation:
        if isinstance(node.from_node[0], Relu) and \
        node.from_node[0].get_propagation_count() == 0:
            return Equation(
                torch.zeros(
                    (torch.sum(flag), 0),
                    dtype=node.config.PRECISION,
                    device=node.config.DEVICE
                ),
                Equation.derive_const(node, flag),
                node.config
            )

        return None

    @staticmethod
    def _derive_matrix(
            node: Node,
            out_flag: torch.Tensor=None,
            in_flag: torch.Tensor=None,
            sparse: bool=False
    ):    
        if isinstance(node, Conv):
            return Equation._derive_conv_matrix(node, out_flag, in_flag, sparse)

        if isinstance(node, Gemm):
            return Equation._derive_gemm_matrix(node, out_flag, in_flag)

        if isinstance(node, MatMul):
            return Equation._derive_matmul_matrix(node, out_flag, in_flag)
        
        if isinstance(node, Add):
            return Equation._derive_add_matrix(node, out_flag)

        if isinstance(node, Flatten):
            return Equation._derive_flatten_matrix(node, out_flag, in_flag)

        if isinstance(node, BatchNormalization):
            return Equation._derive_batch_normalization_matrix(node, out_flag, in_flag)

        if isinstance(node, Slice):
            return Equation._derive_slice_matrix(node, out_flag)

        raise NotImplementedError(f'{type(node)} equations')

    @staticmethod 
    def _derive_conv_matrix(
            node: Node,
            out_flag: torch.Tensor,
            in_flag: torch.Tensor=None,
            sparse: bool=False
    ):
        flag_size = torch.sum(out_flag).item()

        prop_flag = torch.zeros(
            node.get_input_padded_size(), dtype=torch.bool, device=node.config.DEVICE
        )
        if in_flag is None:
            prop_flag[node.get_non_pad_idxs()] = True
            max_index = node.input_size
        else:
            prop_flag[node.get_non_pad_idxs()] = in_flag.flatten()
            max_index = node.from_node[0].get_propagation_count()

        pad = torch.ones(
            node.get_input_padded_size(),
            dtype=torch.long,
            device=node.config.DEVICE
        ) * max_index
        pad[prop_flag] = torch.arange(max_index, device=node.config.DEVICE)

        im2col = Conv.im2col(
            pad.reshape(node.get_input_padded_shape()),
            (node.krn_height, node.krn_width),
            node.strides,
            device=node.config.DEVICE
        )
        indices = torch.arange(
            node.out_ch_sz,
            device=node.config.DEVICE
        ).repeat(node.out_ch)[out_flag]
        conv_indices = im2col[:, indices]

        indices = torch.repeat_interleave(
            torch.arange(node.out_ch, device=node.config.DEVICE), node.out_ch_sz, dim=0
        )[out_flag]
        conv_weights = node.kernels.permute(1, 2, 3, 0).reshape(-1, node.out_ch)[:, indices]
       
        if sparse is True:
            matrix =[
                {
                    conv_indices[i, eq]: conv_weights[i, eq].item()
                    for i in range(np.prod(node.kernels.shape[1:]))
                    if not conv_indices[i, eq] == max_index
                }
                for eq in range(flag_size)
            ]
        
        else:
            matrix = torch.zeros(
                (flag_size, max_index + 1),
                dtype=node.config.PRECISION,
                device=node.config.DEVICE
            )
            matrix[torch.arange(flag_size), conv_indices] = conv_weights
            matrix = matrix[:, :max_index]

        return matrix

    @staticmethod 
    def _derive_gemm_matrix(
            node: Node, out_flag: torch.Tensor, in_flag: torch.Tensor=None
    ):
        if in_flag is None:
            # matrix = node.weights[:, out_flag].T 
            matrix = torch.zeros((node.output_size, node.input_size))
            for i in range(np.prod(node.input_shape[:-1])):
                start1, end1 = i * node.weights.shape[1], (i + 1) * node.weights.shape[1]
                start2, end2 = i * node.weights.shape[0], (i + 1) * node.weights.shape[0]
                temp = matrix[start1 : end1, :]
                temp[:, start2 : end2] = node.weights.T
                matrix[start1 : end1, :] = temp
        else:
            raise NotImplementedError('Gemm with specified input indices')

            # matrix = torch.tile(node.weights.T, node.input_shape[:-1] + (1, 1))
            # matrix = matrix[]
            # matrix = node.weights[in_flag, :][:, out_flag].T

        return matrix[out_flag, :]


    @staticmethod 
    def _derive_matmul_matrix(
            node: Node, out_flag: torch.Tensor, in_flag:torch.Tensor=None
    ):
        if in_flag is None:
            matrix = node.weights[:, out_flag].T
        else:
            matrix = node.weights[in_flag, :][:, out_flag].T

        return matrix

    @staticmethod 
    def _derive_batch_normalization_matrix(
            node: Node, out_flag: torch.Tensor, in_flag: torch.Tensor=None
    ):
        in_ch_sz = node.in_ch_sz()
        scale = torch.tile(node.scale, (in_ch_sz, 1)).T.flatten()
        bias = torch.tile(node.bias, (in_ch_sz, 1)).T.flatten()
        input_mean = torch.tile(node.input_mean, (in_ch_sz, 1)).T.flatten()
        var = torch.sqrt(node.input_var + node.epsilon)
        var = torch.tile(var, (in_ch_sz, 1)).T.flatten()
        scale_var = scale / var

        matrix = torch.zeros(
            (node.output_size, node.output_size),
            dtype=node.config.PRECISION,
            device=node.config.DEVICE
        )
        matrix[torch.eye(node.output_size).bool()] = scale_var

        return matrix[out_flag, :]

    @staticmethod 
    def _derive_add_matrix(node: Node, flag: torch.Tensor):
        return  torch.eye(
            node.input_size, dtype=node.config.PRECISION, device=node.config.DEVICE
        )[flag, :]

    @staticmethod 
    def _derive_flatten_matrix(
        node: Node, out_flag:torch.tensor, in_flag: torch.tensor=None
    ):
        if in_flag is None:
            matrix = torch.eye(
                node.output_size,
                dtype=node.config.PRECISION,
                device=node.config.DEVICE
            ).squeeze()[out_flag, :]
        else:
            matrix = torch.eye(
                node.output_size,
                dtype=node.config.PRECISION,
                device=node.config.DEVICE
            ).squeeze()[out_flag, :][:, in_flag]

        return matrix

    @staticmethod 
    def _derive_slice_matrix(
        node: Node, out_flag:torch.tensor
    ):
        matrix = torch.eye(
            node.input_size,
            dtype=node.config.PRECISION,
            device=node.config.DEVICE
        ).squeeze()
        shape = (node.input_size,) + node.input_shape_no_batch()
        slices = [slice(0, node.input_size)] + node.slices
        matrix = matrix.reshape(shape)[slices]
        matrix = matrix.reshape(node.input_size, -1).T

        return matrix

    @staticmethod
    def derive_const(node: Node, flag: torch.Tensor=None):
        if isinstance(node, Conv):
            return Equation._derive_conv_const(node, flag)

        if isinstance(node, ConvTranspose):
            return Equation._derive_convtranspose_const(node, flag)

        if isinstance(node, Gemm):
            return Equation._derive_gemm_const(node, flag)

        if isinstance(node, BatchNormalization):
            return Equation._derive_batch_normalization_const(node, flag)
        
        if type(node) in [Add, Flatten, Slice, MatMul]:
            return Equation._derive_zero_const(node, flag)

        raise NotImplementedError(f'{type(node)} equations')

    @staticmethod 
    def _derive_gemm_const(node: Node, flag: torch.Tensor): 
        if node.has_bias() is True:
            const = torch.tile(node.bias, (np.prod(node.input_shape[:-1]),))
            const = const if flag is None else const[flag]
        else:
            size = node.output_size if flag is None else torch.sum(flag)
            const = torch.zeros(
                size, dtype=node.config.PRECISION, device=node.confog.DEVICE
            )
     
        return const

    @staticmethod 
    def _derive_conv_const(node: Node, flag: torch.Tensor):
        if node.has_bias() is True:
            if flag is None:
                const =  torch.tile(node.bias, (node.out_ch_sz, 1)).T.flatten()
            else:
                const = torch.tile(node.bias, (node.out_ch_sz, 1)).T.flatten()[flag]
        
        else:
            size = node.output_size if flag is None else torch.sum(flag)
            const = torch.zeros(
                size, dtype=node.config.PRECISION, device=node.config.DEVICE
            )

        return const

    @staticmethod 
    def _derive_convtranspose_const(node: Node, flag: torch.Tensor):
        if node.has_bias() is True:
            if flag is None:
                const = torch.tile(node.bias, (node.out_ch_sz, 1)).T.flatten()
            else:
                const = torch.tile(node.bias, (node.out_ch_sz, 1)).T.flatten()[flag]
        
        else:
            size = node.output_size if flag is None else torch.sum(flag)
            const = torch.zeros(
                size, dtype=node.config.PRECISION, device=node.confog.DEVICE
            )

        return const

    @staticmethod 
    def _derive_zero_const(node: Node, flag: torch.Tensor):
        if flag is None:
            const =  torch.zeros(
                node.output_size, dtype=node.config.PRECISION, device=node.config.DEVICE
            )
        else:
            const = torch.zeros(
                torch.sum(flag), dtype=node.config.PRECISION, device=node.config.DEVICE
            )

        return const

    @staticmethod
    def _derive_batch_normalization_const(node: Node, flag: torch.Tensor):
        in_ch_sz = node.in_ch_sz()

        scale = torch.tile(node.scale, (in_ch_sz, 1)).T.flatten()
        bias = torch.tile(node.bias, (in_ch_sz, 1)).T.flatten()
        input_mean = torch.tile(node.input_mean, (in_ch_sz, 1)).T.flatten()
        var = torch.sqrt(node.input_var + node.epsilon)
        var = torch.tile(var, (in_ch_sz, 1)).T.flatten()

        const = - input_mean / var * scale + bias
        if flag is not None:
            const = const[flag]

        return const
