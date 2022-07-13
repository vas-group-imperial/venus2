"""
# File: os_sip.py
# Top contributors (to current version):
# 	Panagiotis Kouvaros (panagiotis.kouvaros@gmail.com)
# This file is part of the Venus project.
# Copyright: 2019-2021 by the authors listed in the AUTHORS file in the
# top-level directory.
# License: BSD 2-Clause (see the file LICENSE in the top-level directory).
# Description: One Step Symbolic Interval Propagation.
"""

from venus.network.node import *
from venus.bounds.bounds import Bounds
from venus.bounds.equation import Equation
from venus.common.logger import get_logger
from venus.common.configuration import Config

torch.set_num_threads(1)

class OSSIP:

    logger = None

    def __init__(self, prob, config: Config):
        """
        Arguments:

            prob:
                The verification problem.
            config:
                Configuration.
        """
        self.prob = prob
        self.config = config
        if OSSIP.logger is None and config.LOGGER.LOGFILE is not None:
            OSSIP.logger = get_logger(__name__, config.LOGGER.LOGFILE)

        self.lower_eq, self.upper_eq = {}, {}
        self.current_lower_eq, self.current_upper_eq = None, None
    
    def calc_bounds(self, node: Node) -> Bounds:
        input_lower = self.prob.spec.input_node.bounds.lower.flatten()
        input_upper = self.prob.spec.input_node.bounds.upper.flatten()

        lower = self.current_lower_eq.min_values(
            input_lower, input_upper
        ).reshape(node.output_shape)
        lower = torch.max(node.bounds.lower, lower)

        upper = self.current_upper_eq.max_values(
            input_lower, input_upper
        ).reshape(node.output_shape)
        upper = torch.min(node.bounds.upper, upper)

        return Bounds(lower, upper)


    def clear_equations(self):
        max_depth = max([
            self.prob.nn.node[i].depth for i in self.lower_eq
        ])
        idxs = []

        for i in self.lower_eq:
            cond1 = self.prob.nn.node[i].get_prv_non_relu().depth >= max_depth - 1
            cond2 = self.prob.nn.node[i].has_sip_branching_node()
            cond3 = np.any([
                j.is_sip_branching_node() and j.depth >= max_depth - 1
                for j in self.prob.nn.node[i].to_node
            ])
            if cond1 or (cond2 and cond3):
                idxs.append(i)

        self.lower_eq = {
            i: self.lower_eq[i] for i in idxs
        }
        self.upper_eq = {
            i: self.upper_eq[i] for i in idxs
        }

    def forward(self, node: Node, slopes: tuple=None) -> None:
        non_linear_starting_depth = self.prob.nn.get_non_linear_starting_depth()
        if node.depth == 1:
            lower_eq = Equation.derive(node, self.config)
            upper_eq = lower_eq

        elif node.depth < non_linear_starting_depth:
            lower_eq = self._forward(node)
            upper_eq = lower_eq

        else: 
            if slopes is not None and isinstance(node, Relu):
                l_slopes, u_slopes = slopes[0][node.id], slopes[1][node.id]
            else:
                l_slopes, u_slopes = None, None

            if node.depth == non_linear_starting_depth + 1:
                for i, j in self.upper_eq.items():
                    if self.prob.nn.node[i].depth == non_linear_starting_depth \
                    and self.prob.nn.node[i] in node.from_node:
                        j = self.lower_eq[i].copy()
                        self.current_upper_eq = j
            lower_eq = self._int_forward(node, 'lower', l_slopes)
            upper_eq = self._int_forward(node, 'upper', u_slopes)

        self.current_lower_eq, self.current_upper_eq = lower_eq, upper_eq
        self.lower_eq[node.id], self.upper_eq[node.id] = lower_eq, upper_eq

    def _forward(self, node: Node) -> Equation:
        equation = self.lower_eq[node.from_node[0].id]

        if isinstance(node, Gemm):
            f_equation =  self._forward_gemm(equation, node)

        elif type(node) in [Conv, ConvTranspose]:
            f_equation = self._forward_conv(equation, node)

        elif isinstance(node, MatMul):
            f_equation = self._forward_matmul(equation, node)

        elif isinstance(node, AveragePool):
            f_equation = self._forward_average_pool(equation, node)

        elif isinstance(node, Slice):
            f_equation = self._forward_slice(equation, node)

        elif isinstance(node, Pad):
            f_equation = self._forward_pad(equation, node)

        elif isinstance(node, Concat):
            equation = [self.lower_eq[i.id] for i in node.from_node]
            f_equation = self._forward_concat(equation, node)

        elif type(node) in [Flatten, Reshape, Unsqueeze]:
            f_equation = equation

        else:
            raise NotImplementedError(f'One step equation forward for {type(node)}')

        return f_equation
    
    def _forward_gemm(self, equation: Equation, node: Node):
        shape = (equation.coeffs_size,) + node.input_shape_no_batch()
        matrix = node.forward(equation.matrix.T.reshape(shape), add_bias=False)
        matrix = matrix.reshape(equation.coeffs_size, -1).T
        const = node.forward(
            equation.const.reshape(node.input_shape), 
            add_bias=True
        ).flatten()

        return Equation(matrix, const, self.config)

    def _forward_conv(self, equation: Equation, node: Node):
        shape = (equation.coeffs_size,) + node.input_shape_no_batch()
        matrix = node.forward(equation.matrix.T.reshape(shape), add_bias=False)
        matrix = matrix.reshape(equation.coeffs_size, -1).T
        const = node.forward(
            equation.const.reshape(node.input_shape), add_bias=True
        ).flatten()

        return Equation(matrix, const, self.config)
    
    def _forward_matmul(self, equation: Equation, node: Node):
        shape = (equation.coeffs_size,) + node.input_shape_no_batch()
        matrix = node.forward(equation.matrix.T.reshape(shape))
        matrix = matrix.reshape(equation.coeffs_size, -1).T
        const = node.forward(
            equation.const.reshape(node.input_shape)
        ).flatten()

        return Equation(matrix, const, self.config)

    def _forward_average_pool(self, equation: Equation, node: Node):
        shape = (equation.coeffs_size,) + node.input_shape_no_batch()
        matrix = node.forward(equation.matrix.T.reshape(shape))
        matrix = matrix.reshape(equation.coeffs_size, -1).T
        const = node.forward(
            equation.const.reshape(node.input_shape)
        ).flatten()

        return Equation(matrix, const, self.config)

    def _forward_slice(self, equation: Equation, node: Node):
        shape = (equation.coeffs_size,) + node.input_shape_no_batch()
        slices = [slice(0, equation.coeffs_size)] + node.slices
        matrix = equation.matrix.T.reshape(shape)[slices]
        matrix = matrix.reshape(equation.coeffs_size, -1).T
        const = equation.const.reshape(node.input_shape)[node.slices].flatten()

        return Equation(matrix, const, self.config)

    def _forward_pad(self, equation: Equation, node: Node):
        shape = (equation.coeffs_size,) + node.input_shape_no_batch()
        matrix = node.forward(equation.matrix.T.reshape(shape))
        matrix = matrix.reshape(equation.coeffs_size, -1).T
        const = node.forward(
            equation.const.reshape(node.input_shape)
        ).flatten()

        return Equation(matrix, const, self.config)


    def _forward_concat(self, equations: Equation, node: Node):
        shape = (equations[0].coeffs_size,) + node.input_shape_no_batch()
        matrix = torch.cat(
            [i.matrix.T.reshape(shape) for i in equations],
            axis=node.axis + 1
        ).reshape(equations[0].coeffs_size, -1).T
        const = torch.cat(
            [i.const.reshape(node.input_shape) for i in equations],
            axis=node.axis
        ).flatten()

        return Equation(matrix, const, self.config)

    def _int_forward(self,  node: Node, bound: str, slopes: tuple=None):         
        lower = self.lower_eq[node.from_node[0].id]
        upper = self.upper_eq[node.from_node[0].id]
        in_equation = lower if bound == 'lower' else upper
        if isinstance(node, Relu): 
            equation = self._int_forward_relu(
                in_equation, node, bound, slopes
            )

        elif isinstance(node, MaxPool):
            equation = self._int_forward_max_pool(
                in_equation, node, bound
            )

        elif isinstance(node, AveragePool):
            equation = self._forward_average_pool(
                in_equation, node
            )

        elif isinstance(node, Pad):
            equation = self._forward_pad(
                in_equation, node
            )

        elif isinstance(node, Gemm):
            equation = self._int_forward_gemm(
                lower, upper, node, bound
            )

        elif type(node) in [Conv, ConvTranspose]:
            equation = self._int_forward_conv(
                lower, upper, node, bound
            )

        elif isinstance(node, BatchNormalization):
            equation = self._int_forward_batch_normalization(
                lower, upper, node, bound
            )

        elif isinstance(node, Add):
            equation = self._int_forward_add(
                in_equation, node, bound
            )

        elif isinstance(node, Concat):
            equations = self.lower_eq if bound == 'lower' else self.upper_eq
            equations = [equations[i.id] for i in node.from_node]
            return self._forward_concat(equations, node)

        elif type(node) in [Flatten, Reshape, Unsqueeze]:
            equation = in_equation

            # new_lower_eq  = self.lower_eq[node.from_node[0].id].forward(node)
 
            # elif type(node) in [BatchNormalization, Conv, Gemm]:
                # new_lower_eq  = Equation.interval_forward(
                    # node, 
                    # 'lower',
                    # self.lower_eq[node.from_node[0].id],
                    # self.upper_eq[node.from_node[0].id]
                # )
                # new_upper_eq  = Equation.interval_forward(
                    # node,
                    # 'upper',
                    # self.lower_eq[node.from_node[0].id],
                    # self.upper_eq[node.from_node[0].id]
                # )

        else:
            raise NotImplementedError(
                f'One step bound equation interval forward for node {type(node)}'
            )

        return equation

    def _int_forward_gemm(
        self, lower_eq: Equation, upper_eq: Equation, node: Node, bound: str
    ):
        shape = (lower_eq.coeffs_size,) + node.input_shape_no_batch()

        if  bound == 'lower':
            matrix = node.forward(
                lower_eq.matrix.T.reshape(shape), clip='+', add_bias=False
            )
            matrix += node.forward(
                upper_eq.matrix.T.reshape(shape), clip='-', add_bias=False
            )
            const = node.forward(
                lower_eq.const.reshape(node.input_shape), clip='+', add_bias=False
            )
            const += node.forward(
                upper_eq.const.reshape(node.input_shape), clip='-', add_bias=True
            )
            const = const.flatten()

        elif bound == 'upper':
            matrix = node.forward(
                lower_eq.matrix.T.reshape(shape), clip='-', add_bias=False
            )
            matrix += node.forward(
                upper_eq.matrix.T.reshape(shape), clip='+', add_bias=False
            )
            const = node.forward(
                lower_eq.const.reshape(node.input_shape), clip='-', add_bias=False
            )
            const += node.forward(
                upper_eq.const.reshape(node.input_shape), clip='+', add_bias=True
            )
            const = const.flatten()

        else:
            raise ValueError(f"Bound type {bound} could not be recognised.")

        matrix = matrix.reshape(lower_eq.coeffs_size, -1).T

        return Equation(matrix, const, node.config)

    def _int_forward_conv(
        self, lower_eq: Equation, upper_eq: Equation, node: Node, bound: str
    ):
        shape = (lower_eq.coeffs_size,) + node.input_shape_no_batch()

        if  bound == 'lower':
            matrix = node.forward(
                lower_eq.matrix.T.reshape(shape), clip='+', add_bias=False
            ) + node.forward(
                upper_eq.matrix.T.reshape(shape), clip='-', add_bias=False
            )
            const = node.forward(
                lower_eq.const.reshape(node.input_shape), clip='+', add_bias=False
            ).flatten()
            const += node.forward(
                upper_eq.const.reshape(node.input_shape), clip='-', add_bias=True
            ).flatten()

        elif bound == 'upper':
            matrix = node.forward(
                lower_eq.matrix.T.reshape(shape), clip='-', add_bias=False
            ) + node.forward(
                upper_eq.matrix.T.reshape(shape), clip='+', add_bias=False
            )
            const = node.forward(
                lower_eq.const.reshape(node.input_shape), clip='-', add_bias=False
            ).flatten()
            const += node.forward(
                upper_eq.const.reshape(node.input_shape), clip='+', add_bias=True
            ).flatten()

        else:
            raise ValueError(f"Bound type {bound} could not be recognised.")

        matrix = matrix.reshape(lower_eq.coeffs_size, -1).T

        return Equation(matrix, const, node.config)

    def _int_forward_batch_normalization(
        self, lower_eq: Equation, upper_eq, node: Node, bound: str
    ):
        in_ch_sz = node.in_ch_sz()
        
        scale = torch.tile(node.scale, (in_ch_sz, 1)).T.flatten()
        bias = torch.tile(node.bias, (in_ch_sz, 1)).T.flatten()
        input_mean = torch.tile(node.input_mean, (in_ch_sz, 1)).T.flatten()
        var = torch.sqrt(node.input_var + node.epsilon)
        var = torch.tile(var, (in_ch_sz, 1)).T.flatten()
        scale_var = scale / var

        matrix = torch.zeros(
            lower_eq.matrix.shape,
            dtype=lower_eq.config.PRECISION,
            device=node.config.DEVICE
        )
        const = torch.zeros(
            lower_eq.const.shape,
            dtype=lower_eq.config.PRECISION,
            device=node.config.DEVICE
        )

        if bound == 'lower':
            idxs = scale_var < 0
            matrix[idxs, :] = (upper_eq.matrix[idxs, :].T * scale_var[idxs]).T
            const[idxs] = upper_eq.const[idxs] * scale_var[idxs]
            idxs = scale_var >= 0
            matrix[idxs, :] = (lower_eq.matrix[idxs, :].T * scale_var[idxs]).T
            const[idxs] = lower_eq.const[idxs] * scale_var[idxs]
            
        elif bound == 'upper':
            idxs = scale_var < 0
            matrix[idxs, :] = (lower_eq.matrix[idxs, :].T * scale_var[idxs]).T
            const[idxs] = lower_eq.const[idxs] * scale_var[idxs]
            idxs = scale_var >= 0
            matrix[idxs, :] = (upper_eq.matrix[idxs, :].T * scale_var[idxs]).T
            const[idxs] = upper_eq.const[idxs] * scale_var[idxs]

        else:
            raise ValueError(f"Bound type {bound} could not be recognised.")

        batch_const = - input_mean / var * scale + bias
        const += batch_const 

        return Equation(matrix, const, node.config)
 
    def _int_forward_relu(
            self, equation: Equation, node: Node, bound: str, slopes: tuple=None
    ):
        slope = equation.get_relu_slope(node.from_node[0], bound, bound, None, slopes)
        matrix = (equation.matrix.T * slope).T
        const = equation.get_relu_const(
            node.from_node[0], equation.const, bound, slope
        )

        return Equation(matrix, const, self.config)
    
    def _int_forward_max_pool(self, equation: Equation, node: Node, bound: str):
        lower, indices = node.forward(
            node.from_node[0].bounds.lower, return_indices=True
        )
        idx_correction = torch.tensor(
            [
                i * node.from_node[0].in_ch_sz() 
                for i in range(node.from_node[0].in_ch())
            ],
            dtype=torch.long,
            device=self.config.DEVICE    
        ).reshape((node.from_node[0].in_ch(), 1, 1))
        if node.has_batch_dimension():
            idx_correction = idx_correction[None, :]
        indices = indices + idx_correction

        lower, indices = lower.flatten(), indices.flatten()
        upper = node.forward(node.from_node[0].bounds.upper).flatten()
        lower_max  = lower > upper
        not_lower_max = torch.logical_not(lower_max)

        matrix = torch.zeros(
            (node.output_size, node.input_size),
            dtype=node.config.PRECISION, 
            device=node.config.DEVICE
        )
        const = torch.zeros(
            node.output_size, dtype=node.config.PRECISION, device=node.config.DEVICE
        )

        if bound == 'lower':
            matrix = equation.matrix[indices, :]
            const = equation.const[indices]

        elif bound == 'upper':
            matrix[lower_max, :] = equation.matrix[indices, :][lower_max, :]
            const[lower_max] = equation.const[indices][lower_max]
            const[not_lower_max] = \
                node.from_node[0].bounds.upper.flatten()[indices][not_lower_max]

        else:
            raise ValueError(f"Bound type {bound} could not be recognised.")

        return Equation(matrix, const, self.config)
    
    def _int_forward_add(self, equation: Equation, node: Node, bound: str):
        if node.const is None:
            if bound == 'lower':
                summand = self.lower_eq[node.from_node[1].id]
            elif bound == 'upper':
                summand = self.upper_eq[node.from_node[1].id]
            else:
                raise ValueError(f"Bound type {bound} could not be recognised.")

            matrix = equation.matrix + summand.matrix
            const = equation.const + summand.const

        else:
            matrix = equation.matrix.clone()  
            const = equation.const + node.const

        return equation
