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
from venus.bounds.os_sip import OSSIP
from venus.bounds.equation import Equation
from venus.common.logger import get_logger
from venus.common.configuration import Config

torch.set_num_threads(1)

class BSSIP:

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
        if BSSIP.logger is None and config.LOGGER.LOGFILE is not None:
            BSSIP.logger = get_logger(__name__, config.LOGGER.LOGFILE)
 

    def set_bounds(
        self,
        node: Node,
        lower_slopes: dict=None,
        upper_slopes: dict=None,
        os_sip: OSSIP=None
    ) -> Bounds:
        out_flag = self._get_out_prop_flag(node)

        symb_eq = self._derive_symb_eq(node)
        lower_bounds, lower_flag = self.back_substitution(
            symb_eq, node, 'lower', out_flag, lower_slopes, os_sip
        )


        if lower_bounds is None:
            return

        # reduce symbolic equation if instability was reduced by
        # back_substitution.
        if os_sip is not None:
            flag = lower_flag if out_flag is None else lower_flag[out_flag]
            symb_eq = Equation(
                symb_eq.matrix[flag, :], symb_eq.const[flag], self.config,
            )
            concr = lower_flag
            lower_bounds = torch.max(
                node.bounds.lower[lower_flag].flatten(), lower_bounds
            )

        else:
            concr = out_flag
            lower_bounds = torch.max(
                node.bounds.lower[out_flag].flatten(), lower_bounds
            )

        upper_bounds, upper_flag = self.back_substitution(
            symb_eq, node, 'upper', concr, upper_slopes, os_sip
        )

        if upper_bounds is None:
            return
        
        if os_sip is not None: 
            upper_bounds = torch.min(node.bounds.upper[upper_flag], upper_bounds)
            lower_bounds = lower_bounds[upper_flag[lower_flag]]

        else:
            upper_bounds = torch.min(
                node.bounds.upper[out_flag].flatten(), upper_bounds
            )
            upper_flag = out_flag
        
        self._set_bounds(
            node,
            Bounds(lower_bounds, upper_bounds),
            lower_slopes,
            upper_slopes,
            upper_flag
        )

        if node.has_fwd_relu_activation() is True:
            return node.get_next_relu().get_unstable_count()

        return 0

    def _set_bounds(
        self,
        node: None,
        bounds: Bounds,
        lower_slopes: torch.Tensor=None,
        upper_slopes: torch.Tensor=None,
        out_flag: torch.tensor=None
    ):
        if node.has_fwd_relu_activation() and \
        lower_slopes is not None and \
        upper_slopes is not None:
            # relu node with custom slope - leave slope as is but remove from
            # it newly stable nodes.
            old_fl = node.get_next_relu().get_unstable_flag() 
            new_fl = out_flag[old_fl]

            lower_slopes[node.get_next_relu().id] = \
                lower_slopes[node.get_next_relu().id][new_fl]
            upper_slopes[node.get_next_relu().id] = \
                upper_slopes[node.get_next_relu().id][new_fl]

        node.update_bounds(bounds, out_flag)

    def _get_out_prop_flag(self, node: Node):
        if node.has_fwd_relu_activation():
            return node.get_next_relu().get_unstable_flag()

        elif len(node.to_node) == 0:
            return self.prob.spec.get_output_flag(
                node.output_shape, device = self.config.DEVICE
            )        

        return None 

    def _get_stability_flag(self, node: Node):
        stability = node.get_propagation_count()
        if stability / node.output_size >= self.config.SIP.STABILITY_FLAG_THRESHOLD:
            return  None
            
        return node.get_propagation_flag()

    def _derive_symb_eq(self, node: Node):
        in_flag = self._get_stability_flag(node.from_node[0])
        out_flag = self._get_out_prop_flag(node)
        return Equation.derive(
            node,
            self.config,
            None if out_flag is None else out_flag,
            None if in_flag is None else in_flag
        )

    def back_substitution(
        self,
        equation: Equation,
        node: None,
        bound: str, 
        i_flag: torch.Tensor=None,
        slopes: dict=None,
        os_sip: OSSIP=None
    ):
        """
        Substitutes the variables in an equation with input variables.

        Arguments:
            equation:
                The equation.
            node:
                The input node to the node corresponding to the equation.
            bound:
                Bound the equation expresses - either lower or upper.
            i_flag:
                Instability flag from IA.
            slopes:
                Relu slopes to use. If none then the default of minimum area
                approximation are used.

        Returns:
            The concrete bounds of the equation after back_substitution of the
            variables.
        """
        if bound not in ['lower', 'upper']:
            raise ValueError("Bound type {bound} not recognised.")

        equation, i_flag = self._back_substitution(
            equation, node, node, bound, i_flag, slopes, os_sip
        )

        if equation is None:
            return None, i_flag

        if self.config.SIP.ONE_STEP_SYMBOLIC is True and os_sip is not None:
            update = i_flag.flatten()
            if bound == 'lower':
                os_sip.current_lower_eq.matrix[update, :] = equation.matrix
                os_sip.current_lower_eq.const[update] = equation.const

            else:
                os_sip.current_upper_eq.matrix[update, :] = equation.matrix
                os_sip.current_upper_eq.const[update] = equation.const

        concr_values = equation.concrete_values(
            self.prob.spec.input_node.bounds.lower.flatten(),
            self.prob.spec.input_node.bounds.upper.flatten(),
            bound
        )

        return concr_values, i_flag

    def _back_substitution(
        self,
        equation: Equation,
        base_node: Node,
        cur_node: Node,
        bound: str,
        i_flag: torch.Tensor=None,
        slopes: torch.Tensor=None,
        os_sip: OSSIP=None
    ):
        """
        Helper function for back_substitution
        """
        if cur_node.is_sip_branching_node() is True:
            equation, i_flag = self._back_substitution_branching(
                equation, base_node, cur_node, bound, i_flag, slopes, os_sip
            )
        else:
            equation, i_flag = self._back_substitution_linear(
                equation, base_node, cur_node.from_node[0], bound, i_flag, slopes, os_sip
            )

        return equation, i_flag

    def _back_substitution_linear(
        self,
        equation: Equation,
        base_node: Node,
        cur_node: Node,
        bound: str,
        i_flag: torch.Tensor=None,
        slopes: torch.Tensor=None,
        os_sip: OSSIP=None
    ):
        """
        Helper function for back_substitution
        """
        # if base_node.id == 23:
            # print('linear', cur_node,  cur_node.id, cur_node.output_shape, equation.matrix.shape)
            # input()
        if isinstance(cur_node, Input):
            return  equation, i_flag

        if cur_node.has_relu_activation() is True and \
        slopes is not None and \
        cur_node.to_node[0].id in slopes:
            node_slopes = slopes[cur_node.id]
        else:
            node_slopes = None
        non_linear_cond = cur_node.has_relu_activation() or \
            type(cur_node) in [MaxPool, BatchNormalization]
        tranposed = False
 
        if type(cur_node) in [Relu, Flatten, Unsqueeze, Reshape]:
            pass
        
        else:
            in_flag = self._get_stability_flag(cur_node.from_node[0])
            out_flag = self._get_stability_flag(cur_node)

            if non_linear_cond is True:
                equation = self._int_backward(
                    equation, cur_node, bound, in_flag, out_flag, node_slopes
                )
            else:
                equation = self._backward(equation, cur_node, in_flag, out_flag)

            tranposed = True

        if os_sip is not None and tranposed is True:
            equation, i_flag = self._update_back_subs_eqs(
                equation, base_node, cur_node.from_node[0], bound, i_flag, os_sip
            )
            if torch.sum(i_flag) == 0:
                return None, i_flag
        
        if cur_node.from_node[0].is_sip_branching_node() is True:
            equation, i_flag = self._back_substitution_branching(
                equation, base_node, cur_node.from_node[0], bound, i_flag, slopes, os_sip
            )
        else:
            equation, i_flag = self._back_substitution_linear(
                equation, base_node, cur_node.from_node[0], bound, i_flag, slopes, os_sip
            )

        return equation, i_flag
        # return self._back_substitution(
            # equation, base_node, cur_node, bound, i_flag, slopes, os_sip
        # )

    def _back_substitution_branching(
        self,
        equation: Equation,
        base_node: Node,
        cur_node: Node,
        bound: str,
        i_flag: torch.Tensor=None,
        slopes: torch.Tensor=None,
        os_sip: OSSIP=None
    ):
        """
        Helper function for back_substitution
        """
        # if base_node.id == 23:
            # print('branching', cur_node,  cur_node.id, cur_node.output_shape, equation.matrix.shape)
            # input()
        if isinstance(cur_node, Add):
            prev_node = cur_node.from_node[0].get_prv_non_relu()
            if prev_node.is_sip_branching_node() is True:
                eq1, i_flag = self._back_substitution_branching(
                    equation, base_node, prev_node, bound, i_flag, slopes, os_sip
                )
            else:
                eq1, i_flag = self._back_substitution_linear(
                    equation, base_node, prev_node, bound, i_flag, slopes, os_sip
                )

            prev_node = cur_node.from_node[1].get_prv_non_relu()
            if prev_node.is_sip_branching_node() is True:
                eq2, i_flag = self._back_substitution_branching(
                    equation, base_node, prev_node, bound, i_flag, slopes, os_sip
                )
            else:
                eq2, i_flag = self._back_substitution_linear(
                    equation, base_node, prev_node, bound, i_flag, slopes, os_sip
                )
            
            return eq1.add(eq2), i_flag

        # if isinstance(cur_node, Add):
            # prev_node = cur_node.from_node[0].get_prv_non_relu()
            # if equation.zero() is True:
                # eq1 = self._derive_symb_eq(prev_node)
            # else:
                # eq1 = equation.copy()
            # print('1', cur_node.id, prev_node, prev_node.output_size, eq1.matrix.shape)
            # eq1, i_flag = self._back_substitution(
                # eq1, base_node, prev_node, bound, i_flag, slopes, os_sip
            # )
            # prev_node = cur_node.from_node[1].get_prv_non_relu()
            # print('2', cur_node.id, prev_node, prev_node.output_size, eq1.matrix.shape)
            # if equation.zero() is True:
                # eq2 = self._derive_symb_eq(prev_node)
            # else:
                # eq2 = equation
            # print('3', cur_node.id, prev_node, prev_node.output_size, eq2.matrix.shape)
            # eq2, i_flag = self._back_substitution(
                # eq2, base_node, prev_node, bound, i_flag, slopes, os_sip
            # )
            # print('4', cur_node.id, prev_node, prev_node.output_size, eq2.matrix.shape)
            # input()
            
            # return eq1.add(eq2), i_flag


        elif isinstance(cur_node, Concat):
            idx = cur_node.from_node[0].output_size
            b_eq = Equation(
                eq.matrix[:, torch.arange(0, cur_node.from_node[0].output_size)],
                torch.zeros(
                    eq.size, dtype=self.config.PRECISION, device=self.config.DEVICE
                ),
                self.config
            )
            b_eq, i_flag = self._back_substitution(
                b_eq, base_node, cur_node.from_node[0], bound, i_flag, slopes, os_sip
            )

            for i in cur_node.from_node[1:]:
                part_eq = Equation(
                    eq.matrix[:, torch.arange(idx, cur_node.from_node[0].output_size)],
                    torch.zeros(
                        eq.size, dtype=self.config.PRECISION, device=self.config.DEVICE
                    ),
                    self.config
                )
                part_eq, i_flag = self._back_substitution(
                    part_eq, base_node, i,  bound, i_flag, slopes, os_sip
                )
                b_eq = b_eq.add(part_eq)

            return b_eq, i_flag

        else:
            raise TypeError(
                f'Backsubstitution-branching is not supported for node {type(cur_node)}'
            )

    def _update_back_subs_eqs(
        self,
        equation: Equation,
        base_node: Node,
        input_node: Node,
        bound: str,
        instability_flag: torch.Tensor,
        os_sip: OSSIP
    ) -> Equation:

        if base_node.has_fwd_relu_activation() is not True:
            return equation, instability_flag
     
        if bound == 'lower':
            if input_node.id not in os_sip.lower_eq:
                return equation, instability_flag 

            concr_bounds = equation.interval_dot(
                'lower', os_sip.lower_eq[input_node.id], os_sip.upper_eq[input_node.id]
            ).min_values(
                self.prob.spec.input_node.bounds.lower.flatten(),
                self.prob.spec.input_node.bounds.upper.flatten()
            )
            unstable_idxs = concr_bounds < 0
            stable_idxs = torch.logical_not(unstable_idxs)
            flag = torch.zeros(
                base_node.output_shape, dtype=torch.bool, device=self.config.DEVICE
            )
            flag[instability_flag] = stable_idxs
            base_node.bounds.lower[flag] = torch.max(
                base_node.bounds.lower[flag], concr_bounds[stable_idxs]
            )
            base_node.get_next_relu().reset_state_flags()

        elif bound == 'upper':
            if input_node.id not in os_sip.upper_eq:
                return equation, instability_flag 

            concr_bounds = equation.interval_dot(
                'upper', os_sip.lower_eq[input_node.id], os_sip.upper_eq[input_node.id]
            ).max_values(
                self.prob.spec.input_node.bounds.lower.flatten(),
                self.prob.spec.input_node.bounds.upper.flatten()
            )
            unstable_idxs = concr_bounds > 0
            stable_idxs = torch.logical_not(unstable_idxs)
            flag = torch.zeros(
                base_node.output_shape, dtype=torch.bool, device=self.config.DEVICE
            )
            flag[instability_flag] = stable_idxs
            base_node.bounds.upper[flag] = torch.min(
                base_node.bounds.upper[flag], concr_bounds[stable_idxs]
            )
            base_node.get_next_relu().reset_state_flags()
        else:
            raise Exception(f"Bound type {bound} not recognised.")
  
        flag = torch.zeros(
            base_node.output_shape, dtype=torch.bool, device=self.config.DEVICE
        )
        flag[instability_flag] = unstable_idxs
        reduced_eq = Equation(
            equation.matrix[unstable_idxs, :],
            equation.const[unstable_idxs],
            self.config
        )

        return reduced_eq, flag

    def optimise(self, node: Node):
        l_slopes, u_slopes = self.prob.nn.get_lower_relaxation_slopes(gradient=True)
        l_slopes =  self._optimise(node, 'lower', l_slopes)
        u_slopes =  self._optimise(node, 'upper', u_slopes)

        return lower_slopes, upper_slopes

    def _optimise(self, node: Node, bound: str, slopes: torch.tensor):
        in_flag = self._get_stability_flag(node.from_node[0])
        out_flag = self._get_out_prop_flag(node)
        if out_flag is None:
            out_flag = torch.ones(
                node.output_shape, dtype=torch.bool, device=self.config.DEVICE
            )
        else:
            out_flag = out_flag
            
        equation = Equation.derive(node, self.config, out_flag, None)

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

    def _backward_matrix(
        self,
        matrix: torch.Tensor,
        node: Node,
        out_flag: torch.Tensor=None,
        in_flag: torch.Tensor=None
    ) -> torch.Tensor:
        if type(node) in [Conv, ConvTranspose]:
            b_matrix =  self._backward_affine_matrix(matrix, node, out_flag, in_flag)
        
        elif isinstance(node, Gemm):
            b_matrix = self._backward_gemm_matrix(matrix, node, out_flag, in_flag)

        elif isinstance(node, MatMul):
            b_matrix = self._backward_matmul_matrix(matrix, node, out_flag, in_flag)

        elif isinstance(node, Sub):
            b_matrix = self._backward_sub_matrix(matrix, node)

        elif isinstance(node, Add):
            b_matrix = self._backward_add_matrix(matrix, node)

        elif isinstance(node, BatchNormalization):
            b_matrix = self._backward_batch_normalization_matrix(matrix, node, in_flag)

        else:
            raise NotImplementedError(f'Matrix backward for {type(node)}')

        return b_matrix

    def _backward_conv_matrix(
        self,
        matrix: torch.Tensor,
        node: Node,
        out_flag: torch.tensor=None,
        in_flag: torch.tensor=None
    ) -> torch.Tensor:
        batch = matrix.shape[0]
        if out_flag is None:
            shape = (batch,) + node.output_shape_no_batch()
        else:
            shape = matrix.shape
        b_matrix =  node.transpose(matrix.reshape(shape), out_flag, in_flag)
        b_matrix = b_matrix.reshape(batch, -1)

        return b_matrix

    def _backward_gemm_matrix(
        self,
        matrix: torch.Tensor,
        node: Node,
        out_flag: torch.tensor=None,
        in_flag: torch.tensor=None
    ) -> torch.Tensor:
        size = matrix.shape[0]

        matrix = matrix.reshape((size,) + node.output_shape)
        matrix = matrix.permute(*torch.arange(matrix.ndim - 1, -1, -1))
        matrix = node.transpose(matrix, out_flag, in_flag)
        matrix = matrix.permute(*torch.arange(matrix.ndim - 1, -1, -1))
        matrix = matrix.reshape(size, -1)
        return matrix


    def _backward_matmul_matrix(
        self,
        matrix: torch.Tensor,
        node: Node,
        out_flag:torch.tensor=None,
        in_flag: torch.tensor=None
    ) -> torch.Tensor:
        return node.transpose(matrix.T, out_flag, in_flag).T

    def _backward_sub_matrix(self, matrix: torch.Tensor, node:Node) -> torch.Tensor:
        if node.const is None:
            b_matrix = torch.hstack([matrix, -matrix])
        else:
            b_matrix = matrix.clone()

        return b_matrix
 
    def _backward_add_matrix(self, matrix: torch.Tensor, node:Node) -> torch.Tensor:
        if node.const is None:
            b_matrix = torch.hstack([matrix, matrix])
        else:
            matrix = matrix.clone()

        return b_matrix

    def _backward_batch_normalization_matrix(
        self, matrix: torch.Tensor, node: Node, in_flag: torch.Tensor=None
    ) -> torch.Tensor:
        in_ch_sz = node.in_ch_sz()

        scale = torch.tile(node.scale, (in_ch_sz, 1)).T.flatten()
        bias = torch.tile(node.bias, (in_ch_sz, 1)).T.flatten()
        input_mean = torch.tile(node.input_mean, (in_ch_sz, 1)).T.flatten()
        var = torch.sqrt(node.input_var + node.epsilon)
        var = torch.tile(var, (in_ch_sz, 1)).T.flatten()

        scale_var = scale / var
        idxs = scale_var < 0
        scale_var[idxs] = - scale_var[idxs]

        if in_flag is None:
            b_matrix = matrix * scale_var
        else:
            prop_flag = in_flag.flatten()
            b_matrix = matrix[:, prop_flag] * scale_var[prop_flag]

        return b_matrix
    
    def _backward(
        self,
        equation: Equation,
        node: Node,
        out_flag: torch.Tensor=None,
        in_flag: torch.Tensor=None
    ) -> Equation:
        if type(node) in [Gemm, Conv, ConvTranspose]:
            b_equation =  self._backward_affine(equation, node, out_flag, in_flag)

        elif isinstance(node, MatMul):
            b_equation = self._backward_matmul(equation, node, out_flag, in_flag)

        elif isinstance(node, AveragePool):
            b_equation = self._backward_average_pool(equation, node)

        elif isinstance(node, Slice):
            b_equation = self._backward_slice(equation, node)

        elif isinstance(node, Sub):
            b_equation = self._backward_sub(equation, node)

        elif isinstance(node, Add):
            b_equation = self._backward_add(equation, node)

        else:
            raise NotImplementedError(f'Equation backward for {type(node)}')

        return b_equation 

    def _backward_affine(
        self,
        equation: Equation,
        node: Node,
        out_flag: torch.tensor=None,
        in_flag: torch.tensor=None
    ) -> Equation:
        matrix = self._backward_affine_matrix(equation.matrix, node, out_flag, in_flag)
        const = Equation.derive_const(node, out_flag)
        const = (equation.matrix @ const) + equation.const

        return Equation(matrix, const, self.config)

    def _backward_matmul(
        self,
        equation: Equation,
        node: Node,
        out_flag:torch.tensor=None,
        in_flag: torch.tensor=None
    ) -> Equation:
        matrix = self._backward_matmul_matrix(equation.matrix, node, out_flag, in_flag)

        return Equation(matrix, equation.const, self.config)
 
    def _backward_slice(self, equation: Equation, node: Node):
        matrix = torch.zeros(
            (equation.size,) + node.input_shape,
            dtype=node.config.PRECISION,
            device=self.config.DEVICE
        )
        slices = [slice(0, equation.size)] + node.slices
        matrix[slices] = equation.matrix.reshape((equation.size,) + node.output_shape)
        matrix = matrix.reshape(equation.size, -1)

        return Equation(matrix, equation.const, self.config)
 
    def _backward_sub(self, equation: Equation, node: Node):
        matrix = self._backward_sub_matrix(equation.matrix, node)
        if node.const is None:
            const = equation.const.clone()
        else:
            const = node.const.flatten() + equation.const

        return Equation(matrix, const, self.config)
 
    def _backward_add(self, equation: Equation, node: Node):
        matrix = self._backward_add_matrix(equation.matrix, node)
        if node.const is None:
            const = equation.const.clone()
        else:
            const = node.const.flatten() + equation.const

        return Equation(matrix, const, self.config)

    def _backward_average_pool(self, equation: Equation, node: Node):
        krn = torch.ones(
            (node.in_ch(),) + node.kernel_shape,
            dtype=self.config.PRECISION,
            device=self.config.DEVICE
        )
        conv_node = Conv(
            [], [], node.input_shape, krn, None, node.pads, node.strides, self.config
        )
        matrix = self._backward_affine_matrix(equation.matrix, node)
        matrix /= np.prod(node.kernel_shape)

        return Equation(matrix, equation.const, self.config)

    def _int_backward(
        self,
        equation: Equation,
        node: Node,
        bound: str,
        out_flag: torch.Tensor=None,
        in_flag: torch.Tensor=None,
        slopes: torch.tensor=None
    ):
        if node.has_relu_activation():
            b_equation = self._int_backward_relu(
                equation, node, bound, out_flag, in_flag, slopes
            )

        elif isinstance(node, BatchNormalization):
            b_equation = self._int_backward_batch_normalization(
                equation, node, in_flag
            )

        elif isinstance(node, MaxPool):
            b_equation = self._int_backward_maxpool(
                equation, node, bound
            )

        else:
            raise NotImplementedError(f'Interval forward for node {type(node)}')

        return b_equation

    def _int_backward_relu(
        self,
        equation: Equation, 
        node: Node,
        bound: str,
        out_flag: torch.tensor=None,
        in_flag: torch.tensor=None,
        slopes: torch.tensor = None
    ):
        lower_slope = equation.get_relu_slope(node, 'lower', bound, out_flag, slopes)
        upper_slope = equation.get_relu_slope(node, 'upper', bound, out_flag, slopes)
        out_flag = None if out_flag is None else out_flag.flatten()
        lower_const = Equation.derive_const(node, out_flag)
        upper_const = lower_const.clone()
        lower_const = equation.get_relu_const(
            node, lower_const, 'lower', lower_slope, out_flag
        )
        upper_const =  equation.get_relu_const(
            node, upper_const, 'upper', upper_slope, out_flag
        )

        _plus, _minus = equation.get_plus_matrix(), equation.get_minus_matrix()

        if bound == 'lower':
            plus, minus = _plus * lower_slope, _minus * upper_slope
            const = _plus @ lower_const + _minus @ upper_const

        elif bound == 'upper':
            plus, minus = _plus  * upper_slope, _minus * lower_slope
            const = _plus @ upper_const + _minus @ lower_const

        else:
            raise ValueError(f'Bound {bound} is not recognised.')

        matrix = self._backward_matrix(plus, node, out_flag, in_flag)
        matrix += self._backward_matrix(minus, node, out_flag, in_flag)

        const += equation.const

        return Equation(matrix, const, self.config)

    def _int_backward_batch_normalization(
        self,
        equation: Equation,
        node: Node,
        in_flag: torch.tensor
    ):
        in_ch_sz = node.in_ch_sz()
        scale = torch.tile(node.scale, (in_ch_sz, 1)).T.flatten()
        bias = torch.tile(node.bias, (in_ch_sz, 1)).T.flatten()
        input_mean = torch.tile(node.input_mean, (in_ch_sz, 1)).T.flatten()
        var = torch.sqrt(node.input_var + node.epsilon)
        var = torch.tile(var, (in_ch_sz, 1)).T.flatten()

        matrix = self._backward_batch_normalization_matrix(
            equation.matrix, node, in_flag
        )

        batch_const = - input_mean / var * scale + bias
        const = equation.matrix @ batch_const + equation.const

        return Equation(matrix, const, self.config)

    def _int_backward_maxpool(self, equation: Equation, node: Node, bound:str):
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

        _plus, _minus = equation.get_plus_matrix(), equation.get_minus_matrix()

        upper_const = torch.zeros(
            node.output_size, dtype=self.config.PRECISION, device=self.config.DEVICE
        )
        upper_const[not_lower_max] = \
            node.from_node[0].bounds.upper.flatten()[indices][not_lower_max]

        if bound == 'lower':
            plus_lower = torch.zeros(
                (equation.size, node.input_size),
                dtype=self.config.PRECISION,
                device=self.config.DEVICE
            )
            plus_lower[:, indices] = _plus
            minus_upper = torch.zeros(
                (equation.size, node.input_size),
                dtype=self.config.PRECISION,
                device=self.config.DEVICE
            )
            temp = minus_upper[:, indices]
            temp[:, lower_max] = _minus[:, lower_max]
            minus_upper[:, indices] = temp
            del temp
            matrix = plus_lower + minus_upper

            const = _minus @ upper_const + equation.const

        elif bound == 'upper':
            minus_lower = torch.zeros(
                (equation.size, node.input_size),
                dtype=self.config.PRECISION,
                device=self.config.DEVICE
            )
            minus_lower[:, indices] = _minus
            plus_upper = torch.zeros(
                (equation.size, node.input_size),
                dtype=self.config.PRECISION,
                device=self.config.DEVICE
            )
            temp =  plus_upper[:, indices]
            temp[:, lower_max] = _plus[:, lower_max]
            plus_upper[:, indices] = temp
            del temp
            matrix = minus_lower + plus_upper

            const =  _plus @ upper_const + equation.const

        else:
            raise ValueError(f'Bound {bound} is not recognised.')

        return Equation(matrix, const, self.config)
     
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


"""
    def __update_back_subs_eqs(
        self,
        equation: Equation,
        base_node: Node,
        cur_node: Node,
        input_node: Node,
        bound: str,
        instability_flag: torch.Tensor
    ) -> Equation:
        assert base_node.has_fwd_relu_activation() is True, \
            "Update concerns only nodes with relu activation."
        if bound == 'lower':
            concr_bounds = equation.min_values(
                input_node.bounds.lower.flatten(), input_node.bounds.upper.flatten()
            )
            unstable_idxs = concr_bounds < 0
            stable_idxs = torch.logical_not(unstable_idxs)
            flag = torch.zeros(
                base_node.output_shape, dtype=torch.bool, device=self.config.DEVICE
            )
            flag[instability_flag] = stable_idxs
            base_node.bounds.lower[flag] = concr_bounds[stable_idxs] 
            # print('l', base_node.to_node[0].get_unstable_count())
            base_node.to_node[0].reset_state_flags()
            # print('l', base_node.to_node[0].get_unstable_count())
            # input()

        elif bound == 'upper':
            concr_bounds = equation.max_values(
                input_node.bounds.lower.flatten(), input_node.bounds.upper.flatten()
            )
            unstable_idxs = concr_bounds > 0
            stable_idxs = torch.logical_not(unstable_idxs)
            flag = torch.zeros(
                base_node.output_shape, dtype=torch.bool, device=self.config.DEVICE
            )
            flag[instability_flag] = stable_idxs
            base_node.bounds.upper[flag] = concr_bounds[stable_idxs]
            # print('u', base_node.to_node[0].get_unstable_count())
            base_node.to_node[0].reset_state_flags()
            # print('u', base_node.to_node[0].get_unstable_count())
            # input()
        else:
            raise Exception(f"Bound type {bound} not recognised.")
  
        flag = torch.zeros(
            base_node.output_shape, dtype=torch.bool, device=self.config.DEVICE
        )
        flag[instability_flag] = unstable_idxs
        reduced_eq = Equation(
            equation.matrix[unstable_idxs, :],
            equation.const[unstable_idxs],
            self.config
        )

        return reduced_eq, flag
"""
