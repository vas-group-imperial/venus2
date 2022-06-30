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
        in_flag = self._get_stability_flag(node.from_node[0])
        out_flag = self._get_out_prop_flag(node)
        symb_eq = Equation.derive(
            node,
            self.config,
            None if out_flag is None else out_flag.flatten(),
            None if in_flag is None else in_flag.flatten(),
        )

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
            node, Bounds(lower_bounds, upper_bounds), lower_slopes, upper_slopes, upper_flag
        )

        if node.has_relu_activation() is True:
            return node.to_node[0].get_unstable_count()

        return 0

    def _set_bounds(
        self,
        node: None,
        bounds: Bounds,
        lower_slopes: torch.Tensor=None,
        upper_slopes: torch.Tensor=None,
        out_flag: torch.tensor=None
    ):
        if node.has_relu_activation() and \
        lower_slopes is not None and \
        upper_slopes is not None:
            # relu node with custom slope - leave slope as is but remove from
            # it newly stable nodes.
            old_un_flag = node.to_node[0].get_unstable_flag() 
            new_un_flag = out_flag[old_un_flag]

            lower_slopes[node.to_node[0].id] = lower_slopes[node.to_node[0].id][new_fl]
            upper_slopes[node.to_node[0].id] = upper_slopes[node.to_node[0].id][new_fl]

        node.update_bounds(bounds, out_flag)

    def _get_out_prop_flag(self, node: Node):
        if node.has_relu_activation():
            return node.to_node[0].get_unstable_flag()

        elif len(node.to_node) == 0:
            return self.prob.spec.get_output_flag(
                node.output_shape, device = self.config.SIP.DEVICE
            )
        
        return None 

    def _get_stability_flag(self, node: Node):
        stability = node.get_propagation_count()
        if stability / node.output_size >= self.config.SIP.STABILITY_FLAG_THRESHOLD:
            return  None
            
        return node.get_propagation_flag()

    def back_substitution(
        self,
        eq: Equation,
        node: None,
        bound: str, 
        instability_flag: torch.Tensor=None,
        slopes: dict=None,
        os_sip: OSSIP=None
    ):
        """
        Substitutes the variables in an equation with input variables.

        Arguments:
            eq:
                The equation.
            node:
                The input node to the node corresponding to the equation.
            bound:
                Bound the equation expresses - either lower or upper.
            instability_flag:
                Flag of unstable nodes from IA.
            slopes:
                Relu slopes to use. If none then the default of minimum area
                approximation are used.

        Returns:
            The concrete bounds of the equation after back_substitution of the
            variables.
        """
        if bound not in ['lower', 'upper']:
            raise ValueError("Bound type {bound} not recognised.")

        eqs, instability_flag = self._back_substitution(
            eq, node, node.from_node[0], bound, instability_flag, slopes, os_sip
        )

        if eqs is None:
            return None, instability_flag

        sum_eq = eqs[0]
        for i in eqs[1:]:
            sum_eq = sum_eq.add(i)

        if self.config.SIP.ONE_STEP_SYMBOLIC is True and os_sip is not None:
            update = instability_flag.flatten()
            if bound == 'lower':
                os_sip.current_lower_eq.matrix[update, :] = sum_eq.matrix
                os_sip.current_lower_eq.const[update] = sum_eq.const

            else:
                os_sip.current_upper_eq.matrix[update, :] = sum_eq.matrix

                os_sip.current_upper_eq.const[update] = sum_eq.const

        concr_values = sum_eq.concrete_values(
            self.prob.spec.input_node.bounds.lower.flatten(),
            self.prob.spec.input_node.bounds.upper.flatten(),
            bound
        )

        return concr_values, instability_flag


    def _back_substitution(
        self,
        eq: Equation,
        base_node: Node,
        cur_node: Node,
        bound: str,
        instability_flag: torch.Tensor=None,
        slopes: torch.Tensor=None,
        os_sip: OSSIP=None
    ):
        """
        Helper function for back_substitution
        """
        if isinstance(cur_node, Input):
            return  [eq], instability_flag

        _tranposed = False

        if cur_node.has_relu_activation() or isinstance(cur_node, MaxPool):
            if slopes is not None and cur_node.to_cur_node[0].id in slopes:
                cur_node_slopes = slopes[cur_node.to_cur_node[0].id]
            else:
                cur_node_slopes = None
                
            in_flag = self._get_stability_flag(cur_node.from_node[0])
            out_flag = self._get_stability_flag(cur_node)
            eq = eq.interval_transpose(
                cur_node, bound, out_flag, in_flag, cur_node_slopes
            )
            _tranposed = True

        elif type(cur_node) in [Relu, Flatten, Unsqueeze, Reshape]:
            eq = [eq]

        else:
            in_flag = self._get_stability_flag(cur_node.from_node[0])
            out_flag = self._get_stability_flag(cur_node)
            eq = eq.transpose(cur_node, out_flag, in_flag)
            _tranposed = True

        eqs = []

        for i, j in enumerate(eq):
            if os_sip is not None and _tranposed is True:
            # and cur_node.depth == 1 and bound == 'lower':
                j, instability_flag = self._update_back_subs_eqs(
                    j,
                    base_node,
                    cur_node,
                    cur_node.from_node[i],
                    bound,
                    instability_flag,
                    os_sip
                )
                if torch.sum(instability_flag) == 0:
                    return None, instability_flag

            back_subs_eq, instability_flag = self._back_substitution(
                j,
                base_node,
                cur_node.from_node[0],
                bound,
                instability_flag=instability_flag,
                slopes=slopes,
                os_sip=os_sip
            )

            if back_subs_eq is None:
                return None, instability_flag
        
            eqs.extend(back_subs_eq)

        return eqs, instability_flag

    def _update_back_subs_eqs(
        self,
        equation: Equation,
        base_node: Node,
        cur_node: Node,
        input_node: Node,
        bound: str,
        instability_flag: torch.Tensor,
        os_sip: OSSIP
    ) -> Equation:

        if base_node.has_relu_activation() is not True:
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
                base_node.output_shape, dtype=torch.bool, device=self.config.SIP.DEVICE
            )
            flag[instability_flag] = stable_idxs
            base_node.bounds.lower[flag] = torch.max(
                base_node.bounds.lower[flag], concr_bounds[stable_idxs]
            )
            base_node.to_node[0].reset_state_flags()

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
                base_node.output_shape, dtype=torch.bool, device=self.config.SIP.DEVICE
            )
            flag[instability_flag] = stable_idxs
            base_node.bounds.upper[flag] = torch.min(
                base_node.bounds.upper[flag], concr_bounds[stable_idxs]
            )
            base_node.to_node[0].reset_state_flags()
        else:
            raise Exception(f"Bound type {bound} not recognised.")
  
        flag = torch.zeros(
            base_node.output_shape, dtype=torch.bool, device=self.config.SIP.DEVICE
        )
        flag[instability_flag] = unstable_idxs
        reduced_eq = Equation(
            equation.matrix[unstable_idxs, :],
            equation.const[unstable_idxs],
            self.config
        )

        return reduced_eq, flag


    def __update_back_subs_eqs(
        self,
        equation: Equation,
        base_node: Node,
        cur_node: Node,
        input_node: Node,
        bound: str,
        instability_flag: torch.Tensor
    ) -> Equation:
        assert base_node.has_relu_activation() is True, \
            "Update concerns only nodes with relu activation."
        if bound == 'lower':
            concr_bounds = equation.min_values(
                input_node.bounds.lower.flatten(), input_node.bounds.upper.flatten()
            )
            unstable_idxs = concr_bounds < 0
            stable_idxs = torch.logical_not(unstable_idxs)
            flag = torch.zeros(
                base_node.output_shape, dtype=torch.bool, device=self.config.SIP.DEVICE
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
                base_node.output_shape, dtype=torch.bool, device=self.config.SIP.DEVICE
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
            base_node.output_shape, dtype=torch.bool, device=self.config.SIP.DEVICE
        )
        flag[instability_flag] = unstable_idxs
        reduced_eq = Equation(
            equation.matrix[unstable_idxs, :],
            equation.const[unstable_idxs],
            self.config
        )

        return reduced_eq, flag
