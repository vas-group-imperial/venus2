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
        
    def forward(
        self, node: Node,
        lower_slopes: torch.Tensor=None,
        upper_slopes: torch.Tensor=None
    ) -> None:
        non_linear_starting_depth = self.prob.nn.get_non_linear_starting_depth()

        if node.depth == 1:
            new_lower_eq = Equation.derive(node, self.config)
            new_upper_eq = new_lower_eq

        elif node.depth < non_linear_starting_depth:
            new_lower_eq = self.current_lower_eq.forward(node)
            new_upper_eq = new_lower_eq

        else:
            if node.depth == non_linear_starting_depth + 1:
                self.current_upper_eq = self.current_lower_eq.copy()
                
            if isinstance(node, Relu):
                new_lower_eq = self.current_lower_eq.forward(
                    node, 'lower', lower_slopes
                )
                new_upper_eq = self.current_upper_eq.forward(
                    node, 'upper', upper_slopes
                )

            elif type(node) in [Reshape, Flatten, Sub, Add, Slice, Unsqueeze, Concat]:
                new_lower_eq  = self.current_lower_eq.forward(node)
                new_upper_eq  = self.current_upper_eq.forward(node)
 
            else:
                new_lower_eq  = Equation.interval_forward(
                    node, 'lower', self.current_lower_eq, self.current_upper_eq
                )
                new_upper_eq  = Equation.interval_forward(
                    node, 'upper', self.current_lower_eq, self.current_upper_eq
                )

        self.current_lower_eq, self.current_upper_eq = new_lower_eq, new_upper_eq
        self.lower_eq[node.id], self.upper_eq[node.id] = new_lower_eq, new_upper_eq
 
    def set_bounds(self, node: Node) -> int:
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

        bounds = Bounds(lower, upper)
        node.update_bounds(bounds)

        if node.has_relu_activation() is True:
            return node.to_node[0].get_unstable_count()

        return 0

    def clear_equations(self):
        self.lower_eq, self.upper_eq = {}, {}
