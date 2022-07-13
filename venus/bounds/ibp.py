"""
# File: ibp.py
# Top contributors (to current version):
# 	Panagiotis Kouvaros (panagiotis.kouvaros@gmail.com)
# This file is part of the Venus project.
# Copyright: 2019-2021 by the authors listed in the AUTHORS file in the
# top-level directory.
# License: BSD 2-Clause (see the file LICENSE in the top-level directory).
# Description: Interal Bound Propagation
,"""

import math
import numpy as np
from venus.network.node import *
from venus.bounds.bounds import Bounds
from venus.bounds.equation import Equation
from venus.common.logger import get_logger
from venus.common.configuration import Config

torch.set_num_threads(1)

class IBP:

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
        if IBP.logger is None and config.LOGGER.LOGFILE is not None:
            IBP.logger = get_logger(__name__, config.LOGGER.LOGFILE)
        
    def calc_bounds(self, node: Node) -> Bounds:
        if type(node) in [
                Relu, MaxPool, AveragePool, Slice, Unsqueeze, Reshape, Flatten, Pad,
                ReduceSum, Sigmoid
        ]:
            lower, upper = self._calc_non_gemm_bounds(node)

        elif isinstance(node, BatchNormalization):
            lower, upper = self._calc_batch_normalisation_bounds(node)

        elif type(node) in [Concat, Mul, Div]:
            lower, upper = self._calc_branching_bounds(node)

        elif isinstance(node, Sub):
            lower, upper = self._calc_sub_bounds(node)

        elif isinstance(node, Add):
            lower, upper = self._calc_add_bounds(node)

        elif type(node) in [Gemm, Conv, ConvTranspose]:
            lower, upper = self._calc_affine_bounds(node)
         
        elif isinstance(node, MatMul):
            lower, upper = self._calc_linear_bounds(node)

        else:
            raise TypeError(f"IA Bounds computation for {type(node)} is not supported")
        
        bounds = Bounds(
            lower.reshape(node.output_shape), upper.reshape(node.output_shape)
        )

        return bounds

    def _calc_non_gemm_bounds(self, node: Node):
        inp = node.from_node[0].bounds

        return node.forward(inp.lower), node.forward(inp.upper)

    def _calc_batch_normalisation_bounds(self, node: Node):
        inp = node.from_node[0].bounds
        f_l, f_u = node.forward(inp.lower), node.forward(inp.upper)

        return torch.min(f_l, f_u), torch.max(f_l, f_u)

    def _calc_branching_bounds(self, node: Node):
        inp_lower = [i.bounds.lower for i in node.from_node]
        inp_upper = [i.bounds.upper for i in node.from_node]
        
        return node.forward(inp_lower), node.forward(inp_upper)

    def _calc_sub_bounds(self, node: Node):
        inp = node.from_node[0].bounds

        if node.const is None:
            const_lower = node.from_node[1].bounds.lower
            const_upper = node.from_node[1].bounds.upper
        else:
            const_lower = node.const
            const_upper = node.cons

        return inp.lower - const_upper, inp.upper - const_lower

    def _calc_add_bounds(self, node: Node):
        inp = node.from_node[0].bounds

        if node.const is None:
            const_lower = node.from_node[1].bounds.lower
            const_upper = node.from_node[1].bounds.upper
        else:
            const_lower = node.const
            const_upper = node.cons

        return inp.lower + const_lower, inp.upper + const_upper
    
    def _calc_affine_bounds(self, node: Node):
        inp = node.from_node[0].bounds

        lower = node.forward(inp.lower, clip='+', add_bias=False)
        lower += node.forward(inp.upper, clip='-', add_bias=True)
            
        upper = node.forward(inp.lower, clip='-', add_bias=False)
        upper += node.forward(inp.upper, clip='+', add_bias=True)

        return lower, upper

    def _calc_linear_bounds(self, node: Node):
        inp = node.from_node[0].bounds

        lower = node.forward(inp.lower, clip='+')
        lower += node.forward(inp.upper, clip='-')
            
        upper = node.forward(inp.lower, clip='-')
        upper += node.forward(inp.upper, clip='+')

        return lower, upper


