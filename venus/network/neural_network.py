"""
# File: neural_network.py
# Top contributors (to current version): 
# 	Panagiotis Kouvaros (panagiotis.kouvaros@gmail.com)
# This file is part of the Venus project.
# Copyright: 2019-2021 by the authors listed in the AUTHORS file in the
# top-level directory.
# License: BSD 2-Clause (see the file LICENSE in the top-level directory).
# Description: Neural Network class.
"""


import torch
import numpy as np
import os
import itertools

from venus.input.onnx_parser import ONNXParser
from venus.specification.specification import Specification
from venus.specification.formula import TrueFormula
from venus.network.node import Input, Gemm, Relu, MatMul, Add, Constant, Conv
from venus.bounds.sip import SIP
from venus.common.logger import get_logger
from venus.common.configuration import Config

class NeuralNetwork:
    
    logger = None

    def __init__(self, model_path: str, config: Config):
        """
        Arguments:

            model_path: str of path to neural network model.

            name: name of the neural network.
        """
        self.model_path = model_path
        self.config = config
        if NeuralNetwork.logger is None and not config.LOGGER.LOGFILE is None:
            NeuralNetwork.logger = get_logger(__name__, config.LOGGER.LOGFILE)
        self.head = None
        self.tail = None
        self.node = {}
        self.input_simplified = False
        self.batched = False

    def load(self):
        """
        Loads a neural network into Venus.

        Arguments:

            model_path: str of path to neural network model.

        Returns

            None

        """
        _,model_format = os.path.splitext(self.model_path)
        if not model_format == '.onnx':
            raise Exception('only .onnx models are supported')
        
        onnx_parser = ONNXParser(self.config)
        self.head, self.tail, self.node = onnx_parser.load(self.model_path)

        # print(self.head.input_shape)
        # s = 0
        # for i in range(self.tail.depth):
            # nodes = self.get_node_by_depth(i)
            # for j in nodes:
                # if j.has_relu_activation() or j.has_max_pool():
                    # s += j.output_size
                # print(i, j, j.output_size)
        # print(s)
        
        # import sys
        # sys.exit()

    def copy(self):
        """
        Copies the calling object.
        """
        nn = NeuralNetwork(self.model_path, self.config)

        for i in self.node:
            nn.node[i] = self.node[i].copy()
            # to edit
            nn.node[i].grads = self.node[i].grads

        nn.head =  [nn.node[i.id] for i in self.head]
        nn.tail = nn.node[self.tail.id]
 
        for i, j in self.node.items():
            nn.node[i].from_node = [
                nn.node[k.id]
                for k in j.from_node
                if isinstance(k, Input) is not True
            ]
            nn.node[i].to_node = [nn.node[k.id] for k in j.to_node]
            nn.node[i].grads = self.node[i].grads


        nn.input_simplified = self.input_simplified
        nn.batched = self.batched


        return nn

    def get_node_by_depth(self, depth: int) -> list:
        """
        Finds all nodes of certain depth.

        Arguments:
                
            depth:
                the depth. 

        Returns:
            List of nodes with specified depth.
        """
        return [self.node[i] for i in self.node if self.node[i].depth == depth]


    def clean_vars(self):
        """
        Nulls out all MILP variables associate with the network.
        """
        for _, i in self.node.items():
            i.clean_vars()

    def clean_outputs(self):
        """
        Nulls out all outputs of the nodes of the network.
        """
        for i in self.head:
            i.from_node[0].output = None

        for _, i in self.node.items():
            i.output = None

    def detach(self):
        """
        Detaches and clones potentially gradient required tensors. 
        """
        for _, i in self.node.items():
            i.detach()

    def cuda(self):
        """
        Moves all data to gpu memory
        """
        for _, i in self.node.items():
            i.cuda()

    def cpu(self):
        """
        Moves all data to cpu memory
        """
        for _, i in self.node.items():
            i.cpu()

    def set_batch_size(self, size: int=1):
        """
        Sets the batch size.

        Arguments:
            size: the batch size.
        """
        for _, i in self.node.items():
            i.set_batch_size(size)

        self.batched = True

    def cache_bounds(self):
        for _, i in self.node.items():
            i.set_cache_bounds()

    def use_cache_bounds(self):
        for _, i in self.node.items():
            i.use_cache_bounds()


    def predict(self, inp: np.array, mean: float=0, std: float=1):
        """
        Computes the output of the network on a given input.
        
        Arguments:
                
            input:
                input vector to the network.
            mean:
                normalisation mean.
            std:
                normalisation standard deviation.
        
        Returns:

            vector of the network's output on input.
        """
        nn = self.copy()
        input_layer = Input(inp, inp)
        spec = Specification(input_layer, TrueFormula())
        spec.normalise(mean, std)
        config = Config()
        config.SIP.OSIP_CONV = False
        config.SIP.OSIP_FC = False
        sip = SIP([input_layer] + nn.layers, config)
        sip.set_bounds()

        return nn.layers[-1].post_bounds.lower

    def classify(self, inp: np.array, mean: float=0, std: float=1):
        """
        Computes the classification of a given input by the network.
        
        Arguments:
                
            input:
                input vector to the network.
            mean:
                normalisation mean.
            std:
                normalisation standard deviation.
        
        Returns 

            int of the class of the input 
        """
        pred = self.predict(input, mean, std)
        return np.argmax(pred)

    def is_fc(self):
        for i in self.node:
            if not isinstance(i, Gemm) and not isinstance(i, Relu):
                return False
        return True

    def get_n_relu_nodes(self):
        """
        Computes the number of ReLU nodes in the network.

        Returns:

            int of the number of ReLU nodes.
        """
        return sum([
            self.node[i].output_size for i in self.node if isinstance(self.node[i], Relu)
        ])

    def get_n_stabilised_nodes(self):
        """
        Computes the number of stabilised ReLU nodes in the network.

        Returns:

            int of the number of stabilised ReLU nodes.
        """
        return sum([
            self.node[i].get_stable_count() 
            for i in self.node if isinstance(self.node[i], Relu)
        ])

    def get_n_non_stabilised_nodes(self):
        """
        Computes the number of non-stabilised ReLU nodes in the network.

        Returns:

            int of the number of stabilised ReLU nodes.
        """
        return sum([
            self.node[i].get_unstable_count() 
            for i in self.node if isinstance(self.node[i], Relu)
        ])

    def get_stability_ratio(self):
        """
        Computes the ratio of stabilised ReLU nodes to the total number of ReLU nodes.

        Returns:

            float of the ratio.
        """
        relu_count = self.get_n_relu_nodes()
        return  self.get_n_stabilised_nodes() / relu_count if relu_count > 0 else 0


    def get_output_range(self):
        """
        Computes the  output range of the network.

        Returns:

            float of the range.
        """

        diff = self.tail.bounds.upper - self.tail.bounds.lower
        return torch.mean(diff).item()
        

    def calc_neighbouring_units(self, p_node, s_node, index):
        """
        Given a unit it determines its neighbouring units from the previous
        node.

        Arguments:
            p_node:
                the preceding node.
            s_node:
                the subsequent node.
            index: 
                the index of the node.
        Returns:
            a list of indices of neighbouring units.
        """
        if isinstance(s_node, Gemm):
            return p_node.get_outputs()

        elif isinstance(s_node, Conv):
            height_start = index[-2] * s_node.strides[-2] - s_node.pads[0]
            height_rng = range(height_start, height_start + s_node.krn_height)
            height = [i for i in height_rng if i >= 0 and i < s_node.krn_height]
            width_start = index[-1] * s_node.strides[-1] - s_node.pads[1]
            width_rng = range(width_start, width_start + s_node.krn_width)
            width = [i for i in width_rng if i >= 0 and i < s_node.krn_width]
            channel = list(range(s_node.in_ch))

            if len(p_node.output_shape) == 3:
                shape = [channel, height, width]
            else:
                shape = [[0], channel, height, width]

            return list(itertools.product(*shape))

    def forward(self, inp, save_gradient: bool=False):
        """
        Computes the output of the network given an input.

        Arguments:
            inp:
                The input.
            save_gradient:
                whether to save the gradients of the nodes.
        Returns
            The output given inp.
        """
        for i in self.head:
            i.from_node[0].output = inp
        for i in range(self.tail.depth + 1):
            nodes = self.get_node_by_depth(i)
            for j in nodes:
                j.forward(save_output=True, save_gradient=save_gradient) 
                # if save_gradient is True and j.grads is not None:
                    # if self.batched is True:
                        # j.grads = torch.mean(j.grads, dim=0).flatten()
                    # j.grads = torch.argsort(j.grads, descending=True).tolist()


        output = self.tail.output
        self.clean_outputs()

        return output

    def has_custom_relaxation_slope(self):
        """
        Returns whether any relu relaxation slope in the network is not the
        default.
        """
        for _, i in self.node.items():
            if isinstance(i, Relu) and i.has_custom_relaxation_slope():
                return True

        return False

    def reset_relaxation_slope(self):
        """
        Resets to default the relu relaxation slopes in the network.
        """
        for _, i in self.node.items():
            if isinstance(i, Relu) and i.has_custom_relaxation_slope():
                i.set_lower_relaxation_slope(
                    approx=self.config.SIP.RELU_APPROXIMATION
                )

    def clear_relaxation_slope(self):
        """
        Clears the relu relaxation slope.
        """
        for _, i in self.node.items():
            if isinstance(i, Relu) and i.has_custom_relaxation_slope():
                i.clear_lower_relaxation_slope()

    def get_non_linear_starting_depth(self):
        """
        Returns the shallowest depth with a non linear activation function.
        """
        for i in range(self.tail.depth):
            nodes = self.get_node_by_depth(i)
            for j in nodes:
                if j.has_relu_activation() or j.has_max_pool():
                    return i
        return i


    def get_lower_relaxation_slopes(self, gradient=False):
        """
        Builds a dictionary of the lower relaxation slopes of the relu nodes.

        Arguments:
        
            gradient:
                Whether the slopes require gradient.
        """
        lower, upper = {}, {}
        for _, i in self.node.items():
            if isinstance(i, Relu):
                slope = i.get_lower_relaxation_slope()
                if slope[0] is not None:
                    lower[i.id] = slope[0].detach().clone().requires_grad_(gradient)
                if slope[1] is not None:
                    upper[i.id] = slope[1].detach().clone().requires_grad_(gradient)

        return [lower, upper]

    def set_lower_relaxation_slopes(
            self, lower_slopes: torch.Tensor, upper_slopes: torch.Tensor
    ):
        """
        Sets the lower relaxation slopes of the relu nodes.

        Arguments:
        
            slopes:
                A dictionary of the lower relaxation slopes of the relu nodes.
        """
        for i in lower_slopes:
            self.node[i].set_lower_relaxation_slope(
                lower_slopes[i], upper_slopes[i]
            )
