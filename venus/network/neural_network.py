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


import numpy as np
import os
import itertools

from venus.input.onnx_parser import ONNXParser
from venus.specification.specification import Specification
from venus.specification.formula import TrueFormula
from venus.network.layers import Input, FullyConnected, Conv2D
from venus.bounds.sip import SIP
from venus.common.logger import get_logger
from venus.network.activations import Activations
from venus.common.configuration import Config

class NeuralNetwork():
    
    logger = None

    def __init__(self, model_path, config: Config):
        """
        Arguments:

            model_path: str of path to neural network model.

            name: name of the neural network.
        """
        self.model_path = model_path
        self.config = config
        if NeuralNetwork.logger is None and not config.LOGGER.LOGFILE is None:
            NeuralNetwork.logger = get_logger(__name__, config.LOGGER.LOGFILE)
        self.layers = None
        self.mean = 0
        self.std = 1


    def load(self):
        """
        Loads a neural network into Venus.

        Arguments:

            model_path: str of path to neural network model.

        Returns

            None

        """
        _,model_format = os.path.splitext(self.model_path)
        if not model_format in ['.h5', '.onnx', '.onnx.gz']:
            raise Exception('only .h5 and .onnx models are supported')
        if model_format == '.h5':
            keras_parser = KerasParser()
            self.layers = keras_parser.load(self.model_path)
        else:
            onnx_parser = ONNXParser(self.config)
            self.layers = onnx_parser.load(self.model_path)
            self.mean = onnx_parser.mean
            self.std = onnx_parser.std

        self.simplify()
 
    def copy(self):
        """
        Returns: 

            a copy of the calling object 
        """
        nn = NeuralNetwork(self.model_path, self.config)
        nn.layers = [layer.copy() for layer in self.layers]
        
        return nn

    def clean_vars(self):
        """
        Nulls out all MILP variables associate with the network.

        Returns 

            None
        """
        for layer in self.layers + [self.input] + [self.output]:
            layer.clean_vars()

    def predict(self, input, mean=0, std=1):
        """
        Computes the output of the network on a given input.
        
        Arguments:
                
            input: input vector to the network.

            mean: normalisation mean.

            std: normalisation standard deviation.
        
        Returns 

            vector of the network's output on input
        """
        nn = self.copy()
        input_layer = Input(input,input)
        spec = Specification(input_layer,TrueFormula())
        spec.normalise(mean, std)
        config = Config()
        config.SIP.OSIP_CONV = False
        config.SIP.OSIP_FC = False
        sip = SIP([input_layer] + nn.layers, config)
        sip.set_bounds()

        return nn.layers[-1].post_bounds.lower

    def classify(self, input, mean=0, std=1):
        """
        Computes the classification of a given input by the network.
        
        Arguments:
                
            input: input vector to the network.

            mean: normalisation mean.

            std: normalisation standard deviation.
        
        Returns 

            int of the class of the input 
        """
        pred = self.predict(input, mean, std)
        return np.argmax(pred)

    def is_fc(self):
        for layer in self.layers:
            if not isinstance(layer, FullyConnected):
                return False
        return True

    def get_n_relu_nodes(self):
        """
        Computes the number of ReLU nodes in the network.

        Returns:

            int of the number of ReLU nodes.
        """
        count = 0
        for layer in self.layers:
            if layer.activation == Activations.relu:
                count += len(layer.get_outputs())

        return count

    def get_n_stabilised_nodes(self):
        """
        Computes the number of stabilised ReLU nodes in the network.

        Returns:

            int of the number of stabilised ReLU nodes.
        """
        count = 0 
        for layer in self.layers:
            if layer.activation == Activations.relu:
                count += len(layer.get_stable())

        return count

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

        diff = self.layers[-1].post_bounds.upper - self.layers[-1].post_bounds.lower
        rng = np.average(diff)
        
        return rng


    def neighbours_from_p_layer(self, depth, node):
        """
        Determines the neighbouring nodes to the given node from the previous
        layer.

        Arguments:

            depth: the depth of the layer of the node.

            node: the index of the node.

        Returns:

            a list of neighbouring nodes.
        """
        l, p_l = self.layers[depth - 1], self.layers[depth - 2]
        if isinstance(l, FullyConnected):
            return p_l.get_outputs()
        elif isinstance(l, Conv2D):
            x_start = node[0] * l.strides[0] - l.padding[0]
            x_rng = range(x_start, x_start + l.kernels.shape[0])
            x = [i for i in x_rng if i >= 0 and i < l.input_shape[0]]
            y_start = node[1] * l.strides[1] - l.padding[1]
            y_rng = range(y_start, y_start + l.kernels.shape[1])
            y = [i for i in y_rng if i>=0 and i < l.input_shape[1]]
            z = [i for i in range(l.kernels.shape[2])]
            return [i for i in itertools.product(*[x,y,z])]

    def simplify(self):
        if self.layers is None:
            return

        # find first non linear layer
        non_linear_idx = 0
        for i in self.layers:
            if i.activation != Activations.relu:
                non_linear_idx += 1
            else:
                break
 
        if non_linear_idx == 0:
            return 

        # merge linear layers into one fully connected
        weights, bias  = self._simplify_layer(self.layers[non_linear_idx])
        for i in range(non_linear_idx - 1, -1, -1):
            w, b = self._simplify_layer(self.layers[i])
            bias = weights @ b + bias 
            weights = weights @ w 
       
        # simplify 
        fc = FullyConnected(
            (self.layers[0].input_size, ), 
            (self.layers[non_linear_idx].output_size, ),
            weights,
            bias,
            self.layers[non_linear_idx].activation,
            1
        )
        for i in self.layers[non_linear_idx + 1:]:
            i.depth -= non_linear_idx

        self.layers = [fc] + self.layers[non_linear_idx + 1:]

    def _simplify_layer(self, layer):
        if isinstance(layer, FullyConnected):
            weights, bias =  layer.weights, layer.bias
        elif isinstance(layer, Conv2D):
            M, N, O, K = layer.kernels.shape
            sz_kernel = int(layer.output_size / K)
            pad = Conv2D.pad(
                np.arange(layer.input_size).reshape(layer.input_shape),
                layer.padding, 
                values=(layer.input_size, layer.input_size)
            )
            im2col = np.repeat(
                Conv2D.im2col(pad, (M, N), layer.strides), 
                K,
                axis=1
            )
            kernel = np.array(
                [
                    layer.kernels[:,:,:,i].flatten()
                    for j in range(sz_kernel) for i in range(K)
                ]
            ).T  
            weights = np.zeros((layer.output_size, layer.input_size+1), dtype='float64')
            weights[range(layer.output_size), im2col] = kernel
            weights = np.delete(weights, layer.input_size, 1)
            bias = np.array([layer.bias for i in range(sz_kernel)], dtype='float64').flatten()
        else:
            assert False, 'Layer not currently supported'

        return weights, bias



