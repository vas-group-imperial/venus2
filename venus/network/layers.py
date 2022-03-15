# ************
# File: layers.py
# Top contributors (to current version): 
# 	Panagiotis Kouvaros (panagiotis.kouvaros@gmail.com)
# This file is part of the Venus project.
# Copyright: 2019-2021 by the authors listed in the AUTHORS file in the
# top-level directory.
# License: BSD 2-Clause (see the file LICENSE in the top-level directory).
# Description: classes for layers supported by Venus.
# ************

from __future__ import annotations
import itertools
import math
import numpy as np
import tensorflow as tf

from venus.bounds.bounds import Bounds
from venus.network.activations import Activations
from venus.network.activations import ReluState

class Layer:

    def __init__(self, input_shape, output_shape, depth, config):
        """
        Arguments:

            input_shape: shape of the input tensor to the layer.
            
            output_shape: shape of the output tensor to the layer.

            depth: depth of the layer in the network.
        """
        self.input_shape = input_shape
        self.input_size = np.prod(input_shape)
        self.output_shape = output_shape 
        self.output_size = np.prod(output_shape)
        self.depth = depth
        self.config = config
        self.out_vars =  np.empty(0)
        self.delta_vars = np.empty(0)
        self.pre_bounds = Bounds()
        self.post_bounds = Bounds()
        self.activation = None
        self.outputs = None

    def get_outputs(self):
        """
        Constructs a list of the indices of the nodes in the layer.

        Returns: 

            list of indices.
        """
        if self.outputs is not None:
            outputs = self.outputs
        else:
            if len(self.output_shape)>1:
                outputs =  [i for i in itertools.product(*[range(j) for j in self.output_shape])]
            else:
                outputs = list(range(self.output_size))

        return outputs

    def clean_vars(self):
        """
        Nulls out all MILP variables associate with the network.

        Returns 

            None
        """
        self.out_vars = np.empty(0)
        self.delta_vars = np.empty(0)


class Input(Layer):
    def __init__(self, lower, upper, name='input'):
        """
        Argumnets:
            
            lower: array of lower bounds of input nodes.
            
            upper: array of upper bounds of input nodes.
        """
        self.depth = 0
        self.out_vars = np.empty(0)
        self.delta_vars = np.empty(0)
        self.pre_bounds = self.post_bounds = Bounds(lower,upper)
        self.input_shape = self.output_shape = lower.shape
        self.input_size = self.output_size = np.prod(self.input_shape)
        self.activation = None
        self.outputs = None
        self.name=name

    def copy(self):
        """
        Returns: 

            a copy of the calling object 
        """
        return Input(self.post_bounds.lower, self.post_bounds.upper)

    def get_active_flag(self):
        """
        Returns an array of activity statuses for each node.
        """
        return np.ones(self.output_size, bool)

    def get_active_count(self):
        """
        Returns the total number of active nodes.
        """
        return self.output_size

    def get_inactive_flag(self):
        """
        Returns an array of inactivity statuses for each node.
        """
        return np.zeros(self.output_size, bool)

    def get_inactive_count(self):
        """
        Returns the total number of inactive nodes.
        """
        return 0

    def get_unstable_flag(self):
        """
        Returns an array of instability statuses for each node.
        """
        return np.zeros(self.output_size, bool)

    def get_unstable_count(self):
        """
        Returns the total number of unstable nodes.
        """
        return 0

    def get_propagation_flag(self):
        """
        Returns an array of sip propagation statuses for each node.
        """
        return np.ones(self.output_size, bool)
         
    def get_propagation_count(self):
        """
        Returns the total number of sip propagation nodes.
        """
        return self.output_size


class GlobalAveragePooling(Layer):
    def __init__(self, input_shape, output_shape, depth):
        """
        Arguments:

            input_shape: shape of the input tensor to the layer.
            
            output_shape: shape of the output tensor to the layer.

            depth: depth of the layer in the network.
        """
        super().__init__(input_shape,output_shape,depth)

    def copy(self):
        """
        Returns: 

            a copy of the calling object 
        """
        return GlobalAveragePooling(self.input_shape, self.output_shape, self.depth)


class MaxPooling(Layer):
    def __init__(self, input_shape, output_shape, pool_size, depth):
        """
        Arguments:

            input_shape: shape of the input tensor to the layer.
            
            output_shape: shape of the output tensor to the layer.

            pool_size: pair of int for the width and height of the pooling.

            depth: depth of the layer in the network.
        """
        super().__init__(input_shape,output_shape,depth)
        self.pool_size = pool_size

    def copy(self):
        """
        Returns: 

            a copy of the calling object 
        """
        return MaxPooling(self.input_shape, 
                          self.output_shape, 
                          self.pool_size,
                          self.depth)


class AffineLayer(Layer):
    def __init__(self, input_shape, output_shape, weights, bias, activation, depth, config):
        """
        Arguments:

            input_shape: shape of the input tensor to the layer.
            
            output_shape: shape of the output tensor to the layer.

            weights: weight matrix.

            bias: bias vector.

            activation: Activation.

            depth: depth of the layer in the network.
        """
        super().__init__(input_shape, output_shape, depth, config)
        self.weights = weights
        self.bias = bias
        self.activation = activation
        self.active_flag = None
        self.inactive_flag = None
        self.stable_flag = None
        self.unstable_flag = None
        self.propagation_flag = None
        self.propagation_flag = None
        self.unstable_count = None
        self.active_count = None
        self.inactive_count = None
        self.propagation_count = None
        if activation == Activations.relu:
            self.state = np.array([ReluState.UNSTABLE]*self.output_size,dtype=ReluState).reshape(self.output_shape)
            self.dep_root = np.array([False]*self.output_size).reshape(self.output_shape)
            self.dep_consistency = [True for i in range(self.output_size)]

    def set_activation(self, activation):
        """
        Sets the activation of the layer.

        Arguments:
            
            activation: Activation.
        """
        self.activation = activation
        if activation == Activations.linear:
            self.state = self.dep_root = self.dep_consistency = None
        else:
            self.state = np.array([ReluState.UNSTABLE]*self.output_size,dtype=ReluState).reshape(self.output_shape)
            self.dep_root = np.array([False]*self.output_size).reshape(self.output_shape)
            self.dep_consistency = [True for i in range(self.output_size)]

    def reset_state_flags(self):
        """
        Resets calculation flags for relu states
        """
        self.active_flag = None
        self.inactive_flag = None
        self.stable_flag = None
        self.unstable_flag = None
        self.propagation_flag = None
        self.propagation_flag = None
        self.unstable_count = None
        self.active_count = None
        self.inactive_count = None
        self.propagation_count = None
          
    def is_active(self, node):
        """
        Detemines whether a given ReLU node is stricly active.

        Arguments:

            node: index of the node whose active state is to be determined.

        Returns:

            bool expressing the active state of the given node.
        """
        if not self.activation == Activations.relu:
            return []

        return self.pre_bounds.lower[node] >= 0 or self.state[node] == ReluState.ACTIVE

    def is_inactive(self, node):
        """
        Determines whether a given ReLU node is strictly inactive.

        Arguments:

            node: index of the node whose inactive state is to be determined.

        Returns:

            bool expressing the inactive state of the given node.
        """
        if not self.activation == Activations.relu:
            return None

        return self.pre_bounds.upper[node] <= 0 or self.state[node] == ReluState.INACTIVE

    def is_stable(self, node, delta_val=None):
        """
        Determines whether a given ReLU node is stable.

        Arguments:

            node: index of the node whose active state is to be determined.

            delta_val: value of the binary variable associated with the node.
            if set, the value is also used in conjunction with the node's
            bounds to determined its stability.

        Returns:

            bool expressing the stability of the given node.
        """
        if not self.activation == Activations.relu:
            return None

        cond1 = self.pre_bounds.lower[node] >= 0 or self.pre_bounds.upper[node] <= 0
        cond2 = not self.state[node] == ReluState.UNSTABLE
        cond3 = False if delta_val is None else (delta_val == 0 or delta_val == 1)

        return cond1 or cond2 or cond3


    def get_active_flag(self):
        """
        Returns an array of activity statuses for each node.
        """
        if self.active_flag is None:
            if self.activation != Activations.relu:
                self.active_flag = np.ones(self.output_size, bool)
            else:
                self.active_flag = self.pre_bounds.lower.flatten() > 0

        return self.active_flag

    def get_active_count(self):
        """
        Returns the total number of active nodes.
        """
        if self.active_count is None:
            if self.activation != Activations.relu:
                self.active_count = self.output_size
            else:
                self.active_count = np.sum(self.get_active_flag())

        return self.active_count

    def get_inactive_flag(self):
        """
        Returns an array of inactivity statuses for each node.
        """
        if self.inactive_flag is None:
            if self.activation != Activations.relu:
                self.inactive_flag = np.zeros(self.output_size, bool)
            else:
                self.inactive_flag = self.pre_bounds.upper.flatten() <= 0

        return self.inactive_flag

    def get_inactive_count(self):
        """
        Returns the total number of inactive nodes.
        """
        if self.inactive_count is None:
            if self.activation != Activations.relu:
                self.inactive_count =  0
            else:
                self.inactive_count = np.sum(self.get_inactive_flag())

        return self.inactive_count

    def get_unstable_flag(self):
        """
        Returns an array of instability statuses for each node.
        """
        if self.unstable_flag is None:
            if self.activation != Activations.relu:
                self.unstable_flag = np.zeros(self.output_size, bool)
            else:
                self.unstable_flag = np.logical_and(
                    self.pre_bounds.lower < 0,
                    self.pre_bounds.upper > 0
                ).flatten()

        return self.unstable_flag

    def get_unstable_count(self):
        """
        Returns the total number of unstable nodes.
        """
        if self.unstable_count is None:
            if self.activation != Activations.relu:
                self.unstable_count = 0
            else:
                self.unstable_count = np.sum(self.get_unstable_flag())

        return self.unstable_count

    def get_propagation_flag(self):
        """
        Returns an array of sip propagation statuses for each node.
        """
        if self.propagation_flag is None:
            if self.activation != Activations.relu:
                self.propagation_flag = np.ones(self.output_size, bool)
            else:
                self.propagation_flag = np.logical_or(
                    self.get_active_flag(),
                    self.get_unstable_flag()
                ).flatten()

        return self.propagation_flag

    def get_propagation_count(self):
        """
        Returns the total number of sip propagation nodes.
        """
        if self.propagation_count is None:
            if self.activation != Activations.relu:
                self.propagation_count = self.output_size
            else:
                self.propagation_count = np.sum(self.get_propagation_flag())

        return self.propagation_count
 

    def get_active(self):
        """
        Determines the active nodes of the layer.

        Returns: 

            A list of indices of  the active nodes.
        """
        if not self.activation == Activations.relu:
            return []
        return [i for i in range(self.output_size) is self.is_active(i)]


    def get_inactive(self):
        """
        Determines the inactive nodes of the layer.

        Returns: 

            A list of indices of  the inactive nodes.
        """
        if  self.activation != Activations.relu:
            return []

        return [i for i in range(self.output_size) if self.is_inactive(i)]


    def get_unstable(self, delta_vals=None):
        """
        Determines the unstable nodes of the layer.

        Arguments:

            delta_vals: values of the binary variables associated with the nodes.
            if set, the values are also used in conjunction with the nodes'
            bounds to determined their instability.
         
        Returns: 

            A list of indices of the unstable nodes.
        """
        if not self.activation == Activations.relu:
            return []
        if delta_vals is None:
            return [i for i in self.outputs() if not self.is_stable(i)]
        else:
            return [i for i in self.outputs() if not self.is_stable(i, delta_vals[i])]


    def get_stable(self, delta_vals=None):
        """
        Determines the stable nodes of the layer.

        Arguments:

            delta_vals: values of the binary variables associated with the nodes.
            if set, the values are also used in conjunction with the nodes'
            bounds to determined their stability.
         
        Returns: 

            A list of indices of the stable nodes.
        """
        if not self.activation == Activations.relu:
            return []

        if delta_vals is None:
            return [i for i in self.get_outputs() if self.is_stable(i)]
        else:
            return [i for i in self.get_outputs() if self.is_stable(i, delta_vals[i])]

    def get_upper_relu_slope(self):
        slope = np.zeros(self.output_size, dtype=self.config.PRECISION)
        upper = self.pre_bounds.upper.flatten()[self.get_unstable_flag()]
        lower = self.pre_bounds.lower.flatten()[self.get_unstable_flag()]
        slope[self.get_unstable_flag()] = upper /  (upper - lower)
        slope[self.get_active_flag()] = 1.0
        
        return slope

    def get_lower_relu_slope(self):
        slope = np.ones(self.output_size, dtype=self.config.PRECISION)
        upper = self.pre_bounds.upper.flatten()
        lower = self.pre_bounds.lower.flatten()
        idxs = abs(lower) >=  upper
        slope[idxs] = 0.0
        slope[self.get_inactive_flag()] = 0.0
        slope[self.get_active_flag()] = 1.0

        return slope


class FullyConnected(AffineLayer):
    def __init__(self, input_shape, output_shape, weights, bias, activation, depth, config):
        """
        Arguments:

            input_shape: shape of the input tensor to the layer.
            
            output_shape: shape of the output tensor to the layer.

            weights: weight matrix.

            bias: bias vector.

            activation: Activation.

            depth: depth of the layer in the network.
        """
        super().__init__(input_shape, output_shape, weights, bias, activation, depth, config)
        self.vars = {'out': np.empty(0), 'delta': np.empty(0)}

    def copy(self):
        """
        Returns: 

            a copy of the calling object 
        """
        fc =  FullyConnected(self.input_shape, self.output_shape, self.weights, self.bias, self.activation, self.depth, self.config)
        fc.pre_bounds = self.pre_bounds.copy()
        fc.post_bounds = self.post_bounds.copy()
        if self.activation == Activations.relu:
            fc.state = self.state.copy()
            fc.dep_root = self.dep_root.copy()
            fc.dep_consistency = self.dep_consistency.copy()
        return fc

    def neighbours(self, l, node):
        """
        Determines the neighbouring nodes to the given node from the previous
        layer.

        Arguments:

            node: the index of the node.

        Returns:

            a list of neighbouring nodes.
        """
        if len(l.output_shape)>1:
            return [i for i in itertools.product(*[range(j) for j in l.output_shape])]
        else:
            return list(range(self.input_size))

    def get_bias(self, node):
        """
        Returns the bias of the given node.

        Arguments:

            node: the index of the node.

        Returns:
            
            the bias 
        """
        return self.bias[node]

    def edge_weight(self, l, node1, node2):
        """
        Returns the weight of the edge between node1 of the current layer and
        node2 of the previous layer.

        Arguments:

            node: the index of the node.

        Returns:
            
            the bias.
        """
        if isinstance(l, Conv2D) and isinstance(node2, tuple) and len(node2) > 1:
            node2 = l.flattened_index(node2)

        return self.weights[node1][node2]
    
    def intra_connected(self, n1, n2):
        """
        Determines whether two given nodes share connections with nodes from
        the previous layers.

        Arguments:

            n1: index of node 1

            n2: index of node 2

        Returns:
            
            bool expressing whether the nodes are intra-connected
        """
        return True

    def joint_product(self, inp, n1, n2):
        return inp, (self.weights[n1,:], self.weights[n2,:]), (self.bias[n1],self.bias[n2])


    def forward(self, input: np.array, clip=None, add_bias=True) -> np.array:

        if clip is None:
            pass

        elif clip == '+':
            weights = np.clip(self.weights, 0, math.inf)

        elif clip == '-':
            weights = np.clip(self.weights, -math.inf, 0)

        else:
            raise ValueError(f'Kernel clip value {kernel_clip} not recognised')


        product = weights.dot(input.flatten())
        if add_bias is True:
            product += self.bias

        return product


    def transpose(self, inp) -> np.array:
        return np.dot(inp, self.weights)


class Conv2D(AffineLayer):
    def __init__(
        self, 
        input_shape, 
        output_shape, 
        kernels, 
        bias, 
        padding, 
        strides, 
        activation, 
        depth,
        config
    ):
        """
        Arguments:

            input_shape: shape of the input tensor to the layer.
            
            output_shape: shape of the output tensor to the layer.

            kernerls: kernels : matrix.

            bias: bias vector.

            padding: pair of int for the width and height of the padding.
            
            strides: pair of int for the width and height of the strides.

            activation: Activation.

            depth: depth of the layer in the network.
        """
        super().__init__(input_shape, output_shape, kernels, bias, activation, depth, config)
        self.vars = {'out': np.empty(0), 'delta': np.empty(0)}
        self.kernels = kernels
        self.bias = bias
        self.padding = padding
        self.strides = strides
        self.depth = depth
        self._non_pad_idxs = None
        self._input_padded_size = None
        self._input_padded_shape = None

    def copy(self):
        """
        Returns: 

            a copy of the calling object 
        """
        conv = Conv2D(self.input_shape, self.output_shape, self.kernels, self.bias, self.padding, self.strides, self.activation, self.depth, self.config)
        conv.pre_bounds = self.pre_bounds.copy()
        conv.post_bounds = self.post_bounds.copy()
        if self.activation == Activations.relu:
            conv.state = self.state.copy()
            conv.dep_root = self.dep_root.copy()
            conv.dep_consistency = self.dep_consistency.copy()
        return conv

    def flattened_index(self, node):
        """
        Computes the flattened index of the given nodes.

        Arguments:

            node: tuple of the index of the node.

        Returns:

            int of the flattensed index of the node.
        """
        X,Y,Z = self.output_shape
        return node[0] * Y * Z + node[1] * Z + node[2]
       
    def neighbours(self, node):
        """
        Determines the neighbouring nodes to the given node from the previous
        layer.

        Arguments:

            node: the index of the node.

        Returns:

            a list of neighbouring nodes.
        """
        x_start = node[0] * self.strides[0] - self.padding[0]
        x_rng = range(x_start, x_start + self.kernels.shape[0])
        x = [i for i in x_rng if i >= 0 and i<self.input_shape[0]]
        y_start = node[1] * self.strides[1] - self.padding[1]
        y_rng = range(y_start, y_start + self.kernels.shape[1])
        y = [i for i in y_rng if i>=0 and i<self.input_shape[1]]
        z = [i for i in range(self.kernels.shape[2])]
        return [i for i in itertools.product(*[x,y,z])]


    def intra_connected(self, n1, n2):
        """
        Determines whether two given nodes share connections with nodes from
        the previous layers.

        Arguments:

            n1: index of node 1

            n2: index of node 2

        Returns:
            
            bool expressing whether the nodes are intra-connected
        """
        n_n1 = self.neighbours(n1)
        n_n2 = self.neighbours(n2)
        return len(set(n_n1) & set(n_n2)) > 0

    def get_bias(self, node):
        """
        Returns the bias of the given node.

        Arguments:

            node: the index of the node.

        Returns:
            
            the bias 
        """
        return self.bias[node[-1]]

    def edge_weight(self, l, node1, node2):
        """
        Returns the weight of the edge between node1 of the current layer and
        node2 of the previous layer.

        Arguments:

            node: the index of the node.

        Returns:
            
            the bias.
        """
        x_start = node1[0] * self.strides[0] - self.padding[0]
        x = node2[0] - x_start
        y_start = node1[1] * self.strides[1] - self.padding[1]
        y = node2[1] - y_start
        return self.kernels[x][y][node2[2]][node1[2]]


    def forward(self, input: np.array, clip=None, add_bias=True) -> np.array:
        if clip is None:
            pass

        elif clip == '+':
            kernels = np.clip(self.kernels, 0, math.inf)

        elif clip == '-':
            kernels = np.clip(self.kernels, -math.inf, 0)

        else:
            raise ValueError(f'Kernel clip value {kernel_clip} not recognised')

        padded_input = Conv2D.pad(
            input.reshape(self.input_shape),
            self.padding
        )[np.newaxis, ...]

        product = tf.nn.convolution(
            padded_input,
            kernels,
            strides=self.strides,
            padding='VALID'
        ).numpy().flatten()

        if add_bias is True:
            product += np.array(
                [self.bias for i in range(int(self.output_size / self.kernels.shape[-1]))],
                dtype=self.config.PRECISION
            ).flatten()

        return product

    def get_non_pad_idxs(self) -> np.array:
        """
        Returns:

                Indices of the original input whithin the padded one.
        """
        if self._non_pad_idxs is not None:
            return self._non_pad_idxs

        height, width, channels = self.input_shape
        pad_height, pad_width = self.padding
        size = self.get_input_padded_size()
        self._non_pad_idxs = np.arange(size, dtype=np.uint).reshape(
            height + 2 * pad_height,
            width + 2 * pad_width,
            channels
        )[pad_height:height + pad_height, pad_width :width + pad_width].flatten()

        return self._non_pad_idxs

    def get_input_padded_size(self) -> int:
        """
        Returns:

                Size of the padded input.
        """
        if self._input_padded_size is not None:
            return self._input_padded_size

        height, width, channels = self.input_shape
        pad_height, pad_width = self.padding
        self._input_padded_size =  (height + 2 * pad_height) * (width + 2 * pad_width) * channels
        
        return self._input_padded_size

    def get_input_padded_shape(self) -> tuple:
        """
        Returns:

                Shape of the padded input.
        """
        if self._input_padded_shape is not None:
            return self._input_padded_shape

        height, width, channels = self.input_shape
        pad_height, pad_width = self.padding
        self._input_padded_shape =  (
            height + 2 * pad_height,
            width + 2 * pad_width,
            channels
        )
        
        return self._input_padded_shape

    @staticmethod
    def compute_output_shape(in_shape, weights_shape, pads, strides):
        """
        Computes the output shape of a convolutional layer.

        Arguments:

            in_shape: shape of the input tensor to the layer.

            weights_shape: shape of the kernels of the layer.

            padding: pair of int for the width and height of the padding.
            
            strides: pair of int for the width and height of the strides.

        Returns:

            tuple of int of the output shape
        """
        x,y,z = in_shape
        X,Y,_,K = weights_shape
        p,q  = pads
        s,r = strides
        out_x = int(math.floor( (x-X+2*p) / s + 1 ))
        out_y = int(math.floor( (y-Y+2*q) / r + 1 ))
        
        return (out_x,out_y,K)

    @staticmethod
    def pad(A: np.array, PS: tuple, values: tuple=(0,0)) -> np.array:
        """
        Pads a given matrix with zeroes

        Arguments:
            
            A: matrix:
            
            PS: tuple denoting the padding size

        Returns

            padded A.
        """
        if PS == (0, 0):
            return A
        else:
            return np.pad(A, (PS, PS, (0, 0)), 'constant', constant_values=values)



    def transpose(self, inp) -> np.array:
        return tf.nn.conv2d_transpose(
            inp.reshape((inp.shape[0],) + self.output_shape),
            self.kernels,
            (inp.shape[0], ) + self.input_shape,
            self.strides,
            padding = [
                [0, 0], 
                [self.padding[0], self.padding[0]], 
                [self.padding[1], self.padding[1]], 
                [0, 0]
            ]
        ).numpy().reshape(inp.shape[0], -1)


    @staticmethod
    def im2col(matrix: np.array, kernel_shape: tuple, strides: tuple, indices: bool=False) -> np.array:
        """
        MATLAB's im2col function.

        Arguments:

            matrix:
                The matrix.
            kernel_shape:
                The kernel shape.
            strides:
                The strides of the convolution.
            indices:
                Whether to select indices (true) or values (false) of the matrix.

        Returns:

            im2col matrix
        """
        assert len(matrix.shape) in [3, 4], f"Dimension {len(matrix.shape)} is not supported."
        width, height, filters = matrix.shape[0:3]
        if len(matrix.shape) == 4:
            _matrix = matrix.reshape(-1, matrix.shape[-1])
            axis = 0
        else:
            _matrix = matrix
            axis = None
        col_extent = height - kernel_shape[1] + 1
        row_extent = width - kernel_shape[0] + 1

        # starting block indices
        start_idx = np.arange(kernel_shape[0])[:, None] * height * filters + \
            np.arange(kernel_shape[1] * filters)

        # offsetted indices across the height and width of A
        offset_idx = np.arange(row_extent)[:, None][::strides[0]] * height * filters + \
            np.arange(0, col_extent * filters, strides[1] * filters)

        # actual indices
        if indices is True:
            return start_idx.ravel()[:, None] + offset_idx.ravel()

        return np.take(_matrix, start_idx.ravel()[:, None] + offset_idx.ravel(), axis=axis)



