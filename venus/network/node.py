# ************
# File: node.py
# Top contributors (to current version): 
# 	Panagiotis Kouvaros (panagiotis.kouvaros@gmail.com)
# This file is part of the Venus project.
# Copyright: 2019-2021 by the authors listed in the AUTHORS file in the
# top-level directory.
# License: BSD 2-Clause (see the file LICENSE in the top-level directory).
# Description: Node classes.
# ************


import itertools 
import math
import numpy as np
import tensorflow as tf

from venus.bounds.bounds import Bounds
from venus.common.configuration import Config
from venus.common.utils import ReluState


class Node:
    
    id_iter = itertools.count()

    def __init__(
        self,
        from_node: list,
        to_node: list,
        input_shape: tuple,
        output_shape: tuple,
        config: Config,
        depth=None,
        bounds=Bounds(),
        id=None,
    ):
        """
        Arguments:

            from_node:
                list of input nodes.
            to_node:
                list of output nodes.
            input_shape:
                shape of the input tensor to the node.
            output_shape: 
                shape of the output tensor to the node.
            config:
                configuration.
            depth:
                the depth of the node.
            bounds:
                concrete bounds for the node.
        """
        self.id = next(Node.id_iter) if id is None else id
        self.from_node = from_node
        self.to_node = to_node
        self.input_shape = input_shape
        self.input_size = np.prod(input_shape)
        self.output_shape = output_shape
        self.output_size = np.prod(output_shape)
        self.depth = depth
        self.config = config
        self.out_vars =  np.empty(0)
        self.delta_vars = np.empty(0)
        self.outputs = None
        self.bounds = bounds

    def get_outputs(self):
        """
        Constructs a list of the indices of the nodes in the layer.

        Returns: 

            list of indices.
        """
        if self.outputs is not None:
            outputs = self.outputs
        else:
            if len(self.output_shape) > 1:
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

    def has_relu_activation(self) -> bool:
        """
        Returns:

            Whether the output of the node is fed to a relu node.
        """
        for i in self.to_node:
            if isinstance(i, Relu):
                return True

        return False

    def add_from_node(self, node):
        """
        Adds an input node.

        Arguments:
            
            node:
                The input node.
        """
        if self.has_from_node(node.id) is not True:
            self.from_node.append(node)

    def add_to_node(self, node):
        """
        Adds an output node.

        Arguments:
            
            node:
                The output node.
        """
        if self.has_to_node(node.id) is not True:
            self.to_node.append(node)


    def has_to_node(self, id: int) -> bool:
        """
        Checks whether the node points to another.

        Arguments:
            
            id:
                The id of the node with which connection to check.

        Returns:
            
            Whether the current node points to node with id "id".
        """
        for i in self.to_node:
            if i.id == id:
                return True

        return False

    def has_from_node(self, id: int) -> bool:
        """
        Checks whether the node is pointed to by another.

        Arguments:
            
            id:
                The id of the node with which connection to check.

        Returns:
            
            Whether the node with id "id" points to the current node.
        """
        for i in self.from_node:
            if i.id == id:
                return True

        return False

    def is_head_node(self) -> bool:
        """
        Returns:
            Whether the node is the first in the network.
        """
        return isinstance(self.from_node[0], Input)

    def is_tail_node(self) -> bool:
        """
        Returns:
            Whether the node is the first in the network.
        """
        return len(self.to_node) == 0


class Constant(Node):
    def __init__(self, to_node: list, const: np.array, config: Config, id: int=None):
        """
        Argumnets:
            
            const:
                Constant matrix.
            to_node:
                list of output nodes.
            depth:
                the depth of the node.
        """
        super().__init__(
            [],
            to_node,
            const.shape,
            const.shape,
            config,
            depth=0,
            bounds=Bounds(const, const),
            id=id
        )

    def copy(self):
        """
        Returns: 

            a copy of the calling object 
        """
        return Constant(
            self.bounds.lower.copy(),
            self.to_node,
            self.config,
            id=self.id
        )



class Input(Node):
    def __init__(self, lower:np.array, upper:np.array, config: Config, id: int=None):
        """
        Argumnets:
            
            lower:
                array of lower bounds of input nodes.
            upper:
                array of upper bounds of input nodes.
        """
        super().__init__(
            [],
            [],
            lower.shape,
            lower.shape,
            config,
            depth=0,
            bounds=Bounds(lower, upper),
            id=id
        )

    def copy(self):
        """
        Returns: 

            a copy of the calling object 
        """
        return Input(
            self.bounds.lower.copy(),
            self.bounds.upper.copy(),
            self.config,
            id=self.id
        )

class Gemm(Node):
    def __init__(
        self,
        from_node: list,
        to_node: list,
        input_shape: tuple,
        output_shape: tuple,
        weights: np.array,
        bias: np.array,
        config: Config,
        depth=None,
        bounds=Bounds(),
        id=None
    ):
        """
        Arguments:

            from_node:
                list of input nodes.
            to_node:
                list of output nodes.
            input_shape:
                shape of the input tensor to the node. 
            output_shape: 
                shape of the output tensor to the node.
            weights:
                weight matrix.
            bias:
                bias vector.
            config:
                configuration.
            depth:
                the depth of the node.
            bounds:
                concrete bounds for the node.
        """
        super().__init__(
            from_node,
            to_node,
            input_shape,
            output_shape,
            config,
            depth=depth,
            bounds=bounds,
            id=id
        )
        self.weights = weights
        self.bias = bias

    def copy(self):
        """
        Returns:

            a copy of the calling object 
        """
        return Gemm(
            self.from_node,
            self.to_node,
            self.input_shape,
            self.output_shape,
            self.weights,
            self.bias,
            self.config,
            depth=self.depth,
            bounds=self.bounds.copy(),
            id=self.id
        )

    def get_bias(self, index: int) -> float:
        """
        Returns the bias of the given output.

        Arguments:

            index: 
                the index of the output.

        Returns:
            
            the bias.
        """
        return self.bias[index]

    def edge_weight(self, index1: int, index2: int) -> float:
        """
        Returns the weight of the edge between output with index1 of the
        current node and output with index2 of the previous node.

        Arguments:

            index1:
                index of the output of the current node.
            index2:
                index of the output of the previous node.

        Returns:
         
            the weight.
        """
        return self.weights[index1, index2]
    

    def forward(self, inp: np.array, clip: str=None, add_bias: bool=True) -> np.array:
        """
        Computes the output of the node given an input.

        Arguments:

            inp:
                the input.
            clip:
                clips the weights to positive values if set to '+' and to
                negatives if set to '-'
            add_bias:
                whether to add bias
                
        Returns:
            
            the output of the node.
        """

        if clip is None:
            weights = self.weights

        elif clip == '+':
            weights = np.clip(self.weights, 0, math.inf)

        elif clip == '-':
            weights = np.clip(self.weights, -math.inf, 0)

        else:
            raise ValueError(f'Kernel clip value {clip} not recognised')


        output = weights.dot(inp.flatten())

        if add_bias is True:
            output += self.bias

        return output.reshape(self.output_shape)

    def transpose(self, inp: np.array) -> np.array:
        """
        Computes the input to the node given an output.

        Arguments:

            inp:
                the output.
                
        Returns:
            
            the input of the node.
        """
        return np.dot(inp, self.weights)


class MatMul(Node):
    def __init__(
        self,
        from_node: list,
        to_node: list,
        input_shape: tuple,
        output_shape: tuple,
        weights: np.array,
        config: Config,
        depth=None,
        bounds=Bounds(),
        id=None
    ):
        """
        Arguments:

            from_node:
                list of input nodes.
            to_node:
                list of output nodes.
            input_shape:
                shape of the input tensor to the node. 
            output_shape: 
                shape of the output tensor to the node.
            weights:
                weight matrix.
            config:
                configuration.
            depth:
                the depth of the node.
            bounds:
                concrete bounds for the node.
        """
        super().__init__(
            from_node,
            to_node,
            input_shape,
            output_shape,
            config,
            depth=depth,
            bounds=bounds,
            id=id
        )
        self.weights = weights

    def copy(self):
        """
        Returns:

            a copy of the calling object 
        """
        return MatMul(
            self.from_node,
            self.to_node,
            self.input_shape,
            self.output_shape,
            self.weights,
            self.config,
            depth=self.depth,
            bounds=self.bounds.copy(),
            id=self.id
        )


    def edge_weight(self, index1: int, index2: int) -> float:
        """
        Returns the weight of the edge between output with index1 of the
        current node and output with index2 of the previous node.

        Arguments:

            index1:
                index of the output of the current node.
            index2:
                index of the output of the previous node.

        Returns:
         
            the weight.
        """
        return self.weights[index1, index2]
    

    def forward(self, inp: np.array, clip: str=None) -> np.array:
        """
        Computes the output of the node given an input.

        Arguments:

            inp:
                the input.
            clip:
                clips the weights to positive values if set to '+' and to
                negatives if set to '-'
            add_bias:
                whether to add bias
                
        Returns:
            
            the output of the node.
        """

        if clip is None:
            weights = self.weights

        elif clip == '+':
            weights = np.clip(self.weights, 0, math.inf)

        elif clip == '-':
            weights = np.clip(self.weights, -math.inf, 0)

        else:
            raise ValueError(f'Kernel clip value {clip} not recognised')


        return weights.dot(inp.flatten())


    def transpose(self, inp: np.array) -> np.array:
        """
        Computes the input to the node given an output.

        Arguments:

            inp:
                the output.
                
        Returns:
            
            the input of the node.
        """
        return np.dot(inp, self.weights)



class Conv(Node):
    def __init__(
        self,
        from_node: list, 
        to_node: list,
        input_shape: tuple,
        output_shape: tuple,
        kernels: np.array,
        bias: np.array,
        padding: tuple,
        strides: tuple,
        config: Config,
        depth=None,
        bounds=Bounds(),
        id=None
    ):
        """
        Arguments:

            from_node:
                list of input nodes.
            to_node:
                list of output nodes.
            input_shape:
                shape of the input tensor to the node. 
            output_shape:
                shape of the output tensor to the node.
            kernels:
                the kernels.
            bias:
                bias vector.
            padding:
                the padding.
            strides:
                the strides.
            config:
                configuration.
            depth:
                the depth of the node.
            bounds:
                the concrete bounds of the node.
        """
        super().__init__(
            from_node,
            to_node,
            input_shape,
            output_shape,
            config,
            depth=depth,
            bounds=bounds,
            id=id
        )
        self.kernels = kernels
        self.bias = bias
        self.padding = padding
        self.strides = strides
        self._non_pad_idxs = None
        self._input_padded_size = None
        self._input_padded_shape = None
        self.out_ch,  self.in_ch, self.width, self.height = kernels.shape

    def copy(self):
        """
        Returns:

            a copy of the calling object 
        """
        return Conv(
            self.from_node,
            self.to_node,
            self.input_shape,
            self.output_shape,
            self.kernels,
            self.bias,
            self.padding,
            self.strides,
            self.config,
            depth=self.depth,
            bounds=self.bounds.copy(),
            id=self.id
        )
         

    def get_bias(self, index: tuple):
        """
        Returns the bias of the given output.

        Arguments:

            index: 
                the index of the output.

        Returns:
            
            the bias.
        """
        return self.bias[index[-1]]

    def edge_weight(self, index1: tuple, index2: tuple):
        """
        Returns the weight of the edge between output with index1 of the
        current node and output with index2 of the previous node.

        Arguments:

            index1:
                index of the output of the current node.
            index2:
                index of the output of the previous node.

        Returns:
         
            the weight.
        """
        x_start = index1[0] * self.strides[0] - self.padding[0]
        x = index2[0] - x_start
        y_start = index1[1] * self.strides[1] - self.padding[1]
        y = index2[1] - y_start

        return self.kernels[index1[2]][index2[2]][y][x]


    def forward(self, inp: np.array, clip=None, add_bias=True) -> np.array:
        """
        Computes the output of the node given an input.

        Arguments:

            inp:
                the input.
            clip:
                clips the weights to positive values if set to '+' and to
                negatives if set to '-'
            add_bias:
                whether to add bias
                
        Returns:
            
            the output of the node.
        """
        if clip is None:
            pass

        elif clip == '+':
            kernels = np.clip(self.kernels, 0, math.inf)

        elif clip == '-':
            kernels = np.clip(self.kernels, -math.inf, 0)

        else:
            raise ValueError(f'Kernel clip value {kernel_clip} not recognised')

        padded_input = Conv.pad(
            inp.reshape(self.input_shape),
            self.padding
        )[np.newaxis, ...]

        output = tf.nn.convolution(
            padded_input,
            kernels.transpose(2, 3, 1, 0),
            strides=self.strides,
            padding='VALID',
        ).numpy().flatten()


        if add_bias is True:
            output += np.array(
                [self.bias for i in range(int(self.output_size / self.out_ch))],
                dtype=self.config.PRECISION
            ).flatten()

        return output


    def transpose(self, inp: np.array) -> np.array:
        """
        Computes the input to the node given an output.

        Arguments:

            inp:
                the output.
                
        Returns:
            
            the input of the node.
        """
        return tf.nn.conv2d_transpose(
            inp.reshape((inp.shape[0],) + self.output_shape),
            self.kernels.transpose(2, 3, 1, 0),
            (inp.shape[0], ) + self.input_shape,
            self.strides,
            padding = [
                [0, 0],
                [self.padding[0], self.padding[0]],
                [self.padding[1], self.padding[1]],
                [0, 0]
            ]
        ).numpy().reshape(inp.shape[0], -1)


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
    def compute_output_shape(in_shape: tuple, weights_shape: tuple, padding: tuple, strides: tuple) -> tuple:
        """
        Computes the output shape of a convolutional layer.

        Arguments:

            in_shape:
                shape of the input tensor to the layer.
            weights_shape:
                shape of the kernels of the layer.
            padding:
                pair of int for the width and height of the padding.
            strides:
                pair of int for the width and height of the strides.

        Returns:

            tuple of the output shape
        """
        x, y, z = in_shape
        K, _, Y, X = weights_shape
        # X, Y, _, K = weights_shape
        p, q  = padding
        s, r = strides
        out_x = int(math.floor((x - X + 2 * p) / s + 1))
        out_y = int(math.floor((y - Y + 2 * q) / r + 1))
        
        return (out_x, out_y,K)

    @staticmethod
    def pad(inp: np.array, padding: tuple, values: tuple=(0,0)) -> np.array:
        """
        Pads a given matrix with constants.

        Arguments:
            
            inp:
                matrix.
            padding:
                the padding.
            values:
                the constants.

        Returns

            padded inp.
        """
        if padding == (0, 0):
            return inp
        
        return np.pad(
            inp,
            (padding, padding, (0, 0)),
            'constant',
            constant_values=values
        )

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


class Relu(Node):
    def __init__(
        self,
        from_node: list,
        to_node: list,
        input_shape: tuple,
        config: Config,
        depth=None,
        bounds=Bounds(),
        id=None
    ):
        """
        Arguments:

            from_node:
                list of input nodes.
            to_node:
                list of output nodes.
            input_shape:
                shape of the input tensor to the node. 
            config:
                configuration.
            depth:
                the depth of the node.
            bounds:
                concrete bounds for the node.
        """
        super().__init__(
            from_node,
            to_node,
            input_shape,
            input_shape,
            config,
            depth=depth,
            bounds=bounds,
            id=id
        )
        self.state = np.array(
            [ReluState.UNSTABLE] * self.output_size,
            dtype=ReluState
        ).reshape(self.output_shape)
        self.dep_root = np.array(
            [False] * self.output_size,
            dtype=bool
        ).reshape(self.output_shape)
        self.active_flag = None
        self.inactive_flag = None
        self.stable_flag = None
        self.unstable_flag = None
        self.propagation_flag = None
        self.stable_count = None
        self.unstable_count = None
        self.active_count = None
        self.inactive_count = None
        self.propagation_count = None

    def copy(self):
        """
        Returns:

            a copy of the calling object 
        """
        relu = Relu(
            self.from_node,
            self.to_node,
            self.input_shape,
            self.config,
            depth=self.depth,
            bounds=self.bounds.copy(),
            id=self.id
        )
        relu.state = self.state.copy()
        relu.dep_root = self.dep_root.copy()

        return relu

    def forward(self, inp: np.array) -> np.array:
        """
        Computes the output of the node given an input.

        Arguments:

            inp:
                the input.
                
        Returns:
            
            the output of the node.
        """

        return np.clip(inp, 0, math.inf)


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

    def is_active(self, index):
        """
        Detemines whether a given ReLU node is stricly active.

        Arguments:

            index: 
                the index of the node.

        Returns:

            bool expressing the active state of the given node.
        """
        cond1 = self.from_node[0].bounds.lower[index] >= 0
        cond2 = self.state[index] == ReluState.ACTIVE

        return cond1 or cond2

    def is_inactive(self, index: tuple):
        """
        Determines whether a given ReLU node is strictly inactive.

        Arguments:

            index:
                the index of the node.

        Returns:

            bool expressing the inactive state of the given node.
        """
        cond1 = self.from_node[0].bounds.upper[index] <= 0
        cond2 = self.state[index] == ReluState.INACTIVE

        return cond1 or cond2

    def is_stable(self, index: tuple, delta_val: float=None):
        """
        Determines whether a given ReLU node is stable.

        Arguments:

            index:
                the index of the node.

            delta_val:
                the value of the binary variable associated with the node. if
                set, the value is also used in conjunction with the node's
                bounds to determined its stability.

        Returns:

            bool expressing the stability of the given node.
        """
        cond0a = self.from_node[0].bounds.lower[index] >= 0
        cond0b = self.from_node[0].bounds.upper[index] <= 0
        cond1 = cond0a or cond0b
        cond2 = self.state[index] != ReluState.UNSTABLE
        cond3 = False if delta_val is None else delta_val in [0, 1]

        return cond1 or cond2 or cond3

    def get_active_flag(self) -> np.array:
        """
        Returns an array of activity statuses for each ReLU node.
        """
        if self.active_flag is None:
            self.active_flag = self.from_node[0].bounds.lower.flatten() > 0

        return self.active_flag

    def get_active_count(self) -> int:
        """
        Returns the total number of active Relu nodes.
        """
        if self.active_count is None:
            self.active_count = np.sum(self.get_active_flag())

        return self.active_count

    def get_inactive_flag(self) -> np.array:
        """
        Returns an array of inactivity statuses for each ReLU node.
        """
        if self.inactive_flag is None:
            self.inactive_flag = self.from_node[0].bounds.upper.flatten() <= 0

        return self.inactive_flag

    def get_inactive_count(self) -> int:
        """
        Returns the total number of inactive ReLU nodes.
        """
        if self.inactive_count is None:
            self.inactive_count = np.sum(self.get_inactive_flag())

        return self.inactive_count

    def get_unstable_flag(self) -> np.array:
        """
        Returns an array of instability statuses for each ReLU node.
        """
        if self.unstable_flag is None:
            self.unstable_flag = np.logical_and(
                self.from_node[0].bounds.lower < 0,
                self.from_node[0].bounds.upper > 0
            ).flatten()

        return self.unstable_flag

    def get_unstable_count(self) -> int:
        """
        Returns the total number of unstable nodes.
        """
        if self.unstable_count is None:
            self.unstable_count = np.sum(self.get_unstable_flag())

        return self.unstable_count

    def get_stable_flag(self) -> np.array:
        """
        Returns an array of instability statuses for each ReLU node.
        """
        if self.stable_flag is None:
            self.stable_flag = np.logical_or(
                self.get_active_flag(),
                self.get_inactive_flag()
            ).flatten()

        return self.stable_flag

    def get_stable_count(self) -> int:
        """
        Returns the total number of unstable nodes.
        """
        if self.stable_count is None:
            self.stable_count = np.sum(self.get_stable_flag())

        return self.stable_count

    def get_propagation_flag(self) -> np.array:
        """
        Returns an array of sip propagation statuses for each node.
        """
        if self.propagation_flag is None:
            self.propagation_flag = np.logical_or(
                self.get_active_flag(),
                self.get_unstable_flag()
            ).flatten()

        return self.propagation_flag

    def get_propagation_count(self) -> int:
        """
        Returns the total number of sip propagation nodes.
        """
        if self.propagation_count is None:
            self.propagation_count = np.sum(self.get_propagation_flag())

        return self.propagation_count
 
    def get_upper_relaxation_slope(self) -> np.array:
        """
        Returns:

        The upper relaxation slope for each of the ReLU nodes.
        """
        slope = np.zeros(self.output_size, dtype=self.config.PRECISION)
        upper = self.from_node[0].bounds.upper.flatten()[self.get_unstable_flag()]
        lower = self.from_node[0].bounds.lower.flatten()[self.get_unstable_flag()]
        slope[self.get_unstable_flag()] = upper /  (upper - lower)
        slope[self.get_active_flag()] = 1.0
        
        return slope

    def get_lower_relaxation_slope(self):
        """
        Returns:

        The upper relaxation slope for each of the ReLU nodes.
        """
        slope = np.ones(self.output_size, dtype=self.config.PRECISION)
        upper = self.from_node[0].bounds.upper.flatten()
        lower = self.from_node[0].bounds.lower.flatten()
        idxs = abs(lower) >=  upper
        slope[idxs] = 0.0
        slope[self.get_inactive_flag()] = 0.0
        slope[self.get_active_flag()] = 1.0

        return slope


class Reshape(Node):
    def __init__(
        self,
        from_node: list, 
        to_node: list,
        input_shape: tuple,
        output_shape: tuple,
        config: Config,
        depth=None,
        bounds=Bounds(),
        id=None
    ):
        """
        Arguments:

            from_node:
                list of input nodes.
            to_node:
                list of output nodes.
            input_shape:
                shape of the input tensor to the node. 
            output_shape: 
                shape of the output tensor to the node.
            config:
                configuration.
            depth:
                the depth of the node.
            bounds:
                concrete bounds for the node.
        """
        super().__init__(
            from_node,
            to_node,
            input_shape,
            output_shape,
            config,
            depth=self.depth,
            bounds=self.bounds,
            id=id
        )

    def copy(self):
        """
        Returns: 

            a copy of the calling object.
        """
        return Reshape(
            self.from_node,
            self.to_node,
            self.input_shape,
            self.output_shape,
            self.config,
            depth=self.depth,
            bounds=self.bounds.copy(),
            id=self.id
        )


class Flatten(Node):
    def __init__(
        self,
        from_node: list, 
        to_node: list,
        input_shape: tuple,
        config: Config,
        depth=None,
        bounds=Bounds(),
        id=None
    ):
        """
        Arguments:

            from_node:
                list of input nodes.
            to_node:
                list of output nodes.
            input_shape:
                shape of the input tensor to the node. 
            config:
                configuration.
            depth:
                the depth of the node.
            bounds:
                concrete bounds for the node.
        """
        super().__init__(
            from_node,
            to_node,
            input_shape,
            (np.prod(input_shape), ),
            config,
            depth=depth,
            bounds=bounds,
            id=id
        )

    def copy(self):
        """
        Returns:

            a copy of the calling object.
        """
        return Flatten(
            self.from_node,
            self.to_node,
            self.input_shape,
            self.config,
            depth=self.depth,
            bounds=self.bounds.copy(),
            id=self.id
        )

class Sub(Node):
    def __init__(
        self,
        from_node: list,
        to_node: list,
        input_shape: tuple,
        config: Config,
        const=None,
        depth=None,
        bounds=Bounds(),
        id=None
    ):
        """
        Arguments:

            from_node:
                list of input nodes.
            to_node:
                list of output nodes.
            input_shape:
                shape of the input tensor to the node.
            config:
                configuration.
            const:
                Constant second term is set.
            depth:
                the depth of the node.
            bounds:
                concrete bounds for the node.
        """
        super().__init__(
            from_node,
            to_node,
            input_shape,
            input_shape,
            config,
            depth=depth,
            bounds=bounds,
            id=id
        )
        self.const = const

    def copy(self):
        """
        Returns:

            a copy of the calling object.
        """
        return Sub(
            self.from_node,
            self.to_node,
            self.input_shape,
            self.config,
            const=None if self.const is None else self.const.copy(),
            depth=self.depth,
            bounds=self.bounds,
            id=self.id
        )


    def forward(self, inp1: np.array, inp2: np.array=None) -> np.array:
        """
        Computes the output of the node given an input.

        Arguments:

            inp1:
                the first input.
            inp2:
                the second input. If not set then the const of the node is
                taken as second input.
                
        Returns:
            
            the output of the node.
        """
        assert inp2 is not None or self.const is not None, "Second input is not specified"
        inp2 = self.const if inp2 is None else inp2

        return inp1 - inp2

    def transpose(self, inp: np.array) -> np.array:
        """
        Computes the input to the node given an output.

        Arguments:

            inp:
                the output.
                
        Returns:
            
            the input of the node.
        """
        if self.const is not None:
            return inp - self.const.flatten()

        sub = np.zeros(
            self.output_size, 2 * self.output_size,
            dtype=self.config.PRECISION
        )

        return np.hstack([inp, - inp])

class Add(Node):
    def __init__(
        self,
        from_node: list,
        to_node: list,
        input_shape: tuple,
        config: Config,
        const=None,
        depth=None,
        bounds=Bounds(),
        id=None
    ):
        """
        Arguments:

            from_node:
                list of input nodes.
            to_node:
                list of output nodes.
            input_shape:
                shape of the input tensor to the node.
            config:
                configuration.
            const:
                Constant second term is set.
            depth:
                the depth of the node.
            bounds:
                concrete bounds for the node.
        """
        super().__init__(
            from_node,
            to_node,
            input_shape,
            input_shape,
            config,
            depth=depth,
            bounds=bounds,
            id=id
        )
        self.const = const

    def copy(self):
        """
        Returns:

            a copy of the calling object.
        """
        return Add(
            self.from_node,
            self.to_node,
            self.input_shape,
            self.config,
            const = None if self.const is None else self.const.copy(),
            depth=self.depth,
            bounds=self.bounds.copy(),
            id=self.id
        )

    def forward(self, inp1: np.array, inp2: np.array=None) -> np.array:
        """
        Computes the output of the node given an input.

        Arguments:

            inp1:
                the first input.
            inp2:
                the second input. If not set then the const of the node is
                taken as second input.
                
        Returns:
            
            the output of the node.
        """
        assert inp2 is not None or self.const is not None, "Second input is not specified"
        inp2 = self.const if inp2 is None else inp2

        return inp1 + inp2

    def transpose(self, inp: np.array) -> np.array:
        """
        Computes the input to the node given an output.

        Arguments:

            inp:
                the output.
                
        Returns:
            
            the input of the node.
        """
        if self.const is not None:
            return inp + self.const.flatten()

        return np.hstack([inp, inp])


class BatchNormalization(Node):
    def __init__(
        self,
        from_node: list,
        to_node: list,
        input_shape: tuple,
        scale: np.array,
        bias: np.array,
        input_mean: np.array,
        input_var: np.array,
        epsilon: float,
        config: Config,
        depth=None,
        bounds=Bounds(),
        id=None
    ):
        """
        Arguments:

            from_node:
                list of input nodes.
            to_node:
                list of output nodes.
            input_shape:
                shape of the input tensor to the node.
            scale:
                the scale tensor.
            bias:
                the bias tensor.
            input_mean:
                the mean tensor.
            input_var:
                the variance tensor.
            epsilon:
                the epsilon value.
            config:
                configuration.
            depth:
                the depth of the node.
            bounds:
                concrete bounds for the node.
        """
        super().__init__(
            from_node,
            to_node,
            input_shape,
            input_shape,
            config,
            depth=depth,
            bounds=bounds,
            id=id
        )
        self.scale = scale
        self.bias = bias
        self.input_mean = input_mean
        self.input_var = input_var
        self.epsilon = epsilon

    def copy(self):
        """
        Returns:

            a copy of the calling object.
        """
        return BatchNormalization(
            self.from_node,
            self.to_node,
            self.input_shape,
            self.scale,
            self.bias,
            self.input_mean,
            self.input_var,
            self.epsilon,
            self.config,
            depth=self.depth,
            bounds=self.bounds.copy(),
            id=self.id
        )

    def forward(self, inp: np.array) -> np.array:
        """
        Computes the output of the node given an input.

        Arguments:

            inp:
                the input.
                
        Returns:
            
            the output of the node.
        """

        return (inp - self.input_mean) / math.sqrt(self.input_var + self.epsilon) * self.scale + self.bias

    def transpose(self, inp: np.array) -> np.array:
        """
        Computes the input to the node given an output.

        Arguments:

            inp:
                the output.
                
        Returns:
            
            the input of the node.
        """
        return

