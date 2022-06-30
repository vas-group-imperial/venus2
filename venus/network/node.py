# *********** File: node.py
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
import torch

from venus.bounds.bounds import Bounds
from venus.common.configuration import Config
from venus.common.utils import ReluState, ReluApproximation
from venus.split.split_strategy import SplitStrategy

torch.set_num_threads(1)

class Node:
    
    id_iter = itertools.count()

    def __init__(
        self,
        from_node: list,
        to_node: list,
        input_shape: tuple,
        output_shape: tuple,
        config: Config,
        depth=0,
        bounds=Bounds(),
        id=None,
        device=torch.device('cpu')
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
            device:
                device for torch operations.
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
        self.output = None
        self.bounds = bounds
        self.device = device
        self._milp_var_indices = None

    def get_outputs(self):
        """
        Constructs a list of the indices of the units in the node.

        Returns: 
            list of indices.
        """ 
        if len(self.output_shape) > 1:
            return  [
                i for i in itertools.product(*[range(j) for j in self.output_shape])
            ]
            
        return list(range(self.output_size))


    def clean_vars(self):
        """
        Nulls out all MILP variables associate with the network.
        """
        self.out_vars = torch.empty(0)
        self.delta_vars = torch.empty(0)

    def get_milp_var_indices(self, var_type: str='out'):
        """
        Returns the starting and ending indices of the milp variables encoding
        the node.

        Arguments:
            var_type: either 'out' for output variables or 'delta' for binary
            variables.
        """
        if self._milp_var_indices is not None:
            return self._milp_var_indices

        if len(self.from_node) > 0:
            start = self.from_node[-1].get_milp_var_indices()[0]
        else:
            start = 0
        end = start + self.output_size

        self._milp_var_indices = (start, end)

        return self._milp_var_indices
 

        
    def has_non_linear_op(self) -> bool:
        """
        Determines whether the output of the node is fed to a non-linear operation.
        """
        return self.has_relu_activation() or self.has_max_pool()


    def has_relu_activation(self) -> bool:
        """
        Determines whether the output of the node is fed to a relu node.
        """
        for i in self.to_node:
            if isinstance(i, Relu):
                return True

        return False

    def has_max_pool(self) -> bool:
        """
        Determines whether the output of the node is fed to a maxpool node.
        """
        for i in self.to_node:
            if isinstance(i, MaxPool):
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

    def remove_from_node(self, node):
        """
        Removes an input node.

        Arguments:
            node:
                The input node.
        """
        if self.has_from_node(node.id) is True:
            self.from_node.remove(node)

    def remove_to_node(self, node):
        """
        Removes an output node.

        Arguments:
            node:
                The input node.
        """
        if self.has_to_node(node.id) is True:
            self.to_node.remove(node)

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

    def update_bounds(self, bounds: Bounds, flag: torch.Tensor=None) -> None:
        """
        Updates the bounds.

        Arguments:
            bounds:
                the new bounds.
            flag:
                the units of the new bounds; if None then the new bounds should
                be for all units.
        """
        if flag is None:
            self.bounds = bounds.cpu()
        else:
            self.bounds.lower[flag] = bounds.lower.cpu()
            self.bounds.upper[flag] = bounds.upper.cpu()

        if self.has_relu_activation():
            self.to_node[0].reset_state_flags()

    def clear_bounds(self) -> None:
        """
        Nulls out the bounds of the node.
        """
        if isinstance(self, Input) or self.has_relu_activation():
            return

        else:
            self.bounds = Bounds()

    def has_batch_dimension(self) -> int:
        """
        Determines whether the node has batch dimension.
        """
        if len(self.input_shape) in [1, 3]:
            return False

        return True


    def input_shape_no_batch(self) -> int:
        """
        Returns the input shape without the batch dimension. 
        """
        if self.has_batch_dimension():
            return self.input_shape[1:]

        return self.input_shape

    def output_shape_no_batch(self) -> int:
        """
        Returns the output shape without the batch dimension. 
        """
        if self.has_batch_dimension():
            return self.output_shape[1:]

        return self.output_shape

    def in_ch_sz(self) -> int:
        """
        Computes the size of an input channel.
        """
        if self.has_batch_dimension() is True:
            return np.prod(self.input_shape[2:])

        return np.prod(self.input_shape[1:])

    
    def in_ch(self) -> int:
        """
        Computes the number of input channels.
        """
        if self.has_batch_dimension() is True:
            return self.input_shape[1]

        return self.input_shape[0]

    def get_propagation_flag(self) -> torch.Tensor:
        if self.has_relu_activation():
            return self.to_node[0].get_propagation_flag()

        return torch.ones(
            self.output_shape, dtype=torch.bool, device=self.device
        )
 
    def get_propagation_count(self) -> torch.Tensor:
        if self.has_relu_activation():
            return self.to_node[0].get_propagation_count()

        return self.output_size    

class Constant(Node):
    def __init__(self, to_node: list, const: torch.Tensor, config: Config, id: int=None):
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
        self.const = const

    def copy(self):
        """
        Copies the node.
        """
        return Constant(
            self.bounds.lower.detach().clone(),
            self.to_node,
            self.config,
            id=self.id
        )

    def get_milp_var_indices(self, var_type: str):
        """
        Returns the starting and ending indices of the milp variables encoding
        the node.

        Arguments:
            var_type: either 'out' for output variables or 'delta' for binary
            variables.
        """
        return None

class Input(Node):
    def __init__(self, bounds:torch.Tensor, config: Config, id: int=None):
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
            bounds.lower.shape,
            bounds.lower.shape,
            config,
            depth=0,
            bounds=bounds,
            id=id
        )

    def copy(self):
        """
        Copies the node. 
        """
        return Input(
            self.bounds.copy(), self.config, id=self.id
        )

    def get_milp_var_size(self):
        """
        Returns the number of milp variables required for the milp encoding of
        the node.
        """
        return self.output_size

class Gemm(Node):
    def __init__(
        self,
        from_node: list,
        to_node: list,
        input_shape: tuple,
        output_shape: tuple,
        weights: torch.Tensor,
        bias: torch.Tensor,
        config: Config,
        depth=0,
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
        Copies the node.
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
        return self.bias[index].item()

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
        return self.weights[index1, index2].item()
    

    def forward(
        self,
        inp: torch.Tensor=None,
        clip: str=None,
        add_bias: bool=True,
        save_output=False
    ) -> torch.Tensor:
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
            save_output:
                Whether to save the output in the node. 
        Returns: 
            the output of the node.
        """
        assert inp is not None or self.from_node[0].output is not None
        inp = self.from_node[0].output if inp is None else inp

        if isinstance(inp, torch.Tensor):
            return self._forward_torch(inp, clip, add_bias, save_output)

        elif isinstance(inp, np.ndarray):
            return self._forward_numpy(inp, save_output)

        else:
            raise TypeError("Forward supports only numpy arrays and torch.Tensors.")

    def _forward_torch(
        self,
        inp: torch.Tensor=None,
        clip: str=None,
        add_bias: bool=True,
        save_output=False
    ) -> torch.Tensor:
        """
        Torch implementation of forward.

        Arguments:
            inp:
                the input.
            clip:
                clips the weights to positive values if set to '+' and to
                negatives if set to '-'
            add_bias:
                whether to add bias
            save_output:
                Whether to save the output in the node. 
        Returns: 
            the output of the node.
        """
        if clip is None:
            weights = self.weights

        elif clip == '+':
            weights = torch.clamp(self.weights, 0, math.inf)

        elif clip == '-':
            weights = torch.clamp(self.weights, -math.inf, 0)

        else:
            raise ValueError(f'Kernel clip value {clip} not recognised')
     
        output = weights @ inp

        if add_bias is True:
            output += self.bias

        if save_output:
            self.output = output

        return output

    def _forward_numpy(self, inp: np.array=None, save_output=False) -> np.array:
        """
        Numpy implementation of forward.

        Arguments:
            inp:
                the input.
            save_output:
                Whether to save the output in the node. 
        Returns: 
            the output of the node.
        """
        output = self.weights.cpu().numpy() @ inp + self.bias.cpu().numpy()

        if save_output is True:
            self.output = output

        return output

    def transpose(
        self,
        inp: torch.Tensor,
        in_flag: torch.Tensor=True,
        out_flag: torch.Tensor=True
    ) -> torch.Tensor:
        """
        Computes the input to the node given an output.

        Arguments:
            inp:
                the output.
            out_flag:
                boolean flag of the units in output.
            in_flag:
                boolean flag of the units in input.
        Returns:
            the input of the node.
        """
        in_flag = range(self.weights.shape[1]) if in_flag is None else in_flag
        out_flag = range(self.weights.shape[0]) if out_flag is None else out_flag

        return inp @ self.weights[out_flag, :][:, in_flag]


class MatMul(Node):
    def __init__(
        self,
        from_node: list,
        to_node: list,
        input_shape: tuple,
        output_shape: tuple,
        weights: torch.Tensor,
        config: Config,
        depth=0,
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
        Copies the node.
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

    def numpy(self):
        """
        Copies the node with with numpy data.
        """
        return MatMul(
            self.from_node,
            self.to_node,
            self.input_shape,
            self.output_shape,
            self.weights.cpu().numpy(),
            self.config,
            depth=self.depth,
            bounds=self.bounds,
            id=self.id
        )

    def get_milp_var_size(self):
        """
        Returns the number of milp variables required for the milp encoding of
        the node.
        """
        return self.output_size

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
    
    def forward(
        self, inp: torch.Tensor=None, clip: str=None, save_output=False
    ) -> torch.Tensor:
        """
        Computes the output of the node given an input.

        Arguments:
            inp:
                the input.
            clip:
                clips the weights to positive values if set to '+' and to
                negatives if set to '-'
            save_output:
                Whether to save the output in the node. 
        Returns: 
            the output of the node.
        """
        assert inp is not None or self.from_node[0].output is not None
        inp = self.from_node[0].output if inp is None else inp

        if isinstance(inp, torch.Tensor):
            return self._forward_torch(inp, clip, save_output)

        elif isinstance(inp, np.ndarray):
            return self._forward_numpy(inp, save_output)

        else:
            raise TypeError("Forward supports only numpy arrays and torch.Tensors.")


    def _forward_torch(
        self, inp: torch.Tensor=None, clip: str=None, save_output=False
    ) -> torch.Tensor:
        """
        Torch implementation of forward.

        Arguments:
            inp:
                the input.
            clip:
                clips the weights to positive values if set to '+' and to
                negatives if set to '-'
            save_output:
                Whether to save the output in the node. 
        Returns: 
            the output of the node.
        """
        if clip is None:
            weights = self.weights

        elif clip == '+':
            weights = torch.clamp(self.weights, 0, math.inf)

        elif clip == '-':
            weights = torch.clamp(self.weights, -math.inf, 0)

        else:
            raise ValueError(f'Kernel clip value {clip} not recognised')


        output = weights @ inp

        if save_output:
            self.output = output

        return output

    def _forward_numpy(self, inp: np.array=None, save_output=False) -> np.array:
        """
        Numpy implementation of forward.

        Arguments:
            inp:
                the input.
            save_output:
                Whether to save the output in the node. 
        Returns: 
            the output of the node.
        """
        output = self.weights.cpu().numpy() @ inp 

        if save_output is True:
            self.output = output

        return output

    def transpose(
        self,
        inp: torch.Tensor,
        in_flag: torch.Tensor=True,
        out_flag: torch.Tensor=True
    ) -> torch.Tensor:
        """
        Computes the input to the node given an output.

        Arguments:
            inp:
                the output.
            out_flag:
                boolean flag of the units in output.
            in_flag:
                boolean flag of the units in input.
        Returns:
            the input of the node.
        """
        in_flag = range(self.weights.shape[1]) if in_flag is None else in_flag
        out_flag = range(self.weights.shape[0]) if out_flag is None else out_flag

        return inp @ self.weights[out_flag, :][:, in_flag]


class ConvBase(Node):
    def __init__(
        self,
        from_node: list, 
        to_node: list,
        input_shape: tuple,
        kernels: torch.Tensor,
        bias: torch.Tensor,
        pads: tuple,
        strides: tuple,
        config: Config,
        depth=0,
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
            kernels:
                the kernels.
            bias:
                bias vector.
            pads:
                the pads.
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
            None,
            config,
            depth=depth,
            bounds=bounds,
            id=id
        )
        self.kernels = kernels
        self.bias = bias
        self.pads = pads
        self.strides = strides
        self.in_height, self.in_width = input_shape[-2:]

    def get_milp_var_size(self):
        """
        Returns the number of milp variables required for the milp encoding of
        the node.
        """
        return self.output_size


    @staticmethod
    def get_padded_size(input_shape: tuple, pads: tuple) -> int:
        """
        Computes the size of the padded input.
        """
        in_ch, in_height, in_width = input_shape
        pad_height, pad_width = pads
        
        return (in_height + 2 * pad_height) * (in_width + 2 * pad_width) * in_ch

    @staticmethod
    def get_padded_shape(input_shape: tuple, pads: tuple) -> int:
        """
        Computes the shape of padded input.
        """  
        in_ch, in_height, in_width = input_shape
        pad_height, pad_width = pads

        return (
            in_ch,
            in_height + 2 * pad_height,
            in_width + 2 * pad_width
        )

    @staticmethod
    def get_non_pad_idxs(
        input_shape: tuple, pads: tuple, device: str='cpu'
    ) -> torch.Tensor:
        """
        Computes the indices of the original input whithin the padded one.
        """
        in_ch, in_height, in_width = input_shape
        pad_height, pad_width = pads
        size = ConvBase.get_padded_size(input_shape, pads)
        non_pad_idxs = torch.arange(
            size, dtype=torch.long, device=device
        ).reshape(
            in_ch,
            in_height + 2 * pad_height,
            in_width + 2 * pad_width
        )[
            :, pad_height:in_height + pad_height, pad_width :in_width + pad_width
        ].flatten()

        return non_pad_idxs


    @staticmethod
    def get_non_pad_idx_flag(
        input_shape: tuple, pads: tuple, device: str='cpu'
    ) -> torch.Tensor:
        """
        Computes the index flag of the original input whithin the padded one.
        """
        non_pad_idxs = ConvBase.get_non_pad_idxs(input_shape, pads, device=device)
        flag = torch.zeros(
            Conv.get_padded_size(input_shape, pads),
            device=device,
            dtype=torch.bool
        )
        flag[non_pad_idxs] = True

        return flag

    @staticmethod
    def pad(inp: torch.Tensor, pads: tuple, values: tuple=(0,0)) -> torch.Tensor:
        """
        Pads a given matrix with constants.

        Arguments:
            inp:
                matrix.
            pads:
                the pads.
            values:
                the constants.
        Returns
            padded inp.
        """
        assert(len(inp.shape)) in [3, 4]

        if pads == (0, 0):
            return inp
       
        if isinstance(inp, np.ndarray):
            if len(inp.shape) == 3:
                pad_par = ((0, 0), pads, pads)
            else:
                pad_par = ((0, 0), (0, 0), pads, pads)

            return np.pad(inp, pad_par, 'constant', constant_values=values)

        elif isinstance(inp, torch.Tensor):
            pad_par = (pads[1], pads[1], pads[0], pads[0])

            return torch.nn.functional.pad(inp, pad_par, 'constant', 0)

        else:
            raise TypeError(f"Unsupported type {type(inp)} of input")


    @staticmethod
    def im2col(
        matrix: torch.Tensor, kernel_shape: tuple, strides: tuple, device: str='cpu'
    ) -> torch.Tensor:
        """
        im2col function.

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
        assert len(matrix.shape) in [3, 4], f"{len(matrix.shape)}-D is not supported."
        assert type(matrix) in [torch.Tensor, np.ndarray], f"{type(matrix)} matrices are not supported."

        assert type(matrix in [np.ndarray, torch.Tensor])

        filters, height, width = matrix.shape[-3:]
        row_extent = height - kernel_shape[0] + 1
        col_extent = width - kernel_shape[1] + 1

        # starting block indices
        if isinstance(matrix, torch.Tensor):
            start_idx = torch.arange(
                kernel_shape[0], device=device
            )[:, None] * width + torch.arange(kernel_shape[1], device=device)
            offset_filter = torch.arange(
                0, filters * height * width, height * width, device=device
            ).reshape(-1, 1)
        else:
            start_idx = np.arange(
                kernel_shape[0]
            )[:, None] * width + np.arange(kernel_shape[1])
            offset_filter = np.arange(
                0, filters * height * width, height * width
            ).reshape(-1, 1)

        start_idx = start_idx.flatten()[None, :]
        start_idx = start_idx + offset_filter

        # offsetted indices across the height and width of A
        if isinstance(matrix, torch.Tensor):
            offset_idx = torch.arange(
                row_extent, device=device
            )[:, None][::strides[0]] * width + torch.arange(
                0, col_extent, strides[1], device=device
            )
        else: 
            offset_idx = np.arange(
                row_extent
            )[:, None][::strides[0]] * width + np.arange(
                0, col_extent, strides[1]
            )

        index = start_idx.ravel()[:, None] + offset_idx.ravel()

        if len(matrix.shape) == 4:
            if isinstance(matrix, np.ndarray):
                return np.take(
                    matrix.reshape(matrix.shape[0], -1), index, axis=1
                ).transpose(1, 2, 0)

            else:
                return torch.index_select(
                    matrix.reshape(matrix.shape[0], -1), 1, index.flatten()
                ).reshape((matrix.shape[0],) + index.shape).permute(1, 2, 0)

        else:  
            if isinstance(matrix, np.ndarray):
                return np.take(matrix, index)
            else:
                return torch.take(matrix, index)


class Conv(ConvBase):
    def __init__(
        self,
        from_node: list, 
        to_node: list,
        input_shape: tuple,
        kernels: torch.Tensor,
        bias: torch.Tensor,
        pads: tuple,
        strides: tuple,
        config: Config,
        depth=0,
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
            kernels:
                the kernels.
            bias:
                bias vector.
            pads:
                the pads.
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
            kernels,
            bias,
            pads,
            strides,
            config,
            depth=depth,
            bounds=bounds,
            id=id
        )
        self.out_ch,  self.in_ch, self.krn_height, self.krn_width = kernels.shape
        self.output_shape = self._compute_output_shape()
        self.output_size = np.prod(self.output_shape)
        self.out_height, self.out_width = self.output_shape[-2:]
        self.out_ch_sz = int(self.output_size / self.out_ch)

    def copy(self):
        """
        Copies the node.
        """
        return Conv(
            self.from_node,
            self.to_node,
            self.input_shape,
            self.kernels,
            self.bias,
            self.pads,
            self.strides,
            self.config,
            depth=self.depth,
            bounds=self.bounds.copy(),
            id=self.id
        )

    def numpy(self):
        """
        Copies the node with numpy data.
        """
        return Conv(
            self.from_node,
            self.to_node,
            self.input_shape,
            self.output_shape,
            self.kernels.cpu().numpy(),
            self.bias.cpu().numpy(),
            self.pads,
            self.strides,
            self.config,
            depth=self.depth,
            bounds=self.bounds,
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
        height_start = index1[0] * self.strides[0] - self.pads[0]
        height = index2[0] - height_start
        width_start = index1[1] * self.strides[1] - self.pads[1]
        width = index2[1] - width_start

        return self.kernels[index1[0]][index2[0]][height][width]


    def forward(
        self,
        inp: torch.Tensor=None,
        clip=None,
        add_bias=True,
        save_output=False
    ) -> torch.Tensor:
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
            save_output:
                Whether to save the output in the node. 
        Returns: 
            the output of the node.
        """
        assert inp is not None or self.from_node[0].output is not None
        inp = self.from_node[0].output if inp is None else inp

        if isinstance(inp, torch.Tensor):
            return self._forward_torch(inp, clip, add_bias, save_output)

        elif isinstance(inp, np.ndarray):
            return self._forward_numpy(inp, save_output)

        else:
            raise TypeError("Forward supports only numpy arrays and torch.Tensors.")


    def _forward_torch(
        self,
        inp: torch.Tensor=None,
        clip=None,
        add_bias=True,
        save_output=False
    ) -> torch.Tensor:
        """
        Torch implementation of forward.

        Arguments:
            inp:
                the input.
            clip:
                clips the weights to positive values if set to '+' and to
                negatives if set to '-'
            add_bias:
                whether to add bias
            save_output:
                Whether to save the output in the node. 
        Returns: 
            the output of the node.
        """
        if clip is None:
            kernels = self.kernels

        elif clip == '+':
            kernels = torch.clamp(self.kernels, 0, math.inf)

        elif clip == '-':
            kernels = torch.clamp(self.kernels, -math.inf, 0)

        else:
            raise ValueError(f'Kernel clip value {kernel_clip} not recognised')

        output = torch.nn.functional.conv2d(
            inp,
            kernels,
            bias = self.bias if add_bias is True else None,
            stride=self.strides,
            padding=self.pads
        )
            
        if save_output:
            self.output = output

        return output

    def _forward_numpy(self, inp: np.array=None, save_output=False) -> np.array:
        """
        Numpy implementation of forward.

        Arguments:
            inp:
                the input.
            save_output:
                Whether to save the output in the node. 
        Returns: 
            the output of the node.
        """
        padded_inp = Conv.pad(inp, self.pads)

        inp_strech = Conv.im2col(
            padded_inp, 
            (self.krn_height, self.krn_width),
            self.strides,
            device=self.device
        )

        kernel_strech = self.kernels.reshape(self.out_ch, -1).cpu().numpy()

        output = np.tensordot(kernel_strech, inp_strech, axes=([1], [0]))
        output = output.flatten() + np.tile(
            self.bias.cpu().numpy(), (self.out_ch_sz, 1)
        ).T.flatten()
        output = output.reshape(self.output_shape)

        if save_output is True:
            self.output = output

        return output


    def transpose(
        self,
        inp: torch.Tensor,
        in_flag: torch.Tensor,
        out_flag: torch.Tensor
    ) -> torch.Tensor:
        if out_flag is None:
            filled_inp = inp
        else:
            batch = inp.shape[0]
            filled_inp = torch.zeros(
                (batch, self.output_size), dtype=inp.dtype, device=self.device
            ) 
            filled_inp[:, out_flag.flatten()] = inp
            filled_inp = filled_inp.reshape((batch,) + self.output_shape_no_batch())

        if in_flag is None:
            return self._transpose(filled_inp)

        return self._transpose_partial(filled_inp, in_flag)

    def _transpose_partial(
        self,
        inp: torch.Tensor,
        in_flag: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the input to the node given an output.

        Arguments:
            inp:
                the output.
            in_flag:
                boolean flag of the units in input.
        Returns:
            the input of the node.
        """
        batch = inp.shape[0]

        padded_height = self.in_height + 2 * self.pads[0] + self.krn_height - 1
        padded_width = self.in_width + 2 * self.pads[1] + self.krn_width - 1
        padded_inp = torch.zeros(
            (batch, self.out_ch, padded_height, padded_width),
            dtype=inp.dtype,
            device = self.device
        ) 
        slices = tuple(
            [
                slice(0, batch, 1),
                slice(0, self.out_ch, 1),
                slice(
                    self.krn_height - 1,
                    padded_height - self.krn_height + 1, 
                    self.strides[0]
                ),
                slice(
                    self.krn_width - 1,
                    padded_width - self.krn_width + 1,
                    self.strides[1]
                )
            ]
        )

        padded_inp[slices] = inp

        inp_stretch = Conv.im2col(
            padded_inp,
            (self.krn_height, self.krn_width),
            (1, 1),
            device=self.device
        )

        kernel_stretch = torch.flip(
            self.kernels.permute(1, 0, 2, 3), dims=[2, 3]
        ).reshape(self.in_ch, -1)

        flag = in_flag.reshape(self.in_ch, -1)
        pad_flag = self.get_non_pad_idx_flag().reshape(self.in_ch, -1)
 
        result = torch.empty(
            (0, inp_stretch.shape[-1]),
            dtype=self.config.PRECISION,
            devide=self.device
        )

        for i in range(self.in_ch):
            partial_result =  torch.tensordot(
                kernel_stretch[i, :],
                inp_stretch[:, pad_flag[i, :], :][:, flag[i, :], :], 
                dims=([0], [0])
            )
            result = torch.cat((result, partial_result), 0)

        return result.T

    def _transpose(self, inp: torch.Tensor) -> torch.Tensor:
        """
        Computes the input to the node given an output.

        Arguments:
            inp:
                the output.
        Returns:
            the input of the node.
        """
        out_pad_height = self.in_height - (self.out_height - 1) * self.strides[0]
        out_pad_height += 2 * self.pads[0] - self.krn_height
        out_pad_width = self.in_width - (self.out_width - 1) * self.strides[0]
        out_pad_width += 2 * self.pads[0] - self.krn_width

        return torch.nn.functional.conv_transpose2d(
            inp.reshape((inp.shape[0], self.out_ch, self.out_height, self.out_width)),
            self.kernels,
            stride=self.strides,
            padding=self.pads,
            output_padding=(out_pad_height, out_pad_width)
        ).reshape(inp.shape[0], - 1)

    def get_input_padded_size(self) -> int:
        """
        Computes the size of the padded input.
        """
        return ConvBase.get_padded_size(
            (self.in_ch, self.in_height, self.in_width), self.pads
        )

    def get_input_padded_shape(self) -> int:
        """
        Computes the shape of padded input.
        """  
        return ConvBase.get_padded_shape(
            (self.in_ch, self.in_height, self.in_width), self.pads
        )

    def get_non_pad_idxs(self) -> torch.Tensor:
        """
        Computes the indices of the original input within the padded one.
        """
        return ConvBase.get_non_pad_idxs(
            (self.in_ch, self.in_height, self.in_width),
            self.pads,
            device=self.device
        )

    def get_non_pad_idx_flag(self) -> torch.Tensor:
        """
        Computes the index flag of the original input whithin the padded one.
        """
        return ConvBase.get_non_pad_idx_flag(
            (self.in_ch, self.in_height, self.in_width),
            self.pads,
            device=self.device
        )

    def _compute_output_shape(self) -> tuple:
        """
        Computes the output shape of the node.
        """
        out_height = int(math.floor(
            (self.in_height - self.krn_height + 2 * self.pads[0]) / self.strides[0] + 1
        ))
        out_width = int(math.floor(
            (self.in_width - self.krn_width + 2 * self.pads[1]) / self.strides[1] + 1
        ))
      
        if self.has_batch_dimension():
            return 1, self.out_ch, out_height, out_width
        else:
            return self.out_ch, out_height, out_width

class ConvTranspose(ConvBase):
    def __init__(
        self,
        from_node: list, 
        to_node: list,
        input_shape: tuple,
        kernels: torch.Tensor,
        bias: torch.Tensor,
        pads: tuple,
        output_pads: tuple,
        strides: tuple,
        config: Config,
        depth=0,
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
            kernels:
                the kernels.
            bias:
                bias vector.
            pads:
                the pads.
            output_pads:
                the output pads.
            strides:
                the strides.
            config:
                configuration.
            depth:
                the depth of the node.
            bounds:
                the concrete bounds of the node.
        """
        self.output_pads = output_pads
        super().__init__(
            from_node,
            to_node,
            input_shape,
            kernels,
            bias,
            pads,
            strides,
            config,
            depth=depth,
            bounds=bounds,
            id=id
        )
        self.in_ch,  self.out_ch, self.krn_height, self.krn_width = kernels.shape
        self.output_shape = self._compute_output_shape()
        self.output_size = np.prod(self.output_shape)
        self.out_height, self.out_width = self.output_shape[-2:]
        self.out_ch_sz = int(self.output_size / self.out_ch)
        self.in_ch_sz = int(self.input_size / self.in_ch)

    def copy(self):
        """
        Copies the node.
        """
        return ConvTranspose(
            self.from_node,
            self.to_node,
            self.input_shape,
            self.kernels,
            self.bias,
            self.pads,
            self.output_pads,
            self.strides,
            self.config,
            depth=self.depth,
            bounds=self.bounds.copy(),
            id=self.id
        )

    def numpy(self):
        """
        Copies the node with numpy data.
        """
        return ConvTranspose(
            self.from_node,
            self.to_node,
            self.input_shape,
            self.kernels.cpu().cpu().numpy(),
            self.bias.cpu().cpu().numpy(),
            self.pads,
            self.output_pads,
            self.strides,
            self.config,
            depth=self.depth,
            bounds=self.bounds,
            id=self.id
        )
    
    def forward(
        self,
        inp: torch.Tensor=None,
        clip=None,
        add_bias=True,
        save_output=False
    ) -> torch.Tensor:
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
            save_output:
                Whether to save the output in the node. 
        Returns: 
            the output of the node.
        """ 
        assert inp is not None or self.from_node[0].output is not None
        inp = self.from_node[0].output if inp is None else inp

        if isinstance(inp, torch.Tensor):
            return self._forward_torch(inp, clip, add_bias, save_output)

        elif isinstance(inp, np.ndarray):
            return self._forward_numpy(inp, save_output)

        else:
            raise TypeError("Forward supports only numpy arrays and torch.Tensors.")

    def _forward_torch(
        self,
        inp: torch.Tensor=None,
        clip=None,
        add_bias=True,
        save_output=False
    ) -> torch.Tensor:
        """
        Torch implementation of forward.

        Arguments:
            inp:
                the input.
            clip:
                clips the weights to positive values if set to '+' and to
                negatives if set to '-'
            add_bias:
                whether to add bias
            save_output:
                Whether to save the output in the node. 
        Returns: 
            the output of the node.
        """
        if clip is None:
            kernels = self.kernels

        elif clip == '+':
            kernels = torch.clamp(self.kernels, 0, math.inf)

        elif clip == '-':
            kernels = torch.clamp(self.kernels, -math.inf, 0)

        else:
            raise ValueError(f'Kernel clip value {kernel_clip} not recognised')
   
        output = torch.nn.functional.conv_transpose2d(
            inp,
            kernels,
            bias = self.bias if add_bias is True else None,
            stride=self.strides,
            padding=self.pads,
            output_padding=self.output_pads
        )

        if save_output:
            self.output = output

        return output

    def _forward_numpy(self, inp: np.ndarray=None, save_output=False) -> np.ndarray:
        """
        Numpy implementation of forward.

        Arguments:
            inp:
                the input.
            save_output:
                Whether to save the output in the node. 
        Returns: 
            the output of the node.
        """
        padded_height = self.out_height + 2 * self.pads[0] + self.krn_height - 1
        padded_width = self.out_width + 2 * self.pads[1] +  self.krn_width - 1
        padded_inp = np.zeros(
            (self.in_ch, padded_height, padded_width), dtype=inp.dtype
        )

        slices = tuple(
            [
                slice(0, self.in_ch, 1),
                slice(
                    self.krn_height - 1,
                    padded_height - self.krn_height + 1,
                    self.strides[0]
                ),
                slice(
                    self.krn_width - 1,
                    padded_width - self.krn_width + 1,
                    self.strides[1]
                )
            ]
        )
        padded_inp[slices] = inp
        if self.has_batch_dimension():
            padded_inp = padded_inp[np.newaxis, ...]

        inp_strech = Conv.im2col(
            padded_inp,
            (self.krn_height, self.krn_width),
            (1, 1),
            devide=self.device
        )

        kernel_strech = torch.flip(
            self.kernels.permute(1, 0, 2, 3), dims=[2, 3]
        ).reshape(self.out_ch, -1).cpu().numpy()

        output = np.tensordot(kernel_strech, inp_strech, axes = ([1], [0]))
        pad_flag = self.get_non_pad_idx_flag().flatten()
        output = output.flatten()[pad_flag]
        output = output + np.tile(
            self.bias.cpu().numpy(), (self.out_ch_sz, 1)
        ).T.flatten()
        output = output.reshape(self.output_shape)

        if save_output is True:
            self.output = output

        return output

    def transpose(
        self,
        inp: torch.Tensor,
        in_flag: torch.Tensor,
        out_flag: torch.Tensor
    ) -> torch.Tensor:
        if out_flag is None:
            filled_inp = inp
        else:
            batch = inp.shape[0]
            filled_inp = torch.zeros(
                (batch, self.output_size),
                dtype=inp.dtype,
                device=self.device
            ) 
            filled_inp[:, out_flag.flatten()] = inp
            filled_inp = filled_inp.reshape((batch,) + self.output_shape_no_batch())

        if in_flag is None:
            return self._transpose(filled_inp)

        return self._transpose_partial(filled_inp, in_flag)


    def _transpose(self, inp: torch.Tensor) -> torch.Tensor:
        """
        Computes the input to the node given an output.

        Arguments:
            inp:
                the output.
        Returns:
            the input of the node.
        """

        return torch.nn.functional.conv2d(
            inp,
            self.kernels,
            stride=self.strides,
            padding=self.pads
        )

    def _transpose_partial(
        self,
        inp: torch.Tensor,
        in_flag: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the input to the node given an output.

        Arguments:
            inp:
                the output.
            out_flag:
                boolean flag of the units in output.
            in_flag:
                boolean flag of the units in input.
        Returns:
            the input of the node.
        """
        # TODO
        # padded_inp = Conv.pad(inp, self.pads)

        # inp_strech = Conv.im2col(
            # padded_inp, (self.krn_height, self.krn_width), self.strides
        # )

        # kernel_strech = self.kernels.reshape(self.in_ch, -1).numpy()

        # output = kernel_strech @ inp_strech
        # output = output.reshape(self.output_shape)


        # return output

   
    def get_output_padded_size(self) -> int:
        """
        Computes the size of the padded input.
        """
        return ConvBase.get_padded_size(
            (self.out_ch, self.out_height, self.out_width), self.pads
        )

    def get_output_padded_shape(self) -> int:
        """
        Computes the shape of padded input.
        """  
        return ConvBase.get_padded_shape(
            (self.out_ch, self.out_height, self.out_width), self.pads
        )

    def get_non_pad_idxs(self) -> torch.Tensor:
        """
        Computes the indices of the original input whithin the padded one.
        """
        return ConvBase.get_non_pad_idxs(
            (self.out_ch, self.out_height, self.out_width),
            self.pads,
            device=self.device
        )

    def get_non_pad_idx_flag(self) -> torch.Tensor:
        """
        Computes the index flag of the original input whithin the padded one.
        """
        return ConvBase.get_non_pad_idx_flag(
            (self.out_ch, self.out_height, self.out_width),
            self.pads,
            device=self.device
        )

    def _compute_output_shape(self) -> tuple:
        """
        Computes the output shape of the node.
        """
        out_height = (self.in_height - 1) * self.strides[0] - 2 * self.pads[0] + self.krn_height + self.output_pads[0]
        out_width = (self.in_width - 1) * self.strides[0] - 2 * self.pads[0] + self.krn_width + self.output_pads[0]
      
        if self.has_batch_dimension():
            return 1, self.out_ch, out_height, out_width
        else:
            return self.out_ch, out_height, out_width


class MaxPool(Node):
    def __init__(
        self,
        from_node: list,
        to_node: list,
        input_shape: tuple,
        kernel_shape: tuple,
        pads: tuple,
        strides: tuple,
        config: Config,
        depth=0,
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
            kernel_shape:
                the kernel shape.
            pads:
                the pads.
            strides:
                the strides.
            config:
                configuration.
            depth:
                the depth of the node.
            bounds:
                the concrete bounds of the node.
        """
        output_shape = MaxPool.compute_output_shape(
            input_shape, kernel_shape, pads, strides
        )
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
        self.kernel_shape = kernel_shape
        self.pads = pads
        self.strides = strides

    def copy(self):
        """
        Copies the node.
        """
        return MaxPool(
            self.from_node,
            self.to_node,
            self.input_shape,
            self.kernel_shape,
            self.pads,
            self.strides,
            self.config,
            depth=self.depth,
            bounds=self.bounds.copy(),
            id=self.id
        )

    def get_milp_var_size(self):
        """
        Returns the number of milp variables required for the milp encoding of
        the node.
        """
        return self.output_size


    def forward(
        self, inp: torch.Tensor=None, return_indices=False, save_output=False
    ) -> np.array:
        """
        Computes the output of the node given an input.

        Arguments:
            inp:
                The input.
            return_indices:
                Whether to return the indices of the max units.
            save_output:
                Whether to save the output in the node. 
        Returns: 
            the output of the node.
        """
        assert inp is not None or self.from_node[0].output is not None
        inp = self.from_node[0].output if inp is None else inp

        if isinstance(inp, torch.Tensor):
            return self._forward_torch(inp, return_indices, save_output)

        elif isinstance(inp, np.ndarray):
            return self._forward_numpy(inp, save_output)

        else:
            raise TypeError("Forward supports only numpy arrays and torch.Tensors.")


    def _forward_torch(
        self, inp: torch.Tensor=None, return_indices=False, save_output=False
    ) -> np.array:
        """
        Torch implementation of forward.

        Arguments:
            inp:
                The input.
            return_indices:
                Whether to return the indices of the max units.
            save_output:
                Whether to save the output in the node. 
        Returns: 
            the output of the node.
        """
        output = torch.nn.functional.max_pool2d(
            inp,
            self.kernel_shape,
            stride=self.strides,
            padding=self.pads,
            return_indices=return_indices
        )

        if save_output:
            self.output = output if return_indices is not True else output[0]

        return output


    def _forward_numpy(self, inp: torch.Tensor=None, save_output=False) -> np.array:
        """
        Numpy implementation of forward.

        Arguments:
            inp:
                The input.
            save_output:
                Whether to save the output in the node. 
        Returns: 
            the output of the node.
        """
        padded_inp = Conv.pad(inp, self.pads).reshape((self.in_ch(), 1) + inp.shape[-2:])
        im2col = Conv.im2col(
            padded_inp, self.kernel_shape, self.strides, device=self.device
        )
        
        output = im2col.max(axis=0).reshape(
             self.output_shape[-2:] + (self.in_ch(),)
        ).transpose(2, 0, 1)
        if self.has_batch_dimension() is True:
            output = np.expand_dims(output, 0)

        if save_output is True:
            self.output = output

        return output

    @staticmethod
    def compute_output_shape(in_shape: tuple, kernel_shape: tuple, pads: tuple, strides: tuple) -> tuple:
        """
        Computes the output shape of the node.

        Arguments:
            in_shape:
                shape of the input tensor to the node.
            kernel_shape:
                shape of the kernel of the node.
            pads:
                pair of int for the width and height of the pads.
            strides:
                pair of int for the width and height of the strides.
        Returns:
            tuple of the output shape
        """
        assert len(in_shape) in [3, 4]

        if len(in_shape) == 3:
            in_ch, in_height, in_width = in_shape
        else:
            batch, in_ch, in_height, in_width = in_shape

        k_height, k_width = kernel_shape

        out_height = int(math.floor(
            (in_height - k_height + 2 * pads[0]) / strides[0] + 1
        ))
        out_width = int(math.floor(
            (in_width - k_width + 2 * pads[1]) / strides[1] + 1
        ))

        if len(in_shape) == 3:
            return in_ch, out_height, out_width
        else:
            return batch, in_ch, out_height, out_width

class Relu(Node):
    def __init__(
        self,
        from_node: list,
        to_node: list,
        input_shape: tuple,
        config: Config,
        depth=0,
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
            [ReluState.UNSTABLE] * self.output_size, dtype=ReluState
        ).reshape(self.output_shape)
        self.dep_root = np.array(
            [False] * self.output_size, dtype=bool
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
        self.forced_states = 0
        self._lower_relaxation_slope = (None, None)
        self._custom_relaxation_slope = False

    def copy(self):
        """
        Copies the node.
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

    def set_state(self, unit: tuple, state: ReluState):
        """
        Forces the state of a relu unit.

        Arguments:
            unit:
                The relu unit.
            state:
                The state of the unit.
        """
        self.state[unit] = state 
        self.forced_states += 1
 
    def set_dep_root(self, unit: tuple, root_status: bool):
        """
        Forces the dependency root status of a relu unit.

        Arguments:
            unit:
                The relu unit.
            root_status:
                The dependency root status.
        """
        self.dep_root[unit] = root_status
       

    def forward(self, inp: torch.Tensor=None, save_output=None) -> torch.Tensor:
        """
        Computes the output of the node given an input.

        Arguments:
            inp:
                the input.
            save_output:
                Whether to save the output in the node.
        Returns:
            the output of the node.
        """
        assert inp is not None or self.from_node[0].output is not None
        inp = self.from_node[0].output if inp is None else inp

        if isinstance(inp, torch.Tensor):
            return self._forward_torch(inp, save_output)

        elif isinstance(inp, np.ndarray):
            return self._forward_numpy(inp, save_output)

        else:
            raise TypeError("Forward supports only numpy arrays and torch.Tensors.")

    def _forward_torch(self, inp: torch.Tensor=None, save_output=None) -> torch.Tensor:
        """
        Torch implementation of forward.

        Arguments:
            inp:
                the input.
            save_output:
                Whether to save the output in the node.
        Returns:
            the output of the node.
        """
        output = torch.clamp(inp, 0, math.inf)

        if save_output:
            self.output = output

        return output

    def _forward_numpy(self, inp: np.array=None, save_output=None) -> np.array:
        """
        Numpy implementation of forward.

        Arguments:
            inp:
                the input.
            save_output:
                Whether to save the output in the node.
        Returns:
            the output of the node.
        """
        assert inp is not None or self.from_node[0].output is not None
        inp = self.from_node[0].output if inp is None else inp

        output = np.clip(inp, 0, math.inf)

        if save_output:
            self.output = output

        return output

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
        cond0a = self.from_node[0].bounds.lower[index].item() >= 0
        cond0b = self.from_node[0].bounds.upper[index].item() <= 0
        cond1 = cond0a or cond0b
        cond2 = self.state[index] != ReluState.UNSTABLE
        cond3 = False if delta_val is None else delta_val in [0, 1]

        return cond1 or cond2 or cond3

    def get_active_flag(self) -> torch.Tensor:
        """
        Returns an array of activity statuses for each ReLU node.
        """
        if self.active_flag is None:
            self.active_flag = self.from_node[0].bounds.lower >= 0
            if self.forced_states > 0:
                self.active_flag = torch.logical_or(
                    self.active_flag, 
                    torch.Tensor(
                        self.state == ReluState.ACTIVE,
                        device=self.device
                    )
                )

        return self.active_flag

    def get_active_count(self) -> int:
        """
        Returns the total number of active Relu nodes.
        """
        if self.active_count is None:
            self.active_count = torch.sum(self.get_active_flag())

        return self.active_count

    def get_inactive_flag(self) -> torch.Tensor:
        """
        Returns an array of inactivity statuses for each ReLU node.
        """
        if self.inactive_flag is None:
            self.inactive_flag = self.from_node[0].bounds.upper <= 0
            if self.forced_states > 0:
                self.inactive_flag = torch.logical_or(
                    self.inactive_flag, 
                    torch.Tensor(
                        self.state == ReluState.INACTIVE,
                        device=self.device
                    )
                )

        return self.inactive_flag

    def get_inactive_count(self) -> int:
        """
        Returns the total number of inactive ReLU nodes.
        """
        if self.inactive_count is None:
            self.inactive_count = torch.sum(self.get_inactive_flag())

        return self.inactive_count

    def get_unstable_flag(self) -> torch.Tensor:
        """
        Returns an array of instability statuses for each ReLU node.
        """
        if self.unstable_flag is None:
            self.unstable_flag = torch.logical_and(
                self.from_node[0].bounds.lower < 0,
                self.from_node[0].bounds.upper > 0
            )
            if self.forced_states > 0:
                self.unstable_flag = torch.logical_or(
                    self.unstable_flag,
                    torch.Tensor(
                        self.state == ReluState.UNSTABLE,
                        device=self.device
                    )
                )

        return self.unstable_flag

    def get_unstable_indices(self) -> torch.Tensor:
        """
        Returns an array of indices of unstable ReLU units.
        """
        return [
            tuple(j.item() for j in tuple(i))
            for i in self.get_unstable_flag().nonzero()
        ]

    def get_active_indices(self) -> torch.Tensor:
        """
        Returns an array of indices of active ReLU units.
        """
        return [
            tuple(j.item() for j in tuple(i))
            for i in self.get_active_flag().nonzero()
        ]

    def get_inactive_indices(self) -> torch.Tensor:
        """
        Returns an array of indices of inactive ReLU units.
        """
        return [
            tuple(j.item() for j in tuple(i))
            for i in self.get_inactive_flag().nonzero()
        ]

    def get_deproot_indices(self) -> torch.Tensor:
        """
        Returns an array of indices of dependency root ReLU units.
        """
        idx = self.dep_root.nonzero()
        return [
            tuple(i[j].item() for i in idx) for j in range(len(idx[0]))
        ]

    def get_unstable_count(self) -> int:
        """
        Returns the total number of unstable nodes.
        """
        if self.unstable_count is None:
            self.unstable_count = torch.sum(self.get_unstable_flag())

        return self.unstable_count

    def get_stable_flag(self) -> torch.Tensor:
        """
        Returns an array of instability statuses for each ReLU node.
        """
        if self.stable_flag is None:
            self.stable_flag = torch.logical_or(
                self.get_active_flag(), self.get_inactive_flag()
            )

        return self.stable_flag

    def get_stable_count(self) -> int:
        """
        Returns the total number of unstable nodes.
        """
        if self.stable_count is None:
            self.stable_count = torch.sum(self.get_stable_flag()).item()

        return self.stable_count

    def get_propagation_flag(self) -> torch.Tensor:
        """
        Returns an array of sip propagation statuses for each node.
        """
        if self.propagation_flag is None:
            self.propagation_flag = torch.logical_or(
                self.get_active_flag(),
                self.get_unstable_flag()
            )

        return self.propagation_flag

    def get_propagation_count(self) -> int:
        """
        Returns the total number of sip propagation nodes.
        """
        if self.propagation_count is None:
            self.propagation_count = torch.sum(self.get_propagation_flag())

        return self.propagation_count

    def get_lower_relaxation_slope(self) -> torch.Tensor:
        """
        Returns the lower bound relaxation slopes.
        """

        if self.bounds.size() == 0:
            return None, None
        
        elif self._lower_relaxation_slope == (None, None):
            self.set_lower_relaxation_slope(
                approx = self.config.SIP.RELU_APPROXIMATION
            )
            
        return self._lower_relaxation_slope

    def set_lower_relaxation_slope(
        self,
        lower: torch.Tensor=None,
        upper: torch.Tensor=None,
        approx = ReluApproximation.MIN_AREA
    ) -> torch.Tensor:
        """
        Derives the lower bound relaxation slopes.
        """
        if lower is not None and upper is not None:
            self._lower_relaxation_slope = (lower, upper)
            self._custom_relaxation_slope = True
        else:
            slope = torch.ones(
                self.get_unstable_count(),
                dtype=self.config.PRECISION,
                device=self.device
            )
            lower = self.from_node[0].bounds.lower[self.get_unstable_flag()]
            upper = self.from_node[0].bounds.upper[self.get_unstable_flag()]
            idxs = abs(lower) >= upper
            slope[idxs] = 0.0
            if approx == ReluApproximation.VENUS:
                idxs = torch.logical_not(idxs)
                slope[idxs] = upper[idxs] / (upper[idxs] - lower[idxs])
 
            self._lower_relaxation_slope = (slope, slope)


    def has_custom_relaxation_slope(self):
        """
        Returns whether the relaxation slope is not the default.
        """
        return self._custom_relaxation_slope

    def get_milp_var_indices(self, var_type: str='all'):
        """
        Returns the starting and ending indices of the milp variables encoding
        the node.

        Arguments:
            var_type: either 'out' for output variables or 'delta' for binary
            variables of 'all' for all.
        """
        if self._milp_var_indices is None:
            if len(self.from_node) > 0:
                start_out = self.from_node[-1].get_milp_var_indices()[0]
                start_delta = self.from_node[-1].get_milp_var_indices()[0] + self.output_size
            else:
                start_out = 0
                start_delta = self.output_size 
            end_out = start_out + self.output_size
            end_delta = start_delta + self.get_unstable_count()

            self._milp_var_indices = {
                'all': (start_out, end_delta),
                'out': (start_out, end_out),
                'delta': (start_delta, end_delta)
            }

        if var_type == 'all': 
            return self._milp_var_indices['all']
        if var_type == 'out': 
            return self._milp_var_indices['out']
        elif var_type == 'delta':
            return self._milp_var_indices['delta']
        else:
            raise ValueError('var_type can only be out or delta or all')

class Reshape(Node):
    def __init__(
        self,
        from_node: list, 
        to_node: list,
        input_shape: tuple,
        output_shape: tuple,
        config: Config,
        depth=0,
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
            depth=depth,
            bounds=bounds,
            id=id
        )

    def copy(self):
        """
        Copies the node.
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

    def forward(self, inp: torch.Tensor=None, save_output=None) -> torch.Tensor:
        """
        Computes the output of the node given an input.

        Arguments:
            inp:
                the input.
            save_output:
                Whether to save the output in the node.
        Returns:
            the output of the node.
        """
        assert inp is not None or self.from_node[0].output is not None
        inp = self.from_node[0].output if inp is None else inp


        output = inp.reshape(self.output_shape)

        if save_output is True:
            self.output = output

        return output

    def get_milp_var_indices(self, var_type: str):
        """
        Returns the starting and ending indices of the milp variables encoding
        the node.

        Arguments:
            var_type: either 'out' for output variables or 'delta' for binary
            variables.
        """
        return self.from_node[-1].get_milp_var_indices()

    def get_propagation_flag(self) -> torch.Tensor:
        if isinstance(self.from_node[0], Relu):
            return self.forward(self.from_node[0].get_propagation_flag())

        return torch.ones(
            self.output_shape, dtype=torch.bool, device=self.device
        )

    def get_propagation_count(self) -> torch.Tensor:
        if isinstance(self.from_node[0], Relu):
            return self.from_node[0].get_propagation_count()

        return self.output_size
 

class Flatten(Node):
    def __init__(
        self,
        from_node: list, 
        to_node: list,
        input_shape: tuple,
        config: Config,
        depth=0,
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
            (np.prod(input_shape),),
            config,
            depth=depth,
            bounds=bounds,
            id=id
        )

    def copy(self):
        """
        Copies the node.
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

    def get_milp_var_indices(self, var_type: str='out'):
        """
        Returns the starting and ending indices of the milp variables encoding
        the node.

        Arguments:
            var_type: either 'out' for output variables or 'delta' for binary
            variables.
        """
        return self.from_node[-1].get_milp_var_indices()

    def forward(self, inp: torch.Tensor=None, save_output=False) -> torch.Tensor:
        """
        Computes the output of the node given an input.

        Arguments:
            inp:
                the input.
            save_output:
                Whether to save the output in the node.
        Returns: 
            the output of the node.
        """
        assert inp is not None or self.from_node[0].output is not None
        inp = self.from_node[0].output if inp is None else inp
        
        if isinstance(inp, torch.Tensor):
            return self._forward_torch(inp, save_output)

        elif isinstance(inp, np.ndarray):
            return self._forward_numpy(inp, save_output)

        else:
            raise TypeError("Forward supports only numpy arrays and torch.Tensors.")


    def _forward_torch(self, inp: torch.Tensor=None, save_output=False) -> torch.Tensor:
        """
        Torch implementation of forward.

        Arguments:
            inp:
                the input.
            save_output:
                Whether to save the output in the node.
        Returns: 
            the output of the node.
        """
        output = inp.flatten()

        if save_output:
            self.output = output

        return output

    def _forward_numpy(self, inp: np.ndarray=None, save_output=False) -> np.ndarray:
        """
        Numpy implementation of forward.

        Arguments:
            inp:
                the input.
            save_output:
                Whether to save the output in the node. 
        Returns:
            the output of the node.
        """
        assert inp is not None or self.from_node[0].output is not None
        inp = self.from_node[0].output if inp is None else inp

        output = inp.flatten()

        if save_output is True:
            self.output = output

        return output

    def get_propagation_flag(self) -> torch.Tensor:
        if isinstance(self.from_node[0], Relu):
            return self.forward(self.from_node[0].get_propagation_flag())

        return torch.ones(
            self.output_shape, dtype=torch.bool, device=self.device
        )
    
    def get_propagation_count(self) -> torch.Tensor:
        if isinstance(self.from_node[0], Relu):
            return self.from_node[0].get_propagation_count()

        return self.output_size


class Sub(Node):
    def __init__(
        self,
        from_node: list,
        to_node: list,
        input_shape: tuple,
        config: Config,
        const=None,
        depth=0,
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
        Copies the node.
        """
        return Sub(
            self.from_node,
            self.to_node,
            self.input_shape,
            self.config,
            const=None if self.const is None else self.const.detach().clone(),
            depth=self.depth,
            bounds=self.bounds,
            id=self.id
        )

    def numpy(self):
        """
        Copies the node with numpy data.
        """
        return Sub(
            self.from_node,
            self.to_node,
            self.input_shape,
            self.config,
            const=None if self.const is None else self.const.cpu().numpy(),
            depth=self.depth,
            bounds=self.bounds,
            id=self.id
        )

    def get_milp_var_size(self):
        """
        Returns the number of milp variables required for the milp encoding of
        the node.
        """
        return self.output_size

    def forward(self, inp1: torch.Tensor=None, inp2: torch.Tensor=None, save_output=False) -> torch.Tensor:
        """
        Computes the output of the node given an input.

        Arguments:
            inp1:
                the first input.
            inp2:
                the second input. If not set then the const of the node is
                taken as second input.
            save_output:
                Whether to save the output in the node.
        Returns:
            the output of the node.
        """

        assert inp1 is not None or self.from_node[0].output is not None
        assert inp2 is not None or self.const is not None

        inp1 = self.from_node[0].output if inp1 is None else inp1
        inp2 = self.const if inp2 is None else inp2

        if isinstance(inp1, torch.Tensor):
            return self._forward_torch(inp1, inp2, save_output)

        elif isinstance(inp1, np.ndarray):
            return self._forward_numpy(inp1, inp2, save_output)

        else:
            raise TypeError("Forward supports only numpy arrays and torch.Tensors.")

    def _forward_torch(self, inp1: torch.Tensor=None, inp2: torch.Tensor=None, save_output=False) -> torch.Tensor:
        """
        Torch implementation of forward.

        Arguments:
            inp1:
                the first input.
            inp2:
                the second input. If not set then the const of the node is
                taken as second input.
            save_output:
                Whether to save the output in the node.
        Returns:
            the output of the node.
        """
        output = inp1 - inp2

        if save_output:
            self.output = output

        return output

    def _forward_numpy(
        self, inp1: np.array=None, inp2: np.array=None, save_output=False
    ) -> np.array:
        """
        Numpy implementation of forward.

        Arguments:
            inp1:
                the first input.
            inp2:
                the second input. If not set then the const of the node is
                taken as second input.
            save_output:
                Whether to save the output in the node.
        Returns:
            the output of the node.
        """
        output = inp1 - inp2

        if save_output is True:
            self.output = output

        return output

    def transpose(self, inp: torch.Tensor) -> torch.Tensor:
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

        return torch.hstack(([inp, - inp]))

class Add(Node):
    def __init__(
        self,
        from_node: list,
        to_node: list,
        input_shape: tuple,
        config: Config,
        const=None,
        depth=0,
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
        Copies the node.
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

    def numpy(self):
        """
        Copies the node with numpy data.
        """
        return Add(
            self.from_node,
            self.to_node,
            self.input_shape,
            self.config,
            const = None if self.const is None else self.const.cpu().numpy(),
            depth=self.depth,
            bounds=self.bounds,
            id=self.id
        )

    def get_milp_var_indices(self, var_type: str):
        """
        Returns the starting and ending indices of the milp variables encoding
        the node.

        Arguments:
            var_type: either 'out' for output variables or 'delta' for binary
            variables.
        """
        if self._milp_var_indices is not None:
            return self._milp_var_indices

        from_idxs = self.from_node[-1].get_milp_var_indices()
        start = from_idxs[0] if var_type == 'out' else from_idxs[0] + self.output_size
        end = start + self.output_size if var_type == 'out' else start + self.get_unstable_count()

        start, end


    def forward(
        self, inp1: torch.Tensor=None, inp2: torch.Tensor=None, save_output=False
    ) -> torch.Tensor:
        """
        Computes the output of the node given an input.

        Arguments:
            inp1:
                the first input.
            inp2:
                the second input. If not set then the const of the node is
                taken as second input.
            save_output:
                Whether to save the output in the node. 
        Returns: 
            the output of the node.
        """
        assert inp1 is not None or self.from_node[0].output is not None
        assert inp2 is not None or self.const is not None

        inp1 = self.from_node[0].output if inp is None else inp1
        inp2 = self.const if inp2 is None else inp2

        if isinstance(inp1, torch.Tensor):
            return self._forward_torch(inp1, inp2, save_output)

        elif isinstance(inp1, np.ndarray):
            return self._forward_numpy(inp1, inp2, save_output)

        else:
            raise TypeError("Forward supports only numpy arrays and torch.Tensors.")


    def _forward_torch(
        self, inp1: torch.Tensor=None, inp2: torch.Tensor=None, save_output=False
    ) -> torch.Tensor:
        """
        Torch implementation of forward.

        Arguments:
            inp1:
                the first input.
            inp2:
                the second input. If not set then the const of the node is
                taken as second input.
            save_output:
                Whether to save the output in the node. 
        Returns: 
            the output of the node.
        """
        output = inp1 + inp2

        if save_output:
            self.output = output

        return output

    def _forward_numpy(
        self, inp1: np.array=None, inp2: np.array=None, save_output=False
    ) -> np.array:
        """
        Numpy implementation of forward.

        Arguments:
            inp1:
                the first input.
            inp2:
                the second input. If not set then the const of the node is
                taken as second input.
            save_output:
                Whether to save the output in the node.
        Returns:
            the output of the node.
        """
        assert inp1 is not None or self.from_node[0].output is not None
        assert inp2 is not None or self.const is not None

        inp2 = self.const.cpu().numpy() if inp2 is None else inp2

        output = inp1 + inp2

        if save_output is True:
            self.output = output

        return output

    def transpose(self, inp: torch.Tensor) -> torch.Tensor:
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

        return torch.hstack(([inp, inp]))


class BatchNormalization(Node):
    def __init__(
        self,
        from_node: list,
        to_node: list,
        input_shape: tuple,
        scale: torch.Tensor,
        bias: torch.Tensor,
        input_mean: torch.Tensor,
        input_var: torch.Tensor,
        epsilon: float,
        config: Config,
        depth=0,
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
        Copies the node.
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

    def get_milp_var_size(self):
        """
        Returns the number of milp variables required for the milp encoding of
        the node.
        """
        return self.output_size


    def forward(self, inp: torch.Tensor=None, save_output=False) -> torch.Tensor:
        """
        Computes the output of the node given an input.

        Arguments:
            inp:
                the input.
            save_output:
                Whether to save the output in the node. 
        Returns: 
            the output of the node.
        """
        assert inp is not None or self.from_node[0].output is not None
        inp = self.from_node[0].output if inp is None else inp

        if isinstance(inp, torch.Tensor):
            return self._forward_torch(inp, save_output)

        elif isinstance(inp, np.ndarray):
            return self._forward_numpy(inp, save_output)

        else:
            raise TypeError("Forward supports only numpy arrays and torch.Tensors.")

    def _forward_torch(self, inp: torch.Tensor=None, save_output=False) -> torch.Tensor:
        """
        Torch implementation of forward.

        Arguments:
            inp:
                the input.
            save_output:
                Whether to save the output in the node. 
        Returns: 
            the output of the node.
        """
        output = torch.nn.functional.batch_norm(
            inp,
            self.input_mean,
            self.input_var,
            bias=self.bias,
            weight=self.scale,
            eps=self.epsilon
        )

        if save_output:
            self.output = output

        return output

    def _forward_numpy(self, inp: np.ndarray=None, save_output=False) -> np.ndarray:
        """
        Numpy implementation of forward.

        Arguments:
            inp:
                the input.
            save_output:
                Whether to save the output in the node. 
        Returns: 
            the output of the node.
        """
        in_ch_sz = self.in_ch_sz()
        
        scale = np.tile(self.scale.cpu().numpy(), (in_ch_sz, 1)).T.flatten()
        bias = np.tile(self.bias.cpu().numpy(), (in_ch_sz, 1)).T.flatten()
        input_mean = np.tile(self.input_mean.cpu().numpy(), (in_ch_sz, 1)).T.flatten()
        var = np.sqrt(self.input_var.cpu().numpy() + self.epsilon)
        var = np.tile(var, (in_ch_sz, 1)).T.flatten()

        output = (inp.flatten() - input_mean) / var * scale + bias
        output = output.reshape(self.output_shape)

        if save_output is True:
            self.output = output

        return output

    def transpose(self, inp: torch.Tensor) -> torch.Tensor:
        """
        Computes the input to the node given an output.

        Arguments:
            inp:
                the output.
        Returns:
            the input of the node.
        """
        #TODO

class Slice(Node):
    def __init__(
        self,
        from_node: list,
        to_node: list,
        input_shape: tuple,
        slices: list,
        config: Config,
        depth=0,
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
            slices:
                a list of slice objets for each dimension.
            config:
                configuration.
            depth:
                the depth of the node.
            bounds:
                concrete bounds for the node.
        """
        output_shape = Slice.compute_output_shape(input_shape, slices)
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
        self.slices = slices

    def copy(self):
        """
        Copies the node.
        """
        return Slice(
            self.from_node,
            self.to_node,
            self.input_shape,
            self.slices,
            self.config,
            depth=self.depth,
            bounds=self.bounds.copy(),
            id=self.id
        )

    def get_milp_var_indices(self, var_type: str):
        """
        Returns the starting and ending indices of the milp variables encoding
        the node.

        Arguments:
            var_type: either 'out' for output variables or 'delta' for binary
            variables.
        """
        raise NotImplementedError('get milp var indices for Slice')


    def forward(self, inp: torch.Tensor=None, save_output=False) -> torch.Tensor:
        """
        Computes the output of the node given an input.

        Arguments:
            inp:
                the input.
            save_output:
                Whether to save the output in the node. 
        Returns: 
            the output of the node.
        """
        assert inp is not None or self.from_node[0].output is not None
        inp = self.from_node[0].output if inp is None else inp

        if isinstance(inp, torch.Tensor):
            return self._forward_torch(inp, save_output)

        elif isinstance(inp, np.ndarray):
            return self._forward_numpy(inp, save_output)

        else:
            raise TypeError("Forward supports only numpy arrays and torch.Tensors.")

    def _forward_torch(self, inp: torch.Tensor=None, save_output=False) -> torch.Tensor:
        """
        Torch implementation of forward.

        Arguments:
            inp:
                the input.
            save_output:
                Whether to save the output in the node. 
        Returns: 
            the output of the node.
        """
        output = inp[self.slices]

        if save_output:
            self.output = output

        return output

    def _forward_numpy(self, inp: np.ndarray=None, save_output=False) -> np.ndarray:
        """
        Numpy implementation of forward.

        Arguments:
            inp:
                the input.
            save_output:
                Whether to save the output in the node. 
        Returns: 
            the output of the node.
        """
        output = inp[tuple(self.slices)]

        if save_output is True:
            self.output = output

        return output

    @staticmethod
    def compute_output_shape(input_shape: tuple, slices: list):
        """
        Computes the output shape of the node.

        Arguments:
            in_shape:
                shape of the input tensor to the node.
            slices:
                a list of slice objets for each dimension.
        Returns:
            tuple of the output shape
        """

        return tuple(
            len(range(*slices[i].indices(input_shape[i]))) 
            for i in range(len(input_shape))
        )

class Unsqueeze(Node):
    def __init__(
        self,
        from_node: list,
        to_node: list,
        input_shape: tuple,
        axes: list,
        config: Config,
        depth=0,
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
            axes:
                the list of axes
            config:
                configuration.
            depth:
                the depth of the node.
            bounds:
                concrete bounds for the node.
        """
        output_shape = Unsqueeze.compute_output_shape(input_shape, axes)
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
        self.axes = axes

    def copy(self):
        """
        Copies the node.
        """
        return Unsqueeze(
            self.from_node,
            self.to_node,
            self.input_shape,
            self.axes,
            self.config,
            depth=self.depth,
            bounds=self.bounds.copy(),
            id=self.id
        )

    def get_milp_var_indices(self, var_type: str):
        """
        Returns the starting and ending indices of the milp variables encoding
        the node.

        Arguments:
            var_type: either 'out' for output variables or 'delta' for binary
            variables.
        """
        raise NotImplementedError('get milp var indices for Unsqueeze')


        if isinstance(inp, torch.Tensor):
            return self._forward_torch(inp, save_output)

        elif isinstance(inp1, np.ndarray):
            return self._forward_numpy(inp, save_output)

        else:
            raise TypeError("Forward supports only numpy arrays and torch.Tensors.")
    
    def forward(self, inp: torch.Tensor=None, save_output=False) -> torch.Tensor:
        """
        Computes the output of the node given an input.

        Arguments:
            inp:
                the input.
            save_output:
                Whether to save the output in the node. 
        Returns: 
            the output of the node.
        """
        assert inp is not None or self.from_node[0].output is not None
        inp = self.from_node[0].output if inp is None else inp

        if isinstance(inp, torch.Tensor):
            return self._forward_torch(inp, save_output)

        elif isinstance(inp1, np.ndarray):
            return self._forward_numpy(inp, save_output)

        else:
            raise TypeError("Forward supports only numpy arrays and torch.Tensors.")

    def _forward_torch(self, inp: torch.Tensor=None, save_output=False) -> torch.Tensor:
        """
        Torch implementation of forward.

        Arguments:
            inp:
                the input.
            save_output:
                Whether to save the output in the node. 
        Returns: 
            the output of the node.
        """
        output = inp.clone()
        for i, j in enumerate(self.axes):
            output = torch.unsqueeze(output, j + i)

        if save_output:
            self.output = output

        return output

    def _forward_numpy(self, inp: np.ndarray=None, save_output=False) -> np.ndarray:
        """
        Numpy implementation of forward.

        Arguments:
            inp:
                the input.
            save_output:
                Whether to save the output in the node. 
        Returns: 
            the output of the node.
        """
        output = inp.copy()
        for i, j in enumerate(self.axes):
            output = np.expand_dims(output, j + i)

        if save_output is True:
            self.output = output

        return output

    @staticmethod
    def compute_output_shape(input_shape: tuple, axes: list):
        """
        Computes the output shape of the node.

        Arguments:
            in_shape:
                shape of the input tensor to the node.
            axes:
                a list of axes for inserting new dimensions.
        Returns:
            tuple of the output shape
        """
        temp = np.empty(input_shape)
        for i, j in enumerate(axes):
            temp = np.expand_dims(temp, j - i)

        return temp.shape

class Concat(Node):
    def __init__(
        self,
        from_node: list,
        to_node: list,
        input_shape: tuple,
        output_shape: tuple,
        axis: int,
        config: Config,
        depth=0,
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
                list of shapes of the input tensors to the node.
            output_shape:
                shape of the output tensor of the node.
            axis:
                axis where concatenation is performed.
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
        self.axis = axis

    def copy(self):
        """
        Copies the node.
        """
        return BatchNormalization(
            self.from_node,
            self.to_node,
            self.input_shape,
            self.output_shape,
            self.axis,
            self.config,
            depth=self.depth,
            bounds=self.bounds.copy(),
            id=self.id
        )

    def get_milp_var_indices(self, var_type: str):
        """
        Returns the starting and ending indices of the milp variables encoding
        the node.

        Arguments:
            var_type: either 'out' for output variables or 'delta' for binary
            variables.
        """
        raise NotImplementedError('get milp var indices for Slice')


    def forward(self, inp: list=None, save_output=False) -> torch.Tensor:
        """
        Computes the output of the node given an input.

        Arguments:
            inp:
                list of inputs to the node.
            save_output:
                Whether to save the output in the node. 
        Returns: 
            the output of the node.
        """
        assert inp is not None or self.from_node[0].output is not None
        inp = [i.output for i in self.from_node] if inp is None else inp

        if np.all([isinstance(i, torch.Tensor) for i in inp]):
            return self._forward_torch(inp, save_output)

        elif np.all([isinstance(i, np.ndarray) for i in inp]):
            return self._forward_numpy(inp, save_output)

        else:
            raise TypeError("Forward supports only numpy arrays and torch.Tensors.")

    def _forward_torch(self, inp: list=None, save_output=False) -> torch.Tensor:
        """
        Torch implementation of forward.

        Arguments:
            inp:
                list of inputs to the node.
            save_output:
                Whether to save the output in the node. 
        Returns: 
            the output of the node.
        """
        output = torch.cat(inp, self.axis)

        if save_output is True:
            self.output = output

        return output

    def _forward_numpy(self, inp: list=None, save_output=False) -> np.ndarray:
        """
        Numpy implementation of forward.

        Arguments:
            inp:
                list of inputs to the node.
            save_output:
                Whether to save the output in the node. 
        Returns: 
            the output of the node.
        """
        output = np.concatenate(inp, self.axis)

        if save_output is True:
            self.output = output

        return output
