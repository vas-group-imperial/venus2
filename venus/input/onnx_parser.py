# ************
# File: onnx_parser.py
# Top contributors (to current version): 
# 	Panagiotis Kouvaros (panagiotis.kouvaros@gmail.com)
# This file is part of the Venus  project.
# Copyright: 2019-2021 by the authors listed in the AUTHORS file in the
# top-level directory.
# License: BSD 2-Clause (see the file LICENSE in the top-level directory).
# Description: Parses an ONNX network into a Venus network
# ************


import torch
import onnx
import onnx.numpy_helper
from onnx import NodeProto, ModelProto, ValueInfoProto, AttributeProto

from venus.common.configuration import Config
from venus.network.node import *

class ONNXParser:
    SUPPORTED_NODES = ['Flatten', 'Shape', 'Constant', 'Concat', 'Unsqueeze',
                       'Gather', 'Relu', 'Gemm', 'Conv', 'Transpose', 'MatMul',
                       'Add', 'Div', 'Sub', 'BatchNormalization', 'Slice',
                       'MaxPool','ConvTranspose', 'Cast']

    def __init__(self, config: Config):
        self.config = config

    def load(self, model_path: str):
        venus_nodes = {}
        model = onnx.load(model_path)
        dictnode = {i.output[0]: i for i in model.graph.node} 
        init = {i.name: i for i in model.graph.initializer}
        [inp] = [i for i in model.graph.input if i.name not in init]
        [inp_node] = [i for i in model.graph.node if len(i.input) > 0 and i.input[0] == inp.name]
        queue = [i for i in model.graph.node if len(i.input) == 0] + [inp_node]

        while len(queue) != 0:
            node = queue.pop(0)

            if self.are_inputs_parsed(node, venus_nodes, init, inp) is True:
                venus_nodes[node.output[0]] = self.parse_node(
                    node,
                    venus_nodes,
                    init,
                    inp
                )
                queue.extend([
                    dictnode[i.output[0]]
                    for i in model.graph.node
                    if len(i.input) > 0 and i.input[0] == node.output[0]
                ])
            else:
                queue.append(node)


        nodes = self.simplify(venus_nodes[inp_node.output[0]])
        [head] = [nodes[i] for i in nodes if len(nodes[i].from_node) == 0]
        [tail] = [nodes[i] for i in nodes if len(nodes[i].to_node) == 0]
        self.update_depth(head)

        return head, tail, nodes


    def are_inputs_parsed(
        self,
        node: NodeProto,
        venus_nodes: dict,
        init: list,
        inp: ValueInfoProto
    ):
        for i in node.input:
            if i not in venus_nodes and i not in init and i != inp.name:
                return False

        return True

    def parse_node(
        self,
        node: NodeProto,
        venus_nodes: dict,
        init: list,
        inp: ValueInfoProto
    ) -> Node:
        assert node.op_type in self.SUPPORTED_NODES, \
            f'node {node.op_type} is not supported'    
        assert node.op_type == 'Constant' or node.input[0] not in init, \
            "First input to non-constant nodes should not be an initializer."

        # determine input shape
        if len(node.input) == 0:
            input_shape = None
        elif node.input[0] == inp.name:
            input_shape = tuple(
                i.dim_value for i in inp.type.tensor_type.shape.dim
            )
            # input_shape = tuple(
                # i.dim_value if i.dim_value != 0 else 1
                # for i in inp.type.tensor_type.shape.dim
            # )
        else:
            input_shape = venus_nodes[node.input[0]].output_shape
        # process node
        if node.output[0] in venus_nodes:
            vnode = venus_nodes[node.output[0]]
        else:
            if node.op_type == 'Gemm':
                vnode = self.parse_gemm(node, input_shape, venus_nodes, init)

            elif node.op_type == 'MatMul':
                vnode = self.parse_matmul(node, input_shape, venus_nodes, init)

            elif node.op_type == 'Conv':
                vnode = self.parse_conv(node, input_shape, venus_nodes, init)

            elif node.op_type == 'ConvTranspose':
                vnode = self.parse_conv_transpose(node, input_shape, venus_nodes, init)

            elif node.op_type == 'MaxPool':
                vnode = self.parse_maxpool(node, input_shape, init)

            elif node.op_type == 'Relu':
                vnode =  Relu([], [], input_shape, self.config)

            elif node.op_type == 'Flatten':
                vnode = Flatten([], [], input_shape, self.config)

            elif node.op_type == 'Constant':
                vnode = self.parse_constant(node)

            elif node.op_type == 'Sub':
                vnode = self.parse_sub(node, input_shape, venus_nodes, init)

            elif node.op_type == 'Add':
                vnode = self.parse_add(node, input_shape, venus_nodes, init)

            elif node.op_type == 'Div':
                vnode = self.parse_div(node, input_shape, venus_nodes, init)
                
            elif node.op_type == 'BatchNormalization':
                vnode = self.parse_batchnormalization(node, input_shape, venus_nodes, init)

            elif node.op_type == 'Shape':
                vnode = self.parse_shape(node, input_shape)

            elif node.op_type == 'Gather':
                vnode = self.parse_gather(node, venus_nodes, init)

            elif node.op_type == 'Cast':
                vnode = self.parse_cast(node, venus_nodes, init)

            elif node.op_type == 'Unsqueeze':
                vnode = self.parse_unsqueeze(node, venus_nodes, init)

            elif node.op_type == 'Slice':
                vnode = self.parse_slice(node, input_shape, venus_nodes, init)

            elif node.op_type == 'Concat':
                vnode = self.parse_concat(node, venus_nodes, init)
 
        # update inputs and outputs
        for i in node.input:
            if i in venus_nodes:
                venus_nodes[i].add_to_node(vnode)
                vnode.add_from_node(venus_nodes[i])

        return vnode

    def parse_gemm(self, node: NodeProto, input_shape:tuple, venus_nodes: list, init: list) -> Node:
        weights = self._to_tensor(node.input[1], venus_nodes, init)

        attr = self._get_attribute(node, 'transB')
        if attr is not None and attr.i == 0:
            weights = torch.transpose(weights, 0, 1)
        
        bias = self._to_tensor(node.input[2], venus_nodes, init)
        
        output_shape = (weights.shape[0],)

        return Gemm(
            [],
            [],
            input_shape,
            output_shape,
            weights,
            bias,
            self.config
        )

    def parse_matmul(self, node: NodeProto, input_shape: tuple, venus_nodes: list, init: list) -> Node:
        weights = self._to_tensor(node.input[1], venus_nodes, init)
        weights = torch.transpose(weights, 0, 1)
        
        output_shape = (weights.shape[0],)

        return MatMul(
            [],
            [],
            input_shape,
            output_shape,
            weights,
            self.config
        )

    def parse_conv(self, node: NodeProto, input_shape: tuple, venus_nodes:list, init: list) -> Node:
        weights = self._to_tensor(node.input[1], venus_nodes, init)
        bias = self._to_tensor(node.input[2], venus_nodes, init)

        pads, strides = (0, 0), (1, 1)

        attr = self._get_attribute(node, "pads")
        if attr is not None:
            pads =  tuple(i for i in attr.ints[0:2])

        attr = self._get_attribute(node, "strides")
        if attr is not None:
            strides = tuple(i for i in attr.ints[0:2])
            
        return Conv(
            [],
            [],
            input_shape,
            weights,
            bias,
            pads,
            strides,
            self.config
        )

    def parse_conv_transpose(self, node: NodeProto, input_shape: tuple, venus_nodes:list, init: list) -> Node:
        weights = self._to_tensor(node.input[1], venus_nodes, init)
        bias = self._to_tensor(node.input[2], venus_nodes, init)

        pads, strides, output_padding = (0, 0), (1, 1), (0, 0)

        attr = self._get_attribute(node, "pads")
        if attr is not None:
            pads =  tuple(i for i in attr.ints[0:2])

        attr = self._get_attribute(node, "strides")
        if attr is not None:
            strides = tuple(i for i in attr.ints[0:2])

        attr = self._get_attribute(node, "output_padding")
        if attr is not None:
            output_padding = tuple(i for i in attr.ints[0:2])
        
        return ConvTranspose(
            [],
            [],
            input_shape,
            weights,
            bias,
            pads,
            output_padding,
            strides,
            self.config
        )

    def parse_maxpool(self, node: NodeProto, input_shape: tuple, init: list) -> Node:
        pads, strides = (0, 0), (1, 1)

        attr = self._get_attribute(node, "kernel_shape")
        kernel_shape = tuple(attr.ints)

        attr = self._get_attribute(node, "pads")
        if attr is not None:
            pads = tuple(i for i in attr.ints[0:2])

        attr = self._get_attribute(node, "strides")
        if attr is not None:
            strides = tuple(i for i in attr.ints[0:2])

        return MaxPool(
            [],
            [],
            input_shape,
            kernel_shape,
            pads,
            strides,
            self.config
        )

    def parse_sub(self, node: NodeProto, input_shape: tuple, venus_nodes: list, init: list) -> Node:
        if node.input[0] in init or node.input[0] in venus_nodes:
            const0 = self._to_tensor(node.input[0], venus_nodes, init)
            const1 = self._to_tensor(node.input[1], venus_nodes, init)

            return Constant([], const0 - const1, self.config)

        else:
            if node.input[1] in init:
                const = self._to_tensor(node.input[1], venus_nodes, init)
            else:
                const = None

            return Sub([], [], input_shape, self.config, const=const)

    def parse_add(self, node: NodeProto, input_shape: tuple, venus_nodes: list, init: list) -> Node:
        if node.input[0] in init or node.input[0] in venus_nodes:
            const0 = self._to_tensor(node.input[0], venus_nodes, init)
            const1 = self._to_tensor(node.input[1], venus_nodes, init)

            return Constant([], const0 + const1, self.config)

        else:
            if node.input[1] in init:
                const = self._to_tensor(node.input[1], venus_nodes, init)
            else:
                const = None

            return Add([], [], input_shape, self.config, const=const)

    def parse_div(self, node: NodeProto, input_shape: tuple, venus_nodes: list, init: list) -> Node:
        const0 = self._to_tensor(node.input[0], venus_nodes, init)
        const1 = self._to_tensor(node.input[1], venus_nodes, init)

        return Constant([], const0 / const1, self.config)

    def parse_batchnormalization(self, node: NodeProto, input_shape: tuple, venus_nodes:list, init: list) -> Node:
        scale = self._to_tensor(node.input[1], venus_nodes, init)
        bias = self._to_tensor(node.input[2], venus_nodes, init)
        input_mean = self._to_tensor(node.input[3], venus_nodes, init)
        input_var = self._to_tensor(node.input[4], venus_nodes, init)

        attr = self._get_attribute(node, "epsilon")
        epsilon = attr.f

        return BatchNormalization(
                [],
                [],
                input_shape,
                scale,
                bias,
                input_mean,
                input_var,
                epsilon,
                self.config
            )

    def parse_constant(self, node: NodeProto) -> Node:
        attr = self._get_attribute(node, "value")
        const =  torch.tensor(
            onnx.numpy_helper.to_array(attr.t),
            dtype=self.config.PRECISION,
            device=self.config.DEVICE
        )

        return Constant([], const, self.config)

    def parse_shape(self, node: NodeProto, input_shape:tuple) -> Node:
        start = 0
        attr = self._get_attribute(node, "start")
        if attr is not None:
            start = attr.i

        end = len(input_shape)
        attr = self._get_attribute(node, "end")
        if attr is not None:
            end = attr.i
      
        shape = torch.tensor(input_shape[start: end], dtype=torch.int32)

        return Constant([], shape, self.config)

    def parse_gather(self, node: NodeProto, venus_nodes: list, init: list) -> Node:
        data = self._to_tensor(node.input[0], venus_nodes, init)
        indices = self._to_tensor(node.input[1], venus_nodes, init)
        axis = 0
        attr = self._get_attribute(node, "axis")
        if attr is not None:
            axis = attr.i

        gather = torch.index_select(data, axis, indices.int())

        return Constant([], gather, self.config)

    def parse_cast(self, node: NodeProto, venus_nodes: list, init: list) -> Node:
        data = self._to_tensor(node.input[0], venus_nodes, init)
        to = self._get_attribute(node, "to").i
        if to == 1:
            data = torch.float
        elif to == 2:
            dtype = torch.uint8
        elif to == 3:
            dtype = torch.int8
        elif to == 5:
            dtype = torch.int16
        elif to == 6:
            dtype = torch.int32
        elif to == 7:
            dtype = torch.int64
        elif to == 9:
            dtype = torch.bool
        elif to == 10:
            dtype = torch.float16
        elif to == 11:
            dtype = torch.float64
        else:
            raise TypeError('Data type is not supported')

        return Constant([], data.type(dtype), self.config)


    def parse_unsqueeze(self, node: NodeProto, venus_nodes: list, init: list) -> Node:
        data = self._to_tensor(node.input[0], venus_nodes, init)

        axes = [0]
        attr = self._get_attribute(node, "axes")
        if attr is not None:
            axes = [i for i in attr.ints]

        for i in axes:
            data = torch.unsqueeze(data, i)

        return Constant([], data, self.config)
        
    def parse_slice(self, node: NodeProto, input_shape: tuple, venus_nodes: list, init: list) -> Node:    
        starts = self._to_tensor(node.input[1], venus_nodes, init).int()
        ends = self._to_tensor(node.input[2], venus_nodes, init).int()
        if node.input[3] in venus_nodes:
            axes = self._to_tensor(node.input[3], venus_nodes, init).int()
        else:
            axes = torch.tensor([
                i for i in range(
                    len(input_shape) - 1, len(input_shape) - len(starts) - 1, -1
                )
            ])
        if node.input[4] in venus_nodes:
            steps = self._to_tensor(node.input[4], venus_nodes, init).int()
        else:
            steps = torch.tensor([1 for i in range(len(starts))])

        slices, cur_axis = [], 0
        for i in range(len(input_shape)):
            if i in axes:
                slices.append(slice(starts[cur_axis], ends[cur_axis], steps[cur_axis]))
                cur_axis += 1
            else:
                slices.append(slice(0, input_shape[i]))

        return Slice([], [], input_shape, slices, self.config)


    def parse_concat(self, node: NodeProto, venus_nodes: list, init: list) -> Node:
        axis = self._get_attribute(node, "axis").i
        input_shapes = [list(venus_nodes[i].output_shape) for i in node.input]
        output_shape = input_shapes[0].copy()
        output_shape[axis] = sum([i[axis] for i in input_shapes])
        output_shape = tuple(output_shape)

        return Concat([], [], input_shapes, output_shape, axis, self.config)

    def simplify(self, node: Node):
        return self._simplify(node, {})

    def _simplify(self, node: Node, dic: list):

        if node in dic:
            return {}

        elif isinstance(node, Constant):
            for i in node.from_node:
                i.to_node.remove(node)
                i.to_node.extend(node.to_node)
            for i in node.to_node:
                i.from_node.remove(node)
                i.from_node.extend(node.from_node)

            return self._simplify(node.to_node[0], dic)

        elif isinstance(node, MatMul) and isinstance(node.to_node[0], Add) and \
        len(node.to_node) == 1 and node.to_node[0].const is not None:
            newnode = Gemm(
                [i for i in node.from_node],
                [i for i in node.to_node[0].to_node],
                node.input_shape,
                node.output_shape,
                node.weights,
                node.to_node[0].const,
                self.config
            )
            for i in node.from_node:
                i.to_node.remove(node)
                i.to_node.insert(0, newnode)
            dic[newnode.id] = newnode 
            for i in node.to_node[0].to_node:
                i.from_node.remove(node.to_node[0])
                i.from_node.insert(0, newnode)
                dic = dic | self._simplify(i, dic)

            return dic

        else:
            dic[node.id] = node
            for i in node.to_node:
                dic = dic | self._simplify(i, dic)

            return dic

    def update_depth(self, head: Node) -> None:
        self._update_depth(head, 1)

    def _update_depth(self, node: Node, depth: int) -> None:
        node.depth = max(node.depth, depth)
        for i in node.to_node:
            self._update_depth(i, depth + 1)

    def _to_tensor(self, const_name:str, venus_nodes: list, init: list) -> torch.tensor:
        if const_name in init:
            return torch.tensor(
                onnx.numpy_helper.to_array(init[const_name]),
                dtype=self.config.PRECISION,
                device=self.config.DEVICE
            )

        elif const_name in venus_nodes:
            assert isinstance(venus_nodes[const_name], Constant)
            return venus_nodes[const_name].const

        else:
            raise NameError(f"Could not find node {const_name}")

    def _get_attribute(self, node: NodeProto, name: str) -> AttributeProto:
        for attr in node.attribute:
            if attr.name == name:
                return attr

        return None


    # def process_reshape(self, model, node, from_node, input_shape, depth):
        # output_shape = [t for t in model.graph.initializer if t.name == node.input[1]] 
        # [shape_node] = [c for c in self.consts if c.output[0] == node.input[1]]
        # output_shape = shape_node.attribute
        # output_shape =  np.squeeze(onnx.numpy_helper.to_array(output_shape[0].t)).copy()  
        # output_shape =  (output_shape[2], output_shape[3], output_shape[1])

        # return Reshape(
            # from_node,
            # [],
            # input_shape,
            # output_shape,
            # depth,
            # self.config
        # )

    # def process_transpose(node, input_shape):
        # for att in node.attribute:
            # if att.name == 'perm':
                # perms = [i for i in att.ints]
                # input_shape = tuple([input_shape[i - 1] for i in perms[1:]])

        # return input_shape
