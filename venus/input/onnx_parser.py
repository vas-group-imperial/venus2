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
from onnx import NodeProto, ModelProto, ValueInfoProto

from venus.common.configuration import Config
from venus.network.node import Node, Gemm, Conv, Relu, Flatten, Sub, Constant, Add, MatMul

class ONNXParser:
    SUPPORTED_NODES = ['Flatten', 'Shape', 'Constant', 'Concat', 'Reshape',
                       'Unsqueeze', 'Gather', 'Relu', 'Gemm', 'Conv',
                       'Transpose', 'MatMul', 'Add', 'Div', 'Sub',
                       'BatchNormalizaton']

    def __init__(self, config: Config):
        self.mean = 0
        self.std = 1
        self.consts = []
        self.config = config


    def load(self, model_path: str):
        venus_nodes = {}
        model = onnx.load(model_path)
        dictnode = {i.output[0]: i for i in model.graph.node} 
        init = {i.name: i for i in model.graph.initializer}
        [inp] = [i for i in model.graph.input if i.name not in init]
        [inp_node] = [i for i in model.graph.node if i.input[0] == inp.name]
        queue = [inp_node]

        while len(queue) != 0:
            node = queue.pop()
            venus_nodes[node.output[0]] = self.parse_node(
                node,
                venus_nodes,
                init,
                inp
            )
            queue.extend([
                dictnode[i.output[0]]
                for i in model.graph.node
                if i.input[0] == node.output[0]
            ])

        lst = self.simplify(venus_nodes[inp_node.output[0]])
        [head] = [i for i in lst if len(i.from_node) == 0]
        [tail] = [i for i in lst if len(i.to_node) == 0]
        self.update_depth(head)

        return head, tail, {i.id: i for i in lst}

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
        if node.input[0] == inp.name:
            input_shape = tuple([i.dim_value for i in inp.type.tensor_type.shape.dim])
        else:
            input_shape = venus_nodes[node.input[0]].output_shape
        # process node
        if node.output[0] in venus_nodes:
            vnode = venus_nodes[node.output[0]]
        else:
            if node.op_type == 'Gemm':
                vnode = self.parse_gemm(node, input_shape, init)

            elif node.op_type == 'MatMul':
                vnode = self.parse_matmul(node, input_shape, init)

            elif node.op_type == 'Conv':
                vnode = self.parse_conv(node, input_shape, init)

            elif node.op_type == 'Relu':
                vnode =  Relu([], [], input_shape, self.config)

            elif node.op_type == 'Flatten':
                vnode = Flatten([], [], input_shape, self.config)

            elif node.op_type == 'Constant':
                vnode = self.parse_constant(node, input_shape, init)

            elif node.op_type == 'Sub':
                vnode = self.parse_sub(node, input_shape, init)

            elif node.op_type == 'Add':
                vnode = self.parse_add(node, input_shape, init)
                
            elif node.op_type == 'BatchNormalizaton':
                vnode = self.parse_batchnormalization(node, input_shape, init)

        # update inputs and outputs
        for i in node.input:
            if i in venus_nodes:
                venus_nodes[i].add_to_node(vnode)
                vnode.add_from_node(venus_nodes[i])

        return vnode

    def parse_gemm(self, node: NodeProto, input_shape:tuple, init: list) -> Node:
        assert node.input[1] in init, \
            "Second input to gemm nodes should be an initializer."
        assert node.input[2] in init, \
            "Third input to gemm nodes should be an initializer."

        [weights] = [
            torch.tensor(
                onnx.numpy_helper.to_array(init[i]),
                dtype=self.config.PRECISION,
                device=self.config.DEVICE
            )
            for i in init if init[i].name == node.input[1]
        ]
        for att in node.attribute:
            if att.name == 'transB' and att.i == 0:
                weights = torch.transpose(weights, 0, 1)
        [bias] = [
            torch.tensor(
                onnx.numpy_helper.to_array(init[i]),
                dtype=self.config.PRECISION,
                device=self.config.DEVICE
            )
            for i in init if init[i].name == node.input[2]
        ]
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

    def parse_matmul(self, node: NodeProto, input_shape: tuple, init: list) -> Node:
        assert node.input[1] in init, \
            "Second input to matmul nodes should be an initializer."

        [weights] = [
            torch.tensor(
                onnx.numpy_helper.to_array(init[i]),
                dtype=self.config.PRECISION,
                device=self.config.DEVICE
            )
            for i in init if init[i].name == node.input[1]
        ]
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

    def parse_conv(self, node: NodeProto, input_shape: tuple, init: list) -> Node:
        assert node.input[1] in init, \
            "Second input to conv nodes should be an initializer."
        assert node.input[2] in init, \
            "Third input to conv nodes should be an initializer."

        [weights] = [
            torch.tensor(
                onnx.numpy_helper.to_array(init[i]),
                dtype=self.config.PRECISION,
                device=self.config.DEVICE
            )
            for i in init if init[i].name == node.input[1]
        ]
        [bias] = [
            torch.tensor(
                onnx.numpy_helper.to_array(init[i]),
                dtype=self.config.PRECISION,
                device=self.config.DEVICE
            )
            for i in init if init[i].name == node.input[2]
        ]
        pads = (0, 0)
        strides = (1, 1)
        for att in node.attribute:
            if att.name == "pads":
                pads = [i for i in att.ints[0:2]]
            elif att.name == "strides":
                strides = [i for i in att.ints[0:2]]
        output_shape = Conv.compute_output_shape(
            input_shape,
            weights.shape + (1,),
            pads,
            strides
        )
        return Conv(
            [],
            [],
            input_shape,
            output_shape,
            weights,
            bias,
            pads,
            strides,
            self.config
        )

    def parse_sub(self, node: NodeProto, input_shape: tuple, init: list) -> Node:
        if node.input[1] in init:
            [const] = [
                torch.tensor(
                    onnx.numpy_helper.to_array(init[i]),
                    dtype=self.config.PRECISION,
                    device=self.config.DEVICE
                )
                for i in init if init[i].name == node.input[1]
            ]
        else:
            const=None

        return Sub([], [], input_shape, self.config, const=const)

    def parse_add(self, node: NodeProto, input_shape: tuple, init: list) -> Node:
        if node.input[1] in init:
            [const] = [
                torch.tensor(
                    onnx.numpy_helper.to_array(init[i]),
                    dtype=self.config.PRECISION,
                    device=self.config.DEVICE
                )
                for i in init if init[i].name == node.input[1]
            ]
        else:
            const = None

        return Add([], [], input_shape, self.config, const=const)

    def parse_batchnormalization(self, node: NodeProto, input_shape: tuple, init: list) -> Node:

        [scale] = [
            onnx.numpy_helper.to_array(init[i]).astype(self.config.PRECISION)
            for i in init if init[i].name == node.input[1]
        ]
        [bias] = [
            onnx.numpy_helper.to_array(init[i]).astype(self.config.PRECISION)
            for i in init if init[i].name == node.input[2]
        ]
        [input_mean] = [
            onnx.numpy_helper.to_array(init[i]).astype(self.config.PRECISION)
            for i in init if init[i].name == node.input[3]
        ]
        [input_var] = [
            onnx.numpy_helper.to_array(init[i]).astype(self.config.PRECISION)
            for i in init if init[i].name == node.input[4]
        ]
        for att in node.attribute:
            if att.name == "epsilon":
                epsilon = tuple(i for i in att.ints[0:2])

        return BatchNormalizaton(
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

    def parse_constant(self, node: NodeProto, init: list) -> Node:
        [const] = [
            torch.tensor(
                onnx.numpy_helper.to_array(init[i]),
                dtype=self.config.PRECISION,
                device=self.config.DEVICE
            )
            for i in init if init[i].name == node.input[1]
        ]

        return Constant([], const, self.config)

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


    def simplify(self, node: Node):
        if isinstance(node, MatMul) and isinstance(node.to_node[0], Add) and \
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
            lst = [newnode]
            for i in node.to_node[0].to_node:
                i.from_node.remove(node.to_node[0])
                i.from_node.insert(0, newnode)
                lst.extend(self.simplify(i))
        else:
            lst = [node]
            for i in node.to_node:
                lst.extend(self.simplify(i))

        return lst

    def update_depth(self, head: Node) -> None:
        self._update_depth(head, 1)

    def _update_depth(self, node: Node, depth: int) -> None:
        node.depth = depth
        for i in node.to_node:
            self._update_depth(i, depth+1)
