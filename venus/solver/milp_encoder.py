# ************
# File: milp_encoder.py
# Top contributors (to current version):
# 	Panagiotis Kouvaros (panagiotis.kouvaros@gmail.com)
# This file is part of the Venus project.
# Copyright: 2019-2021 by the authors listed in the AUTHORS file in the
# top-level directory.
# License: BSD 2-Clause (see the file LICENSE in the top-level directory).
# Description: Builds a Gurobi Model encoding a verification problem.
# ************

import torch
import numpy as np
from gurobipy import *

from venus.network.node import *
from venus.dependency.dependency_graph import DependencyGraph
from venus.dependency.dependency_type import DependencyType
from venus.common.utils import ReluState
from venus.verification.verification_problem import VerificationProblem
from venus.common.configuration import Config
from venus.common.logger import get_logger
from timeit import default_timer as timer

class MILPEncoder:

    logger = None

    def __init__(self, prob: VerificationProblem, config: Config):
        """
        Arguments:
            nn:
                NeuralNetwork.
            spec:
                Specification.
            config:
                Configuration
        """

        self.prob = prob
        self.config = config
        if MILPEncoder.logger is None:
            MILPEncoder.logger = get_logger(__name__, config.LOGGER.LOGFILE)

        self._idx_count = 0

    def encode(self, linear_approx=False):
        """
        Builds a Gurobi Model encoding the  verification problem.

        Arguments:
            linear_approx:
                whether to use linear approximation for Relu nodes.
        Returns:
            Gurobi Model.
        """
        start = timer()

        with gurobipy.Env(empty=True) as env:
            env.setParam('OutputFlag', 0)
            env.setParam('LogToConsole', 0)
            env.start()

            gmodel = Model(env=env)

            self.add_node_vars(gmodel, linear_approx)

            gmodel.update()

            self.add_node_constrs(gmodel, linear_approx)
            self.add_output_constrs(self.prob.nn.tail, gmodel)
            deps_cond = linear_approx is not True and \
                (
                    self.config.SOLVER.INTRA_DEP_CONSTRS is True or \
                    self.config.SOLVER.INTER_DEP_CONSTRS is True
                )
            if deps_cond is True:
                self.add_dep_constrs(gmodel)

            gmodel.update()

            MILPEncoder.logger.info(
                'Encoded verification problem {} into {}, time: {:.2f}'.format(
                    self.prob.id,
                    "LP" if linear_approx is True else "MILP",
                    timer() - start
                )
            )

            return gmodel

    def add_node_vars(self, gmodel: Model, linear_approx: bool=False):
        """
        Assigns MILP variables for encoding each of the outputs of a given
        node.

        Arguments:
            gmodel:
                The gurobi model
            linear_approx:
                whether to use linear approximation for Relu nodes.
        """
        self.add_output_vars(self.prob.spec.input_node, gmodel)

        for i in range(self.prob.nn.tail.depth + 1):
            nodes = self.prob.nn.get_node_by_depth(i)
            for j in nodes:
                if j.has_relu_activation() is True:
                    p_idx = j.from_node[-1].get_milp_var_indices()
                    j.set_milp_var_indices(out_start=p_idx[0], out_end=p_idx[1])

                elif isinstance(j, Relu):
                    self.add_output_vars(j, gmodel)
                    if linear_approx is not True:
                        self.add_relu_delta_vars(j, gmodel)

                elif type(j) in [Flatten, Slice, Unsqueeze, Reshape]:
                    j.out_vars = j.forward(j.from_node[0].out_vars)
                    p_idx = j.from_node[0].get_milp_var_indices()
                    j.set_milp_var_indices(out_start=p_idx[0], out_end=p_idx[1])

                elif isinstance(j, Concat):
                    j.out_vars = j.forward([k.out_vars for k in j.from_node])

                elif type(j) in [
                    Gemm, MatMul, Conv, ConvTranspose, Add, Sub, BatchNormalization,
                    MaxPool, AveragePool
                ]:
                    self.add_output_vars(j, gmodel)

                else:
                    raise TypeError(f'The MILP encoding of node {j} is not supported')

    def add_output_vars(self, node: Node, gmodel: Model):
        """
        Creates a real-valued MILP variable for each of the outputs of a given
        node.

        Arguments:
            node:
                The node.
            gmodel:
                The gurobi model
        """
        if node.bounds.size() > 0:

            node.out_vars = np.array(
                gmodel.addVars(
                    int(node.output_size),
                    lb=node.bounds.lower.flatten(),
                    ub=node.bounds.upper.flatten()
                ).values()
            ).reshape(node.output_shape)
        else:
            if isinstance(node, Relu):
                node.out_vars = np.array(
                    gmodel.addVars(
                        (node.output_size.item(),), lb=0, ub=GRB.INFINITY
                    ).values()
                ).reshape(node.output_shape)
            else:
                node.out_vars = np.array(
                    gmodel.addVars(
                        (node.output_size.item(),), lb=-GRB.INFINITY, ub=GRB.INFINITY
                    ).values()
                ).reshape(node.output_shape)

        new_idx = self._idx_count + int(node.output_size)
        node.set_milp_var_indices(out_start=self._idx_count, out_end=new_idx)
        self._idx_count = new_idx


    def add_relu_delta_vars(self, node: Relu, gmodel: Model):
        """
        Creates a binary MILP variable for encoding each of the units in a given
        ReLU node. The variables are prioritised for branching according to the
        depth of the node.

        Arguments:

            node:
                The Relu node.
            gmodel:
                The gurobi model
        """
        assert(isinstance(node, Relu)), "Cannot add delta variables to non-relu nodes."

        node.delta_vars = np.empty(shape=node.output_size, dtype=Var)
        if node.get_unstable_count() > 0:
            node.delta_vars[node.get_unstable_flag().flatten()] = np.array(
                gmodel.addVars(
                    node.get_unstable_count().item(), vtype=GRB.BINARY
                ).values()
            )
        node.delta_vars = node.delta_vars.reshape(node.output_shape)

        new_idx = self._idx_count + node.get_unstable_count().item()
        node.set_milp_var_indices(delta_start=self._idx_count, delta_end=new_idx)
        self._idx_count = new_idx

    def add_node_constrs(self, gmodel, linear_approx: bool=False):
        """
        Computes the output constraints of a node given the MILP variables of its
        inputs. It assumes that variables have already been added.

        Arguments:

            gmodel:
                The gurobi model.
            linear_approx:
                whether to use linear approximation for Relu nodes.
        """
        for i in range(self.prob.nn.tail.depth + 1):
            nodes = self.prob.nn.get_node_by_depth(i)
            for j in nodes:
                if j.has_relu_activation() is True:
                    continue

                elif isinstance(j, Relu):
                    self.add_relu_constrs(j, gmodel, linear_approx)

                elif type(j) in [Flatten, Concat, Slice, Unsqueeze, Reshape]:
                    pass

                elif type(j) in [
                    Gemm, Conv, ConvTranspose, MatMul, Sub, Add, BatchNormalization,
                    AveragePool
                ]:
                    self.add_affine_constrs(j, gmodel)

                elif isinstance(j, MaxPool):
                    self.add_maxpool_constrs(j, gmodel)

                else:
                    raise TypeError(f'The MILP encoding of node {j} is not supported')


    def add_affine_constrs(self, node: Gemm, gmodel: Model):
        """
        Computes the output constraints of an affine node given the MILP
        variables of its inputs. It assumes that variables have already been
        added.

        Arguments:
            node:
                The node.
            gmodel:
                Gurobi model.
        """
        if type(node) not in [Gemm, Conv, ConvTranspose, MatMul, Sub, Add, BatchNormalization]:
            raise TypeError(f"Cannot compute affine onstraints for {type(node)} nodes.")

        if type(node) in ['Sub', 'Add'] and node.const is not None:
            output = node.forward(
                node.from_node[0].out_vars, node.from_node[1].out_vars
            )

        else:
            output = node.forward(node.from_node[0].out_vars)

        for i in node.get_outputs():
            gmodel.addConstr(node.out_vars[i] == output[i])


    def add_relu_constrs(self, node: Relu, gmodel: Model, linear_approx=False):
        """
        Computes the output constraints of a relu node given the MILP variables
        of its inputs.

        Arguments:
            node:
                Relu node.
            gmodel:
                Gurobi model.
        """
        assert(isinstance(node, Relu)), "Cannot compute relu constraints for non-relu nodes."

        if type(node.from_node[0]) in [Add, Sub] and \
        node.from_node[0].const is None:
            inp = node.from_node[0].forward(
                node.from_node[0].from_node[0].out_vars,
                node.from_node[0].from_node[1].out_vars
            )
        else:
            inp = node.from_node[0].forward(node.from_node[0].from_node[0].out_vars)

        out, delta = node.out_vars, node.delta_vars
        l, u = node.from_node[0].bounds.lower, node.from_node[0].bounds.upper

        # if self.test is True:
            # gmodel.addConstrs(out[i] == inp[i] for i in node.get_active_indices())
            # gmodel.addConstrs(out[i] == 0 for i in node.get_inactive_indices())
            # gmodel.addConstrs(inp[i] <= 0 for i in node.get_deproot_indices())
            # idx = node.get_unstable_indices()
            # if linear_approx is True:
                # gmodel.addConstrs(out[i] >= inp[i] for i in idx)
                # gmodel.addConstrs(out[i] >= 0 for i in idx)
                # gmodel.addConstrs(out[i] <= (u[i].item() / (u[i].item() - l[i].item())) * (inp[i] - l[i].item()) for i in idx)
            # else:
                # gmodel.addConstrs(out[i] >= inp[i] for i in idx)
                # gmodel.addConstrs(out[i] <= inp[i] - l[i].item() * (1 - delta[i]) for i in idx)
                # gmodel.addConstrs(out[i] <= u[i].item() * delta[i] for i in idx)

            # return

        for i in node.get_outputs():
            if l[i] >= 0 or node.state[i] == ReluState.ACTIVE:
                # active node as per bounds or as per branching
                gmodel.addConstr(out[i] == inp[i])

            elif u[i] <= 0:
                # inactive node as per bounds
                gmodel.addConstr(out[i] == 0)

            elif node.dep_root[i] == False and node.state[i] == ReluState.INACTIVE:
                # non-root inactive node as per branching
                gmodel.addConstr(out[i] == 0)

            elif node.dep_root[i] == True and node.state[i] == ReluState.INACTIVE:
                # root inactive node as per branching
                gmodel.addConstr(out[i] == 0)
                gmodel.addConstr(inp[i] <= 0)

            else:
                l_i, u_i = l[i].item(), u[i].item()
                # unstable node
                if linear_approx is True:
                    gmodel.addConstr(out[i] >= inp[i])
                    gmodel.addConstr(out[i] >= 0)
                    gmodel.addConstr(out[i] <= (u_i / (u_i - l_i)) * (inp[i] - l_i))
                else:
                    gmodel.addConstr(out[i] >= inp[i])
                    gmodel.addConstr(out[i] <= inp[i] - l_i * (1 - delta[i]))
                    gmodel.addConstr(out[i] <= u_i * delta[i])


    def add_maxpool_constrs(self, node: MaxPool, gmodel: Model):
        """
        Computes the output constraints of a maxpool node given the MILP variables
        of its inputs.

        Arguments:
            node:
                MaxPool node.
            gmodel:
                Gurobi model.
        """
        assert(isinstance(node, MaxPool)), "Cannot compute maxpool constraints for non-maxpool nodes."

        inp = node.from_node[0].out_vars
        padded_inp = Conv.pad(inp, node.pads).reshape((node.in_ch(), 1) + inp.shape[-2:])
        im2col = Conv.im2col(
            padded_inp, node.kernel_shape, node.strides
        )

        idxs = np.arange(node.output_size).reshape(
            node.output_shape_no_batch()
        ).transpose(1, 2, 0).reshape(-1, node.in_ch())

        for i in itertools.product(*[range(j) for j in idxs.shape]):
            gmodel.addConstr(
                np.take(node.out_vars, idxs[i]) == max_(im2col[:, i[0], i[1]].tolist())
            )

        # for i in node.get_outputs():
            # kernel, height, width = i[-3:]
            # win = []
            # for kh, kw in itertools.product(
                    # range(node.kernel_shape[0]), range(node.kernel_shape[1])
            # ):
                # index_h = height * node.kernel_shape[0] + kh
                # index_w = width * node.kernel_shape[1] + kw
                # if node.from_node[0].has_batch_dimension():
                    # index = (0, kernel, index_h, index_w)
                # else:
                    # index = (kernel, index_h, index_w)
                # win.append(inp[index])

            # gmodel.addConstr(node.out_vars[i] == max_(win))


    def add_output_constrs(self, node: Node, gmodel: Model):
        """
        Creates MILP constraints for the output of the output layer.

        Arguments:

            node:
                The output node.
            gmodel:
                The gurobi model.
        """
        constrs = self.prob.spec.get_output_constrs(gmodel, node.out_vars.flatten())
        for i in constrs:
            gmodel.addConstr(i)

    def add_dep_constrs(self, gmodel):
        """
        Adds dependency constraints.

        Arguments:

            gmodel:
                The gurobi model.
        """
        dg = DependencyGraph(
            self.prob.nn,
            self.config.SOLVER.INTRA_DEP_CONSTRS,
            self.config.SOLVER.INTER_DEP_CONSTRS,
            self.config
        )
        dg.build()

        for i in dg.nodes:
            for j in dg.nodes[i].adjacent:
                # get the nodes in the dependency
                lhs_node, lhs_idx = dg.nodes[i].nodeid, dg.nodes[i].index
                delta1 = self.prob.nn.node[lhs_node].delta_vars[lhs_idx]
                rhs_node, rhs_idx = dg.nodes[j].nodeid, dg.nodes[j].index
                delta2 = self.prob.nn.node[rhs_node].delta_vars[rhs_idx]
                dep = dg.nodes[i].adjacent[j]

                # add the constraint as per the type of the dependency
                if dep == DependencyType.INACTIVE_INACTIVE:
                    gmodel.addConstr(delta2 <= delta1)

                elif dep == DependencyType.INACTIVE_ACTIVE:
                    gmodel.addConstr(1 - delta2 <= delta1)

                elif dep == DependencyType.ACTIVE_INACTIVE:
                    gmodel.addConstr(delta2 <= 1 - delta1)

                elif dep == DependencyType.ACTIVE_ACTIVE:
                    gmodel.addConstr(1 - delta2 <= 1 - delta1)
