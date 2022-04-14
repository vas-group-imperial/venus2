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

from venus.network.node import Node, Relu, Input, Gemm, Conv, Sub, Add, Flatten, MatMul, BatchNormalization
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

    def encode(self):
        """
        Builds a Gurobi Model encoding the  verification problem.
    
        Returns:

            Gurobi Model
        """
        start = timer()

        with gurobipy.Env(empty=True) as env:
            env.setParam('OutputFlag', 0)
            env.setParam('LogToConsole', 0)
            env.start()

            gmodel = Model(env=env)
            self.add_node_vars(gmodel)
            self.add_node_constrs(gmodel)
            self.add_output_constrs(self.prob.nn.tail, gmodel)
            if self.config.SOLVER.INTRA_DEP_CONSTRS is True or \
            self.config.SOLVER.INTER_DEP_CONSTRS is True:
                self.add_dep_constrs(gmodel)

            gmodel.update()
            MILPEncoder.logger.info('Encoded verification problem {} into MILP, time: {:.2f}'.format(self.prob.id, timer() - start))

            return gmodel


    def lp_encode(self):
        """
        constructs the lp encoding of the network
        """

        gmodel = Model()
        layers = [self.prob.spec.input_layer] + self.prob.nn.layers

        # Add LP variables
        for l in layers:
            self.add_node_vars(l, gmodel)
        
        # Add LP constraints
        for i in range(1,len(layers)):
            if isinstance(layers[i],Conv2D):
                preact = self.get_conv_preact(layers[i], layers[i-1].out_vars.reshape(layers[i].input_shape))
            elif isinstance(layers[i], FullyConnected):
                preact = self.get_fc_preact(layers[i], layers[i-1].out_vars.flatten())
            if layers[i].activation == Activations.linear:
                self.add_linear_constrs(layers[i], preact, gmodel)
            else:
                self.add_relu_constrs(layers[i], preact, gmodel, linear_approx=True)

        self.add_output_constrs(layers[-1].out_vars, gmodel)
        gmodel.update()

        return gmodel

    def add_node_vars(self, gmodel):
        """
        Assigns MILP variables for encoding each of the outputs of a given
        node.

        Arguments:

            gmodel:
                The gurobi model
        """
        self.add_output_vars(self.prob.spec.input_node, gmodel)

        for i in range(self.prob.nn.tail.depth + 1):
            nodes = self.prob.nn.get_node_by_depth(i)
            for j in nodes:
                if isinstance(j, Relu):
                    self.add_output_vars(j, gmodel)
                    self.add_relu_delta_vars(j, gmodel)

                elif isinstance(j, Flatten):
                    j.out_vars = j.from_node[0].out_vars.flatten()

                elif type(j) in [Gemm, Conv, Sub, BatchNormalization]:
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
        node.out_vars = np.empty(shape=node.output_shape, dtype=Var)
        for i in node.get_outputs():
            node.out_vars[i] = gmodel.addVar(
                lb=node.bounds.lower[i].item(),
                ub=node.bounds.upper[i].item()
            )
    
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
    
        node.delta_vars = np.empty(shape=node.output_shape, dtype=Var)
        for i in node.get_outputs():
            node.delta_vars[i] = gmodel.addVar(vtype=GRB.BINARY)
            node.delta_vars[i].setAttr(GRB.Attr.BranchPriority, node.depth)

    def add_node_constrs(self, gmodel):
        """
        Computes the output constraints of a node given the MILP variables of its
        inputs. It assumes that variables have already been added.

        Arguments:

            gmodel:
                The gurobi model.
        """
        for i in range(self.prob.nn.tail.depth + 1):
            nodes = self.prob.nn.get_node_by_depth(i)
            for j in nodes:
                if isinstance(j, Relu):
                    self.add_relu_constrs(j, gmodel)

                elif isinstance(j, Flatten):
                    pass

                elif type(j) in [Gemm, Conv, MatMul, Sub, Add, BatchNormalization]:
                    self.add_linear_constrs(j, gmodel)

                else:
                    raise TypeError(f'The MILP encoding of node {j} is not supported')

    

    def add_linear_constrs(self, node: Gemm, gmodel: Model):
        """
        Computes the output constraints of a linar node given the MILP
        variables of its inputs. It assumes that variables have already been
        added.
    
        Arguments:
            
            node: 
                The node. 
            gmodel:
                Gurobi model.
        """
        assert type(node) in [Gemm, Conv, MatMul, Sub, Add, BatchNormalization], f"Cannot compute sub onstraints for {type(j)} nodes."
        
        if type(node) in ['Sub', 'Add'] and node.const is not None:
            output = node.forward_numpy(
                node.from_node[0].out_vars, node.from_node[1].out_vars
            )
 
        else:
            output = node.forward_numpy(node.from_node[0].out_vars)


        for i in node.get_outputs():
            gmodel.addConstr(
                node.out_vars[i] == output[i]
            )
 

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
   
        inp = node.from_node[0].out_vars
        out = node.out_vars
        delta = node.delta_vars
        l = node.from_node[0].bounds.lower
        u = node.from_node[0].bounds.upper

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
     
    def add_output_constrs(self, node: Node, gmodel: Model):
        """
        Creates MILP constraints for the output of the output layer.
   
        Arguments:
            
            node:
                The output node.
            gmodel:
                The gurobi model.
        """
        constrs = self.prob.spec.get_output_constrs(gmodel, node.out_vars)
        for constr in constrs:
            gmodel.addConstr(constr)

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
