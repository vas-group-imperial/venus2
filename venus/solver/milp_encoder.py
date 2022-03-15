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

from gurobipy import *
from venus.network.layers import Input, FullyConnected, Conv2D, GlobalAveragePooling
from venus.dependency.dependency_graph import DependencyGraph
from venus.network.activations import Activations, ReluState
from venus.common.logger import get_logger
from timeit import default_timer as timer
import numpy as np

class MILPEncoder:

    logger = None
    
    def __init__(self, prob, config):
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
            layers = [self.prob.spec.input_layer] + self.prob.nn.layers
            
            # Add MILP variables
            for l in layers:
                self.add_node_vars(l, gmodel)
                if l.activation == Activations.relu:
                    self.add_relu_delta_vars(l, gmodel)
            # Add MILP constraints
            for i in range(1,len(layers)):
                if isinstance(layers[i],GlobalAveragePooling):
                    self.add_global_average_pooling_constrs(layers[i],layers[i-1].out_vars,gmodel)
                else:
                    if isinstance(layers[i],Conv2D):
                        preact = self.get_conv_preact(layers[i], layers[i-1].out_vars.reshape(layers[i].input_shape))
                    elif isinstance(layers[i],FullyConnected):
                        preact = self.get_fc_preact(layers[i], layers[i-1].out_vars.flatten())
                    if layers[i].activation == Activations.linear:
                        self.add_linear_constrs(layers[i], preact, gmodel)
                    else:
                        self.add_relu_constrs(layers[i], preact, gmodel)
            self.add_output_constrs(layers[-1].out_vars, gmodel)
            # Add dependency constraints
            if self.config.SOLVER.INTRA_DEP_CONSTRS or self.config.SOLVER.INTER_DEP_CONSTRS:
                dg = DependencyGraph(
                    self.prob.spec.input_layer,
                    self.prob.nn, 
                    self.config.SOLVER.INTRA_DEP_CONSTRS,
                    self.config.SOLVER.INTER_DEP_CONSTRS,
                    self.config
                )
                self.add_dep_constrs(dg, gmodel) 
                dg.build()
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
     
    def add_node_vars(self, layer, gmodel):
        """
        Creates a real-valued MILP variable for each of the nodes of a given
        layer
   
        Arguments:
            
            layer: any layer  apart from output
            
            gmodel: gurobi model
    
        Returns: 

            None
        """
 
        layer.out_vars = np.empty(shape=layer.output_shape, dtype=Var)
        for i in layer.get_outputs():
            layer.out_vars[i] = gmodel.addVar(lb=layer.post_bounds.lower[i], ub=layer.post_bounds.upper[i])
    
    def add_relu_delta_vars(self, layer, gmodel):
        """
        Creates a binary MILP variable for encoding each of the nodes in a given
        ReLU layer. The variables are prioritised for branching according to the
        depth of the layer.
   
        Arguments: 
        
            layer: any layer with ReLU activation
            
            gmodel: gurobi model
    
        Returns: 

            None
        """
        assert(layer.activation == Activations.relu)
    
        layer.delta_vars = np.empty(shape=layer.output_shape,dtype=Var)
        for i in layer.get_outputs():
            layer.delta_vars[i] = gmodel.addVar(vtype=GRB.BINARY)
            layer.delta_vars[i].setAttr(GRB.Attr.BranchPriority, layer.depth)
   
   
    def get_fc_preact(self, layer, in_vars):
        """
        Computes the pre-activation of a dense layer given the MILP variables of
        the activation of the previous layer.
    
        Arguments:
            
            layer: dense layer.
        
            in_vars: vector of the MILP variables of  the activations of the
            previous layer.
    
        Returns: 

            Vector of the pre-activations of the layer.
        """
        assert(isinstance(layer,FullyConnected))
    
        return layer.weights.dot(in_vars) + layer.bias
    
    def get_conv_preact(self, layer, in_vars):
        """
        Computes the pre-activation of a convolutional layer given the MILP
        variables of the activation of the previous layer.
    
        Arguments:  
            
            layer: convolutional layer.
        
            in_vars: matrix of the MILP variables of  the activations of the
            previous layer.
    
        Returns: 

            Matrix of the pre-activations of the layer.
        """
        assert(isinstance(layer,Conv2D))
        KS = layer.kernels.shape[0:-2]
        inp = Conv2D.pad(in_vars, layer.padding)
        inp_strech = Conv2D.im2col(inp, KS, layer.strides)
        kernel_strech = np.array( [ layer.kernels[:,:,:,i].flatten() for i in range(layer.kernels.shape[-1])],dtype='float64')
        conv = kernel_strech.dot(inp_strech).T.reshape(layer.output_shape) + layer.bias

        return conv

    def add_linear_constrs(self, layer, preact, gmodel): 
        """
        Creates MILP constraints for the output of a layer with linear
        activations.
   
        Arguments:
        
            layer: layer with linear activation.
            
            preact: array of MILP expressions of the pre-activation of the
            layer.
    
        Returns: 
            
            None
        """
        assert(layer.activation == Activations.linear)
    
        for i in layer.get_outputs():
            gmodel.addConstr(layer.out_vars[i] == preact[i])
    
    def add_relu_constrs(self, layer, preact, gmodel,
                         linear_approx=False):
        """
        Creates MILP constraints for the output of a layer with relu
        activations
   
        Arguments: 
            
            layer: layer with relu activation.
        
            preact: array of MILP expressions of the pre-activation of the
            layer.
    
        Returns: 

            None
        """
        assert(layer.activation == Activations.relu)
    
        out = layer.out_vars
        delta = layer.delta_vars
        l = layer.pre_bounds.lower
        u = layer.pre_bounds.upper
        for i in layer.get_outputs():
            if l[i] >= 0 or layer.state[i] == ReluState.ACTIVE:
                # active node as per bounds or as per branching
                gmodel.addConstr(out[i] == preact[i])
            elif u[i] <= 0:
                # inactive node as per bounds
                gmodel.addConstr(out[i] == 0)
            elif layer.dep_root[i]==False and layer.state[i]==ReluState.INACTIVE:
                # non-root inactive node as per branching
                gmodel.addConstr(out[i] == 0)
            elif layer.dep_root[i]==True and layer.state[i] == ReluState.INACTIVE:
                # root inactive node as per branching
                gmodel.addConstr(out[i] == 0)
                gmodel.addConstr(preact[i] <= 0)
            else:
                # unstable node
                if linear_approx:
                    gmodel.addConstr(out[i] >= preact[i])
                    gmodel.addConstr(out[i] >= 0)
                    gmodel.addConstr( out[i] <=
                                (u[i]/(u[i]-l[i]))*(preact[i]-l[i]) )
                else:
                    gmodel.addConstr(out[i] >= preact[i])
                    gmodel.addConstr(out[i] <= preact[i] - l[i] * (1 - delta[i]))
                    gmodel.addConstr(out[i] <= u[i] * delta[i])
    
    
    def add_global_average_pooling_constrs(self, layer, inp, gmodel):
        """
        Creates MILP constraints for a global average pooling layer.  [it is
        assumed that inp is a four dimensional output of a convulutional layer]
   
        Arguments: 

            layer: global average pooling layer.
            
            inp: array of MILP expressions of the input to the layer.
    
        Returns: 

            None
        """
        for i in range(layer.output_shape[-1]):
            gmodel.addConstr(layer.out_vars[i] == np.average(inp[:,:,i]))
    
    def add_output_constrs(self, out_vars, gmodel):
        """
        Creates MILP constraints for the output of the output layer.
   
        Arguments:
            
            out_vars: np.araat of output MILP variables of the output layer.
            
            gmodel: gurobi model
    
        Returns: 

            None
        """
        constrs = self.prob.spec.get_output_constrs(gmodel, out_vars)
        for constr in constrs:
            gmodel.addConstr(constr)

    def add_dep_constrs(self, dg, gmodel):
        """
        Adds dependency constraints.

        Arguments:

            dg: dependency graph.

        Returns:

            None
        """
        for lhs_node in dg.nodes:
            for rhs_node, dep in dg.nodes[lhs_node].adjacent:
                # get the nodes in the dependency
                l1 = node1.layer 
                n1 = node1.index
                delta1 = self.prob.nn.layers[l1].delta_vars[n1]
                l2 = node2.layer 
                n2 = node2.index
                delta1 = self.prob.nn.layers[l2].delta_vars[n2]
                # add the constraint as per the type of the dependency
                if dep == DependencyType.INACTIVE_INACTIVE:
                    gmodel.addConstr(delta2 <= delta1)
                elif dep == DepType.INACTIVE_ACTIVE:
                    gmodel.addConstr(1 - delta2 <= delta1)
                elif dep == DepType.ACTIVE_INACTIVE:
                    gmodel.addConstr(delta2 <= 1 - delta1)
                elif dep == DepType.ACTIVE_ACTIVE:
                    gmodel.addConstr(1 - delta2 <= 1 - delta1)

