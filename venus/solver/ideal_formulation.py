# ************
# File: ideal_formulation.py
# Top contributors (to current version): 
# 	Panagiotis Kouvaros (panagiotis.kouvaros@gmail.com)
# This file is part of the Venus project.
# Copyright: 2019-2021 by the authors listed in the AUTHORS file in the
# top-level directory.
# License: BSD 2-Clause (see the file LICENSE in the top-level directory).
# Description: Constructs ideal cuts as per Anderson et al. Strong
# Mixed-Integer Programming Formulations for Trained Neural Networks.
# ************

from venus.network.activations import Activations
from venus.solver.cuts import Cuts
from venus.common.logger import get_logger
from timeit import default_timer as timer
from gurobipy import *
import numpy as np


class IdealFormulation(Cuts):

    logger = None

    def __init__(self, prob, gmodel, config):
        """
        Arguments:

            prob: 
                VerificationProblem.
            gmodel:
                Gurobi model.
            config:
                Configuration.
        """
        super().__init__(prob, gmodel, config.SOLVER.IDEAL_FREQ, config)
        if IdealFormulation.logger is None:
            IdealFormulation.logger = get_logger(__name__, config.LOGGER.LOGFILE)


    def add_cuts(self):
        """
        Adds ideal cuts.

        Arguments:

            model: Gurobi model.
        """
        cuts = self.build_cuts()
        ts = timer()
        for lhs, rhs in cuts:
            self.gmodel.cbCut(lhs <= rhs)
        te = timer()
        if len(cuts) > 0:
            self.logger.info(f'Added ideal cuts, #cuts: {len(cuts)}, time: {te - ts}')

    def build_cuts(self):
        ts = timer()
        cuts = [] 
        p_l = self.prob.spec.input_layer
        for l in self.prob.nn.layers:
            if not l.activation == Activations.relu or not self.freq_check(l.depth):
                p_l = l
                continue
            counter = 0
            s, e = self.prob.get_var_indices(l.depth, 'delta')
            delta = self.gmodel._vars[s: e]
            _delta = np.asarray(self.gmodel.cbGetNodeRel(delta)).reshape(l.output_shape)
            delta = np.asarray(delta).reshape(l.output_shape)
            for nd in l.get_outputs():
                if not l.is_stable(nd):
                    if _delta[nd] > 0 and _delta[nd] < 1:
                        ineqs = self.get_inequalities(l, p_l, nd, _delta)
                        if self.cut_condition(ineqs, l, p_l, nd, _delta):
                            lhs, rhs = self.build_constraint(ineqs, l, p_l, nd, delta)
                            cuts.append((lhs,rhs))
            p_l = l
        te = timer()
        if len(cuts) > 0:
            self.logger.info(f'Constructed ideal cuts, #cuts: {len(cuts)}, time: {te - ts}')

        return cuts


    def get_inequalities(self, layer, p_layer, node, _delta):
        """
        Derives set of inequality nodes. See Anderson et al. Strong
        Mixed-Integer Programming Formulations for Trained Neural Networks

        Arguments:

            layer: Layer for deriving ideal cuts.

            p_layer: the previous layer of layer

            node: index of node of layer for deriving ideal cuts.

        Returns:
            
            list of indices of nodes of p_layer.
        """

        (s, e) = self.prob.get_var_indices(p_layer.depth, 'out')
        in_vars = self.gmodel._vars[s:e]
        _in = np.asarray(self.gmodel.cbGetNodeRel(in_vars)).reshape(p_layer.output_shape)
        neighbours = self.prob.nn.neighbours_from_p_layer(layer.depth, node)
        pos_connected = [n for n in neighbours if layer.edge_weight(p_layer,node,n) > 0]
        ineqs = []
        for p_node in pos_connected:
            l = self._get_lb(p_layer, layer, p_node, node)
            u = self._get_ub(p_layer, layer, p_node, node)
            w = layer.edge_weight(p_layer,node,p_node) 
            lhs = w * _in[p_node]
            rhs = w * (l * (1 - _delta[node]) + u * _delta[node])
            if lhs < rhs: ineqs.append(p_node)

        return ineqs

    def cut_condition(self, ineqs, layer, p_layer, node, _delta):
        """
        Checks required  inequality condition  on inequality nodes for adding a
        cut.  See Anderson et al. Strong Mixed-Integer Programming Formulations
        for Trained Neural Networks.
        
        Arguments:
            
            ineqs: list of inequality nodes.

            layer: Layer for deriving ideal cuts.

            p_layer: the previous layer of layer

            node: index of node of layer for deriving ideal cuts.

            _delta: value of the binary variable associated with node.
        
        Returns:

            bool expressing whether or not to add cuts.
        """

        (s, e) = self.prob.get_var_indices(p_layer.depth, 'out')
        in_vars = self.gmodel._vars[s:e]
        _in = np.asarray(self.gmodel.cbGetNodeRel(in_vars)).reshape(p_layer.output_shape)
        (s, e) = self.prob.get_var_indices(layer.depth, 'out')
        out_vars = self.gmodel._vars[s:e]
        _out = np.asarray(self.gmodel.cbGetNodeRel(out_vars)).reshape(layer.output_shape)
        s1 = 0
        s2 = 0

        # for p_node in layer.neighbours(node):
        for p_node in self.prob.nn.neighbours_from_p_layer(layer.depth, node):
            l = self._get_lb(p_layer, layer, p_node, node)
            u = self._get_ub(p_layer, layer, p_node, node)
            if p_node in ineqs:
                s1 += layer.edge_weight(p_layer,node,p_node) * (_in[p_node] - l * (1 - _delta[node]))
            else:
                s2 += layer.edge_weight(p_layer,node,p_node) * u * _delta[node]
        p = layer.get_bias(node) * _delta[node]
        
        return True if _out[node] > p + s1 + s2 else False


    def build_constraint(self, ineqs, layer, p_layer, node, delta):
        """
        Builds the linear cut. See Anderson et al. Strong Mixed-Integer
        Programming Formulations for Trained Neural Networks.

        Arguments:
            
            ineqs: list of inequality nodes.

            layer: Layer for deriving ideal cuts.

            p_layer: the previous layer of layer

            node: index of node of layer for deriving ideal cuts.

            _delta: binary variable associated with node.
        
        Returns:

            a pair of Grurobi linear expression for lhs and the rhs of the
            linear cut.
        """

        (s, e) = self.prob.get_var_indices(p_layer.depth, 'out')
        in_vars = np.asarray(self.gmodel._vars[s:e]).reshape(p_layer.output_shape)
        (s, e) = self.prob.get_var_indices(layer.depth, 'out')
        out_vars = np.asarray(self.gmodel._vars[s:e]).reshape(layer.output_shape)
        le = LinExpr()
        s = 0
        # for p_node in layer.neighbours(node):
        for p_node in self.prob.nn.neighbours_from_p_layer(layer.depth, node):
            l = self._get_lb(p_layer, layer, p_node, node)
            u = self._get_ub(p_layer, layer, p_node, node)
            if p_node in ineqs:
                le.addTerms(layer.edge_weight(p_layer,node,p_node), in_vars[p_node])
                le.addConstant(- l * layer.edge_weight(p_layer,node,p_node))
                le.addTerms(l * layer.edge_weight(p_layer,node,p_node), delta[node])
            else:
                s += layer.edge_weight(p_layer,node,p_node) * u
        le.addTerms(s + layer.get_bias(node), delta[node])

        return out_vars[node], le


    def _get_lb(self, p_l, l, p_n, n):
        """
        Helper function. Given two connected nodes, it returns the upper bound
        of the pointing node if the weight of the connecting edge negative;
        otherwise it returns the lower bound. 

        Arguments:

            p_l, l: two consequtive layers.

            p_n, n: indices of nodes within p_l, l.

        Returns:
                
            float of the lower or upper bound of p_n
        """
        
        if l.edge_weight(p_l,n,p_n) < 0:
            return p_l.post_bounds.upper[p_n]
        else:
            return p_l.post_bounds.lower[p_n]

    def _get_ub(self, p_l, l, p_n, n):
        """
        Helper function. Given two connected nodes, it returns the lower bound
        of the pointing node if the weight of the connecting edge negative;
        otherwise it returns the upper bound. 

        Arguments:

            p_l, l: two consequtive layers.

            p_n, n: indices of nodes within p_l, l.

        Returns:
                
            float of the lower or upper bound of p_n
        """
        if l.edge_weight(p_l,n,p_n) < 0:
            return p_l.post_bounds.lower[p_n] 
        else:
            return p_l.post_bounds.upper[p_n]
