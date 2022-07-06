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
        """
        Constructs ideal cuts.
        """
        ts = timer()
        cuts = [] 
        p_l = self.prob.spec.input_node
        for _, i in self.prob.nn.node.items():
            if i.has_relu_activation() is not True or \
            len(i.from_node) != 1 or \
            self.freq_check(i.depth) is not True:
                continue

            counter = 0
            delta, _delta = self.get_var_values(i.to_node[0], 'delta')
            for j in i.to_node[0].get_unstable_indices():
                if _delta[j] > 0 and _delta[j] < 1:
                    ineqs = self.get_inequalities(i, j, _delta)
                    if self.cut_condition(ineqs, i, j, _delta):
                        lhs, rhs = self.build_constraint(ineqs, i, j, delta)
                        cuts.append((lhs,rhs))
        te = timer()

        if len(cuts) > 0:
            self.logger.info(f'Constructed ideal cuts, #cuts: {len(cuts)}, time: {te - ts}')

        return cuts

    def get_inequalities(self, node, unit, _delta):
        """
        Derives set of inequality nodes. See Anderson et al. Strong
        Mixed-Integer Programming Formulations for Trained Neural Networks

        Arguments:

            node:
                node for deriving ideal cuts.
            unit:
                index of the unit in the node.

        Returns:
            list of indices of nodes of p_layer.
        """

        in_vars, _in = self.get_var_values(node.from_node[0], 'out')
        neighbours = self.prob.nn.calc_neighbouring_units(node.from_node[0], node, unit)
        pos_connected = [i for i in neighbours if node.edge_weight(unit, i) > 0]
        ineqs = []
            
        for p_unit in pos_connected:
            l = self._get_lb(node.from_node[0], node, p_unit, unit)
            u = self._get_ub(node.from_node[0], node, p_unit, unit)
            w = node.edge_weight(unit, p_unit) 
            lhs = w * _in[p_unit]
            rhs = w * (l * (1 - _delta[unit]) + u * _delta[unit])
            if lhs < rhs: 
                ineqs.append(p_unit)

        return ineqs

    def cut_condition(self, ineqs, node, unit, _delta):
        """
        Checks required  inequality condition  on inequality nodes for adding a
        cut.  See Anderson et al. Strong Mixed-Integer Programming Formulations
        for Trained Neural Networks.
        
        Arguments:
            
            ineqs:
                list of inequality units.
            node: 
                The node for deriving ideal cuts.
            unit:
                the index of the unit in node.
            _delta:
                the value of the binary variable associated with the unit.
        
        Returns:

            bool expressing whether or not to add cuts.
        """
        in_vars, _in = self.get_var_values(node.from_node[0], 'out')
        out_vars, _out = self.get_var_values(node.to_node[0], 'out')
        s1 = 0
        s2 = 0

        for p_unit in self.prob.nn.calc_neighbouring_units(node.from_node[0], node,  unit):
            l = self._get_lb(node.from_node[0], node, p_unit, unit)
            u = self._get_ub(node.from_node[0], node, p_unit, unit)
            if p_unit in ineqs:
                s1 += node.edge_weight(unit, p_unit) * (_in[p_unit] - l * (1 - _delta[unit]))
            else:
                s2 += node.edge_weight(unit, p_unit) * u * _delta[unit]
           
            if node.has_bias() is True:
                p = node.get_bias(unit) * _delta[unit]
            else:
                p = 0
        
        return bool(_out[unit] > p + s1 + s2)


    def build_constraint(self, ineqs, node, unit, delta):
        """
        Builds the linear cut. See Anderson et al. Strong Mixed-Integer
        Programming Formulations for Trained Neural Networks.

        Arguments:
            
            ineqs:
                list of inequality nodes.
            node: 
                Node for deriving ideal cuts.
            unit: 
                The index of the unit of node.
            _delta:
                The binary variable associated with the unit.
        
        Returns:

            a pair of Grurobi linear expression for lhs and the rhs of the
            linear cut.
        """
        in_vars, _ = self.get_var_values(node.from_node[0], 'out')
        out_vars, _ = self.get_var_values(node.to_node[0], 'out')

        le = LinExpr()
        s = 0
        for p_unit in self.prob.nn.calc_neighbouring_units(node.from_node[0], node, unit):
            l = self._get_lb(node.from_node[0], node, p_unit, unit)
            u = self._get_ub(node.from_node[0], node, p_unit, unit)
            if p_unit in ineqs:
                le.addTerms(node.edge_weight(unit, p_unit), in_vars[p_unit])
                le.addConstant(- l * node.edge_weight(unit, p_unit))
                le.addTerms(l * node.edge_weight(unit, p_unit), delta[unit])
            else:
                s += node.edge_weight(unit, p_unit) * u
       
        if node.has_bias() is True:
            le.addTerms(s + node.get_bias(unit), delta[unit])
        else:
            le.addTerms(s, delta[unit])

        return out_vars[unit], le


    def _get_lb(self, p_n, n, p_idx, idx):
        """
        Helper function. Given two connected nodes, it returns the upper bound
        of the pointing node if the weight of the connecting edge negative;
        otherwise it returns the lower bound. 

        Arguments:

            p_n, n:
                two consequtive nodes.
            p_idx, idx: 
                indices of units in p_n and n.

        Returns:
                
            float of the lower or upper bound of p_idx.
        """
        
        if n.edge_weight(idx, p_idx) < 0:
            return p_n.bounds.upper[p_idx]

        else:
            return p_n.bounds.lower[p_idx]

    def _get_ub(self, p_n, n, p_idx, idx):
        """
        Helper function. Given two connected nodes, it returns the lower bound
        of the pointing node if the weight of the connecting edge negative;
        otherwise it returns the upper bound. 

        Arguments:

            p_n, n:
                two consequtive nodes.
            p_idx, idx: 
                indices of units in p_n and n.

        Returns:
                
            float of the lower or upper bound of p_n
        """
        if n.edge_weight(idx, p_idx) < 0:
            return p_n.bounds.lower[p_idx] 

        else:
            return p_n.bounds.upper[p_idx]
