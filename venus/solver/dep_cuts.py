# ************
# File: dep_cuts.py
# Top contributors (to current version): 
# 	Panagiotis Kouvaros (panagiotis.kouvaros@gmail.com)
# This file is part of the Venus project.
# Copyright: 2019-2021 by the authors listed in the AUTHORS file in the
# top-level directory.
# License: BSD 2-Clause (see the file LICENSE in the top-level directory).
# Description: Constructs dependency cuts.
# ************

from venus.network.activations import Activations
from venus.dependency.dependency_graph import DependencyGraph
from venus.solver.cuts import Cuts
from venus.common.logger import get_logger
from timeit import default_timer as timer
from gurobipy import *
import numpy as np


class DepCuts(Cuts):

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
        super().__init__(prob, gmodel, config.SOLVER.DEP_FREQ, config)
        if DepCuts.logger is None:
            DepCuts.logger = get_logger(__name__, config.LOGGER.LOGFILE)

    def add_cuts(self):
        """
        Adds dependency cuts.

        Arguments:

            model: Gurobi model.
        """
        if not self.freq_check():
            return
        # compute runtime bounds
        delta, _delta = self._get_current_delta()
        _delta_flags = [[_delta[i] == 0, _delta[i] == 1] for i in range(len(_delta))]
        self.prob.set_bounds(_delta_flags)
        # get linear desctiption of the current stabilised binary variables
        le = self._get_lin_descr(delta, _delta)
        # build dependency graph and add dependencies
        dg = DependencyGraph(
            self.prob.spec.input_layer,
            self.prob.nn,
            self.config.SOLVER.INTRA_DEP_CUTS,
            self.config.SOLVER.INTER_DEP_CUTS,
            self.config
        )
        ts = timer()
        for lhs_node in dg.nodes:
            for rhs_node, dep in dg.nodes[lhs_node].adjacent:
                # get the nodes in the dependency
                l1 = node1.layer 
                n1 = node1.index
                delta1 = self.prob.nn.layers[l1].delta_vars[n1]
                l2 = node2.layer 
                n2 = node2.index
                delta2 = self.prob.nn.layers[l2].delta_vars[n2]
                # add the constraint as per the type of the dependency
                if dep == DependencyType.INACTIVE_INACTIVE:
                    self.gmodel.cbCut(delta2 <= le + delta1)
                elif dep == DependencyType.INACTIVE_ACTIVE:
                    self.gmodel.cbCut(1 - delta2 <= le + delta1)
                elif dep == DependencyType.ACTIVE_INACTIVE:
                    self.gmodel.cbCut(delta2 <= le + 1 - delta1)
                elif dep == DependencyType.ACTIVE_ACTIVE:
                    self.gmodel.cbCut(1 - delta2 <= le + 1 - delta1)
        te = timer()
        DepCuts.logger.info(f'Added dependency cuts, #cuts: {dg.get_total_deps_count()}, time: {te - ts}')


    def _get_current_delta(self):
        """
        Fetches the binary variables and their current values.

        Arguments:

            model: Gurobi model.

        Returns:
            
            list of all binary variables, list of their current values.
        """
        delta = []
        _delta = []

        for l in self.prob.nn.layers:
            (s, e) = self.prob.get_var_indices(l.depth, 'delta')
            d = self.gmodel._vars[s:e]
            if len(d) > 0:
                _d = np.asarray(self.gmodel.cbGetNodeRel(d)).reshape(l.output_shape)
                d = np.asarray(d).reshape(l.output_shape)
            else:
                _d = d.copy()
            delta.append(d)
            _delta.append(_d)

        return delta, _delta

    def _get_lin_descr(self, delta, _delta):
        """
        Creates a linear expression of current integral binary variables.
        Current dependency cuts are only sound when this expression is
        satisfied so the cuts are added in conjuction with the expression.

        Arguments:

            delta: list of binary variables.

            _delta: list of values of delta.

        Returns:
            
            LinExpr
        """

        le = LinExpr()
        for i in range(len(self.prob.nn.layers)):
            l = self.prob.nn.layers[i]
            if l.activation == Activations.relu:
                d = delta[i]
                _d = _delta[i]
                for j in range(len(d)):
                    if _d[j] == 0 and not l.is_stable(j):
                        le.addTerms(1, d[j])
                    elif _d[j] == 1 and not l.is_stable(j):
                        le.addConstant(1)
                        le.addTerms(-1, d[j])
     
        return le
