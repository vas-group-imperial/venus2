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

from venus.dependency.dependency_graph import DependencyGraph
from venus.dependency.dependency_type import DependencyType
from venus.network.node import Relu
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
        _delta_flags = {i: [_delta[i] == 0, _delta[i] == 1] for i in _delta}
        self.prob.set_bounds(delta_flags=_delta_flags)
        # get linear desctiption of the current stabilised binary variables
        le = self._get_lin_descr(delta, _delta)
        # build dependency graph and add dependencies
        dg = DependencyGraph(
            self.prob.nn,
            self.config.SOLVER.INTRA_DEP_CUTS,
            self.config.SOLVER.INTER_DEP_CUTS,
            self.config
        )

        ts = timer()
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
        delta, _delta = {}, {}

        for _, i in self.prob.nn.node.items():
            if isinstance(i, Relu):
                d, _d = self.get_var_values(i.to_node[0], 'delta')
                delta[i.id] = d
                _delta[i.id] = _d

        return delta, _delta

    def _get_lin_descr(self, delta, _delta):
        """
        Creates a linear expression of current integral binary variables.
        Current dependency cuts are only sound when this expression is
        satisfied so the cuts are added in conjuction with the expression.

        Arguments:

            delta:
                dict of binary variables.
            _delta:
                dict of values of delta.

        Returns:
            
            LinExpr
        """
        le = LinExpr()

        for _, i in self.prob.nn.node.items():
            if i.has_relu_activation() is True:
                d = delta[i.to_node[0].id]
                _d = _delta[i.to_node[0].id]
                for j in range(len(d)):
                    if _d[j] == 0 and i.to_node[0].is_stable(j) is not True:
                        le.addTerms(1, d[j])
                    elif _d[j] == 1 and i.to_node[0].is_stable(j) is not True:
                        le.addConstant(1)
                        le.addTerms(-1, d[j])

        return le
