# ************
# File: osip.py
# Top contributors (to current version): 
# 	Panagiotis Kouvaros (panagiotis.kouvaros@gmail.com)
# This file is part of the Venus project.
# Copyright: 2019-2021 by the authors listed in the AUTHORS file in the
# top-level directory.
# License: BSD 2-Clause (see the file LICENSE in the top-level directory).
# Description: Symbolic Interval Propagaton.
# ************

from timeit import default_timer as timer
from enum import Enum
import numpy as np
import torch
import math

from venus.common.configuration import Config
from venus.bounds.equation import Equation
from venus.common.logger import get_logger
from venus.network.node import Node, Conv, Gemm
from gurobipy import *

class OSIPMode(Enum):
    """
    Modes of operation.
    """
    OFF = 0
    ON = 1
    SPLIT = 2


class OSIP():

    logger = None

    def __init__(self, node: Node, config: Config):
        """
        """
        self.node = node
        self.config = config

        if OSIP.logger is None:
            OSIP.logger = get_logger(__name__, config.LOGGER.LOGFILE)

    
    def optimise(self):
        start = timer()
        opt_nodes = self.get_opt_nodes()
        if isinstance(self.node, Gemm):
            l_slope = self.opt_fc_slope('lower', opt_nodes)
            u_slope = self.opt_fc_slope('upper', opt_nodes)

        elif isinstance(self.node, Conv):
            l_slope = self.opt_conv_slope('lower', opt_nodes)
            u_slope = self.opt_conv_slope('upper', opt_nodes)

        else:
            OSIP.logger.warning('Cannot optimise slopes for nodes other than Gemm and Conv.')
            return 

        if l_slope is not None:
            print('ls', torch.sum(l_slope))

        if l_slope is None or u_slope is None:
            OSIP.logger.info(f'OSIP timelimit reached, using default slope')

        self.node.from_node[0].set_relaxation_slope(l_slope, u_slope)

        OSIP.logger.info(f'OSIP for node {self.node.id} finished, time {timer() - start}')

    def get_opt_nodes(self):  
        opt_nodes = torch.zeros(self.node.output_size, dtype=bool)

        if isinstance(self.node, Conv):
            num_opt_nodes = self.config.SIP.OSIP_CONV_NODES
        elif isinstance(self.node, Gemm):
            num_opt_nodes = self.config.SIP.OSIP_FC_NODES

        eq = Equation.derive(
            self.node, self.node.to_node[0].get_unstable_flag().flatten(), None, self.config
        )
        lb = eq.min_values(
            self.node.from_node[0].bounds.lower.flatten(),
            self.node.from_node[0].bounds.upper.flatten()
        )
        ub = eq.max_values(
            self.node.from_node[0].bounds.lower.flatten(),
            self.node.from_node[0].bounds.upper.flatten()
        )
        
        diff =  ub - lb

        for j in range(min(num_opt_nodes, self.node.output_size)):
            max_diff = np.argmax(diff)
            opt_nodes[max_diff] = True
            diff[max_diff] = -math.inf

        return opt_nodes

    def opt_conv_slope(self, bound, opt_nodes):  
        start = timer()

        unstable = self.node.from_node[0].get_unstable_flag().flatten()
        unstable_count = self.node.from_node[0].get_unstable_count()
        pbl = self.node.from_node[0].from_node[0].from_node[0].bounds.lower.flatten()
        pbu = self.node.from_node[0].from_node[0].from_node[0].bounds.upper.flatten()

        # create model
        model = Model()
        model.setParam('OutputFlag', 0)
        model.setParam('TimeLimit', self.config.SIP.OSIP_TIMELIMIT)
        slopes = np.ones(self.node.from_node[0].output_size, dtype=Var)
        slopes[unstable] = [model.addVar(lb=0, ub=1) for j in range(unstable_count)]

        eq, eq_low, eq_up = self._derive_conv_eqs(bound, opt_nodes)
        eq = self._conv_eq_dot(bound, eq, eq_low, eq_up, slopes)
        # eq = self._conv_eq_dot(bound, eq, eq_low, eq_up, slopes[self.node.from_node[0].get_propagation_flag().flatten()])

        var_pos_lst, var_neg_lst, var_lst = [], [], []
        for i in range(eq.size):
            var_pos, var_neg, var = {}, {}, {}
            for j in eq.matrix[i]:
                var[j] = model.addVar(lb=-GRB.INFINITY)
                var_pos[j] = model.addVar(lb=0, ub=GRB.INFINITY)
                var_neg[j] = model.addVar(lb=-GRB.INFINITY, ub=0)
                model.addConstr(var[j] == eq.matrix[i][j])
                # model.addConstr(var_pos[j] == max_(var[j], 0))
                # model.addConstr(var_neg[j] == min_(var[j], 0))
            var_lst.append(var)
            var_pos_lst.append(var_pos)
            var_neg_lst.append(var_neg)
    
        product = np.zeros(eq.size, dtype=LinExpr)
        for i in range(eq.size):
            if bound == 'upper':
                product[i] = quicksum(
                    [
                        var_pos_lst[i][j] * pbu[j].item() + var_neg_lst[i][j] * pbl[j].item()
                        for j in eq.matrix[i]
                    ]
                )
            else:
                product[i] = quicksum(
                    [
                        var_pos_lst[i][j] * pbl[j].item() + var_neg_lst[i][j] * pbu[j].item()
                        for j in eq.matrix[i]
                    ]
                )
            product[i] += eq.const[i]
    
        if bound == 'upper':
            model.setObjective(quicksum(product), GRB.MINIMIZE)
        else:
            model.setObjective(quicksum(product), GRB.MAXIMIZE)

        model.optimize()
        if model.status == GRB.OPTIMAL:
            return torch.tensor(
                [slopes[i].x for i in unstable.nonzero()]
            )

        return None
    
    def _derive_conv_eqs(self, bound, opt_nodes):
        eq = Equation.derive(
            self.node,
            opt_nodes,
            # self.node.from_node[0].get_propagation_flag().flatten(),
            None,
            self.config,
            sparse=True
        )
        eq.const = eq.const.numpy()

        eq_low = Equation.derive(
            self.node.from_node[0].from_node[0],
            None,
            # self.node.from_node[0].get_propagation_flag().flatten(),
            None,
            self.config,
            sparse=True
        )
        eq_low.const = eq_low.const.numpy()

        upper_matrix, upper_const = eq_low.matrix.copy(), eq_low.const.copy()
        l_bounds = self.node.from_node[0].from_node[0].bounds.lower.flatten()
        u_bounds = self.node.from_node[0].from_node[0].bounds.upper.flatten()  
        # l_bounds = self.node.from_node[0].from_node[0].bounds.lower[
            # self.node.from_node[0].get_propagation_flag()
        # ].flatten()
        # u_bounds = self.node.from_node[0].from_node[0].bounds.upper[
            # self.node.from_node[0].get_propagation_flag()
        # ].flatten()

        for i in range(eq_low.size):
            l_bound, u_bound = l_bounds[i].item(), u_bounds[i].item()

            if u_bound <= 0:
                upper_matrix[i] = {}
                upper_const[i] = 0
                eq_low.matrix[i] = {}
                eq_low.const[i] = 0

            if l_bound < 0 and u_bound > 0:
                adj =  u_bound / (u_bound - l_bound)
                for k in upper_matrix[i]:
                    upper_matrix[i][k] *= adj
                
                if bound == 'lower':
                    upper_const[i] *= adj
                else:
                    upper_const[i] = upper_const[i] * adj - adj * l_bound

        eq_up = Equation(upper_matrix, upper_const, self.config)

        return eq, eq_low, eq_up

    def _conv_eq_dot(self, bound, eq, eq_low, eq_up, slopes):
        matrix, const = [], np.zeros(eq.const.shape, dtype=LinExpr)

        for i in range(eq.size):
            dic = {}
            const[i] = eq.const[i]
            for j in eq.matrix[i]:
                _min = min(eq.matrix[i][j], 0)
                _max = max(eq.matrix[i][j], 0)

                for k in eq_low.matrix[j]:
                    if bound == 'lower':
                        prod = (_max * slopes[j] * eq_low.matrix[j][k]) + (_min * eq_up.matrix[j][k])
                    else:
                        prod = (_min * slopes[j] * eq_low.matrix[j][k]) + (_max * eq_up.matrix[j][k])
                    dic[k] = dic[k] + prod if k in dic else prod

                if bound == 'lower':
                    const[i] += (_max * slopes[j] * eq_low.const[j]) + (_min * eq_up.const[j])
                else:
                    const[i] += (_max * eq_up.const[j]) + (_min * slopes[j] * eq_low.const[j])

            matrix.append(dic)

        return Equation(matrix, const, self.config)


    def opt_fc_slope(self, bound, opt_nodes):
        start = timer()


        unstable = self.node.from_node[0].get_unstable_flag().flatten()
        unstable_count = self.node.from_node[0].get_unstable_count()
        pbl = self.node.from_node[0].from_node[0].from_node[0].bounds.lower.flatten().numpy()
        pbu = self.node.from_node[0].from_node[0].from_node[0].bounds.upper.flatten().numpy()
       
        # create model
        model = Model()
        model.setParam('OutputFlag', 0)
        model.setParam('TimeLimit', self.config.SIP.OSIP_TIMELIMIT)
        slopes = np.ones(self.node.from_node[0].output_size, dtype=Var)
        slopes[unstable] = [model.addVar(lb=0, ub=1) for j in range(unstable_count)] 

        eq, eq_low, eq_up = self._derive_gemm_eqs(bound, opt_nodes)
        eq = self._gemm_eq_dot(bound, eq, eq_low, eq_up, slopes[self.node.from_node[0].get_propagation_flag()])

        var = np.zeros(eq.matrix.shape, dtype=Var)
        var_pos = np.zeros(eq.matrix.shape, dtype=Var)
        var_neg = np.zeros(eq.matrix.shape,dtype=Var)
        idx = itertools.product(*[range(j) for j in eq.matrix.shape])
        for i in idx:
            var[i] = model.addVar(lb=-GRB.INFINITY)
            var_pos[i] = model.addVar(lb=0, ub=GRB.INFINITY)
            var_neg[i] = model.addVar(lb=-GRB.INFINITY, ub=0)
            model.addConstr(var[i] == eq.matrix[i])
            model.addConstr(var_pos[i] == max_(var[i], 0))
            model.addConstr(var_neg[i] == min_(var[i], 0))
        if bound == 'upper':
            product =  var_pos @ pbu + var_neg @ pbl  + eq.const
            model.setObjective(quicksum(product), GRB.MINIMIZE)
        else:
            product =  var_pos @ pbl + var_neg @ pbu  + eq.const
            model.setObjective(quicksum(product), GRB.MAXIMIZE)
        # optimise
        model.optimize()
        if model.status == GRB.OPTIMAL:
            return torch.tensor(
                [slopes[i].x for i in unstable.nonzero()]
            )

        return None


    def _derive_gemm_eqs(self, bound, opt_nodes):
        eq = Equation.derive(
            self.node,
            opt_nodes, 
            self.node.from_node[0].get_propagation_flag().flatten(),
            self.config,
            sparse=True
        )
        eq.matrix, eq.const = eq.matrix.numpy(), eq.const.numpy()

        eq_low = Equation.derive(
            self.node.from_node[0].from_node[0],
            self.node.from_node[0].get_propagation_flag().flatten(),
            None,
            self.config,
            sparse=True
        )
        eq_low.matrix, eq_low.const = eq_low.matrix.numpy(), eq_low.const.numpy()

        upper_matrix, upper_const = eq_low.matrix.copy(), eq_low.const.copy()
        l_bounds = self.node.from_node[0].from_node[0].bounds.lower[
            self.node.from_node[0].get_propagation_flag()
        ].flatten()
        u_bounds = self.node.from_node[0].from_node[0].bounds.upper[
            self.node.from_node[0].get_propagation_flag()
        ].flatten()

        for i in range(eq_low.size):
            l_bound, u_bound = l_bounds[i].item(), u_bounds[i].item()

            if l_bound < 0:
                adj =  u_bound / (u_bound - l_bound)
                upper_matrix[i] *= adj
                
                if bound == 'lower':
                    upper_const[i] *= adj
                else:
                    upper_const[i] = upper_const[i] * adj - adj * l_bound

        eq_up = Equation(upper_matrix, upper_const, self.config)

        return eq, eq_low, eq_up

    def _gemm_eq_dot(self, bound, eq, eq_low, eq_up, slopes):

        eq_low.matrix = eq_low.matrix * np.expand_dims(slopes, axis=1)
        eq_low.const = eq_low.const * slopes
        pos = np.clip(eq.matrix, 0, math.inf)
        neg = np.clip(eq.matrix, -math.inf, 0)

        if bound == 'lower':
            matrix = pos @ eq_low.matrix + neg @ eq_up.matrix
            const = pos @ eq_low.const + neg @ eq_up.const + eq.const
        else:
            matrix = pos @ eq_up.matrix + neg @ eq_low.matrix
            const = pos @ eq_up.const + neg @ eq_low.const + eq.const
        
        return Equation(matrix, const, self.config)

