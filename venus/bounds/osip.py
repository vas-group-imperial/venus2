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

# from venus.network.layers import FullyConnected, Conv2D
from venus.bounds.equation import Equation
from venus.bounds.mem_equations import MemEquations
from venus.network.activations import Activations
from venus.common.logger import get_logger
from timeit import default_timer as timer
from enum import Enum
import numpy as np
import math


class OSIPMode(Enum):
    """
    Modes of operation.
    """
    OFF = 0
    ON = 1
    SPLIT = 2


class OSIP():

    logger = None

    def __init__(
        self, 
        layer, 
        eq, 
        players, 
        peqs, 
        num_opt_nodes, 
        timelimit, 
        approx,
        logfile, 
        mem_peqs=None):
        """
        Arguments:

        """
        self.layer = layer
        self.input_layer = players[0]
        self.players = players[1:]
        self.eq = eq
        self.peqs = peqs
        self.num_opt_nodes = num_opt_nodes
        self.timelimit = timelimit
        self.approx = approx
        self.logfile = logfile
        self.mem_peqs = mem_peqs
        if OSIP.logger is None:
            OSIP.logger = get_logger(__name__, logfile)

    
    def optimise(self):
        start = timer()
        if len(self.peqs) == 0: 
            return  
        # determine layer to set slope
        l = len(self.players) - 1
        while l >= 0 and self.players[l].activation != Activations.relu:
            l -= 1
        if not self.players[l].activation == Activations.relu \
        or self.peqs[l][0].is_slope_optimised == True:
            return

        opt_nodes = self.get_opt_nodes()
        if isinstance(self.layer, FullyConnected):
            l_slope = self.opt_fc_slope(l, 'lower', opt_nodes)
            u_slope = self.opt_fc_slope(l, 'upper', opt_nodes)
        elif isinstance(self.layer, Conv2D):
            l_slope = self.opt_conv_slope(l, 'lower', opt_nodes)
            u_slope = self.opt_conv_slope(l, 'upper', opt_nodes)
        else:
            OSIP.logger.warning('Cannot optimise slopes for layers other than FullyConnected and Conv2D.')
            return 
  
        if l_slope is None or u_slope is None:
            OSIP.logger.info(f'OSIP timelimit reached, using slope of {self.approx}')
            default_slope = self.peqs[l][0].get_lower_slope(self.players[l].pre_bounds.lower,
                                                            self.players[l].pre_bounds.upper,
                                                            self.approx)
        if l_slope is None: l_slope = default_slope
        if u_slope is None: u_slope = default_slope
        self.peqs[l][0].set_lower_slope(l_slope, u_slope)
        OSIP.logger.info(f'OSIP for layer {self.layer.depth} finished, time {timer() - start}')

    def get_opt_nodes(self):
        opt_nodes = []
        plb = self.players[-1].post_bounds.lower.flatten()
        pub = self.players[-1].post_bounds.upper.flatten()
        lb = self.eq.min_values(plb, pub)
        ub = self.eq.max_values(plb, pub)
        diff =  ub - lb
        for j in range(min(self.num_opt_nodes,self.layer.output_size)):
            max_diff = np.argmax(diff)
            opt_nodes.append(max_diff)
            diff[max_diff] = -math.inf

        return opt_nodes

    def opt_conv_slope(self, layer, bound, opt_nodes):  
        start = timer()
        unstable = self.players[layer].get_unstable()
        if isinstance(self.players[layer], Conv2D):
            unstable = [self.players[layer].flattened_index(i) for i in unstable]
        layers = [self.input_layer] + self.players
        pbl = layers[layer].post_bounds.lower.flatten()
        pbu = layers[layer].post_bounds.upper.flatten()
        # create model
        #
        model = Model()
        model.setParam('OutputFlag', 0)
        model.setParam('TimeLimit', self.timelimit)
        slopes = np.ones(self.mem_peqs[layer][0].size, dtype=Var)
        slopes[unstable] = [model.addVar(lb=0, ub=1) for j in range(len(unstable))]
        cf = (np.array(self.eq.coeffs)[opt_nodes]).tolist()
        c = (np.array(self.eq.const)[opt_nodes]).tolist()
        qdot = MemEquations(cf, c, self.logfile)
        for i in range(len(self.mem_peqs)-1, layer, -1):
            qdot = qdot.dot(bound, self.mem_peqs[i][0], self.mem_peqs[i][1])
        qdot = qdot.dot(bound, self.mem_peqs[layer][0], self.mem_peqs[layer][1], slopes)

        v_p = []
        v_n = []
        v = []
        for k in range(qdot.size):
            d_p = {}
            d_n = {}
            d = {}
            for kp in qdot.coeffs[k]:
                d[kp] = model.addVar(lb=-GRB.INFINITY)
                d_p[kp] = model.addVar(lb=0, ub=GRB.INFINITY)
                d_n[kp] = model.addVar(lb=-GRB.INFINITY, ub=0)
                model.addConstr(d[kp] == qdot.coeffs[k][kp])
                model.addConstr(d_p[kp] == max_(d[kp], 0))
                model.addConstr(d_n[kp] == min_(d[kp], 0))
            v.append(d)
            v_p.append(d_p)
            v_n.append(d_n)
    
        pr = np.zeros(qdot.size, dtype=LinExpr)
        for k in range(qdot.size):
            if bound == 'upper':
                pr[k] = quicksum([v_p[k][kp] * pbu[kp] + v_n[k][kp] * pbl[kp] 
                                  for kp in qdot.coeffs[k]]) 
            else:
                pr[k] = quicksum([v_p[k][kp] * pbl[kp] + v_n[k][kp] * pbu[kp] 
                                  for kp in qdot.coeffs[k]]) 
            pr[k] += qdot.const[k]
    
        if bound == 'upper':
            model.setObjective(quicksum(pr), GRB.MINIMIZE)
        else:
            model.setObjective(quicksum(pr), GRB.MAXIMIZE)

        model.optimize()
        if model.status == GRB.OPTIMAL:
            sl = np.ones((self.peqs[layer][0].size))
            sl[unstable] = [slopes[j].x for j in unstable]
            return sl

        return None

    def opt_fc_slope(self, layer, bound, opt_nodes):
        start = timer()
        # determine layer to set slope
        unstable = self.players[layer].get_unstable()
        if isinstance(self.players[layer], Conv2D):
            unstable = [self.players[layer].flattened_index(i) for i in unstable]
        layers = [self.input_layer] + self.players
        pbl = layers[layer].post_bounds.lower.flatten()
        pbu = layers[layer].post_bounds.upper.flatten()
        # initialise model
        model = Model()
        model.setParam('OutputFlag', 0)
        model.setParam('TimeLimit', self.timelimit)
        # compute dot produce
        slopes = np.ones(self.peqs[layer][0].size, dtype=Var)
        slopes[unstable] = [model.addVar(lb=0, ub=1) for j in range(len(unstable))]
        qdot = Equation(self.eq.coeffs[opt_nodes,:],
                         self.eq.const[opt_nodes],
                         self.logfile)
        for i in range(len(self.peqs)-1, layer, -1):
            qdot = qdot.dot(bound, self.peqs[i][0], self.peqs[i][1])
        qdot = qdot.dot(bound, self.peqs[layer][0], self.peqs[layer][1], slopes)

        # create model
        v = np.zeros(qdot.coeffs.shape,dtype=Var)
        v_p = np.zeros(qdot.coeffs.shape,dtype=Var)
        v_n = np.zeros(qdot.coeffs.shape,dtype=Var)
        x = itertools.product(*[range(j) for j in qdot.coeffs.shape])
        for j in x:
            v[j] = model.addVar(lb=-GRB.INFINITY)
            v_p[j] = model.addVar(lb=0,ub=GRB.INFINITY)
            v_n[j] = model.addVar(lb=-GRB.INFINITY,ub=0)
            model.addConstr(v[j] == qdot.coeffs[j])
            model.addConstr(v_p[j] == max_(v[j],0))
            model.addConstr(v_n[j] == min_(v[j],0))
        if bound == 'upper':
            pr =  v_p @ pbu + v_n @ pbl  + qdot.const
            model.setObjective(quicksum(pr), GRB.MINIMIZE)
        else:
            pr =  v_p @ pbl + v_n @ pbu  + qdot.const
            model.setObjective(quicksum(pr), GRB.MAXIMIZE)
        # optimise
        model.optimize()
        if model.status == GRB.OPTIMAL:
            sl = np.ones(self.peqs[layer][0].size)
            sl[unstable] = [slopes[j].x for j in unstable]
            return sl

        return None

