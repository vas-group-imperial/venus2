# ************
# File: config.py
# Top contributors (to current version): 
# 	Panagiotis Kouvaros (panagiotis.kouvaros@gmail.com)
# This file is part of the Venus project.
# Copyright: 2019-2021 by the authors listed in the AUTHORS file in the
# top-level directory.
# License: BSD 2-Clause (see the file LICENSE in the top-level directory).
# Description: configuration class
# ************

from venus.split.split_strategy import SplitStrategy
from venus.common.utils import ReluApproximation, OSIPMode
import multiprocessing
import torch

class Logger():
    LOGFILE: str = "venus_log.txt"
    SUMFILE: str = "venus_summary.txt"
    VERBOSITY_LEVEL: int = 0

class Solver():
    # Gurobi time limit per MILP in seconds;
    # Default: -1 (No time limit)
    TIME_LIMIT: int = 1800
    # Frequency of Gurobi callbacks for ideal cuts
    # Cuts are added every 1 in pow(milp_nodes_solved, IDEAL_FREQ)
    IDEAL_FREQ: float = 1
    # Frequency of Gurobi callbacks for dependency cuts
    # Cuts are added every 1 in pow(milp_nodes_solved, DEP_FREQ)
    DEP_FREQ: float = 1
    # Whether to use Gurobi's default cuts   
    DEFAULT_CUTS: bool = False
    # Whether to use ideal cuts
    IDEAL_CUTS: bool = True
    # Whether to use inter-depenency cuts
    INTER_DEP_CUTS: bool = True
    # Whether to use intra-depenency cuts
    INTRA_DEP_CUTS: bool = False
    # Whether to use inter-dependency constraints
    INTER_DEP_CONSTRS: bool = True
    # Whether to use intra-dependency constraints
    INTRA_DEP_CONSTRS: bool = True
    # whether to monitor the number of MILP nodes solved and initiate
    # splititng only after BRANCH_THRESHOLD is reached.
    MONITOR_SPLIT: bool = False
    # Number of MILP nodes solved before initiating splitting. Splitting
    # will be initiated only if MONITOR_SPLIT is True.
    BRANCH_THRESHOLD: int = 500
    # Whether to print gurobi output
    PRINT_GUROBI_OUTPUT: bool = False
    # Gurobi feasibility tolerance
    FEASIBILITY_TOL: float = 10e-6

    def callback_enabled(self):
        """
        Returns 

            True iff the MILP SOLVER is using a callback function.
        """
        if self.IDEAL_CUTS or self.INTER_DEP_CUTS or self.INTRA_DEP_CUTS or self.MONITOR_SPLIT:
            return True
        else:
            return False

    def dep_cuts_enabled(self):
        """
        Returns 

            True iff the MILP SOLVER is using dependency cuts.
        """
        if self.INTER_DEP_CUTS or self.INTRA_DEP_CUTS:
            return True
        else:
            return False

    
class Verifier():
    # complete or incomplete verification
    COMPLETE: bool = True
    # number of parallel processes solving subproblems
    # VER_PROC_NUM: int = multiprocessing.cpu_count()
    VER_PROC_NUM: int = multiprocessing.cpu_count()
    # console output
    CONSOLE_OUTPUT: bool = True
    # pgd step size - The epsilon will be divided by this number.
    PGD_EPS: float = 10
    # pgd number of iterations
    PGD_NUM_ITER: int = 10

class Splitter():
    # Maximum  depth for node splitting. 
    BRANCHING_DEPTH: int = 7
    # determinines when the splitting process can idle because there are
    # many unprocessed jobs in the jobs queue
    LARGE_N_OF_UNPROCESSED_JOBS: int = 500
    # sleeping interval for when the splitting process idles
    SLEEPING_INTERVAL: int = 3
    # the number of input dimensions still considered to be small
    # so that the best split can be chosen exhaustively
    SMALL_N_INPUT_DIMENSIONS: int = 6
    # splitting strategy
    SPLIT_STRATEGY: SplitStrategy = SplitStrategy.NODE
    # the stability ratio weight for computing the difficulty of a problem
    STABILITY_RATIO_WEIGHT: float = 1
    # the value of fixed ratio above which the splitting can stop in any
    # case
    STABILITY_RATIO_CUTOFF: float = 0.7
    # the number of parallel splitting processes is 2^d where d is the
    # number of the parameter
    SPLIT_PROC_NUM: int = 2
    # macimum splitting depth
    MAX_SPLIT_DEPTH: int = 1000

class SIP():

    def __init__(self):
        # relu approximation
        self.RELU_APPROXIMATION = ReluApproximation.MIN_AREA
        # optimise memory
        self.OPTIMISE_MEMORY = False
        # formula simplificaton
        self.SIMPLIFY_FORMULA: bool = True
        # whether to use osip for convolutional layers
        self.OSIP_CONV = OSIPMode.OFF
        # number of optimised nodes during osip for convolutional layers
        self.OSIP_CONV_NODES = 200
        # whether to use oSIP for fully connected layers
        self.OSIP_FC = OSIPMode.OFF
        # number of optimised nodes during oSIP for fully connected
        self.OSIP_FC_NODES = 3
        # oSIP timelimit in seconds
        self.OSIP_TIMELIMIT = 7

    def is_osip_enabled(self):
        return self.OSIP_CONV == OSIPMode.ON  or self.OSIP_FC == OSIPMode.ON

    def is_split_osip_enabled(self):
        return self.OSIP_CONV == OSIPMode.SPLIT  or self.OSIP_FC == OSIPMode.SPLIT

    def is_osip_conv_enabled(self):
        return self.OSIP_CONV == OSIPMode.ON

    def is_osip_fc_enabled(self, depth=None):
        return self.OSIP_FC == OSIPMode.ON

    def copy(self):
        sip_cf = SIP()
        sip_cf.RELU_APPROXIMATION = self.RELU_APPROXIMATION
        sip_cf.OSIP_CONV = self.OSIP_CONV
        sip_cf.OSIP_CONV_NODES = self.OSIP_CONV_NODES
        sip_cf.OSIP_FC = self.OSIP_FC
        sip_cf.OSIP_FC_NODES = self.OSIP_FC_NODES
        sip_cf.OSIP_TIMELIMIT = self.OSIP_TIMELIMIT

        return sip_cf


class Config:
    """
    Venus's Parameters
    """

    def __init__(self):
        """
        """
        self.LOGGER = Logger()
        self.SOLVER = Solver()
        self.SPLITTER = Splitter()
        self.VERIFIER = Verifier()
        self.SIP = SIP()
        self.PRECISION = torch.float32
        self.DEVICE = torch.device('cpu')
        self._user_set_params = set()

    def set_param(self, param, value):
        if value is None: return
        self._user_set_params.add(param)
        if param == 'logfile':
            self.LOGGER.LOGFILE = value
        elif param == 'sumfile':
            self.LOGGER.SUMFILE = value
        elif param == 'time_limit':
            self.SOLVER.TIME_LIMIT = int(value)
        elif param == 'intra_dep_constrs':
            self.SOLVER.INTRA_DEP_CONSTRS = value
        elif param == 'intra_dep_cuts':  
            self.SOLVER.INTRA_DEP_CUTS = value
        elif param == 'inter_dep_constrs': 
            self.SOLVER.INTER_DEP_CONSTRS = value
        elif param == 'inter_dep_cuts':
            self.SOLVER.INTER_DEP_CUTS = value
        elif param == 'ideal_cuts':
            self.SOLVER.IDEAL_CUTS = value
        elif param == 'monitor_split':
            self.SOLVER.MONITOR_SPLIT = value
        elif param == 'branching_threshold':
            self.SOLVER.BRANCHING_THRESHOLD = int(value)
        elif param == 'ver_proc_num':
            self.VERIFIER.VER_PROC_NUM = int(value) 
        elif param == 'split_proc_num': 
            self.SPLITTER.SPLIT_PROC_NUM = int(value)
        elif param == 'branching_depth':
            self.SPLITTER.BRANCHING_DEPTH = int(value)
        elif param == 'stability_ratio_cutoff':
            self.SPLITTER.STABILITY_RATIO_CUTOFF = float(value)
        elif param == 'split_strategy':
            if value == 'node':
                self.SPLITTER.SPLIT_STRATEGY = SplitStrategy.NODE
            elif value == 'input':
                self.SPLITTER.SPLIT_STRATEGY = SplitStrategy.INPUT
            elif value == 'inputnodealt':
                self.SPLITTER.SPLIT_STRATEGY = SplitStrategy.INPUT_NODE_ALT
            elif value == 'nodeinput':
                self.SPLITTER.SPLIT_STRATEGY = SplitStrategy.NODE_INPUT
            elif value == 'inputnode':
                self.SPLITTER.SPLIT_STRATEGY = SplitStrategy.INPUT_NODE
            elif value == 'none':
                self.SPLITTER.SPLIT_STRATEGY = SplitStrategy.NONE
        elif param == 'oSIP_conv':
            if value == 'on':
                self.SIP.OSIP_CONV = OSIPMode.ON
            elif value == 'off':
                self.SIP.OSIP_CONV = OSIPMode.OFF
            elif value == 'split':
                self.SIP.OSIP_CONV = OSIPMode.SPLIT
        elif param == 'osip_conv_nodes':
            self.SIP.OSIP_CONV_NODES = int(value) 
        elif param == 'osip_fc':
            if value == 'on':
                self.SIP.OSIP_FC = OSIPMode.ON
            elif value == 'off':
                self.SIP.OSIP_FC = OSIPMode.OFF
            elif value == 'split':
                self.SIP.OSIP_FC = OSIPMode.SPLIT
        elif param == 'osip_fc_nodes':
            self.SIP.OSIP_FC_NODES = int(value) 
        elif param == 'osip_timelimit':
            self.SIP.OSIP_TIMELIMIT = int(value)
        elif param == 'relu_approximation':
            if value == 'min_area':
                self.SIP.RELU_APPROXIMATION = ReluApproximation.MIN_AREA
            elif value == 'identity':
                self.SIP.RELU_APPROXIMATION = ReluApproximation.IDENTITY
            elif value == 'venus':
                self.SIP.RELU_APPROXIMATION = ReluApproximation.VENUS_HEURISTIC
            elif value == 'parallel':
                self.SIP.RELU_APPROXIMATION = ReluApproximation.PARALLEL
            elif value == 'zero':
                self.SIP.RELU_APPROXIMATION = ReluApproximation.ZERO
        elif param == 'complete':
            self.VERIFIER.COMPLETE = value
        elif param == 'console_output':
            self.VERIFIER.CONSOLE_OUTPUT = value
        elif param == 'precision':
            self.PRECISION = value

    def set_param_if_not_set(self, param, value):
        if not param in self._user_set_params:
            self.set_param(param,value)

    def set_nn_defaults(self, nn):
        if nn.is_fc():
            print('asdsada')
            self.set_fc_defaults(nn)
        else: 
            self.set_conv_defaults(nn)

    def set_fc_defaults(self, nn):
        self.set_param_if_not_set('inter_deps', False)
        self.set_param_if_not_set('relu_approximation', 'venus')
        relus = nn.get_n_relu_nodes()
        if nn.head.input_size < 10:
            self.set_param_if_not_set('inter_dep_constrs', False)
            self.set_param_if_not_set('intra_dep_constrs', False)
            self.set_param_if_not_set('inter_dep_cuts', False)
            self.set_param_if_not_set('monitor_split', False)
            self.set_param_if_not_set('stability_ratio_cutoff', 0.75)
            self.set_param_if_not_set('split_strategy', 'input')
            self.set_param_if_not_set('split_proc_num', 2)
        else:   
            self.set_param_if_not_set('split_proc_num', 0)
            self.set_param_if_not_set('inter_dep_constrs', True)
            self.set_param_if_not_set('intra_dep_constrs', True)
            self.set_param_if_not_set('inter_dep_cuts', True)
            self.set_param_if_not_set('monitor_split', True)
            self.set_param_if_not_set('split_strategy', 'node')
            if relus < 1000:
                self.set_param_if_not_set('branching_depth', 2)
                self.set_param_if_not_set('branching_threshold', 10000)
            elif relus < 2000:
                self.set_param_if_not_set('branching_depth', 2)
                self.set_param_if_not_set('branching_threshold', 5000)
            else:
                self.set_param_if_not_set('branching_depth', 7)
                self.set_param_if_not_set('branching_threshold', 300)

    def set_conv_defaults(self, nn):
        self.set_param_if_not_set('stability_ratio_cutoff', 0.9)
        relus = nn.get_n_relu_nodes()
        if relus <= 10000 and len(nn.node) <=5:
            self.set_param_if_not_set('relu_approximation', 'venus')
        else:
            self.set_param_if_not_set('relu_approximation', 'min_area')
        if relus > 4000:
            self.set_param_if_not_set('intra_dep_constrs', False)
            self.set_param_if_not_set('inter_dep_cuts', False)  
        if relus <= 10000:
            self.set_param_if_not_set('branching_depth', 2)
            self.set_param_if_not_set('branching_threshold', 50)
        else:
            self.set_param_if_not_set('monitor_split', False)
            self.set_param_if_not_set('split_strategy', 'none')

    def set_user(self, u_params):
        self.set_param('logfile', u_params.logfile)
        self.set_param('sumfile', u_params.sumfile)
        self.set_param('time_limit', u_params.timeout)
        self.set_param('intra_dep_constrs', u_params.intra_dep_constrs)
        self.set_param('intra_dep_cuts', u_params.intra_dep_cuts)
        self.set_param('inter_dep_constrs', u_params.inter_dep_constrs)
        self.set_param('inter_dep_cuts', u_params.inter_dep_cuts)
        self.set_param('ideal_cuts', u_params.ideal_cuts)
        self.set_param('monitor_split', u_params.monitor_split)
        self.set_param('branching_depth', u_params.branching_depth)
        self.set_param('branching_threshold', u_params.branching_threshold)
        self.set_param('ver_proc_num', u_params.ver_proc_num)
        self.set_param('split_proc_num', u_params.split_proc_num)
        self.set_param('stability_ratio_cutoff', u_params.stability_ratio_cutoff)
        self.set_param('split_strategy', u_params.split_strategy)
        self.set_param('osip_conv', u_params.osip_conv)
        self.set_param('osip_conv_nodes', u_params.osip_conv_nodes)
        self.set_param('osip_fc', u_params.osip_fc)
        self.set_param('osip_fc_nodes', u_params.osip_fc_nodes)
        self.set_param('osip_timelimit', u_params.osip_timelimit)
        self.set_param('relu_approximation', u_params.relu_approximation)
        self.set_param('complete', u_params.complete)
        self.set_param('console_output', u_params.console_output)
