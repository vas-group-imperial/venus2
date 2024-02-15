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

from venus.split.split_strategy import SplitStrategy, NodeSplitStrategy
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
    TIME_LIMIT: int = 7200
    # Frequency of Gurobi callbacks for ideal cuts
    # Cuts are added every 1 in pow(milp_nodes_solved, IDEAL_FREQ)
    IDEAL_FREQ: float = 1
    # Frequency of Gurobi callbacks for dependency cuts
    # Cuts are added every 1 in pow(milp_nodes_solved, DEP_FREQ)
    DEP_FREQ: float = 1
    # Whether to use Gurobi's default cuts
    DEFAULT_CUTS: bool = False
    # Whether to use ideal cuts
    IDEAL_CUTS: bool = False
    # Whether to use inter-depenency cuts
    INTER_DEP_CUTS: bool = False
    # Whether to use intra-depenency cuts
    INTRA_DEP_CUTS: bool = False
    # Whether to use inter-dependency constraints
    INTER_DEP_CONSTRS: bool = False
    # Whether to use intra-dependency constraints
    INTRA_DEP_CONSTRS: bool = False
    # whether to monitor the number of MILP nodes solved and initiate
    # splititng only after BRANCH_THRESHOLD is reached.
    MONITOR_SPLIT: bool = False
    # Number of MILP nodes solved before initiating splitting. Splitting
    # will be initiated only if MONITOR_SPLIT is True.
    BRANCH_THRESHOLD: int = 100
    # Whether to print gurobi output
    PRINT_GUROBI_OUTPUT: bool = False

    def callback_enabled(self):
        """
        Returns True iff the MILP SOLVER is using a callback function.
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
    # whether to use MLP
    MILP: bool = True
    # complete or incomplete verification
    COMPLETE: bool = True
    # number of parallel processes solving subproblems
    # VER_PROC_NUM: int = multiprocessing.cpu_count()
    VER_PROC_NUM: int = multiprocessing.cpu_count()
    # console output
    CONSOLE_OUTPUT: bool = True
    # whether to use lp relaxations
    LP: bool = False
    # whether to try verification via PGD
    PGD: bool = True
    # whether to try verification via PGD on the LP relaxation
    PGD_ON_LP: bool = False
    # pgd step size - The epsilon will be divided by this number.
    PGD_EPS: float = 1
    # pgd number of iterations
    PGD_NUM_ITER: int = 10

class Splitter():
    # Maximum  depth for node splitting.
    BRANCHING_DEPTH: int = 8
    # determinines when the splitting process can idle because there are
    # many unprocessed jobs in the jobs queue
    LARGE_N_OF_UNPROCESSED_JOBS: int = 500
    # sleeping interval for when the splitting process idles
    SLEEPING_INTERVAL: int = 3
    # the number of input dimensions still considered to be small
    # so that the best split can be chosen exhaustively
    SMALL_N_INPUT_DIMENSIONS: int = 16
    # splitting strategy
    SPLIT_STRATEGY: SplitStrategy = SplitStrategy.NODE
    NODE_SPLIT_STRATEGY: NodeSplitStrategy = NodeSplitStrategy.MULTIPLE_SPLITS
    # branching heuristic, either deps or grad
    BRANCHING_HEURISTIC: str = 'deps'
    # the stability ratio weight for computing the difficulty of a problem
    STABILITY_RATIO_WEIGHT: float = 1
    # the value of fixed ratio above which the splitting can stop in any
    # case
    STABILITY_RATIO_CUTOFF: float = 0.8
    # the number of parallel splitting processes is 2^d where d is the
    # number of the parameter
    SPLIT_PROC_NUM: int = 2
    # maximum splitting depth
    MAX_SPLIT_DEPTH: int = 100

class SIP():

    def __init__(self):
        # one step symbolic bounds
        self.ONE_STEP_SYMBOLIC = False
        # symbolic bounds using back-substitution
        self.SYMBOLIC = True
        # whether to concretise bounds during back substitution
        self.CONCRETISATION = False
        # whether to concretise bounds during back substitution using one step
        # equations - for this to work ONE_STEP_SYMBOLIC needs to be true.
        self.EQ_CONCRETISATION = False
        # relu approximation
        self.RELU_APPROXIMATION = ReluApproximation.MIN_AREA
        # formula simplificaton
        self.SIMPLIFY_FORMULA: bool = False
        # whether to use gradient descent optimisation of slopes
        self.SLOPE_OPTIMISATION: bool = False
        # gradient descent learning rate for optimising slopes
        self.GD_LR: float = 1
        # gradient descent steps
        self.GD_STEPS: int = 100
        # STABILITY FLAG THRESHOLD
        self.STABILITY_FLAG_THRESHOLD = 0.0

    def copy(self):
        sip_cf = SIP()
        sip_cf.RELU_APPROXIMATION = self.RELU_APPROXIMATION

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
        # self.BENCHMARK = 'nn4sys'
        # self.BENCHMARK = 'carvana'
        self.BENCHMARK = 'mnistfc'
        # self.BENCHMARK = 'cifar_biasfield'
        # self.BENCHMARK = 'rl_benchmarks'
        # self.BENCHMARK = 'collins_rul_cnn'
        # self.BENCHMARK = 'sri_resnet_a'
        # self.BENCHMARK = 'cifar100_tinyimagenet_resnet'

    def set_param(self, param, value):
        if value is None: return
        self._user_set_params.add(param)
        if param == 'logfile':
            self.LOGGER.LOGFILE = value
        elif param == 'benchmark':
            self.BENCHMARK = value
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
                self.SIP.RELU_APPROXIMATION = ReluApproximation.VENUS
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
            self.set_fc_defaults(nn)
        else:
            self.set_conv_defaults(nn)

    def set_fc_defaults(self, nn):
        self.set_param_if_not_set('inter_deps', False)
        self.set_param_if_not_set('relu_approximation', 'venus')
        relus = nn.get_n_relu_nodes()
        if nn.head[0].input_size < 10:
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
        self.set_param('benchmark', u_params.benchmark)
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


    def set_params(self, n_relus, i_size, batch=1):
        if i_size > 10:
            self.SPLITTER.SPLIT_STRATEGY = SplitStrategy.NODE
            self.SOLVER.MONITOR_SPLIT = True
        else:
            self.SPLITTER.SPLIT_STRATEGY = SplitStrategy.INPUT
            self.SOLVER.MONITOR_SPLIT = False

        self.SOLVER.IDEAL_CUTS = True
        self.SOLVER.INTER_DEP_CUTS = True
        self.SOLVER.INTER_DEP_CONSTRS = True
        if n_relus < 1000:
            self.SPLITTER.BRANCHING_DEPTH = 2
            self.SOLVER.BRANCH_THRESHOLD = 10000
        elif n_relus < 2000:
          self.SPLITTER.BRANCHING_DEPTH = 2
          self.SOLVER.BRANCH_THRESHOLD = 5000
        else:
          self.SPLITTER.BRANCHING_DEPTH = 7
          self.SOLVER.BRANCH_THRESHOLD = 300
        self.VERIFIER.COMPLETE = True
        self.VERIFIER.PGD = True
        self.SIP.ONE_STEP_SYMBOLIC = True
        self.SIP.EQ_CONCRETISATION = False
        self.SIP.SIMPLIFY_FORMULA = True
        if n_relus < 10000:
          self.SIP.SLOPE_OPTIMISATION = True
        else:
          self.SIP.SLOPE_OPTIMISATION = False
        self.SPLITTER.BRANCHING_HEURISTIC = 'deps'

