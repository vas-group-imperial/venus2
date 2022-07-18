# ************
# File: node_splitter.py
# Top contributors (to current version): 
# 	Panagiotis Kouvaros (panagiotis.kouvaros@gmail.com)
# This file is part of the Venus project.
# Copyright: 2019-2021 by the authors listed in the AUTHORS file in the
# top-level directory.
# License: BSD 2-Clause (see the file LICENSE in the top-level directory).
# Description: Splits a verification problem into subproblems resulting from
# branching on the states of the ReLU nodes by heuristically selecting the
# nodes with the highest dependency degrees.
# ************

import random

from venus.dependency.dependency_graph import DependencyGraph
from venus.verification.verification_problem import VerificationProblem
from venus.split.split_strategy import SplitStrategy, NodeSplitStrategy
from venus.network.node import Relu
from venus.common.utils import DFSState, ReluState
from venus.common.logger import get_logger


class NodeSplitter(object):

    logger = None

    def __init__(self, initial_prob, config):
        """
        Arguments:

            initial_prob:
                VerificationProblem to split.

            config:
                configuration.
        """

        self.initial_prob = initial_prob
        self.config = config
        self.split_queue = [initial_prob]
        self.subprobs = []
        if NodeSplitter.logger is None:
            NodeSplitter.logger = get_logger(__name__, config.LOGGER.LOGFILE)

    def split(self):
        """
        Splits the  verification problem on top of the split_queue into a pair
        of subproblems. Splitting is via branching on the states of the ReLU
        node with the maximum dependency degree.

        Returns:
            
            list of VerificationProblem
        """
        if self.initial_prob.depth >= self.config.SPLITTER.BRANCHING_DEPTH:
            return  []

        if self.config.SPLITTER.BRANCHING_HEURISTIC == 'deps':
            dg = DependencyGraph(
                self.initial_prob.nn,
                self.config.SOLVER.INTRA_DEP_CONSTRS,
                self.config.SOLVER.INTER_DEP_CONSTRS,
                self.config
            )
            dg.build()
            sn = dg.sort()
            avail = len(sn)
        else:
            relu_nd = [
                i.id for _, i in self.initial_prob.nn.node.items() 
                if isinstance(i, Relu) and i.from_node[0].grads is not None
            ]
            avail = self.initial_prob.nn.get_n_non_stabilised_nodes()

        next_node, split_flaf = 0, False

        while len(self.split_queue) > 0:
            prob = self.split_queue.pop()
            if prob.depth >= self.config.SPLITTER.BRANCHING_DEPTH:
                self.add_to_subprobs(prob)
                NodeSplitter.logger.info('Depth cutoff for node splitting reached.')
            elif next_node >= avail:
                self.add_to_subprobs((prob))
                NodeSplitter.logger.info('No available nodes for splitting.')
            else:
                if self.config.SPLITTER.BRANCHING_HEURISTIC == 'deps':
                    node = dg.nodes[sn[next_node]]
                    subprobs = self.split_node_deps(prob, dg, node)
                else:
                    nd = random.choice(relu_nd)
                    while self.initial_prob.nn.node[nd].get_unstable_count() == 0:
                        nd = random.choice(relu_nd)

                    node = self.initial_prob.nn.node[nd].from_node[0].grads.pop()
 
                    while self.initial_prob.nn.node[nd].get_unstable_flag()[node].item() is not True:
                        node = self.initial_prob.nn.node[nd].from_node[0].grads.pop()

                    subprobs = self.split_node_grad(prob,  nd, node)

                    i_strat = self.config.SPLITTER.SPLIT_STRATEGY 
                    n_strat = self.config.SPLITTER.NODE_SPLIT_STRATEGY 
                    if i_strat == SplitStrategy.INPUT_NODE_ALT or \
                    n_strat == NodeSplitStrategy.MULTIPLE_SPLITS:
                        return subprobs

                next_node += 1

                # if len(subprobs) != 0 and prob.check_bound_tightness(subprobs):
                if len(subprobs) != 0:
                    for subprob in subprobs:
                        self.add_to_split_queue(subprob)
                    split_flag = True
                else:
                    self.add_to_subprobs(prob)
       
        return self.subprobs if split_flag is True else []


    def split_node_grad(self, prob, node, index):
        """
        Splits a given verification problem  into a pair of subproblems.
        Splitting is via branching on the states of the ReLU node with the
        maximum gradient.

        Arguments:
            prob:
                VerificationProblem to split.
            dg:
                DependencyGraph build for the initial VerificationProblem.
            node:
                dependency graph node to split.

        Returns: 
            list of VerificationProblem
        """
        prob1 = VerificationProblem(
            prob.nn.copy(),
            prob.spec.copy(),
            prob.depth + 1,
            self.config
        )
        prob2 = VerificationProblem(
            prob.nn.copy(),
            prob.spec.copy(),
            prob.depth + 1,
            self.config
        )

        if prob.nn.node[node].has_custom_relaxation_slope():
            lower, upper = prob.nn.node[node].get_lower_relaxation_slope()
            idxs = prob.nn.node[node].get_unstable_flag().clone()
            idxs[index] = False
            idxs = idxs[prob.nn.node[node].get_unstable_flag()]
            prob1.nn.node[node].set_lower_relaxation_slope(lower[idxs], upper[idxs])
            prob2.nn.node[node].set_lower_relaxation_slope(lower[idxs], upper[idxs])

        if prob.nn.batched is True:
            for i in range(prob.nn.node[node].output_shape[0]):
                idx = (i,) + index[1:]
                prob1.nn.node[node].from_node[0].bounds.lower[idx] = 0
                prob1.nn.node[node].set_state(idx, ReluState.ACTIVE)
                prob2.nn.node[node].bounds.upper[idx] = 0
                prob2.nn.node[node].from_node[0].bounds.upper[idx] = 0
                prob2.nn.node[node].set_state(idx, ReluState.INACTIVE)
            prob1.last_split_strategy = SplitStrategy.NODE
            prob2.last_split_strategy = SplitStrategy.NODE

        else:
            prob1.nn.node[node].from_node[0].bounds.lower[index] = 0
            prob1.nn.node[node].set_state(index, ReluState.ACTIVE)
            prob2.nn.node[node].bounds.upper[index] = 0
            prob2.nn.node[node].from_node[0].bounds.upper[index] = 0
            prob2.nn.node[node].set_state(index, ReluState.INACTIVE)
            prob1.last_split_strategy = SplitStrategy.NODE
            prob2.last_split_strategy = SplitStrategy.NODE
        
        return [prob1, prob2]

    def split_node_deps(self, prob, dg, dgnode):
        """
        Splits a given verification problem  into a pair of subproblems.
        Splitting is via branching on the states of the ReLU node with the
        maximum dependency degrree.

        Arguments:
            prob:
                VerificationProblem to split.
            dg:
                DependencyGraph build for the initial VerificationProblem.
            node:
                dependency graph node to split.

        Returns: 
            list of VerificationProblem
        """
        prob1 = VerificationProblem(
            prob.nn.copy(),
            prob.spec.copy(),
            prob.depth + 1,
            self.config
        )
        prob2 = VerificationProblem(
            prob.nn.copy(),
            prob.spec.copy(),
            prob.depth + 1,
            self.config
        )
        if prob.nn.node[dgnode.nodeid].has_custom_relaxation_slope():
            lower, upper = prob.nn.node[dgnode.nodeid].get_lower_relaxation_slope()
            idxs = prob.nn.node[dgnode.nodeid].get_unstable_flag().clone()
            print(lower.shape, torch.sum(idxs))
            idxs[dgnode.index] = False
            idxs = idxs[prob.nn.node[dgnode.nodeid].get_unstable_flag()]
            print(idxs.shape, torch.sum(idxs))
            prob1.nn.node[dgnode.nodeid].set_lower_relaxation_slope(lower[idxs], upper[idxs])
            prob2.nn.node[dgnode.nodeid].set_lower_relaxation_slope(lower[idxs], upper[idxs])

        prob1.nn.node[dgnode.nodeid].set_dep_root(dgnode.index, True)
        prob1.nn.node[dgnode.nodeid].from_node[0].bounds.lower[dgnode.index] = 0
        prob2.nn.node[dgnode.nodeid].set_dep_root(dgnode.index, True)
        prob2.nn.node[dgnode.nodeid].bounds.upper[dgnode.index] = 0
        prob2.nn.node[dgnode.nodeid].from_node[0].bounds.upper[dgnode.index] = 0
        prob1.last_split_strategy = SplitStrategy.NODE
        prob2.last_split_strategy = SplitStrategy.NODE

        subprobs = []
        if self.set_states(prob1, dg, dgnode, ReluState.ACTIVE):
            subprobs.append(prob1)
        if self.set_states(prob2, dg, dgnode, ReluState.INACTIVE):
            subprobs.append(prob2)

        return subprobs

    def set_states(self, prob, dg, dgnode, state):
        """
        Sets the ReLU states of a given verification problem as per the
        dependency chain origininating from a given node. 

        Arguments:

            prob:
                VerificationProblem.
            dg:
                DependencyGraph build for the initial VerificationProblem.
            dgnode:
                dependency graph node from which the dependency chain originates.
            state:
                ReLU state of node.

        Returns:
            
            bool expressing whether the setting of the states was consistent,
            i.e., it was not the case that a stabilised node should (as per the
            dependency chain) be set to a different state.
        """
        if prob.nn.node[dgnode.nodeid].state[dgnode.index] == state:
            return True

        if prob.nn.node[dgnode.nodeid].state[dgnode.index] == ReluState.inverse(state):
            self.logger.warning(f'Inconsisteny in setting states, layer {l}, node {n}.')
            return False
        
        prob.nn.node[dgnode.nodeid].set_state(dgnode.index, state)

        dgnode.dfs_state[state] = DFSState.VISITING
        for key in dgnode.adjacent:
            u = dg.nodes[key]
            t = dg.get_state_for_dep(dgnode, u, state)
            if not t is None and u.dfs_state[t] == DFSState.UNVISITED:
                if not self.set_states(prob, dg, u, t):
                    return False

        dgnode.dfs_state[state] = DFSState.VISITED

        return True

    def add_to_subprobs(self, prob):
        """
        Adds a verification subproblem to the subproblems list.

        Arguments:
            
            prob:
                VerificationProblem

        Returns:
            
            None
        """

        if prob.depth < 2:
            try: 
                prob.nn.reset_relaxation_slope()
                prob.bound_analysis()
                prob.detach() 
            except Exception as error:
                self.logger.warning(f'Bound analysis failed. {error}')
            
        # prob.bounds_ver_done = True
        self.subprobs = [prob] + self.subprobs
        # self.logger.info(f'Added subproblem {prob.id} to node subproblems list.')

    def add_to_split_queue(self, prob):
        """
        Adds a verification subproblem to the split queue.

        Arguments:
            
            prob:
                VerificationProblem

        Returns:
            
            None
        """
        self.split_queue = [prob] + self.split_queue
        # self.logger.info(f'Added subproblem {prob.id} to node split queue.')

