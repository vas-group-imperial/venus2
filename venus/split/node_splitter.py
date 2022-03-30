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

from venus.dependency.dependency_graph import DependencyGraph
from venus.verification.verification_problem import VerificationProblem
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
        dg = DependencyGraph(
            self.initial_prob.nn,
            self.config.SOLVER.INTRA_DEP_CONSTRS,
            self.config.SOLVER.INTER_DEP_CONSTRS,
            self.config
        )
        dg.build()
        sn = dg.sort()
        next_node = 0
        split_flag = False

        while len(self.split_queue) > 0:
            prob = self.split_queue.pop()
            if prob.depth >= self.config.SPLITTER.BRANCHING_DEPTH:
                self.add_to_subprobs(prob)
                NodeSplitter.logger.info('Depth cutoff for node splitting reached.')
            elif next_node >= len(sn):
                self.add_to_subprobs((prob))
                NodeSplitter.logger.info('No available nodes for splitting.')
            else:
                node = dg.nodes[sn[next_node]]
                next_node += 1
                subprobs = self.split_node(prob, dg, node)

                # if len(subprobs) != 0 and prob.check_bound_tightness(subprobs):
                if len(subprobs) != 0: 
                    for subprob in subprobs:
                        self.add_to_split_queue(subprob)
                    split_flag = True
                else:
                    self.add_to_subprobs(prob)
       
        return self.subprobs if split_flag is True else []


    def split_node(self, prob, dg, dgnode):
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
        subprobs = []

        prob1 = VerificationProblem(
            prob.nn.copy(),
            prob.spec.copy(),
            prob.depth + 1,
            self.config
        )
        prob1.nn.node[dgnode.nodeid].dep_root[dgnode.index] = True
        if self.set_states(prob1, dg, dgnode, ReluState.ACTIVE):
            prob1.bound_analysis()
            subprobs.append(prob1)


        prob2 = VerificationProblem(
            prob.nn.copy(),
            prob.spec.copy(),
            prob.depth + 1,
            self.config
        )

        prob2.nn.node[dgnode.nodeid].dep_root[dgnode.index] = True
        if self.set_states(prob2, dg, dgnode, ReluState.INACTIVE):
            prob2.bound_analysis()
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
            # self.logger.warning(f'Inconsisteny in setting states, layer {l}, node {n}.')
            return False
        
        prob.nn.node[dgnode.nodeid].state[dgnode.index] = state

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

