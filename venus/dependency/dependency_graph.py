# ************
# File: dependency_graph.py
# Top contributors (to current version): 
# 	Panagiotis Kouvaros (panagiotis.kouvaros@gmail.com)
# This file is part of the Venus project.
# Copyright: 2019-2021 by the authors listed in the AUTHORS file in the
# top-level directory.
# License: BSD 2-Clause (see the file LICENSE in the top-level directory).
# Description: Dependency graph class.
# ************

import numpy as np
import itertools
import math
import torch


from venus.dependency.dependency_graph_node import DependencyGraphNode
from venus.dependency.dependency_type import DependencyType
from venus.network.node import Gemm
from venus.common.logger import get_logger
from venus.common.utils import ReluState, DFSState
from timeit import default_timer as timer

class DependencyGraph:
    """
    Dependency Graph
    """

    logger = None
    DEP_DEGREE_THRESHOLD = 2

    def __init__(self, nn, intra, inter, config):
        """
        Arguments:

            nn:
                Neural Network.
            intra:
                Whether to use intra-layer dependencies.
            inter:
                Whether to use inter-layer dependencies.
            config:
                Configuration. 
        """

        self.nn = nn
        self.config = config
        self.intra = intra
        self.inter = inter
        self.nodes = {}
        self.inter_deps_count = 0
        self.intra_deps_count = 0
        self.node_count = 0
        if DependencyGraph.logger is None and config.LOGGER.LOGFILE is not None:
            DependencyGraph.logger = get_logger(__name__, config.LOGGER.LOGFILE)

    def build(self, delta_vals=None):
        """
        Builds the dependency graph.

        Arguments: 

            delta_vals: list of current values of Gurobi binary variables;
            required when the dependencies are to be uses as callback cuts.

        Returns:
            
            None
        """
        ts = timer()  
        if self.inter:
            self.build_inter_deps()
        if self.intra:
            self.build_intra_deps()
        te  = timer()

        DependencyGraph.logger.info(
            'Finished building dependency graph, time: {:.2f}, '
            '#intra deps: {}, ' 
            '#inter deps: {}.'.format(
                te-ts,
                self.intra_deps_count,
                self.inter_deps_count
            ))

    def build_inter_deps(self, delta_vals=None):
        """
        Identifies inter-layer dependencies and expresses them in the
        dependency graph.

        Arguments: 

            delta_vals: list of current values of Gurobi binary variables;
            required when the dependencies are to be uses as callback cuts.

        Returns:
            
            None
        """
        for i in range(self.nn.tail.depth): 
            nodes = self.nn.get_node_by_depth(i)
            for j in nodes:
                if j.has_relu_activation() is not True or \
                len(j.to_node[0].to_node) != 1 or \
                j.to_node[0].to_node[0].has_relu_activation() is not True:
                    continue
                v = None if delta_vals is None else delta_vals[j.to_node[0].id]
                n_v = None if delta_vals is None else delta_vals[j.to_node[0].to_node[0].to_node[0].id]
                self._layer_inter_deps(j, j.to_node[0].to_node[0], v, n_v)
                                               

    def _layer_inter_deps(self, n, n_n, v=None, n_v=None):
        """
        Identifies inter-layer dependencies between a pair of consequtive
        layers.

        Arguments:
            
            n:
                a node with relu activatin.
            n_n: 
                a subsequent node with relu activation.
            v:
                list of current values of Gurobi binary variables for node n.
            n_v:
                list of current values of Gurobi binary variables for noden_n.

        Returns:

            None
        """ 
        for j in n_n.to_node[0].get_unstable_indices():
            d = None if n_v is None else n_v[j]
            for i in self.nn.calc_neighbouring_units(n, n_n, j):
                d = None if v is None else v[i]
                if n.to_node[0].is_stable(i, d):
                    continue  
                dep = self._node_inter_deps(n, n_n, i, j)
                if dep is None:
                    continue

                self.add_edge(n.to_node[0].id, i, n_n.to_node[0].id, j, dep)
                self.add_edge(n_n.to_node[0].id, j, n.to_node[0].id, i, DependencyType.inverse(dep))
                self.inter_deps_count += 1

    def _node_inter_deps(self, n, n_n, idx, n_idx):
        """
        Identifies inter-layer dependencies between a pair of units in
        consequtive nodes.

        Arguments:
            
            n: 
                a node with relu activation.
            n_n:
                a subsequent node with relu activation.
            idx:
                the index of the unit in n.
            n_idx:
                the index of the unit in n_n.

        Returns:

            None
        """ 
        weight = n_n.edge_weight(n_idx, idx)
        if weight > 0:
            # inactive -> inactive dependency
            ub = n_n.bounds.upper[n_idx] - weight * n.to_node[0].bounds.upper[idx]
            if ub <= 0:
                return DependencyType.INACTIVE_INACTIVE
        else:
            # inactive -> active dependency
            lb = n_n.bounds.lower[n_idx] - weight * n.to_node[0].bounds.upper[idx]
            if lb >= 0:
                return DependencyType.INACTIVE_ACTIVE

        return None

 
    def build_intra_deps(self, delta_vals=None):
        """
        Identifies intra-layer dependencies and expresses them in the
        dependency graph.

        Arguments: 

            delta_vals: list of current values of Gurobi binary variables;
            required when the dependencies are to be uses as callback cuts.

        Returns:
            
            None
        """
        for i in range(self.nn.tail.depth): 
            nodes = self.nn.get_node_by_depth(i)
            for j in nodes:
                if isinstance(j, Gemm) is not True or j.has_relu_activation() is not True:
                    continue
                dvals = None if delta_vals is None else delta_vals[j.to_node[0].id]
                self._layer_intra_deps(j, dvals)


    def _layer_intra_deps(self, n, v=None):
        """
        Identifies intra-layer dependencies within a layer.
        
        Arguments:
            
            n: 
                a node with relu activation. 
            v: 
                list of current values of Gurobi binary variables for node n.

        Returns:

            None
        """ 
        for (i,j) in list(itertools.combinations(n.to_node[0].get_unstable_indices(), 2)):
            d_i = None if v is None else v[i]
            d_j = None if v is None else v[j]
            dep = self._node_intra_dep(n, i, j)
            if dep is None:
                continue 
            self.add_edge(n.to_node[0].id, i, n.to_node[0].id, j, dep)
            self.add_edge(n.to_node[0].id, j, n.to_node[0].id, i, DependencyType.inverse(dep))
            self.intra_deps_count += 1
    
    def _node_intra_dep(self, n, idx1, idx2):
        """
        Identifies intra-layer dependencies between a pair of units within the
        same node.

        Arguments:
            
            n:
                a node with relu activation.
            idx1:
                the index of the first unit.
            idx2:
                the index of the second unit.
    
        Returns:

            None
        """ 
        bounds = n.from_node[0].bounds
        w = (n.weights[:, idx1].flatten(), n.weights[:, idx2].flatten())
        if n.has_bias() is True:
            b = (n.bias[idx1], n.bias[idx2])
        else:
            b = (0, 0)

        min0, max0 = self._node_intra_dep_helper(w[0], w[1], b[0], b[1], bounds)
        if min0 is None:
            return None

        min1, max1 = self._node_intra_dep_helper(w[1], w[0], b[1], b[0], bounds)
        if min1 is None:
            return None

        if max0 < 0 and max1 < 0:
            return DependencyType.ACTIVE_INACTIVE 

        elif min0 > 0 and min1 > 0:
            return DependencyType.INACTIVE_ACTIVE

        elif max0 < 0 and min1 > 0:
            return DependencyType.ACTIVE_ACTIVE

        elif min0 > 0 and max1 < 0:
            return DependencyType.INACTIVE_INACTIVE

        else:
            return None
 
    def _node_intra_dep_helper(self, w1, w2, b1, b2, bounds):
        """
        Helper for the identification of intra-layer dependencies between a
        pair of nodes within the same layer.

        Arguments:
            
            w1: weight vector for the first node.

            w2: weight vector for the seconf node.

            b1: bias for the first node.

            b2: bias for the second node.

            bounds: Bounds of the nodes from the previous layer having
            connections to both the nodes under analysis.

        Returns:

            None
        """ 
        nonzero_index = 0
        while w1[nonzero_index] == 0 or w2[nonzero_index] == 0:
            nonzero_index += 1
            if nonzero_index == len(w1):
                return None, None

        wp = w1 - (w1[nonzero_index] / w2[nonzero_index]) * w2
        bp = b1 - (w1[nonzero_index] / w2[nonzero_index]) * b2
        weights_plus = np.clip(wp, 0, math.inf)
        weights_minus = np.clip(wp, -math.inf, 0)
        _min = _max = 0
        _min +=  torch.tensordot(
            bounds.lower, weights_plus, dims=([len(bounds.lower.shape) - 1], [0])
        )
        _min += torch.tensordot(
            bounds.upper, weights_minus, dims=([len(bounds.upper.shape) - 1], [0])
        )
        _min += bp
        _max +=  torch.tensordot(
            bounds.upper, weights_plus, dims=([len(bounds.upper.shape) - 1], [0])
        )
        _max +=  torch.tensordot(
            bounds.lower, weights_minus, dims=([len(bounds.upper.shape) - 1], [0])
        )
        _max += bp

        
        # _min +=  weights_plus.dot(bounds.lower) + weights_minus.dot(bounds.upper) + bp
        # _max +=  weights_plus.dot(bounds.upper) + weights_minus.dot(bounds.lower) + bp

        return _min, _max

    def get_total_deps_count(self):
        """
        Returns the total number of dependencies within the dependency graph.
        """
        return self.inter_deps_count + self.intra_deps_count


    def add_node(self, layer, index):
        """
        Adds a node to the graph.
    
        Arguments:
           
            layer: the layer of the node

            index: the index of the node in the layer

        Returns
            
            None
        """
        node = DependencyGraphNode(self.node_count, layer, index)
        self.node_count += 1
        self.nodes[node.id] = node

    def add_edge(self, from_layer, from_index, to_layer, to_index, deptype):
        """
        Adds a directed labeled edge between two nodes. The edge represents a
        dependency between the nodes and the label indicates the type of the
        dependency.

        Arguments:
            
            from_layer: the layer of the node from which the edge is pointing 

            from_index: the index of the node in the from_layer 

            to_layer: the layer of the node to which the edge is pointing at

            to_index: the index of the node in the to_layer 

            deptype: A DependencyType label of the edge

        Returns:
            
            None
        """

        _from = (from_layer, from_index)
        _to = (to_layer, to_index)
        if _from not in self.nodes:
            self.add_node(from_layer, from_index)
        if _to not in self.nodes:
            self.add_node(to_layer, to_index)
        self.nodes[_from].adjacent[_to] = deptype
         

    def reachability_analysis(self):
        """
        Does reachability analysis from each node in the graph to compute the
        dependency degrees of the nodes and to determine whether or not their
        stabilisation leads the dependency graph to inconsistency.
        """
        start = timer()
        dfs_init = {self.nodes[n].id : DFSState.UNVISITED for n in self.nodes}
        for key in self.nodes:
            if self.nodes[key].dep_size[ReluState.ACTIVE] == -1:
                dfs_state = {ReluState.ACTIVE : dfs_init.copy(),
                             ReluState.INACTIVE : dfs_init.copy()}
                self.dfs(self.nodes[key], ReluState.ACTIVE, dfs_state)
            if self.nodes[key].dep_size[ReluState.INACTIVE] == -1:
                dfs_state = {ReluState.ACTIVE : dfs_init.copy(),
                             ReluState.INACTIVE : dfs_init.copy()}
                self.dfs(self.nodes[key], ReluState.INACTIVE, dfs_state)
        # self.logger.info('Reachability analysis finished, time: {:.2f}'.format(timer() - start))

    def dfs(self, u, s, dfs_state):
        """
        DFS from node (u,s) where the dependency degree. The dependency degree
        of the nodes is computed. The validity of the stabilisation of the node
        is determined.

        Arguments:
            
            (u,s): the node in the dependency graph from which dfs is performed.
            
            dfs_state: a list of the DFS states of the nodes.
        
        Returns: 

            the dependency degree of the node if the node is valid; otherwise
            it returns -1.
        """
        if not u.dep_size[s] == -1:
            dfs_state[s][u.id] = DFSState.VISITED
            return u.dep_size[s]
        dfs_state[s][u.id] = DFSState.VISITING
        count = 0
        for key in u.adjacent:
            v = self.nodes[key]
            t = self.get_state_for_dep(u,v,s)
            if not t is None:
                if dfs_state[t][v.id] == DFSState.UNVISITED:
                    c = self.dfs(v,t,dfs_state)
                    count += c + 1
        u.dep_size[s] = max(count,0)
        u.valid[s] = (u.dep_size[s] >  0)
        dfs_state[s][u.id] = DFSState.VISITED

        return count

    def get_state_for_dep(self, u, v, state):
        """
        Determines the state that node v needs to have so that the dependency
        between u and v is satisfied when u has state "state".

        Arguments:

            u, v: Nodes
            
            state: ReluState of u
        
        Returns: 
            
            ReluState of v satisfying the dependency; if no state exists, it
            returns None.
        """

        if state == ReluState.ACTIVE:
            if u.adjacent[v.id] == DependencyType.ACTIVE_ACTIVE:
                return ReluState.ACTIVE
            elif  u.adjacent[v.id] == DependencyType.ACTIVE_INACTIVE:
                return ReluState.INACTIVE
        elif state == ReluState.INACTIVE:
            if u.adjacent[v.id] == DependencyType.INACTIVE_ACTIVE:
                return ReluState.ACTIVE
            elif  u.adjacent[v.id] == DependencyType.INACTIVE_INACTIVE:
                return ReluState.INACTIVE

        return None


    def sort(self):
        """
        Sorts the nodes wrt to their dependency degree.
        """
        self.reachability_analysis()
        lst =  sorted(self.nodes,
                    key=lambda x: self.nodes[x].dep_degree(),
                    reverse=True)
        lst = [lst[i] for i in range(len(lst)) 
               if ((self.nodes[lst[i]].valid[ReluState.ACTIVE]==True) or
                 (self.nodes[lst[i]].valid[ReluState.INACTIVE]==True)) and
             self.nodes[lst[i]].dep_degree()>DependencyGraph.DEP_DEGREE_THRESHOLD]
        # self.logger.info(f'Identified and sorted valid split nodes, #nodes: {len(lst)}')
        return  lst
