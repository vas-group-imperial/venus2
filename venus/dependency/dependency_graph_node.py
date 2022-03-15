# ************
# File: dependency_graph_node.py
# Top contributors (to current version): 
# 	Panagiotis Kouvaros (panagiotis.kouvaros@gmail.com)
# This file is part of the Venus  project.
# Copyright: 2019-2021 by the authors listed in the AUTHORS file in the
# top-level directory.
# License: BSD 2-Clause (see the file LICENSE in the top-level directory).
# Description: Dependency graph node class.
# ************

from venus.network.activations import ReluState
from venus.common.utils import DFSState

class DependencyGraphNode(object):
    """
    Node in a dependency graph
    """

    def __init__(self, id, layer, index):
        """
        Arguments: 
            
            id: int of the id of the node.

            layer: the layer the node.

            index: the index of the node in layer.
        """
        self.layer = layer
        self.index = index
        self.id = (layer,index)
        self.adjacent = {}
        self.dfs_state = {ReluState.ACTIVE: DFSState.UNVISITED, 
                          ReluState.INACTIVE: DFSState.UNVISITED}
        self.dep_size = {ReluState.ACTIVE: -1, ReluState.INACTIVE: -1}
        # marks whether the stabilisation of the nodes leads to an inconsistent
        # dependency graph
        self.valid = {ReluState.ACTIVE: True,  ReluState.INACTIVE: True}

    
    def add_adjacent(self, node_id, dep_type):
        """
        Adds a directed labeled edge from the current node to another node. The
        edge represents that the latter depends on the current node via the
        dependency type indicated by the label.
       
        Arguments:

            node_id: the id of the "to" node.
            dep_type: A DependencyType label of the edge.

        Returns:

            None
        """
        self.adjacent[node_id] = dep_type

    def dep_degree(self):
        """
        Returns: 

            an integer of the depedency degree of the node
        """
        return self.dep_size[ReluState.ACTIVE] + self.dep_size[ReluState.INACTIVE]
