# ************
# File: dependency_type.py
# Top contributors (to current version): 
# 	Panagiotis Kouvaros (panagiotis.kouvaros@gmail.com)
# This file is part of the Venus  project.
# Copyright: 2019-2021 by the authors listed in the AUTHORS file in the
# top-level directory.
# License: BSD 2-Clause (see the file LICENSE in the top-level directory).
# Description: Dependency types class.
# ************

from enum import Enum

class DependencyType(Enum):
    """
    Types of dependencies.
    """
    INACTIVE_INACTIVE = 'inactive-inactive'
    INACTIVE_ACTIVE = 'inactive-active'
    ACTIVE_ACTIVE = 'active-active'
    ACTIVE_INACTIVE = 'active-inactive'

    @staticmethod
    def inverse(dep_type):
        """
        Arguments:
            
            dep_type: an instance of DependencyType 
        
        Returns
            
            the symmetric dependency of dep_type
        """
        if dep_type == DependencyType.INACTIVE_INACTIVE:
            return DependencyType.ACTIVE_ACTIVE
        elif dep_type == DependencyType.ACTIVE_ACTIVE:
            return DependencyType.INACTIVE_INACTIVE
        else:
            return dep_type

    @staticmethod
    def antecedent(dep_type):
        """
        Arguments:
        
            dep_type: an instance of DepType 
        
        Returns: 

            the antecedent node state of the dep_type
        """
        if dep_type == DependencyType.INACTIVE_INACTIVE \
        or dep_type == DependencyType.INACTIVE_ACTIVE:
            return ReluState.INACTIVE
        else:
            return ReluState.ACTIVE

    @staticmethod
    def consequent(dep):
        """
        Arguments:
            
            dep_type: an instance of DepType 
        
        Returns: 

            the consequent  node state of the dep_type
        """
        if dep == DependencyType.INACTIVE_INACTIVE \
        or dep == DependencyType.ACTIVE_INACTIVE:
            return ReluState.INACTIVE
        else:
            return ReluState.ACTIVE
