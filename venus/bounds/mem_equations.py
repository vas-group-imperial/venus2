# ************
# File: mem_quations.py
# Top contributors (to current version): 
# 	Panagiotis Kouvaros (panagiotis.kouvaros@gmail.com)
# This file is part of the Venus  project.
# Copyright: 2019-2021 by the authors listed in the AUTHORS file in the
# top-level directory.
# License: BSD 2-Clause (see the file LICENSE in the top-level directory).
# Description:  The equation class provides a memory efficient representation
# for  the bound equations and implements  relu relaxations and dot products on
# them. 
# ************

from venus.network.activations import ReluApproximation
import numpy as np

class MemEquations:

    def __init__(self, coeffs, const):
        """
        Arguments:

            coeffs: list of size i of dictionaries of pairs (k,v) where k is the
                    index an equation variable and v is its coefficient.
            
            const:  vector (of size n). Each row i represents node's i equation
                    constant term
        """
        self.coeffs = coeffs
        self.const = const
        self.size = len(coeffs)
        self.lower_bounds = None
        self.upper_bounds = None

    def copy(self):
        """
        Returns: 

            a copy of the calling object 
        """
        return Equations(self.coeffs.copy(), self.const.copy())

    def max_values(self, lower, upper):
        """
        Computes the upper bounds of the equations.
    
        Arguments:

            lower: vector of the lower bounds for the variables in the equation
            
            upper: vector of the upper bounds for the variables in the equation
        
        Returns: 
            
            a vector for the upper bounds of the equations 
        """
        if not self.upper_bounds is None:
            return self.upper_bounds

        self.upper_bounds = np.zeros(shape=self.size,dtype='float64')
        for i in range(self.size):
            self.upper_bounds[i] = sum((max(self.coeffs[i][k],0) * upper[k]) + (min(self.coeffs[i][k],0) * lower[k])  for k in self.coeffs[i] )  + self.const[i]

 
        return self.upper_bounds

    def min_values(self, lower, upper):
        """
        Computes the lower bounds of the equations 

        Arguments:
            
            lower: vector of the lower bounds for the variables in the equation
            
            upper: vector of the upper bounds for the variables in the equation
        
        Returns: 

            a vector of upper bounds of the equations 
        """  
        if not self.lower_bounds is None:
            return self.lower_bounds

        self.lower_bounds = np.zeros(shape=self.size,dtype='float64')
        for i in range(self.size):
            self.lower_bounds[i] = sum((max(self.coeffs[i][k],0) * lower[k]) + (min(self.coeffs[i][k],0) * upper[k]) for k in self.coeffs[i]) + self.const[i]

        return self.lower_bounds


    def lower_relu_relax(self, 
                         lower=None, 
                         upper=None, 
                         approx=ReluApproximation.MIN_AREA):
        """
        Derives the ReLU lower linear relaxation of the equations

        Arguments:

            lower: vector of lower bounds of the equations
            
            upper: vector of upper bounds of the equations

            approx: ReluApproximation

        Returns:

            Equations of the relaxation.
        """
        if lower is None:
            if self.lower_bounds is None:
                raise Exception("Missing lower bounds")
            else:
                lower = self.lower_bounds
        if upper is None:
            if self.upper_bounds is None:
                raise Exception("Missing upper bounds")
            else:
                upper = self.upper_bounds

        coeffs = [eq.copy() for eq in self.coeffs]
        const = self.const.copy()
        # compute the coefficients of the linear approximation of out bound
        # equations
        for i in range(self.size):
            if  lower[i] >= 0:
                # Active node - Propagate lower bound equation unaltered
                pass
            elif upper[i] <= 0: 
                # Inactive node - Propagate the zero function
                coeffs[i] = {}
                const[i] =  0
            else:
                # Unstable node - Propagate linear relaxation of lower bound
                # equations
                # 
                if approx == ReluApproximation.ZERO:
                    coeffs[i] = {}
                    const[i] = 0
                elif approx == ReluApproximation.IDENTITY:
                    pass
                elif approx == ReluApproximation.PARALLEL:
                    coeffs[i], const[i] = self.parallel(coeffs[i],
                                                        const[i],
                                                        lower[i], 
                                                        upper[i], 
                                                        'lower') 
                elif approx == ReluApproximation.MIN_AREA:
                    coeffs[i], const[i] = self.min_area(coeffs[i],
                                                        const[i],
                                                        lower[i], 
                                                        upper[i])
                elif approx == ReluApproximation.VENUS_HEURISTIC:
                    coeffs[i], const[i] = self.venus_heuristic(coeffs[i],
                                                               const[i],
                                                               lower[i], 
                                                               upper[i])
                else:
                    pass


        return MemEquations(coeffs, const)

    def upper_relu_relax(self, lower=None, upper=None):
        """
        Derives the ReLU upper linear relaxation of the equations

        Arguments:

            lower: vector of lower bounds of the equations
            
            upper: vector of upper bounds of the equations

        Returns:

            Equations of the relaxation.
        """

        if lower is None:
            if self.lower_bounds is None:
                raise Exception("Missing lower bounds")
            else:
                lower = self.lower_bounds
        if upper is None:
            if self.upper_bounds is None:
                raise Exception("Missing upper bounds")
            else:
                upper = self.upper_bounds

        coeffs = self.coeffs.copy()
        const = self.const.copy()
        # compute the coefficients of the linear approximation of out bound
        # equations
        for i in range(self.size):
            if  lower[i] >= 0:
                # Active node - Propagate lower bound equation unaltered
                pass
            elif upper[i] <= 0:  
                # Inactive node - Propagate the zero function
                coeffs[i], const[i] =  0, 0
            else:
                # Unstable node - Propagate linear relaxation of lower bound equations
                coeffs[i], const[i] = self.parallel(coeffs[i],
                                                    const[i],
                                                    lower[i],
                                                    upper[i],
                                                    'upper')

        return MemEquations(coeffs, const)


    def parallel(self, coeffs, const, l, u, bound_line):
        """
        Parallel ReLU approximation of the given equation.

        Arguments:
           
            coeffs: the coefficients of the equation.
            
            const: the constant term of the equation.
            
            l: the lower bound of the equation.
            
            u: the upper bound of the equation.
            
            bound_line: either 'upper' or 'lower' approximation.

        Reurns:

            the coefficients and constant terms of the relaxation.
        """
        if not bound_line in ['lower','upper']:
            raise Exception('Got invalid bound line')
        
        adj =  u / (u - l)
        for k in coeffs:
            coeffs[k] *= adj
        if bound_line == 'lower':
            const *= adj
        else:
            const  = const * adj - adj * l

        return coeffs, const

    def min_area(self, coeffs, const, l, u):
        """
        Minimum Area ReLU approximation of the given equation.

        Arguments:
           
            coeffs: the coefficients of the equation.
            
            const: the constant term of the equation.
            
            l: the lower bound of the equation.
            
            u: the upper bound of the equation.
            
        Reurns:

            the coefficients and constant terms of the relaxation.
        """
        if abs(l) < u:
            return coeffs, const
        else:
            return {}, 0

    def venus_heuristic(self, coeffs, const, l, u):
        """
        Venus heuristic ReLU approximation of the given equation.

        Arguments:
           
            coeffs: the coefficients of the equation.
            
            const: the constant term of the equation.
            
            l: the lower bound of the equation.
            
            u: the upper bound of the equation.
            
        Reurns:

            the coefficients and constant terms of the relaxation.
        """
        if abs(l) < u:
            return self.parallel(coeffs, const, l, u, 'lower')
        else:
            return {}, 0

    def dot(self, bound, eqlow, equp, slopes = None):
        """
        Computes the dot product of the equations with given lower and upper
        bound equations

        Arguments:
            
            bound: either 'lower' or 'upper'. Determines whether the lower or
            upper bound equation of the dot product will be derived.

            eqlow: lower bound equations.

            equp: upper bound equations.

            slopes: slopes for the lower bound equation.

            eqs: list of indiced of equations for which to carry out the
            procuct. If None, then all equations are considered.

        Returns:
            
            Equations of the dot product.
        """
        if not bound in ['lower','upper']:
            raise Exception('Got invalid bound line')
  
        
        coeffs = []
        const = self.const.copy()

        for eq in range(self.size):
            d = {}
            for k in self.coeffs[eq]:
                _min = min(self.coeffs[eq][k],0) 
                _max = max(self.coeffs[eq][k],0) 
                slope = slopes[k] if not slopes is None else 1
                for kp in eqlow.coeffs[k]:
                    if bound == 'lower':
                        c = (_max * slope * eqlow.coeffs[k][kp]) + (_min * equp.coeffs[k][kp])
                    else:
                        c = (_min * slope * eqlow.coeffs[k][kp]) + (_max * equp.coeffs[k][kp])
                    d[kp] = d[kp] + c if kp in d else c
                if bound == 'lower':
                    const[eq] += (_max * slope * eqlow.const[k]) + (_min * equp.const[k])
                else:
                    const[eq] += (_max * equp.const[k]) + (_min * slope * eqlow.const[k])
            coeffs.append(d)

        return MemEquations(coeffs, const)

