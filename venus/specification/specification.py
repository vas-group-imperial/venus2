# ************
# File: specification.py
# Top contributors (to current version): 
# 	Panagiotis Kouvaros (panagiotis.kouvaros@gmail.com)
# This file is part of the Venus project.
# Copyright: 2019-2021 by the authors listed in the AUTHORS file in the
# top-level directory.
# License: BSD 2-Clause (see the file LICENSE in the top-level directory).
# Description: class for generic specifications.
# ************

from gurobipy import *
import numpy as np
import torch

from venus.specification.formula import *

class Specification:

    def __init__(self, input_node, output_formula, name=None):
        """
        Arguments:
            input_node:
                The input node.
            output_formula:
                The formula encoding output constraints.
        """
        self.input_node = input_node
        self.output_formula = output_formula
        self.name = name

    def get_output_constrs(self, gmodel, output_vars):
        """
        Encodes the output constraints of the spec into MILP

        Arguments:
            gmodel:
                gurobi model.
            output_vars:
                list of gurobi variables of the output of the network.
        Returns:
            list of gurobi constraints encoding the output formula.
        """
        if self.output_formula is None:
            return []
   
        negated_output_formula = NegationFormula(self.output_formula).to_NNF()  
        return self.get_constrs(negated_output_formula, gmodel, output_vars)

    def get_constrs(self, formula, gmodel, output_vars):
        """
        Encodes a given formula into MILP

        Arguments:
            formula:
                formula to encode.
            gmodel:
                gurobi model.
            output_vars:
                list of gurobi variables of the output of the network. 
        Returns:
            list of gurobi constraints encoding the given formula.
        """
        assert isinstance(formula, Formula), f'Got {type(formula)} instead of Formula'

        if isinstance(formula, Constraint):
            return [self.get_atomic_constr(formula, output_vars)]
        elif isinstance(formula, ConjFormula):
            return self.get_conj_formula_constrs(formula, gmodel, output_vars)
        elif isinstance(formula, NAryConjFormula):
            return self.get_nary_conj_formula_constrs(formula, gmodel, output_vars)
        elif isinstance(formula, DisjFormula):
            return self.get_disj_formula_constrs(formula, gmodel, output_vars)
        elif isinstance(formula, NAryDisjFormula):
            return self.get_nary_disj_formula_constrs(formula, gmodel, output_vars)
        elif isinstance(formula, FalseFormula):
            return [output_vars[0] == GRB.INFINITY]
        else:
            raise Exception(f'Unexpected type {type(formula)} of formula')

    def get_atomic_constr(self, constraint, output_vars):
        """
        Encodes an atomic constraint into MILP

        Arguments:
            constraint:
                constraint to encode.
            output_vars:
                list of gurobi variables of the output of the network.
        Returns:
            list of gurobi constraints encoding the given constraint.
        """

        assert isinstance(constraint, Constraint), f'Got {type(constraint)} instead of Constraint'

        sense = constraint.sense
        if isinstance(constraint, VarVarConstraint):
            op1 = output_vars[constraint.op1.i]
            op2 = output_vars[constraint.op2.i]
        elif isinstance(constraint, VarConstConstraint):
            op1 = output_vars[constraint.op1.i]
            op2 = constraint.op2
        elif isinstance(constraint, LinExprConstraint):
            op1 = 0
            for i, c in constraint.op1.coord_coeff_map.items():
                op1 += c * vars[i]
            op2 = constraint.op2
        else:
            raise Exception("Unexpected type of atomic constraint", constraint)
        
        if sense == Formula.Sense.GE:
            return op1 >= op2
        elif sense == Formula.Sense.LE:
            return op1 <= op2
        elif sense == Formula.Sense.EQ:
            return op1 == op2
        else:
            raise Exception("Unexpected type of sense", sense)


    def get_conj_formula_constrs(self, formula, gmodel, output_vars):
        """
        Encodes a conjunctive formula into MILP

        Arguments:

            formula:
                conjunctive formula to encode.
            gmodel:
                gurobi model
            output_vars:
                list of gurobi variables of the output of the network.
        Returns:
            list of gurobi constraints encoding the given formula
        """
        assert isinstance(formula, ConjFormula), f'Got {type(formula)} instead of ConjFormula'

        return self.get_constrs(formula.left, gmodel, output_vars) + self.get_constrs(formula.right, gmodel, output_vars)

    def get_nary_conj_formula_constrs(self, formula, gmodel, output_vars):
        """
        Encodes an nary conjunctive formula into MILP

        Arguments:
            formula:
                nary conjunctive formula to encode.
            gmodel:
                gurobi model.
            output_vars:
                list of gurobi variables of the output of the network. 
        Returns:
            list of gurobi constraints encoding the given formula
        """
        assert isinstance(formula, NAryConjFormula), f'Got {type(formula)} instead of NAryConjFormula'

        constrs = []
        for subformula in formula.clauses:
            constrs += self.get_constrs(subformula, gmodel, output_vars)

        return constrs

    def get_disj_formula_constrs(self, formula, gmodel, output_vars):
        """
        Encodes a disjunctive formula into MILP

        Arguments:
            formula:
                disjunctive formula to encode
            gmodel:
                gurobi model.
            output_vars:
                list of gurobi variables of the output of the network. 
        Returns:
            list of gurobi constraints encoding the given formula
        """
        assert isinstance(formula, DisjFormula), f'Got {type(formula)} instead of DisjFormula'

        split_var = gmodel.addVar(vtype=GRB.BINARY)
        clause_vars = [gmodel.addVars(len(output_vars), lb=-GRB.INFINITY), 
                       gmodel.addVars(len(output_vars), lb=-GRB.INFINITY)]
        constr_sets = [self.get_constrs(formula.left, gmodel, clause_vars[0]), 
                       self.get_constrs(formula.right, gmodel, clause_vars[1])]
        constrs = []
        for i in [0, 1]:
            for j in range(len(output_vars)):
                constrs.append((split_var == i) >> (output_vars[j] == clause_vars[i][j]))
            for disj_constr in constr_sets[i]:
                constrs.append((split_var == i) >> disj_constr)

        return constrs

    def get_nary_disj_formula_constrs(self, formula, gmodel, output_vars):
        """
        Encodes an nary disjunctive formula into MILP

        Arguments:

            formula:
                nary disjunctive formula to encode.
            gmodel:
                gurobi model.
            output_vars:
                list of gurobi variables of the output of the network.
        Returns:
            list of gurobi constraints encoding the given formula
        """
        assert isinstance(formula, NAryDisjFormula), f'Got {type(formula)} instead of NAryDisjFormula'

        clauses = formula.clauses
        split_vars = gmodel.addVars(len(clauses), vtype=GRB.BINARY)
        clause_vars = [gmodel.addVars(len(output_vars), lb=-GRB.INFINITY) 
                       for _ in range(len(clauses))]
        constr_sets = []
        constrs = []

        for i, o in enumerate(clauses):
            constr_sets.append(self.get_constrs(o, gmodel, clause_vars[i]))
            for j, p in enumerate(output_vars):
                constrs.append((split_vars[i] == 1) >> (p == clause_vars[i][j]))
            for disj_constr in constr_sets[i]:
                constrs.append((split_vars[i] == 1) >> disj_constr)

        # exactly one variable must be true
        constrs.append(quicksum(split_vars) == 1)
            
        return constrs

    def copy(self, input_node=None):
        """
        Returns a copy of the specificaton

        Arguments:
            input_node:
                The input node to optionally update in the copy
        Returns:
            Specification
        """
        innode = input_node if input_node is not None else self.input_node.copy()

        return Specification(innode, self.output_formula, self.name)

    def normalise(self, mean, std):
        """
        Normalises the input bounds

        Arguments:
            mean:
                normalisation mean
            std:
                normalisation standard deviation
        Returns:
            None
        """
        self.input_layer.post_bounds.normalise(mean, std)

    def clip(self, min_value, max_value):
        """
        Clips the input bounds

        Arguments:

            min_value:
                valid lower bound
            max_value:
                valid upper bound
        """
        self.input_layer.post_bounds.clip(min_value, max_value)

    def is_satisfied(self, lower_bounds, upper_bounds):
        """
        Checks whether the specificaton is satisfied given the network's output
        lower and upper bounds.

        Arguments:
            lower_bounds:
                The lower bounds.
            upper_bounds:
                The upper bounds.
        Returns:
            Whether or not the specification is satisfied.
        """
        return self._is_satisfied(
            self.output_formula, lower_bounds.flatten(), upper_bounds.flatten()
        )

    def _is_satisfied(self, formula, lower_bounds, upper_bounds):
        """
        Helper function for is_satisfied.

        Arguments:
            formula:
                formula whose satisfaction to check.
            lower_bounds:
                The lower bounds.
            upper_bounds:
                The upper bounds.
        Returns:
            Whether or not the given formula is satisfied.
        """
        if isinstance(formula, TrueFormula):
            return True
        elif isinstance(formula, Constraint):
            sense = formula.sense
            if sense == Formula.Sense.LT:
                if isinstance(formula, VarVarConstraint):
                    return upper_bounds[formula.op1.i].item() < lower_bounds[formula.op2.i].item()
                if isinstance(formula, VarConstConstraint):
                    return upper_bounds[formula.op1.i].item() < formula.op2
            elif sense == Formula.Sense.GT:
                if isinstance(formula, VarVarConstraint):
                    return lower_bounds[formula.op1.i].item() > upper_bounds[formula.op2.i].item()
                if isinstance(formula, VarConstConstraint):
                    return lower_bounds[formula.op1.i].item() > formula.op2
            else:
                raise Exception('Unexpected sense', formula.sense)
        elif isinstance(formula, ConjFormula):
            return self._is_satisfied(formula.left, lower_bounds, upper_bounds) and \
                   self._is_satisfied(formula.right, lower_bounds, upper_bounds)
        elif isinstance(formula, NAryConjFormula):
            for clause in formula.clauses:
                if not self._is_satisfied(clause, lower_bounds, upper_bounds):
                    return False
            return True
        elif isinstance(formula, DisjFormula):
            return self._is_satisfied(formula.left, lower_bounds, upper_bounds) or \
                   self._is_satisfied(formula.right, lower_bounds, upper_bounds)
        elif isinstance(formula, NAryDisjFormula):
            for clause in formula.clauses:
                if self._is_satisfied(clause, lower_bounds, upper_bounds):
                    return True
            return False
        else:
            raise Exception("Unexpected type of formula", type(formula))

    def get_mse_loss(self, output):
        """
        Computes the mean squared error of the output. 

        Arguments:
            output:
                The output.
        Returns:
            MSE of the output.
        """
        padded_output = torch.hstack((torch.zeros(1), output))
        pos_dims, neg_dims, consts = self._get_mse_loss(
            self.output_formula,
            padded_output
        )
        pos_dims = torch.tensor(pos_dims, dtype=torch.long)
        neg_dims = torch.tensor(neg_dims, dtype=torch.long)
        consts = torch.tensor(consts)

        loss = torch.mean((padded_output[pos_dims] - padded_output[neg_dims] - consts) ** 2)

        return loss

    def _get_mse_loss(self, formula, output):
        """
        Helper function for get_mse_loss.

        Arguments:
            formula:
                subformula of the output formula.
            output:
                The output.
        Returns:
            MSE Loss.
        """
        if isinstance(formula, TrueFormula):
            return [0], [0], [0]

        elif isinstance(formula, Constraint):
            sense = formula.sense

            if sense == Formula.Sense.LT:
                if isinstance(formula, VarVarConstraint):
                    return [formula.op2.i + 1], [formula.op1.i + 1], [0]
                if isinstance(formula, VarConstConstraint):
                    return [0], [formula.op1.i + 1], [- formula.op2]

            elif sense == Formula.Sense.GT:
                if isinstance(formula, VarVarConstraint):
                    return [formula.op1.i + 1], [formula.op2.i + 1], [0]
                if isinstance(formula, VarConstConstraint):
                    return [formula.op1.i], [0], [formula.op2]
            else:
                raise Exception('Unexpected sense', formula.sense)

        elif type(formula) in [ConjFormula, DisjFormula]:
            pos_dims1, neg_dims1, consts1 = self._get_mse_loss(formula.left, output)
            pos_dims2, neg_dims2, consts2 = self._get_mse_loss(formula.right, output)

            return pos_dims1 + pos_dims2, neg_dims1 + neg_dims2, consts1 + consts2

        elif type(formula) in [NAryConjFormula, NAryDisjFormula]:
            pos_dims, neg_dims, consts = [], [], []
            for clause in formula.clauses:
                pos_dims2, neg_dims2, consts2 = self._get_mse_loss(clause, output)
                pos_dims += pos_dims2
                neg_dims += neg_dims2
                consts += consts2

            return pos_dims, neg_dims, consts

        else:
            raise Exception("Unexpected type of formula", type(formula))

    def is_adversarial_robustness(self):
        """
        Checks whether the output constraints of the specificaton refer to an
        adversarial robustness property.
        """
        return self._is_adversarial_robustness(self.output_formula)

    def _is_adversarial_robustness(self, formula):
        """
        Helper function for is_satisfied.

        Arguments:
            formula:
                formula whose satisfaction to check.
            lower_bounds:
                The lower bounds.
            upper_bounds:
                The upper bounds.
        Returns:
            Whether or not the given formula is satisfied.
        """
        if isinstance(formula, TrueFormula):
            return -1
        elif isinstance(formula, Constraint):
            if formula.sense == Formula.Sense.LT:
                return formula.op2.i if isinstance(formula, VarVarConstraint) else -1
            elif formula.sense == Formula.Sense.GT:
                return formula.op1.i if isinstance(formula, VarVarConstraint) else -1
            else:
                raise Exception('Unexpected sense', formula.sense)
        elif type(formula) in [ConjFormula, DisjFormula]:
            label1 = self._is_adversarial_robustness(formula.left)
            if label1 == -1:
                return -1 
            label2 = self._is_adversarial_robustness(formula.right)
            if label2 == -1:
                return -1 
            return -1 if label1 != label2 else label1
        elif type(formula) in [NAryConjFormula, NAryDisjFormula]:
            label1 = self._is_adversarial_robustness(formula.clauses[0])
            if label1 == -1:
                return -1
            for clause in formula.clauses[1:]:
                label2 = self._is_adversarial_robustness(clause)
                if label2 != label1:
                    return -1
            return label1
        else:
            raise Exception("Unexpected type of formula", type(formula))

    def get_output_flag(self, output_shape: tuple):
        """
        Creates a boolean flag of the outputs units that the specification refers to.

        Arguments:
            output_shape:
                the output shape of the network.
        Returns:
            Boolean flag of whether each output concerns the specification.
        """
        flag = self._get_output_flag(
            self.output_formula, torch.zeros(np.prod(output_shape), dtype=torch.bool)
        )

        return flag.reshape(output_shape)

    def _get_output_flag(self, formula: Formula, flag: torch.tensor):
        """
        Helper function for get_output_flag.
        """
        if isinstance(formula, TrueFormula):
            return flag
        elif isinstance(formula, Constraint):
            if isinstance(formula.op1, StateCoordinate):
                flag[formula.op1.i] = True
            if isinstance(formula.op2, StateCoordinate):
                flag[formula.op2.i] = True
            return flag
        elif type(formula) in [ConjFormula, DisjFormula]:
            flag = self._get_output_flag(formula.left, flag)
            flag = self._get_output_flag(formula.right, flag)
            return flag
        elif type(formula) in [NAryConjFormula, NAryDisjFormula]:
            for clause in formula.clauses:
                flag = self._get_output_flag(clause, flag)
            return flag
        else:
            raise Exception("Unexpected type of formula", type(formula))

    def to_string(self):
        """
        Returns:
            str describing the specification.
        """
        return self.name
