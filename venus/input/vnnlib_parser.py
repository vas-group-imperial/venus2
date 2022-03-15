# ************
# File: vnnlib_parser.py
# Top contributors (to current version): 
# 	Panagiotis Kouvaros (panagiotis.kouvaros@gmail.com)
# This file is part of the Venus project.
# Copyright: 2019-2021 by the authors listed in the AUTHORS file in the
# top-level directory.
# License: BSD 2-Clause (see the file LICENSE in the top-level directory).
# Description: Parses a VNNLIB specification.
# ************

import os
import numpy as np
import math
import collections
from sly import Lexer
from sly import Parser
from venus.specification.formula import Formula, StateCoordinate, VarConstConstraint, VarVarConstraint, ConjFormula, \
    DisjFormula, NAryConjFormula, NAryDisjFormula, NegationFormula
from venus.specification.specification import Specification
from venus.network.layers import Input



class VNNLIBParser:

    def __init__(self, pf, X_SZ, Y_SZ, config):
        self.pf = pf
        self.X_SZ = X_SZ
        self.Y_SZ = Y_SZ
        self.config = config

    def parse(self):
        with open(self.pf, 'r') as f:
            s = f.read()
        lexer = VNNLexer()
        parser = VNNParser(self.X_SZ, self.Y_SZ, self.config)
        i_b, o_f, i_cl = parser.parse(lexer.tokenize(s))
        specs = []
        if len(i_cl) == 0:
            specs.append(
                Specification(
                    Input(i_b[0], i_b[1]),
                    NegationFormula(o_f).to_NNF(),
                    os.path.basename(self.pf)[1]
                )
            )
        else:
            for clause in i_cl:
                if not o_f is None and not clause[1] is None:
                    f = ConjFormula(o_f, clause[1])
                elif not o_f is None and clause[1] is None:
                    f = o_f
                elif o_f is None and not clause[1] is None:
                    f = clause[1]
                else:
                    raise Exception('No output constraints found')

                specs.append(
                    Specification(
                        Input(clause[0][0], clause[0][1]),
                        NegationFormula(f).to_NNF(),
                        os.path.basename(self.pf)[1]
                    )
                )

        return specs



class VNNLexer(Lexer):
    tokens = { LE, GE, ASSERT, AND, OR, INPUT, OUTPUT, NUM, NUM, LPAR, RPAR, UNDERSCR, CONST, REAL}
    ignore = ' \t'

    LE = r'<='
    GE = r'>='
    ASSERT = r'assert'
    AND = r'and'
    OR = r'or'
    INPUT = r'X'
    OUTPUT = r'Y'
    LPAR = r'\('
    RPAR = r'\)'
    UNDERSCR = r'_'
    CONST = 'declare-const'
    REAL = r'Real'

    # @_(r'[-+]?([0-9]*\.[0-9]+|[0-9]+)')
    @_(r'[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?')
    def NUM(self, t):
        t.value = float(t.value)
        return t

    @_(r';.*')
    def COMMENT(self, t):
        pass
  
    # newline tracking
    @_(r'\n+')
    def newline(self, t):
        self.lineno = t.value.count('\n')


class VNNParser(Parser):
    tokens = VNNLexer.tokens
    TermTuple = collections.namedtuple('term_tuple', ['type', 'index', 'sense', 'bound'])

    def __init__(self, X_SZ, Y_SZ, config):
        self.env = { }
        self.X_SZ = X_SZ
        self.Y_SZ = Y_SZ
        self.config = config
        self.i_b = [np.ones(self.X_SZ, dtype=config.PRECISION) * -math.inf, 
                    np.ones(self.X_SZ, dtype=config.PRECISION) * math.inf]
        self.o_f = None
        self.i_cl = []

    @_('statement')
    def statements(self, p):
        return  self.i_b, self.o_f, self.i_cl

    @_('statement statements')
    def statements(self, p):
        return self.i_b, self.o_f, self.i_cl

    @_('input_statement')
    def statement(self, p):
        pass

    @_('output_statement')
    def statement(self, p):
        if self.o_f is None:
            self.o_f = p.output_statement
        else:
            self.o_f = ConjFormula(self.o_f, p.output_statement)

    @_('LPAR CONST input_id REAL RPAR')
    def statement(self, p):
        pass

    @_('LPAR CONST output_id REAL RPAR')
    def statement(self, p):
        pass

    @_('LPAR ASSERT input_term RPAR')
    def input_statement(self, p):
        pass

    @_('LPAR ASSERT LPAR OR input_and_clauses RPAR RPAR')
    def input_statement(self, p):
        self.i_cl = p.input_and_clauses

    @_('input_and_clause')
    def input_and_clauses(self, p):
        return [p.input_and_clause]

    @_('input_and_clause input_and_clauses')
    def input_and_clauses(self, p):
        return p.input_and_clauses + [p.input_and_clause]

    @_('LPAR AND iio_terms RPAR')
    def input_and_clause(self, p):
        i_b = [np.ones(self.X_SZ) * -math.inf,  np.ones(self.X_SZ) * math.inf]
        o_f_terms =  []
        for term in p.iio_terms:
            if term.type == 'input':
                if term.sense == 'le':
                    i_b[1][term.index] = term.bound
                elif term.sense == 'ge':
                    i_b[0][term.index] = term.bound
                else:
                    raise Exception(f'Unexpected term sense {term.sense}')
            elif term.type == 'output':
                if term.sense == 'le':
                    constr = VarConstConstraint(term.index, Formula.Sense.LE, term.bound)
                elif term.sense == 'ge':
                    constr = VarConstConstraint(term.index, Formula.Sense.GE, term.bound)
                else:
                    raise Exception(f'Unexpected term sense {term.sense}')
                o_f_terms.append(constr)
            else:
                raise Exception(f'Unexpected term type {term.type}')
        o_f = None if len(o_f_terms) == 0 else NAryConjFormula(o_f_terms)
        
        return (i_b, o_f)

    @_('io_input_term  io_terms')
    def iio_terms(self, p):
        return [p.io_input_term] + p.io_terms

    @_('io_term')
    def io_terms(self, p):
        return [p.io_term]

    @_('io_term io_terms')
    def io_terms(self, p):
        return [p.io_term] + p.io_terms

    @_('io_input_term')
    def io_term(self, p):
        return p.io_input_term

    @_('io_output_term')
    def io_term(self, p):
        return p.io_output_term

    @_('LPAR LE  input_id  NUM RPAR')
    def io_input_term(self, p):
        return VNNParser.TermTuple('input',  p.input_id, 'le', p.NUM)

    @_('LPAR GE  input_id  NUM RPAR')
    def io_input_term(self, p):
        return VNNParser.TermTuple('input',  p.input_id, 'ge', p.NUM)

    @_('LPAR LE output_id  NUM RPAR')
    def io_output_term(self, p):
        return VNNParser.TermTuple('output', p.output_id, 'le', p.NUM)

    @_('LPAR GE output_id  NUM RPAR')
    def io_output_term(self, p):
        return VNNParser.TermTuple('output', p.output_id, 'ge', p.NUM)

    @_('LPAR LE output_id  output_id RPAR')
    def io_output_term(self, p):
        return VNNParser.TermTuple('output', p.output_id0, 'le', p.output_id1)

    @_('LPAR GE output_id  output_id RPAR')
    def io_output_term(self, p):
        return VNNParser.TermTuple('output', p.output_id0, 'ge', p.output_id1)


    @_('LPAR LE  input_id  NUM RPAR')
    def input_term(self, p):
        self.i_b[1][p.input_id] = p.NUM

    @_('LPAR GE  input_id NUM RPAR')
    def input_term(self, p):
        self.i_b[0][p.input_id] = p.NUM

    @_('INPUT UNDERSCR NUM')
    def input_id(self, p):
        return int(p.NUM)

    @_('LPAR ASSERT output_term RPAR')
    def output_statement(self, p):
        return p.output_term

    @_('LPAR ASSERT  output_logic_clause RPAR')
    def output_statement(self, p):
        return p.output_logic_clause

    @_('output_and_clause')
    def output_logic_clause(self, p):
        return p.output_and_clause

    @_('output_or_clause')
    def output_logic_clause(self, p):
        return p.output_or_clause

    @_('output_logic_clause')
    def output_logic_clauses(self, p):
        return [p.output_logic_clause]

    @_('output_logic_clause output_logic_clauses')
    def output_logic_clauses(self, p):
        return p.output_logic_clauses + [p.output_logic_clause]


    @_('LPAR AND output_logic_clauses RPAR')
    def output_and_clause(self, p):
        if len(p.output_logic_clauses) == 1:
            return p.output_logic_clauses[0]
        elif len(p.output_logic_clauses) == 2:
            return ConjFormula(p.output_logic_clauses[0], p.output_logic_clauses[1])
        else:
            return NAryConjFormula(p.output_logic_clauses)

    @_('LPAR AND output_terms RPAR')
    def output_and_clause(self, p):
        if len(p.output_terms) == 1:
            return p.output_terms[0]
        elif len(p.output_terms) == 2:
            return ConjFormula(p.output_terms[0], p.output_terms[1])
        else:
            return NAryConjFormula(p.output_terms)

    @_('LPAR OR output_logic_clauses RPAR')
    def output_or_clause(self, p):
        if len(p.output_logic_clauses) == 1:
            return p.output_logic_clauses[0]
        elif len(p.output_logic_clauses) == 2:
            return DisjFormula(p.output_logic_clauses[0], p.output_logic_clauses[1])
        else:
            return NAryDisjFormula(p.output_logic_clauses)


    @_('LPAR OR output_terms RPAR')
    def output_or_clause(self, p):
        if len(p.output_terms) == 1:
            return p.output_terms[0]
        elif len(p.output_terms) == 2:
            return DisjFormula(p.output_terms[0], p.output_terms[1])
        else:
            return NAryDisjFormula(p.output_terms)

    @_('output_term output_terms')
    def output_terms(self, p):
        return p.output_terms + [p.output_term]

    @_('output_term')
    def output_terms(self, p):
        return [p.output_term]
        
    @_('LPAR LE output_id  NUM RPAR')
    def output_term(self, p):
        return VarConstConstraint(p.output_id, Formula.Sense.LE, p.NUM)

    @_('LPAR GE output_id  NUM RPAR')
    def output_term(self, p):
        return VarConstConstraint(p.output_id, Formula.Sense.GE, p.NUM)

    @_('LPAR LE output_id  output_id RPAR')
    def output_term(self, p):
        return VarVarConstraint(p.output_id0, Formula.Sense.LE, p.output_id1)

    @_('LPAR GE output_id  output_id RPAR')
    def output_term(self, p):
        return VarVarConstraint(p.output_id0, Formula.Sense.GE, p.output_id1)

    @_('OUTPUT UNDERSCR NUM')
    def output_id(self, p):
        return StateCoordinate(int(p.NUM))
