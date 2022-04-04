# ************
# File: mnistfc.py
# Top contributors (to current version): 
# 	Panagiotis Kouvaros (panagiotis.kouvaros@gmail.com)
# This file is part of the Venus project.
# Copyright: 2019-2021 by the authors listed in the AUTHORS file in the
# top-level directory.
# License: BSD 2-Clause (see the file LICENSE in the top-level directory).
# Description: Tests verification results for MNIST Fully Connected.
# ************


import unittest

from venus.verification.venus import Venus
from venus.tests.verification.satisfaction_reader import SatisfactionReader

class TestMnistFC(unittest.TestCase):

    def setUp(self):
        self.true_results = SatisfactionReader('venus/tests/data/mnistfc/queries.csv').read_results()
        self.properties = {
            1 : [(1, x, y) for x in range(1, 6) for y in range(1, 10)],
            2 : [(2, x, y) for x in range(1, 6) for y in range(1, 10)],
            3 : [(3, x, y) for x in range(1, 6) for y in range(1, 10)],
            4 : [(4, x, y) for x in range(1, 6) for y in range(1, 10)],
            5 : [(5, 1, 1)],
            6 : [(6, 1, 1)],
            7 : [(7, 1, 9)],
            8 : [(8, 2, 9)],
            9 : [(9, 3, 3)],
            10 : [(10, 4, 5)]
        }


    def test_fc(self):
        """
        Tests the verification results MNISTFC.
        """
        for (nn, spec) in self.true_results:
            venus = Venus(
                nn='venus/tests/data/mnistfc/' + nn,
                spec='venus/tests/data/mnistfc/' + spec
            )
            report = venus.verify()
            self.assertTrue(self.true_results[(nn, spec)] == report.result)
