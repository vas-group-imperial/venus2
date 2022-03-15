# ************
# File: split_report.py
# Top contributors (to current version): 
# 	Panagiotis Kouvaros (panagiotis.kouvaros@gmail.com)
# This file is part of the Venus project.
# Copyright: 2019-2021 by the authors listed in the AUTHORS file in the
# top-level directory.
# License: BSD 2-Clause (see the file LICENSE in the top-level directory).
# Description: Data regarding a split procedure.
# ************

class SplitReport:
    def __init__(self, id, jobs_count, node_split_count, input_split_count):
        self.id = id,
        self.jobs_count = jobs_count
        self.node_split_count = node_split_count
        self.input_split_count = input_split_count
