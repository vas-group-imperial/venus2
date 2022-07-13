# ************
# File: verification_problem.py
# Top contributors (to current version): 
# 	Panagiotis Kouvaros (panagiotis.kouvaros@gmail.com)
# This file is part of the Venus project.
# Copyright: 2019-2021 by the authors listed in the AUTHORS file in the
# top-level directory.
# License: BSD 2-Clause (see the file LICENSE in the top-level directory).
# Description: Verification problem class.
# ************

import torch

from venus.bounds.sip import SIP

class VerificationProblem(object):

    prob_count = 0

    def __init__(self, nn, spec, depth, config):
        """
        Arguments:
            nn:
                NeuralNetwork.
            spec:
                Specification.
            depth:
                Depth of the problem in the branch-and-bound tree.
            config:
                Configuration.
        """
        VerificationProblem.prob_count += 1
        self.id = VerificationProblem.prob_count
        self.nn = nn
        self.spec = spec
        # couple neural network with spec
        for i in self.nn.head:
            i.from_node.insert(0, spec.input_node)
        self.depth = depth
        self.config = config
        self.stability_ratio = -1
        self.output_range = 0
        self.bounds_ver_done = False
        self.inc_ver_done = False
        self.pgd_ver_done = False
        self.lp_ver_done = False
        self._sip_bounds_computed = False
        self.device = torch.device('cpu')


    def bound_analysis(self, delta_flags=None):
        """
        Computes bounds the network.

        Arguments:
            delta_flags:
                list of current values of Gurobi binary variables; required
                when the bounds are computed at runtime
        """
        sip = self.set_bounds(delta_flags)

        if sip is not None:
            self.stability_ratio = self.nn.get_stability_ratio()
            self.output_range = self.nn.get_output_range()

            return True

        else:
            return False

    def set_bounds(self, delta_flags=None):
        """
        Computes bounds the network.

        Arguments:
            delta_flags:
                list of current values of Gurobi binary variables; required
                when the bounds are computed at runtime
        """
        # check if bounds are already computed
        if delta_flags is None:
            if self._sip_bounds_computed:
                return None

        # compute bounds
        sip = SIP(self, self.config, delta_flags)
        sip.set_bounds()
        # flag the computation
        if delta_flags is None:
            self._sip_bounds_computed = True

        return sip

    def score(self, initial_fixed_ratio):
        return (self.stability_ratio - initial_fixed_ratio) 

    def worth_split(self, subprobs, initial_fixed_ratio):
        pscore0 = self.score(initial_fixed_ratio)
        pscore1 = subprobs[0].score(initial_fixed_ratio)
        pscore2 = subprobs[1].score(initial_fixed_ratio)

        _max = max(pscore1, pscore2)
        _min = min(pscore1, pscore2) 

        if pscore0 >= _max:
            return False
        elif _min > pscore0:
            return True
        elif  (pscore1 + pscore2)/2 > pscore0:
            return True
        else:
            return False
 
    def check_bound_tightness(self, subprobs):
        out0 = self.nn.layers[-1]
        for sp in subprobs:
            if sp.output_range > self.output_range:
                return False
        return True

        for sp in subprobs:
            out1 = sp.nn.layers[-1]
            for i in out0.get_outputs():
                b0 = out0.post_bounds.lower[i]
                b1 = out1.post_bounds.lower[i]
                if b1 < b0:
                    return False
                b0 = out0.post_bounds.upper[i]
                b1 = out1.post_bounds.upper[i]
                if b1 > b0:
                    return False
        return True

    def lp_analysis(self):
        ver_report = VerificationReport()
        if self.spec.output_formula is None:
            return True
        self.bound_analysis()
        lower_bounds = self.nn.layers[-1].post_bounds.lower
        upper_bounds = self.nn.layers[-1].post_bounds.upper
        if self.satisfies_spec(self.spec.output_formula, lower_bounds, upper_bounds):  
            ver_report.result = SolveResult.SAFE

    def satisfies_spec(self):
        if self.spec.output_formula is None:
            return True
        if not self._sip_bounds_computed:
            raise Exception('Bounds not computed')
        
        return self.spec.is_satisfied(
            self.nn.tail.bounds.lower,
            self.nn.tail.bounds.upper
        )

    def get_var_indices(self, nodeid, var_type):
        """
        Returns the indices of the MILP variables associated with a given
        layer.

        Arguments:
                
            nodeid:
                the id of the node for which to retrieve the indices of the
                MILP variables.
            var_type:
                either 'out' for the output variables or 'delta' for the binary
                variables.

        Returns:
        
            pair of ints indicating the start and end positions of the indices
        """
        assert nodeid in self.nn.node or nodeid == self.spec.input_npde.id, \
            f"Node id {nodeid} not  recognised."

        if  nodeid == self.spec.input_node.id:
            return 0, self.spec.input_node.out_vars.size

        start, end = self.spec.input_node.out_vars.size, 0

        for i in range(self.nn.tail.depth + 1):
            nodes = self.nn.get_node_by_depth(i)
            for j in nodes:
                if j.id == nodeid:

                    if var_type == 'out':
                        end = start + j.out_vars.size

                    elif var_type == 'delta':
                        start += j.out_vars.size
                        end = start + j.get_unstable_count()

                    else:
                        raise ValueError(f'Var type {var_type} is not recognised')

                    return start, end

                else:
                    start += j.get_milp_var_size()

    def detach(self):
        """
        Detaches and clones the bound tensors.
        """
        self.nn.detach()
        self.spec.detach()

    def clean_vars(self):
        """
        Nulls out all MILP variables associate with the verification problem.
        """
        self.nn.clean_vars()
        self.spec.clean_vars()
        
    def cuda(self):
        """
        Moves all data to gpu memory
        """
        if self.device == torch.device('cpu'):
            self.nn.cuda()
            self.spec.cuda()
            self.device = torch.device('cuda')

    def cpu(self):
        """
        Moves all data to cpu memory
        """
        if self.device == torch.device('cuda'):
            self.nn.cpu()
            self.spec.cpu()
            self.device = torch.device('cpu')

    def set_batch_size(self, size: int=1):
        """
        Sets the batch size.

        Arguments:
            size: the batch size.
        """
        self.nn.set_batch_size(size)
        self.spec.set_batch_size(size)

    def to_string(self):
        return self.nn.model_path  + ' against ' + self.spec.to_string()

