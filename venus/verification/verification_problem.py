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


from venus.bounds.sip import SIP

class VerificationProblem(object):

    prob_count = 0

    def __init__(self, nn, spec, depth, config):
        VerificationProblem.prob_count += 1
        self.id = VerificationProblem.prob_count
        self.nn = nn
        self.spec = spec
        self.depth = depth
        self.config = config
        self.stability_ratio = -1
        self.output_range = 0
        self._sip_bounds_computed = False
        self._osip_bounds_computed = False


    def bound_analysis(self, delta_flags=None):
        sip = self.set_bounds(delta_flags)
        if sip is not None:
            self.stability_ratio = self.nn.get_stability_ratio()
            self.output_range = self.nn.get_output_range()
            if self.config.SIP.SIMPLIFY_FORMULA is True:
                self.spec.output_formula = sip.simplify_formula(self.spec.output_formula)
            return True
        else:
            return False

    def set_bounds(self, delta_flags=None):
        # check if bounds are already computed
        if delta_flags is None:
            if self.config.SIP.is_osip_enabled():
                if self._osip_bounds_computed:
                    return None
            else:
                if self._sip_bounds_computed:
                    return None
        # compute bounds
        sip = SIP([self.spec.input_layer] + self.nn.layers, self.config, delta_flags)
        sip.set_bounds()
        # flag the computation
        if delta_flags is None:
            if self.config.SIP.is_osip_enabled():
                self._osip_bounds_computed = True
            else:
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
        if not self._sip_bounds_computed and not self.osip_bounds_computed:
            raise Exception('Bounds not computed')
        lower_bounds = self.nn.layers[-1].post_bounds.lower
        upper_bounds = self.nn.layers[-1].post_bounds.upper
        return self.spec.is_satisfied(lower_bounds, upper_bounds)

    def get_var_indices(self, layer, var_type):
        """
        Returns the indices of the MILP variables associated with a given
        layer.

        Arguments:
                
            layer: int of the index of the layer  for which to retrieve the
            indices of the MILP variables

            var_type: str: either 'out' for the output variables or 'delta' for
            the binary variables.

        Returns:
        
            pair of ints indicating the start and end positions of the indices
        """
        layers = [self.spec.input_layer] + self.nn.layers
        start = 0
        end = 0
        for i in range(layer):
            start += layers[i].out_vars.size + layers[i].delta_vars.size
        if var_type == 'out':
            end = start + layers[layer].out_vars.size
        elif var_type == 'delta':
            start += layers[layer].out_vars.size
            end = start + layers[layer].delta_vars.size

        return start, end


    def to_string(self):
        return self.nn.model_path  + ' against ' + self.spec.to_string()

