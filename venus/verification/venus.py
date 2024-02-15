# ************
# File: venus# Top contributors (to current version):
# 	Panagiotis Kouvaros (panagiotis.kouvaros@gmail.com)
# This file is part of the Venus project.
# Copyright: 2019-2021 by the authors listed in the AUTHORS file in the
# top-level directory.
# License: BSD 2-Clause (see the file LICENSE in the top-level directory).
# Description: Main verification class.
# ************

import os
from tqdm import tqdm

from venus.input.query_reader import QueryReader
from venus.input.vnnlib_parser import VNNLIBParser
from venus.network.neural_network import NeuralNetwork
from venus.verification.verifier import Verifier
from venus.solver.solve_result import SolveResult
from venus.common.configuration import Config
from venus.bounds.bounds import Bounds
from venus.specification.specification import Specification


import torch
import numpy as np
from venus.verification.verification_problem import VerificationProblem
from venus.verification.verification_report import VerificationReport
from timeit import default_timer as timer
from venus.bounds.sip import SIP

import math

class Venus:

    def __init__(
        self,
        queries=None,
        nn=None,
        spec=None,
        config=None):
        """
        Arguments:

            queries:
                csv file of queries.

            nn:
                network file or folder of networks.

            spec:
                specification file or folder of specifications.

            config:
                Configuration.
        """
        if queries is not  None:
            self.queries = QueryReader().read_from_csv(queries)
        elif nn is not None and spec is not None:
            self.queries = QueryReader().read_from_file(nn, spec)
        else:
            assert False, "Expected queries file or neural network and specification files."
        self.config = Config() if config is None else config
        with open(self.config.LOGGER.LOGFILE, 'w'):
            pass
        with open(self.config.LOGGER.SUMFILE, 'w'):
            pass


    def verify(self):
        results = []
        safe, unsafe, undecided, timeout = 0, 0, 0, 0
        total_time, total_safe_time, total_unsafe_time = 0, 0, 0

        print('\n\n')
        pbar = tqdm(self.queries, desc='Verifying ', ncols=125)
        for query in pbar:
            pbar.set_description('Verifying ' +
                                 os.path.basename(query[0]) +
                                 ' against ' +
                                 os.path.basename(query[1]) +
                                 '...')
            if self.config.BENCHMARK == 'vgg16_2022':
                self.config.PRECISION = torch.float64
            # load model
            nn = NeuralNetwork(query[0], self.config)
            nn.load()

            # load spec
            vnn_parser = VNNLIBParser(
                query[1],
                nn.head[0].input_shape,
                self.config
            )
            spec = vnn_parser.parse()


            if self.config.BENCHMARK == 'carvana':
                head_id = nn.head[0].id
                nn.head[0] = nn.head[0].to_node[0]
                nn.head[0].from_node = []
                del nn.node[head_id]
                spec[0].carvana_out_vals = spec[0].input_node.bounds.lower[:, 3, ...]
                spec[0].input_node.bounds.lower = spec[0].input_node.bounds.lower[:,0:3,...]
                spec[0].input_node.bounds.upper = spec[0].input_node.bounds.upper[:,0:3,...]
                shape = spec[0].input_node.input_shape
                shape = (shape[0], 3, shape[2], shape[3])
                spec[0].input_node.input_shape = shape
                nn.head[0].input_shape = shape


            # spec[0].input_node.bounds.upper = spec[0].input_node.bounds.lower.clone()
            # import onnx
            # import onnxruntime.backend as rt
            # m = onnx.load('vnncomp2022_benchmarks-main/benchmarks/collins_rul_cnn/onnx/NN_rul_full_window_20.onnx')
            # runnable = rt.prepare(m, 'CPU')
            # pred = runnable.run(spec[0].input_node.bounds.lower.numpy().reshape(nn.head.input_shape))
            # print(pred)
            # import sys
            # sys.exit()

            # verify
            ver_report = self.verify_query(nn, spec)
            if ver_report.result == SolveResult.SAFE:
                safe += 1
                total_safe_time += ver_report.runtime
            elif ver_report.result == SolveResult.UNSAFE:
                unsafe += 1
                total_unsafe_time += ver_report.runtime

                # import onnx
                # import onnxruntime.backend as rt
                # m = onnx.load(query[0])
                # runnable = rt.prepare(m, 'CPU')
                # cex = np.expand_dims(ver_report.cex.numpy(), 0)
                # cex = ver_report.cex.numpy()
                # pred = runnable.run(cex)
                # print('*****', pred)
                # import sys
                # sys.exit()

            elif ver_report.result == SolveResult.UNDECIDED:
                undecided += 1
            elif ver_report.result == SolveResult.TIMEOUT:
                timeout += 1
            total_time += ver_report.runtime
            results.append(ver_report)

            with open(self.config.LOGGER.SUMFILE, 'w') as f:
                if ver_report.result == SolveResult.UNSAFE:
                    res = 'sat\n'
                    cex = ver_report.cex.flatten()
                    res += f'((X_0 {cex[0]})'
                    for i, j in enumerate(cex[1:]):
                        res += f'\n (X_{i + 1} {j})'
                    res += ')'
                elif ver_report.result == SolveResult.SAFE:
                    res = 'unsat\n'
                elif ver_report.result == SolveResult.TIMEOUT:
                    res = 'timeout\n'
                else:
                    res = 'unknown'

                f.write(res)

                # if res == SolveResult.SATISFIED:
                    # if.write('holds\n')
                # elif res == SolveResult.UNSATISFIED:
                    # f.write('violated\n')
                # elif res == SolveResult.TIMEOUT:
                    # f.write('timeout\n')
                # else:
                    # f.write('unknown\n')

                # f.write('{:<12}{:6.4f}\n'.format(ver_report.result.value, ver_report.runtime))

        # avg_safe_time = 0 if safe == 0 else total_safe_time / safe
        # avg_unsafe_time = 0 if unsafe == 0 else total_unsafe_time / unsafe

        # with open(self.config.LOGGER.SUMFILE, 'a') as f:
            # f.write('\n\nVerified: {}\tSAFE: {}\tUNSAFE: {}\tUndecided: {}\tTimeouts: {}\n\n'.format(
                # safe + unsafe,
                # safe,
                # unsafe,
                # undecided,
                # timeout
            # ))
            # f.write(
                # 'Total Time:       {:6.4f}\tAvg Time:       {:6.4f}\n'.format(
                    # total_time,
                    # total_time / len(self.queries)
                # )
            # )
            # f.write(
                # 'Total SAFE Time:   {:6.4f}\tAvg SAFE Time:   {:6.4f}\n'.format(
                    # total_safe_time,
                    # avg_safe_time
                # )
            # )
            # f.write(
                # 'Total UNSAFE Time: {:6.4f}\tAvg UNSAFE Time: {:6.4f}\n\n'.format(
                    # total_unsafe_time,
                    # avg_unsafe_time
                # )
            # )

        if self.config.VERIFIER.CONSOLE_OUTPUT:
            print(
              '\nVerified: {}\tSAFE: {}\tUNSAFE: {}\tUndecided: {}\tTimeouts: {}\t Total Time: {}\n'.format(
                    safe + unsafe,
                    safe,
                    unsafe,
                    undecided,
                    timeout,
                    total_time
                )
            )
            # print(
                # 'Total Time:       {:6.4f}\tAvg Time:       {:6.4f}'.format(
                    # total_time,
                    # total_time / len(self.queries)
                # )
            # )
            # print(
                # 'Total SAFE Time:   {:6.4f}\tAvg SAFE Time:   {:6.4f}'.format(
                    # total_safe_time,
                    # avg_safe_time
                # )
            # )
            # print(
                # 'Total UNSAFE Time: {:6.4f}\tAvg UNSAFE Time: {:6.4f}'.format(
                    # total_unsafe_time,
                    # avg_unsafe_time
                # )
            # )

        return results[0] if len(results) == 1 else results

    def verify_query(self, nn, spec):
        if len(spec) > 1 and self.config.BENCHMARK == 'nn4sys':
            return self.verify_batch(nn, spec)

        return self.verify_sequence(nn, spec)




    def verify_batch(self, nn, spec):
        size, batch = len(spec), 50
        ver_report = VerificationReport()
        start = timer()

        input_node = spec[0].input_node

        for i in range(0, size, batch):
            until = min(size, i + batch)
            batch_lower = torch.vstack(
                tuple(
                    j.input_node.bounds.lower for j in spec[i: until]
                )
            )
            batch_upper = torch.vstack(
                tuple(
                    j.input_node.bounds.upper for j in spec[i: until]
                )
            )
            input_node.bounds = Bounds(batch_lower, batch_upper)
            input_node.input_shape = input_node.output_shape = batch_lower.shape
            input_node.input_size = input_node.output_size = np.prod(batch_lower.shape)
            batch_formula = [j.output_formula for j in spec[i: until]]
            batch_spec = Specification(
                spec[0].input_node, batch_formula, self.config
            )

            verifier = Verifier(nn, batch_spec, self.config, batch=until-i)
            sub_ver_report = verifier.verify()
            ver_report.runtime += sub_ver_report.runtime
            if sub_ver_report.result == SolveResult.UNSAFE:
                ver_report.result = SolveResult.UNSAFE
                for idx in range(until - i):
                    if batch_spec.is_form_satisfied(
                        batch_spec.output_formula[idx],
                        sub_ver_report.cex[idx, ...].flatten(),
                        sub_ver_report.cex[idx, ...].flatten()
                    ) is not True:
                        ver_report.cex = sub_ver_report.cex[idx, ...]

                return ver_report
            else:
                time_left = self.config.SOLVER.TIME_LIMIT - sub_ver_report.runtime
                if time_left <= 0:
                    ver_report.result = SolveResult.TIMEOUT
                    return ver_report
                else:
                    self.config.SOLVER.TIME_LIMIT =  time_left

        ver_report.result = SolveResult.SAFE
        ver_report.result = SolveResult.SATISFIED
        return ver_report

    def verify_sequence(self, nn, spec):
        time_elapsed = 0
        ver_report = None

        for subspec in spec:
            # create verifier
            verifier = Verifier(nn, subspec, self.config)
            sub_ver_report = verifier.verify()
            if ver_report is None:
                ver_report = sub_ver_report
            else:
                ver_report.runtime += sub_ver_report.runtime
                if sub_ver_report.result == SolveResult.UNSAFE:
                    ver_report.result = SolveResult.UNSAFE
                    return ver_report
                else:
                    time_left = self.config.SOLVER.TIME_LIMIT - sub_ver_report.runtime
                    if time_left <= 0:
                        ver_report.result = SolveResult.TIMEOUT
                        return ver_report
                    else:
                        self.config.SOLVER.TIME_LIMIT =  time_left

        return ver_report
