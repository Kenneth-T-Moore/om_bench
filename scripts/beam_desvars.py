"""
Generate benchmarking data for the Beam problem.
"""
from __future__ import print_function

import numpy as np

from om_bench.bench import Bench

from openmdao.test_suite.test_examples.beam_optimization.multipoint_beam_stress import MultipointBeamGroup


class BeamBench(Bench):

    def setup(self, problem, ndv, nstate, nproc, flag):
        E = 1.
        L = 1.
        b = 0.1
        volume = 0.01
        max_bending = 100.0
        num_elements = nstate
        num_cp = ndv
        num_load_cases = 32

        problem.model = MultipointBeamGroup(E=E, L=L, b=b, volume=volume, max_bending=max_bending,
                                            num_elements=num_elements,
                                            num_cp=num_cp, num_load_cases=num_load_cases
                                            max_bending=max_bending)

    def post_run(problem):
        # Check stuff here.
        pass


if __name__ == "__main__":

    desvars = [1, 2, 4, 8, 16, 32]
    desvars = [item * 4 for item in desvars]
    states = [250]
    procs = [1]

    bench = BeamBench(desvars, states, procs, mode='rev', name='beam')
    bench.num_averages = 5

    bench.run_benchmark()








