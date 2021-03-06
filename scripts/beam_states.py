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
        num_elements = nstate
        max_bending = 100.0
        num_cp = ndv * 16
        num_load_cases = 32

        problem.model = MultipointBeamGroup(E=E, L=L, b=b, volume=volume, num_elements=num_elements,
                                            num_cp=num_cp, num_load_cases=num_load_cases,
                                            max_bending=max_bending)

    def post_run(problem):
        # Check stuff here.
        pass


if __name__ == "__main__":

    desvars = [1]
    states = [1, 2, 4, 8, 16, 32]
    states = [item * 50 for item in states]
    procs = [1]

    bench = BeamBench(desvars, states, procs, name='beam')

    bench.run_benchmark()








