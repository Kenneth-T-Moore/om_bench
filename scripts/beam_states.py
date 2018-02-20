"""
Generate benchmarking data for the Beam problem.
"""
from __future__ import print_function

import numpy as np

from om_bench.bench import Bench

from openmdao.test_suite.test_examples.beam_optimization.beam_group import BeamGroup


class BeamBench(Bench):

    def setup(self, problem, ndv, nstate, nproc):
        E = 1.
        L = 1.
        b = 0.1
        volume = 0.01
        num_elements = 50 * nstate

        problem.model = BeamGroup(E=E, L=L, b=b, volume=volume, num_elements=num_elements)

    def post_run(problem):
        pass


desvars = [1]
states = [1, 5, 10, 50]
procs = [1]

bench = BeamBench(desvars, states, procs, name='beam')

bench.run_benchmark()








