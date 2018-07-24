"""
Generate benchmarking data for the AMD problem.

Study of Parallel Derivs here.
"""
from __future__ import print_function

import numpy as np

from om_bench.bench import Bench


class MyBench(Bench):

    def setup(self, problem, ndv, nstate, nproc, flag):

        par_derivs = flag
    def post_setup(self, prob, ndv, nstate, nproc, flag):
        pass

    def post_run(problem):
        # Check stuff here.
        pass


if __name__ == "__main__":

    desvars = [1]
    states = [1, 2]
    states = [item * 10 for item in states]
    procs = [1]

    bench = MyBench(desvars, states, procs, mode='auto', name='minTimeClimb', use_flag=True)
    bench.num_averages = 1
    bench.time_linear = True
    bench.time_driver = False

    bench.run_benchmark()








