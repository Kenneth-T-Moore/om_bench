"""
Generate benchmarking data for several pycycle models with assembled an mvp jacobians.
"""
from __future__ import print_function
import os

import numpy as np


class MyBench(Bench):

    def setup(self, problem, ndv, nstate, nproc, flag):
        pass

    def post_setup(self, prob, ndv, nstate, nproc, flag):
        pass

    def post_run(problem):
        # Check stuff here.
        pass


if __name__ == "__main__":

    desvars = [1]

    # These are model numbers. Just generating a chart.
    states = [1, 2]#, 3, 4]
    procs = [1]

    bench = MyBench(desvars, states, procs, mode='fwd', name='pycycleMisc', use_flag=True)
    bench.num_averages = 5
    bench.time_linear = True
    bench.time_driver = False

    bench.run_benchmark()