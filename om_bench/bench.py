"""
Class that assists with generating scaling benchmarking data for OpenMDAO models.
"""
from six.moves import range
import os
from time import time

import numpy as np

from openmdao.api import Problem


class Bench(object):

    def __init__(self, desvars, states, procs, name='bench'):
        """
        Initialize the benchmark assistant class.
        """
        ndv = len(desvars)
        nstate = len(states)
        nproc = len(procs)
        if ndv + nstate + nproc > np.max([ndv, nstate, nproc]) + 2:
            raise ValueError("For now, please only vary one of [states, procs, desvars]")

        if ndv > 1:
            self.mode = 'desvar'
        elif nstate > 1:
            self.mode = 'state'
        elif nproc > 1:
            self.mode = 'proc'

        self.name = name
        #self.basedir = basedir

        self.desvars = desvars
        self.states = states
        self.procs = procs

        # Options
        self.num_averages = 5
        self.time_nonlinear = False
        self.time_linear = True
        self.time_driver = False

        self.base_dir = os.getcwd()

    def setup(self, problem, ndv, nstate, nproc):
        """
        Set up the problem.

        This method is overriden by the user, and is used to build the problem prior to setup.
        """
        pass

    def post_setup(self, problem, ndv, nstate, nproc):
        """
        Perform all post-setup operations before final setup.

        This method is overriden by the user.
        """
        pass

    def run_benchmark(self):
        """
        Run benchmarks and save data.
        """
        desvars = self.desvars
        states = self.states
        procs = self.procs

        data = {}

        for nproc in procs:
            for nstate in states:
                for ndv in desvars:

                    print("\n")
                    print('Running: dv=%d, state=%d, proc=%d' % (ndv, nstate, nproc))
                    print("\n")

                    t1_sum = 0.0
                    t3_sum = 0.0
                    t5_sum = 0.0
                    for j in range(self.num_averages):
                        t1, t3, t5 = self._run_nl_ln_dr(ndv, nstate, nproc)
                        t1_sum += t1
                        t3_sum += t3
                        t5_sum += t5

                    t1_av = t1_sum / (j + 1)
                    t3_av = t3_sum / (j + 1)
                    t5_av = t5_sum / (j + 1)

                    data[ndv, nstate, nproc] = (t1_av, t3_av, t5_av)

        os.chdir(self.base_dir)

    def _run_nl_ln_dr(self, ndv, nstate, nproc):
        """
        Benchmark a single point.

        Nonlinear solve is always run. Linear Solve and Driver are optional.
        """
        prob = Problem()

        # User hook pre setup
        self.setup(prob, ndv, nstate, nproc)

        prob.setup()

        # User hook post setup
        self.post_setup(prob, ndv, nstate, nproc)

        prob.final_setup()

        # Time Execution
        t0 = time()
        prob.run_model()
        t1 = time() - t0
        print("Nonlinear Execution complete:", t1, 'sec')

        if self.time_linear:
            t2 = time()
            prob.compute_totals()
            t3 = time() - t2
            print("Linear Execution complete:", t3, 'sec')
        else:
            t3 = 0.0

        if self.time_driver:
            t4 = time()
            prob.run_driver()
            t5 = time() - t4
            print("Linear Execution complete:", t3, 'sec')
        else:
            t5 = 0.0

        return t1, t3, t5


def PostProccss(filename):
    pass
