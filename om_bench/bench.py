"""
Class that assists with generating scaling benchmarking data for OpenMDAO models.
"""
from six import iteritems
from six.moves import range
from collections import Iterable
import os
import subprocess
import sys
from time import time

import numpy as np

from openmdao.core.problem import Problem

from om_bench.templates import qsub_template, run_template


class Bench(object):

    def __init__(self, desvars, states, procs, name='bench'):
        """
        Initialize the benchmark assistant class.
        """
        if not isinstance(desvars, Iterable):
            desvars = [desvars]
        if not isinstance(states, Iterable):
            states = [states]
        if not isinstance(procs, Iterable):
            procs = [procs]

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
        self.time_nonlinear = True
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

    def post_run(self, problem):
        """
        Perform any post benchmark activities, like testing the result.

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

        # This method only supports single proc.
        if len(procs) > 1 or procs[0] > 1:
            msg = 'This method only supports a single proc. Use run_benchmark_mpi instead.'
            raise RuntimeError(msg)

        data = []

        nproc = 1
        for nstate in states:
            for ndv in desvars:

                print("\n")
                print('Running: dv=%d, state=%d, proc=%d' % (ndv, nstate, nproc))
                print("\n")

                t1_sum = 0.0
                t3_sum = 0.0
                t5_sum = 0.0
                for j in range(self.num_averages):
                    t1, t3, t5 = self._run_nl_ln_drv(ndv, nstate, nproc)
                    t1_sum += t1
                    t3_sum += t3
                    t5_sum += t5

                t1_av = t1_sum / (j + 1)
                t3_av = t3_sum / (j + 1)
                t5_av = t5_sum / (j + 1)

                data.append((ndv, nstate, nproc, t1_av, t3_av, t5_av))

        os.chdir(self.base_dir)

        name = self.name
        mode = self.mode
        op = []
        if self.time_nonlinear:
            op.append('nl')
        if self.time_linear:
            op.append('ln')
        if self.time_driver:
            op.append('drv')
        op = '_'.join(op)

        filename = '%s_%s_%s.dat' % (name, mode, op)

        outfile = open(filename, 'w')
        outfile.write(name)
        outfile.write('\n')
        outfile.write(mode)
        outfile.write('\n')
        outfile.write('%s, %s, %s' % (self.time_nonlinear, self.time_linear, self.time_driver))
        outfile.write('\n')

        for ndv, nstate, nproc, t1, t3, t5 in data:
            outfile.write('%d, %d, %d, %f, %f, %f' % (ndv, nstate, nproc, t1, t3, t5))
            outfile.write('\n')
        outfile.close()

    def run_benchmark_mpi(self, walltime=4):
        """
        Create and submit jobs that run benchmarks and save data.
        """
        self.walltime = walltime

        desvars = self.desvars
        states = self.states
        procs = self.procs

        mode = self.mode
        op = []
        if self.time_nonlinear:
            op.append('nl')
        if self.time_linear:
            op.append('ln')
        if self.time_driver:
            op.append('drv')
        op = '_'.join(op)

        data = []

        for nproc in procs:
            for nstate in states:
                for ndv in desvars:
                    for j in range(self.num_averages):

                        name = '_%s_%s_%s_%d_%d_%d_%d' % (self.name, mode, op, ndv, nstate, nproc, j)

                        # Prepare python code
                        self._prepare_run_script(nproc, nstate, ndv, j, name)

                        # Prepare job submission file
                        self._prepare_pbs_job(nproc, nstate, ndv, j, name)

                        # Submit job
                        p = subprocess.Popen(["qsub", '%s.sh' % name])
                        #command = ". ~/.bashrc; qsub", '%s.py' % name
                        #p = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)

        print("All jobs submitted.")

    def _run_nl_ln_drv(self, ndv, nstate, nproc, use_mpi=False):
        """
        Benchmark a single point.

        Nonlinear solve is always run. Linear Solve and Driver are optional.
        """
        prob = Problem()

        # User hook pre setup
        self.setup(prob, ndv, nstate, nproc)

        # Do this here so that we don't get rejected from running on a head node.
        if use_mpi:
            from openmdao.api import PETScVector
            vector_class = PETScVector
        else:
            from openmdao.vectors.default_vector import DefaultVector
            vector_class = DefaultVector

        vector_class = PETScVector if use_mpi else DefaultVector
        prob.setup(vector_class=vector_class)

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
            print("Driver Execution complete:", t3, 'sec')
        else:
            t5 = 0.0

        self.post_run()

        return t1, t3, t5

    def _prepare_run_script(self, ndv, nstate, nproc, average, name):
        """
        Output run script for mpi submission using template.
        """
        tp = run_template
        tp = tp.replace('<ndv>', str(ndv))
        tp = tp.replace('<nstate>', str(nstate))
        tp = tp.replace('<nproc>', str(nproc))
        tp = tp.replace('<average>', str(average))

        # We need to import from the file that is running.
        module = sys.argv[0].split('/')[-1].strip('.py')
        classname = self.__class__.__name__
        tp = tp.replace('<module>', module)
        tp = tp.replace('<classname>', classname)
        tp = tp.replace('<name>', self.name)
        tp = tp.replace('<filename>', name)
        tp = tp.replace('<time_linear>', str(self.time_linear))
        tp = tp.replace('<time_driver>', str(self.time_driver))

        outname = '%s.py' % name
        outfile = open(outname, 'w')
        outfile.write(tp)
        outfile.close()

    def _prepare_pbs_job(self, ndv, nstate, nproc, average, name):
        """
        Output PBS run submission file using template.
        """
        tp = qsub_template
        proc_node = 24.0

        tp = tp.replace('<name>', name)
        tp = tp.replace('<walltime>', str(self.walltime))

        local = os.getcwd()
        tp = tp.replace('<local>', local)

        # Figure out the number of nodes and procs
        node = int(np.ceil(nproc/proc_node))

        tp = tp.replace('<node>', str(node))
        tp = tp.replace('<nproc>', str(nproc))

        outname = '%s.sh' % name
        outfile = open(outname, 'w')
        outfile.write(tp)
        outfile.close()

