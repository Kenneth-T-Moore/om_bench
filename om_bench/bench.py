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

from om_bench.templates import qsub_template, run_template, qsub_template_single_file, qsub_template_amd


class Bench(object):
    """
    Attributes
    ----------
    ln_of : bool
        Allows override of 'of' list during compute_totals. Default is None, which uses driver vars.
    ln_wrt : bool
        Allows override of 'wrt' list during compute_totals. Default is None, which uses driver vars.
    mode : str
        Derivatives mode string passed into openmdao setup. Can be ('fwd', 'rev')
    num_averages : int
        Number of time to repeat each calculation and save the average time.
    single_file : bool
        If True, then mpi submissions are placed in a single qsub file and submitted as one job;
        if False, then they are submitted separately.
    time_driver : bool(False)
        If True, run the driver (i.e., optimizer) and save timings.
    time_linear : bool(True)
        If True, run the linear solve and save timings.
    time_nonlinear : bool(True)
        If True, save timings from the nonlinear solve. Nonlinear solve always runs regardless.
    _desvars : list
        List of ascending integers that are individually passed in to the problem to request the
        number of design variables.
    _name : string
        Name for this problem. Should be unix-safe but not contain underscores.
    _procs : list
        List of ascending integers that are individually passed in to the problem to request the
        number of processors during mpi execution.
    _run_mode : str
        Determination of which quantity (state, dv, proc) we are varying.
    _states : list
        List of ascending integers that are individually passed in to the problem to request the
        number of states. States should be independent of design variables.
    """

    def __init__(self, desvars, states, procs, name='bench', mode='fwd', use_flag=False):
        """
        Initialize the benchmark assistant class.

        Parameters
        ----------
        desvars : list
            List of ascending integers that are individually passed in to the problem to request the
            number of design variables.
        states : list
            List of ascending integers that are individually passed in to the problem to request the
            number of states. States should be independent of design variables.
        procs : list
            List of ascending integers that are individually passed in to the problem to request the
            number of processors during mpi execution.
        name : string
            Name for this problem. Should be unix-safe but not contain underscores.
        mode : str
            Derivatives mode string passed into openmdao setup. Can be ('fwd', 'rev')
        use_flag : bool
            Set to True to enable a single flag to be turned on and off. All cases will be executed
            with flag set to False and True.
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
            self._run_mode = 'desvar'
        elif nstate > 1:
            self._run_mode = 'state'
        elif nproc > 1:
            self._run_mode = 'proc'
        else:
            self._run_mode = 'state'

        self._name = name
        #self.basedir = basedir

        self._desvars = desvars
        self._states = states
        self._procs = procs
        self._use_flag = use_flag

        # Options
        self.num_averages = 5
        self.time_nonlinear = True
        self.time_linear = True
        self.time_driver = False
        self.single_batch = False
        self.mode = mode
        self.auto_queue_submit = True

        # Custom specification of of/wrt for linear solution.
        self.ln_of = None
        self.ln_wrt = None

        self.base_dir = os.getcwd()

    def setup(self, problem, ndv, nstate, nproc, flag):
        """
        Set up the problem.

        This method is overriden by the user, and is used to build the problem prior to setup.

        Parameters
        ----------
        problem : <Problem>
            Clean OpenMDAO problem object.
        ndv : int
            Number of design variables requested.
        nstate : int
            Number of states requested.
        nproc : int
            Number of processors requested.
        flag : bool
            User assignable flag that will be False or True.
        """
        pass

    def post_setup(self, problem, ndv, nstate, nproc, flag):
        """
        Perform all post-setup operations before final setup.

        This method is overriden by the user.

        Parameters
        ----------
        problem : <Problem>
            Clean OpenMDAO problem object.
        ndv : int
            Number of design variables requested.
        nstate : int
            Number of states requested.
        nproc : int
            Number of processors requested.
        flag : bool
            User assignable flag that will be False or True.
        """
        pass

    def post_run(self, problem):
        """
        Perform any post benchmark activities, like testing the result.

        This method is overriden by the user.

        Parameters
        ----------
        problem : <Problem>
            Clean OpenMDAO problem object.
        ndv : int
            Number of design variables requested.
        nstate : int
            Number of states requested.
        nproc : int
            Number of processors requested.
        """
        pass

    def run_benchmark(self):
        """
        Run benchmarks and save data.
        """
        desvars = self._desvars
        states = self._states
        procs = self._procs

        # This method only supports single proc.
        if len(procs) > 1 or procs[0] > 1:
            msg = 'This method only supports a single proc. Use run_benchmark_mpi instead.'
            raise RuntimeError(msg)

        data = []

        flags = [False]
        if self._use_flag:
            flags.append(True)

        nproc = 1
        for nstate in states:
            for ndv in desvars:
                for flag in flags:

                    print("\n")
                    print('Running: dv=%d, state=%d, proc=%d, flag=%s' % (ndv, nstate, nproc,
                                                                          flag))
                    print("\n")

                    t1_sum = 0.0
                    t3_sum = 0.0
                    t5_sum = 0.0
                    for j in range(self.num_averages):
                        t1, t3, t5 = self._run_nl_ln_drv(ndv, nstate, nproc, flag)
                        t1_sum += t1
                        t3_sum += t3
                        t5_sum += t5

                    t1_av = t1_sum / (j + 1)
                    t3_av = t3_sum / (j + 1)
                    t5_av = t5_sum / (j + 1)

                    data.append((ndv, nstate, nproc, flag, t1_av, t3_av, t5_av))

        os.chdir(self.base_dir)

        name = self._name
        mode = self._run_mode
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

        for ndv, nstate, nproc, flag, t1, t3, t5 in data:
            outfile.write('%d, %d, %d, %s, %f, %f, %f' % (ndv, nstate, nproc, str(flag), t1, t3, t5))
            outfile.write('\n')
        outfile.close()

    def run_benchmark_mpi(self, walltime=4):
        """
        Create and submit jobs that run benchmarks and save data.

        Parameters
        ----------
        walltime : int
            Amount of walltime for the mpi jobs in hours.
        """
        self.walltime = walltime

        desvars = self._desvars
        states = self._states
        procs = self._procs

        mode = self._run_mode
        op = []
        if self.time_nonlinear:
            op.append('nl')
        if self.time_linear:
            op.append('ln')
        if self.time_driver:
            op.append('drv')
        op = '_'.join(op)

        flags = [False]
        if self._use_flag:
            flags.append(True)

        data = []
        commands = []
        for nproc in procs:
            for nstate in states:
                for ndv in desvars:
                    for flag in flags:

                        for j in range(self.num_averages):

                            name = '_%s_%s_%s_%d_%d_%d_%s_%d' % (self._name, mode, op, ndv,
                                                                 nstate, nproc, str(flag), j)

                            # Prepare python code
                            self._prepare_run_script(ndv, nstate, nproc, flag, j, name)

                            if self.single_batch is True:
                                command = "mpiexec -n %d python -u %s.py" % (nproc, name)
                                commands.append(command)

                            else:
                                # Prepare job submission file
                                self._prepare_pbs_job(ndv, nstate, nproc, j, name)

                                # Submit job
                                if self.auto_queue_submit:
                                    p = subprocess.Popen(["qsub", '%s.sh' % name])

        if self.single_batch is True:
            name = '_%s_%s_%s_all' % (self._name, mode, op)

            # Prepare job submission file
            self._prepare_pbs_job_single_file(procs, name, commands)

            # Submit job
            if self.auto_queue_submit:
                p = subprocess.Popen(["qsub", '%s.sh' % name])

        print("All jobs submitted.")

    def _run_nl_ln_drv(self, ndv, nstate, nproc, flag):
        """
        Benchmark a single point.

        Nonlinear solve is always run. Linear Solve and Driver are optional.

        Parameters
        ----------
        ndv : int
            Number of design variables requested.
        nstate : int
            Number of states requested.
        nproc : int
            Number of processors requested.
        flag : bool
            User assignable flag that will be False or True.
        """
        prob = Problem()

        # User hook pre setup
        self.setup(prob, ndv, nstate, nproc, flag)

        prob.setup(mode=self.mode)

        # User hook post setup
        self.post_setup(prob, ndv, nstate, nproc, flag)

        prob.final_setup()

        # Time Execution
        t0 = time()
        prob.run_model()
        t1 = time() - t0
        print("Nonlinear Execution complete:", t1, 'sec')

        if self.time_driver:
            t4 = time()
            prob.run_driver()
            t5 = time() - t4
            print("Driver Execution complete:", t5, 'sec')
        else:
            t5 = 0.0

        if self.time_linear:
            t2 = time()
            prob.compute_totals(of=self.ln_of, wrt=self.ln_wrt, return_format='dict')
            t3 = time() - t2
            print("Linear Execution complete:", t3, 'sec')
        else:
            t3 = 0.0

        self.post_run(prob, ndv, nstate, nproc, flag)

        return t1, t3, t5

    def _prepare_run_script(self, ndv, nstate, nproc, flag, average, name):
        """
        Output run script for mpi submission using template.

        Parameters
        ----------
        ndv : int
            Number of design variables requested.
        nstate : int
            Number of states requested.
        nproc : int
            Number of processors requested.
        flag : bool
            User assignable flag that will be False or True.
        average : int
            Which average we are on.
        name : string
            Unique filename for the output data.
        """
        tp = run_template
        tp = tp.replace('<ndv>', str(ndv))
        tp = tp.replace('<nstate>', str(nstate))
        tp = tp.replace('<nproc>', str(nproc))
        tp = tp.replace('<flag>', str(flag))
        tp = tp.replace('<average>', str(average))

        # We need to import from the file that is running.
        module = sys.argv[0].split('/')[-1].strip('.py')
        classname = self.__class__.__name__
        tp = tp.replace('<module>', module)
        tp = tp.replace('<classname>', classname)
        tp = tp.replace('<name>', self._name)
        tp = tp.replace('<filename>', name)
        tp = tp.replace('<mode>', self.mode)
        tp = tp.replace('<time_linear>', str(self.time_linear))
        tp = tp.replace('<time_driver>', str(self.time_driver))

        outname = '%s.py' % name
        outfile = open(outname, 'w')
        outfile.write(tp)
        outfile.close()

    def _prepare_pbs_job(self, ndv, nstate, nproc, flag, average, name):
        """
        Output PBS run submission file using template.

        Parameters
        ----------
        ndv : int
            Number of design variables requested.
        nstate : int
            Number of states requested.
        nproc : int
            Number of processors requested.
        flag : bool
            User assignable flag that will be False or True.
        average : int
            Which average we are on.
        name : string
            Unique filename for the output data.
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
        tp = tp.replace('<flag>', str(flag))

        outname = '%s.sh' % name
        outfile = open(outname, 'w')
        outfile.write(tp)
        outfile.close()

    def _prepare_pbs_job_single_file(self, procs, name, commands):
        """
        Output PBS run submission file using template, but running all python scripts in one file.
        """
        #tp = qsub_template_single_file
        tp = qsub_template_amd
        proc_node = 24.0

        tp = tp.replace('<name>', name)
        tp = tp.replace('<walltime>', str(self.walltime))

        local = os.getcwd()
        tp = tp.replace('<local>', local)

        # Figure out the number of nodes and procs
        node = int(np.ceil(np.max(procs)/proc_node))

        tp = tp.replace('<node>', str(node))

        cmd_txt = '\n'.join(commands)
        tp = tp.replace('<commands>', cmd_txt)

        outname = '%s.sh' % name
        outfile = open(outname, 'w')
        outfile.write(tp)
        outfile.close()

