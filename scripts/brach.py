"""
Generate benchmarking data for the minTimeClimb problem.
"""
from __future__ import print_function
import os

import numpy as np

from openmdao.api import DirectSolver, pyOptSparseDriver
import openmdao.utils.coloring as coloring_mod

from dymos import Phase
from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE

from om_bench.bench import Bench

class ColoringOnly(pyOptSparseDriver):

    def run(self):
        """
        Only does coloring when requested.
        """
        if self.options['dynamic_simul_derivs']:
            coloring_mod.dynamic_simul_coloring(self, do_sparsity=True)


class MyBench(Bench):

    def setup(self, problem, ndv, nstate, nproc, flag):

        simul_derivs = flag
        num_segments = nstate
        transcription = 'radau-ps'
        top_level_jacobian = 'csc'
        transcription_order = 3
        force_alloc_complex=False

        p = problem

        self.phase = phase = Phase(transcription,
                                   ode_class=BrachistochroneODE,
                                   num_segments=num_segments,
                                   transcription_order=transcription_order,
                                   compressed=False)

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

        phase.set_state_options('x', fix_initial=True, fix_final=True)
        phase.set_state_options('y', fix_initial=True, fix_final=True)
        phase.set_state_options('v', fix_initial=True, fix_final=False)

        phase.add_control('theta', continuity=True, rate_continuity=True,
                          units='deg', lower=0.01, upper=179.9)

        phase.add_design_parameter('g', units='m/s**2', opt=False, val=9.80665)

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

        p.model.options['assembled_jac_type'] = top_level_jacobian.lower()
        p.model.linear_solver = DirectSolver(assemble_jac=True)

        p.driver = ColoringOnly()
        p.driver.options['dynamic_simul_derivs'] = simul_derivs

    def post_setup(self, prob, ndv, nstate, nproc, flag):
        phase = self.phase

        prob['phase0.t_initial'] = 0.0
        prob['phase0.t_duration'] = 2.0

        prob['phase0.states:x'] = phase.interpolate(ys=[0, 10], nodes='state_input')
        prob['phase0.states:y'] = phase.interpolate(ys=[10, 5], nodes='state_input')
        prob['phase0.states:v'] = phase.interpolate(ys=[0, 9.9], nodes='state_input')
        prob['phase0.controls:theta'] = phase.interpolate(ys=[0, 100], nodes='control_input')
        prob['phase0.design_parameters:g'] = 9.80665

        # Clean out coloring
        #if os.path.exists('coloring.json'):
        #    os.remove('coloring.json')

    def post_run(self, prob, ndv, nstate, nproc, flag):
        # Check stuff here.
        pass


if __name__ == "__main__":

    desvars = [1]
    states = [1, 2, 4, 6, 12, 24, 48, 100, 200, 500, 1000]
    states = [item * 10 for item in states]
    procs = [1]

    bench = MyBench(desvars, states, procs, mode='auto', name='brach', use_flag=True)
    bench.num_averages = 1
    bench.time_linear = True
    bench.time_driver = True
    bench.single_batch = True
    bench.auto_queue_submit = False

    # Hardcode of/wrt to remove linear constraints form consideration.
    bench.ln_of  = ['phase0.time', 'phase0.collocation_constraint.defects:y', 'phase0.collocation_constraint.defects:x', 'phase0.collocation_constraint.defects:v', 'phase0.continuity_comp.defect_controls:theta', 'phase0.continuity_comp.defect_control_rates:theta_rate']
    bench.ln_wrt = ['phase0.t_duration', 'phase0.controls:theta', 'phase0.states:y', 'phase0.states:x', 'phase0.states:v']

    #bench.run_benchmark()
    bench.run_benchmark_mpi(walltime=8)








