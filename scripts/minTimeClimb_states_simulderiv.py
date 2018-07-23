"""
Generate benchmarking data for the minTimeClimb problem.
"""
from __future__ import print_function

import numpy as np

from openmdao.api import pyOptSparseDriver, DirectSolver

from dymos import Phase
from dymos.examples.min_time_climb.min_time_climb_ode import MinTimeClimbODE

from om_bench.bench import Bench


class MyBench(Bench):

    def setup(self, problem, ndv, nstate, nproc, flag):

        simul_derivs = flag
        num_seg = nstate
        transcription = 'gauss-lobatto'
        top_level_jacobian = 'csc'
        transcription_order = 3
        force_alloc_complex=False

        p = problem

        p.driver = pyOptSparseDriver()
        p.driver.options['optimizer'] = 'SLSQP'

        self.phase = phase = Phase(transcription,
                                   ode_class=MinTimeClimbODE,
                                   num_segments=num_seg,
                                   compressed=True,
                                   transcription_order=transcription_order)

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(fix_initial=True, duration_bounds=(50, 400),
                               duration_ref=100.0)

        phase.set_state_options('r', fix_initial=True, lower=0, upper=1.0E6,
                                scaler=1.0E-3, defect_scaler=1.0E-2, units='m')

        phase.set_state_options('h', fix_initial=True, lower=0, upper=20000.0,
                                scaler=1.0E-3, defect_scaler=1.0E-3, units='m')

        phase.set_state_options('v', fix_initial=True, lower=10.0,
                                scaler=1.0E-2, defect_scaler=1.0E-2, units='m/s')

        phase.set_state_options('gam', fix_initial=True, lower=-1.5, upper=1.5,
                                ref=1.0, defect_scaler=1.0, units='rad')

        phase.set_state_options('m', fix_initial=True, lower=10.0, upper=1.0E5,
                                scaler=1.0E-3, defect_scaler=1.0E-3)

        phase.add_control('alpha', units='deg', lower=-8.0, upper=8.0, scaler=1.0,
                          continuity=True, rate_continuity=True, rate2_continuity=False)

        phase.add_design_parameter('S', val=49.2386, units='m**2', opt=False)
        phase.add_design_parameter('Isp', val=1600.0, units='s', opt=False)
        phase.add_design_parameter('throttle', val=1.0, opt=False)

        phase.add_boundary_constraint('h', loc='final', equals=20000, scaler=1.0E-3, units='m')
        phase.add_boundary_constraint('aero.mach', loc='final', equals=1.0, units=None)
        phase.add_boundary_constraint('gam', loc='final', equals=0.0, units='rad')

        phase.add_path_constraint(name='h', lower=100.0, upper=20000, ref=20000)
        phase.add_path_constraint(name='aero.mach', lower=0.1, upper=1.8)
        phase.add_path_constraint(name='alpha', lower=-8, upper=8)

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final')

        p.driver.options['dynamic_simul_derivs'] = simul_derivs
        p.model.options['assembled_jac_type'] = top_level_jacobian.lower()
        p.model.linear_solver = DirectSolver(assemble_jac=True)

        p.setup(check=True, force_alloc_complex=force_alloc_complex)

    def post_setup(self, prob, ndv, nstate, nproc, flag):
        phase = self.phase

        prob['phase0.t_initial'] = 0.0
        prob['phase0.t_duration'] = 298.46902

        prob['phase0.states:r'] = phase.interpolate(ys=[0.0, 111319.54], nodes='state_input')
        prob['phase0.states:h'] = phase.interpolate(ys=[100.0, 20000.0], nodes='state_input')
        prob['phase0.states:v'] = phase.interpolate(ys=[135.964, 283.159], nodes='state_input')
        prob['phase0.states:gam'] = phase.interpolate(ys=[0.0, 0.0], nodes='state_input')
        prob['phase0.states:m'] = phase.interpolate(ys=[19030.468, 16841.431], nodes='state_input')
        prob['phase0.controls:alpha'] = phase.interpolate(ys=[0.0, 0.0], nodes='control_input')

    def post_run(problem):
        # Check stuff here.
        pass


if __name__ == "__main__":

    desvars = [1]
    states = [1, 2, 4, 8]
    states = [item * 10 for item in states]
    procs = [1]

    bench = MyBench(desvars, states, procs, mode='auto', name='minTimeClimb', use_flag=True)
    bench.num_averages = 1
    bench.time_linear = False
    bench.time_driver = True

    bench.run_benchmark()








