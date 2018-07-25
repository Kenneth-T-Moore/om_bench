"""
Generate benchmarking data for the AMD problem.

Study of Parallel Derivs here.
"""
from __future__ import print_function

import pickle
import os

from six import iteritems

import numpy as np

import amd_om
from amd_om.allocation.airline_networks.prob_128_4 import allocation_data
from amd_om.allocation.airline_networks.general_allocation_data import general_allocation_data

from amd_om.design.utils.flight_conditions import get_flight_conditions

from amd_om.mission_analysis.components.aerodynamics.rans_3d_data import get_aero_smt_model, get_rans_crm_wing
from amd_om.mission_analysis.components.propulsion.b777_engine_data import get_prop_smt_model
from amd_om.mission_analysis.utils.plot_utils import plot_single_mission_altitude, plot_single_mission_data

from amd_om.allocation_mission_design import AllocationMissionDesignGroup

from amd_om.utils.aircraft_data.CRM_full_scale import get_aircraft_data
from amd_om.utils.pyoptsparse_setup import get_pyoptsparse_driver


from om_bench.bench import Bench


class MyBench(Bench):

    def setup(self, problem, ndv, nstate, nproc, flag):

        par_derivs = flag

        allocation_data['num_pt'] = nstate * np.ones(128, int)

        base_dir = amd_om.__file__.split('amd_om/__init__')[0]
        this_dir = os.path.join(base_dir, 'run_scripts/AMD/L3_128_route/')
        grid_dir = os.path.join(base_dir, 'grids/')
        output_dir = this_dir + '_amd_outputs/'

        flight_conditions = get_flight_conditions()

        aeroOptions = {'gridFile' : os.path.join(grid_dir, 'L3_myscaled.cgns'),
                       'writesurfacesolution' : False,
                       'writevolumesolution' : False,
                       'writetecplotsurfacesolution' : False,
                       'grad_scaler' : 10.,
                       }
        meshOptions = {'gridFile' : os.path.join(grid_dir, 'L3_myscaled.cgns')}

        design_variables = ['shape', 'twist', 'sweep', 'area']

        initial_dvs = {}

        optimum_design_filename = '_design_outputs/optimum_design.pkl'
        optimum_design_data = pickle.load(open(os.path.join(this_dir, optimum_design_filename), 'rb'))
        for key in ['shape', 'twist', 'sweep', 'area']:
            initial_dvs[key] = optimum_design_data[key]

        optimum_alloc_filename = '_allocation_outputs/optimum_alloc.pkl'
        optimum_alloc_data = pickle.load(open(os.path.join(this_dir, optimum_alloc_filename), 'rb'))
        for key in ['pax_flt', 'flt_day']:
            initial_dvs[key] = optimum_alloc_data[key]

        initial_mission_vars = {}

        num_routes = allocation_data['num']
        for ind in range(num_routes):
            optimum_mission_filename = '_mission_outputs/optimum_msn_{:03}.pkl'.format(ind)
            optimum_mission_data = pickle.load(open(os.path.join(this_dir, optimum_mission_filename), 'rb'))
            for key in ['h_km_cp', 'M0']:
                initial_mission_vars[ind, key] = optimum_mission_data[key]

        aircraft_data = get_aircraft_data()

        ref_area_m2 = aircraft_data['areaRef_m2']
        Wac_1e6_N = aircraft_data['Wac_1e6_N']
        Mach_mode = 'TAS'

        propulsion_model = get_prop_smt_model()
        aerodynamics_model = get_aero_smt_model()

        xt, yt, xlimits = get_rans_crm_wing()
        aerodynamics_model.xt = xt

        problem.model = AllocationMissionDesignGroup(
            flight_conditions=flight_conditions, aircraft_data=aircraft_data,
            aeroOptions=aeroOptions, meshOptions=meshOptions, design_variables=design_variables,
            general_allocation_data=general_allocation_data, allocation_data=allocation_data,
            ref_area_m2=ref_area_m2, Wac_1e6_N=Wac_1e6_N, Mach_mode=Mach_mode,
            propulsion_model=propulsion_model, aerodynamics_model=aerodynamics_model,
            initial_mission_vars=initial_mission_vars,
            parallel_derivs = par_derivs,
        )

        snopt_file_name = 'SNOPT_print_amd.out'

        problem.driver = get_pyoptsparse_driver()
        problem.driver.opt_settings['Print file'] = os.path.join(output_dir, snopt_file_name)

        self.initial_dvs = initial_dvs

    def post_setup(self, prob, ndv, nstate, nproc, flag):
        # Load initial conditions.
        for key, value in iteritems(self.initial_dvs):
            prob[key] = value

    def post_run(problem):
        # Check stuff here.
        pass


if __name__ == "__main__":

    desvars = [1]
    states = [100, 50]
    procs = [140]

    bench = MyBench(desvars, states, procs, mode='auto', name='AMD', use_flag=True)
    bench.num_averages = 1
    bench.time_linear = True
    bench.time_driver = False
    bench.single_batch = True

    bench.run_benchmark_mpi(walltime=2)








