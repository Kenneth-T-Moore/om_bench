"""
Generate benchmarking data for several pycycle models with assembled an mvp jacobians.
"""
from __future__ import print_function
import os

import numpy as np

from openmdao.api import Group, IndepVarComp, BalanceComp, ExecComp
from openmdao.api import DirectSolver, BoundsEnforceLS, NewtonSolver

from example_cycles.N_plus_3_ref.N3ref import N3
from pycycle.constants import AIR_MIX, AIR_FUEL_MIX
from pycycle.connect_flow import connect_flow
from pycycle.cea import species_data
from pycycle.elements.api import FlightConditions, Inlet, Compressor, Duct, Nozzle, Performance, \
     Combustor, Turbine, Shaft, Performance
from pycycle.maps.axi5 import AXI5
from pycycle.maps.lpt2269 import LPT2269

from om_bench.bench import Bench


class Propulsor(Group):

    def setup(self):

        thermo_spec = species_data.janaf

        self.add_subsystem('fc', FlightConditions(thermo_data=thermo_spec,
                                                  elements=AIR_MIX))

        self.add_subsystem('inlet', Inlet(design=True, thermo_data=thermo_spec, elements=AIR_MIX))
        self.add_subsystem('fan', Compressor(thermo_data=thermo_spec, elements=AIR_MIX, design=True))
        self.add_subsystem('nozz', Nozzle(thermo_data=thermo_spec, elements=AIR_MIX))
        self.add_subsystem('perf', Performance(num_nozzles=1, num_burners=0))

        self.add_subsystem('shaft', IndepVarComp('Nmech', 1., units='rpm'))

        self.add_subsystem('pwr_balance', BalanceComp('W', units='lbm/s', eq_units='hp', val=50., lower=1., upper=500.),
                           promotes_inputs=[('rhs:W', 'pwr_target')])

        connect_flow(self, 'fc.Fl_O', 'inlet.Fl_I')
        connect_flow(self, 'inlet.Fl_O', 'fan.Fl_I')
        connect_flow(self, 'fan.Fl_O', 'nozz.Fl_I')

        self.connect('shaft.Nmech', 'fan.Nmech')

        self.connect('fc.Fl_O:stat:P', 'nozz.Ps_exhaust')
        self.connect('inlet.Fl_O:tot:P', 'perf.Pt2')
        self.connect('fan.Fl_O:tot:P', 'perf.Pt3')
        self.connect('inlet.F_ram', 'perf.ram_drag')
        self.connect('nozz.Fg', 'perf.Fg_0')

        self.connect('pwr_balance.W', 'fc.fs.W')
        self.connect('fan.power', 'pwr_balance.lhs:W')

        newton = self.nonlinear_solver = NewtonSolver()
        newton.options['atol'] = 1e-12
        newton.options['rtol'] = 1e-12
        newton.options['iprint'] = 2
        newton.options['maxiter'] = 10
        newton.options['solve_subsystems'] = True
        newton.options['max_sub_solves'] = 10
        #
        newton.linesearch = BoundsEnforceLS()
        newton.linesearch.options['bound_enforcement'] = 'scalar'


class Turbojet(Group):

    def initialize(self):
        self.options.declare('design', default=True,
                              desc='Switch between on-design and off-design calculation.')
        self.options.declare('statics', default=True,
                              desc='If True, calculate static properties.')

    def setup(self):

        thermo_spec = species_data.janaf
        design = self.options['design']
        statics = self.options['statics']

        self.add_subsystem('fc', FlightConditions(thermo_data=thermo_spec, elements=AIR_MIX))
        self.add_subsystem('inlet', Inlet(design=design, thermo_data=thermo_spec, elements=AIR_MIX))
        self.add_subsystem('duct1', Duct(design=design, thermo_data=thermo_spec, elements=AIR_MIX, statics=statics))
        self.add_subsystem('comp', Compressor(map_data=AXI5, design=design, thermo_data=thermo_spec, elements=AIR_MIX,
                                        bleed_names=['cool1','cool2'], statics=statics, map_extrap=True),promotes_inputs=['Nmech'])
        self.add_subsystem('burner', Combustor(design=design,thermo_data=thermo_spec,
                                        inflow_elements=AIR_MIX,
                                        air_fuel_elements=AIR_FUEL_MIX,
                                        fuel_type='JP-7', statics=statics))
        self.add_subsystem('turb', Turbine(map_data=LPT2269, design=design, thermo_data=thermo_spec, elements=AIR_FUEL_MIX,
                                        bleed_names=['cool1','cool2'], statics=statics, map_extrap=True),promotes_inputs=['Nmech'])
        self.add_subsystem('ab', Combustor(design=design,thermo_data=thermo_spec,
                                        inflow_elements=AIR_FUEL_MIX,
                                        air_fuel_elements=AIR_FUEL_MIX,
                                        fuel_type='JP-7', statics=statics))
        self.add_subsystem('nozz', Nozzle(nozzType='CD', lossCoef='Cv', thermo_data=thermo_spec, elements=AIR_FUEL_MIX))
        self.add_subsystem('shaft', Shaft(num_ports=2),promotes_inputs=['Nmech'])
        self.add_subsystem('perf', Performance(num_nozzles=1, num_burners=2))

        self.connect('duct1.Fl_O:tot:P', 'perf.Pt2')
        self.connect('comp.Fl_O:tot:P', 'perf.Pt3')
        self.connect('burner.Wfuel', 'perf.Wfuel_0')
        self.connect('ab.Wfuel', 'perf.Wfuel_1')
        self.connect('inlet.F_ram', 'perf.ram_drag')
        self.connect('nozz.Fg', 'perf.Fg_0')

        self.connect('comp.trq', 'shaft.trq_0')
        self.connect('turb.trq', 'shaft.trq_1')
        # self.connect('shaft.Nmech', 'comp.Nmech')
        # self.connect('shaft.Nmech', 'turb.Nmech')
        self.connect('fc.Fl_O:stat:P', 'nozz.Ps_exhaust')

        balance = self.add_subsystem('balance', BalanceComp())
        if design:

            #balance.add_balance('W', units='lbm/s', eq_units='lbf')
            #self.connect('balance.W', 'inlet.Fl_I:stat:W')
            #self.connect('perf.Fn', 'balance.lhs:W')

            balance.add_balance('FAR', eq_units='degR', lower=1e-4, val=.017)
            self.connect('balance.FAR', 'burner.Fl_I:FAR')
            self.connect('burner.Fl_O:tot:T', 'balance.lhs:FAR')

            balance.add_balance('turb_PR', val=1.5, lower=1.001, upper=8, eq_units='hp', rhs_val=0.)
            self.connect('balance.turb_PR', 'turb.PR')
            self.connect('shaft.pwr_net', 'balance.lhs:turb_PR')

            # self.set_order(['fc', 'inlet', 'duct1', 'comp', 'burner', 'turb', 'ab', 'nozz', 'shaft', 'perf', 'thrust_balance', 'temp_balance', 'shaft_balance'])
            self.set_order(['balance', 'fc', 'inlet', 'duct1', 'comp', 'burner', 'turb', 'ab', 'nozz', 'shaft', 'perf'])

        else:

            balance.add_balance('FAR', eq_units='degR', lower=1e-4, val=.017)
            self.connect('balance.FAR', 'burner.Fl_I:FAR')
            self.connect('burner.Fl_O:tot:T', 'balance.lhs:FAR')

            balance.add_balance('Nmech', val=1.5, units='rpm', lower=500., eq_units='hp', rhs_val=0.)
            self.connect('balance.Nmech', 'Nmech')
            self.connect('shaft.pwr_net', 'balance.lhs:Nmech')

            #balance.add_balance('W', val=168.0, units='lbm/s', eq_units=None, rhs_val=2.0)
            #self.connect('balance.W', 'inlet.Fl_I:stat:W')
            #self.connect('comp.map.RlineMap', 'balance.lhs:W')

            self.set_order(['balance', 'fc', 'inlet', 'duct1', 'comp', 'burner', 'turb', 'ab', 'nozz', 'shaft', 'perf'])

        if statics:
            connect_flow(self, 'fc.Fl_O', 'inlet.Fl_I', connect_w=False)
            connect_flow(self, 'inlet.Fl_O', 'duct1.Fl_I')
            connect_flow(self, 'duct1.Fl_O', 'comp.Fl_I')
            connect_flow(self, 'comp.Fl_O', 'burner.Fl_I')
            connect_flow(self, 'burner.Fl_O', 'turb.Fl_I')
            connect_flow(self, 'turb.Fl_O', 'ab.Fl_I')
            connect_flow(self, 'ab.Fl_O', 'nozz.Fl_I')
        else:
            connect_flow(self, 'fc.Fl_O', 'inlet.Fl_I', connect_w=False)
            connect_flow(self, 'inlet.Fl_O', 'duct1.Fl_I', connect_stat=False)
            connect_flow(self, 'duct1.Fl_O', 'comp.Fl_I', connect_stat=False)
            connect_flow(self, 'comp.Fl_O', 'burner.Fl_I', connect_stat=False)
            connect_flow(self, 'burner.Fl_O', 'turb.Fl_I', connect_stat=False)
            connect_flow(self, 'turb.Fl_O', 'ab.Fl_I', connect_stat=False)
            connect_flow(self, 'ab.Fl_O', 'nozz.Fl_I', connect_stat=False)

        connect_flow(self, 'comp.cool1', 'turb.cool1', connect_stat=False)
        connect_flow(self, 'comp.cool2', 'turb.cool2', connect_stat=False)

        newton = self.nonlinear_solver = NewtonSolver()
        newton.options['atol'] = 1e-6
        newton.options['rtol'] = 1e-6
        newton.options['iprint'] = 2
        newton.options['maxiter'] = 15
        newton.options['solve_subsystems'] = True
        newton.options['max_sub_solves'] = 100
        newton.linesearch = BoundsEnforceLS()
        newton.linesearch.options['bound_enforcement'] = 'scalar'
        newton.linesearch.options['iprint'] = -1
        self.linear_solver = DirectSolver(assemble_jac=True)


class MyBench(Bench):

    def setup(self, problem, ndv, nstate, nproc, flag):
        prob = problem

        # Propulsor model
        if nstate == 1143:
            des_vars = prob.model.add_subsystem('des_vars', IndepVarComp(), promotes=["*"])
            des_vars.add_output('alt', 10000., units="m")
            des_vars.add_output('MN', .72)
            des_vars.add_output('inlet_MN', .6)
            des_vars.add_output('FPR', 1.2)
            des_vars.add_output('pwr_target', -2600., units='kW')

            design = prob.model.add_subsystem('design', Propulsor())

            prob.model.connect('alt', 'design.fc.alt')
            prob.model.connect('MN', 'design.fc.MN')
            prob.model.connect('inlet_MN', 'design.inlet.MN')
            prob.model.connect('FPR', 'design.fan.map.PRdes')
            prob.model.connect('pwr_target', 'design.pwr_target')

            prob.model.linear_solver = DirectSolver(assemble_jac=flag)

        # J79 model
        elif nstate == 6317:
            des_vars = prob.model.add_subsystem('des_vars', IndepVarComp(), promotes=["*"])

            # FOR DESIGN
            des_vars.add_output('alt', 0.0, units='ft'),
            des_vars.add_output('MN', 0.000001),
            des_vars.add_output('T4max', 2370.0, units='degR'),
            des_vars.add_output('Fn_des', 11800.0, units='lbf'),
            des_vars.add_output('duct1:dPqP', 0.02),
            des_vars.add_output('comp:PRdes', 13.5),
            des_vars.add_output('comp:effDes', 0.83),
            des_vars.add_output('burn:dPqP', 0.03),
            des_vars.add_output('turb:effDes', 0.86),
            des_vars.add_output('ab:dPqP', 0.06),
            des_vars.add_output('nozz:Cv', 0.99),
            des_vars.add_output('shaft:Nmech', 8070.0, units='rpm'),
            des_vars.add_output('inlet:MN_out', 0.60),
            des_vars.add_output('duct1:MN_out', 0.60),
            des_vars.add_output('comp:MN_out', 0.20),
            des_vars.add_output('burner:MN_out', 0.20),
            des_vars.add_output('turb:MN_out', 0.4),
            des_vars.add_output('ab:MN_out',0.4),
            des_vars.add_output('ab:FAR', 0.000),
            des_vars.add_output('comp:cool1:frac_W', 0.0789),
            des_vars.add_output('comp:cool1:frac_P', 1.0),
            des_vars.add_output('comp:cool1:frac_work', 1.0),
            des_vars.add_output('comp:cool2:frac_W', 0.0383),
            des_vars.add_output('comp:cool2:frac_P', 1.0),
            des_vars.add_output('comp:cool2:frac_work', 1.0),
            des_vars.add_output('turb:cool1:frac_P', 1.0),
            des_vars.add_output('turb:cool2:frac_P', 0.0),

            des_vars.add_output('W', 0.0)
            prob.model.connect('W', 'DESIGN.inlet.Fl_I:stat:W')

            # OFF DESIGN 1

            des_vars.add_output('OD1_MN', 0.000001),
            des_vars.add_output('OD1_alt', 0.0, units='ft'),
            des_vars.add_output('OD1_T4', 2370.0, units='degR'),
            des_vars.add_output('OD1_ab_FAR', 0.031523391),
            des_vars.add_output('OD1_Rline', 2.0),
            # OFF DESIGN 2
            des_vars.add_output('OD2_MN', 0.8),
            des_vars.add_output('OD2_alt', 0.0, units='ft'),
            des_vars.add_output('OD2_T4', 2370.0, units='degR'),
            des_vars.add_output('OD2_ab_FAR', 0.022759941),
            des_vars.add_output('OD2_Rline', 2.0),
            # OFF DESIGN 3
            des_vars.add_output('OD3_MN', 1.0),
            des_vars.add_output('OD3_alt', 15000.0, units='ft'),
            des_vars.add_output('OD3_T4', 2370.0, units='degR'),
            des_vars.add_output('OD3_ab_FAR', 0.036849745),
            des_vars.add_output('OD3_Rline', 2.0),
            # OFF DESIGN 4
            des_vars.add_output('OD4_MN', 1.2),
            des_vars.add_output('OD4_alt', 25000.0, units='ft'),
            des_vars.add_output('OD4_T4', 2370.0, units='degR'),
            des_vars.add_output('OD4_ab_FAR', 0.035266091),
            des_vars.add_output('OD4_Rline', 2.0),
            # OFF DESIGN 5
            des_vars.add_output('OD5_MN', 0.6),
            des_vars.add_output('OD5_alt', 35000.0, units='ft'),
            des_vars.add_output('OD5_T4', 2370.0, units='degR'),
            des_vars.add_output('OD5_ab_FAR', 0.020216221),
            des_vars.add_output('OD5_Rline', 2.0),
            # OFF DESIGN 6
            des_vars.add_output('OD6_MN', 1.6),
            des_vars.add_output('OD6_alt', 35000.0, units='ft'),
            des_vars.add_output('OD6_T4', 2370.0, units='degR'),
            des_vars.add_output('OD6_ab_FAR', 0.038532787),
            des_vars.add_output('OD6_Rline', 2.0),
            # OFF DESIGN 7
            des_vars.add_output('OD7_MN', 1.6),
            des_vars.add_output('OD7_alt', 50000.0, units='ft'),
            des_vars.add_output('OD7_T4', 2370.0, units='degR'),
            des_vars.add_output('OD7_ab_FAR', 0.038532787),
            des_vars.add_output('OD7_Rline', 2.0),
            # OFF DESIGN 8
            des_vars.add_output('OD8_MN', 1.8),
            des_vars.add_output('OD8_alt', 70000.0, units='ft'),
            des_vars.add_output('OD8_T4', 2370.0, units='degR'),
            des_vars.add_output('OD8_ab_FAR', 0.038532787),
            des_vars.add_output('OD8_Rline', 2.0),

            # DESIGN CASE
            prob.model.add_subsystem('DESIGN', Turbojet(statics=True))

            prob.model.connect('alt', 'DESIGN.fc.alt')
            prob.model.connect('MN', 'DESIGN.fc.MN')
            #prob.model.connect('Fn_des', 'DESIGN.balance.rhs:W')
            prob.model.connect('T4max', 'DESIGN.balance.rhs:FAR')

            prob.model.connect('duct1:dPqP', 'DESIGN.duct1.dPqP')
            prob.model.connect('comp:PRdes', 'DESIGN.comp.map.PRdes')
            prob.model.connect('comp:effDes', 'DESIGN.comp.map.effDes')
            prob.model.connect('burn:dPqP', 'DESIGN.burner.dPqP')
            prob.model.connect('turb:effDes', 'DESIGN.turb.map.effDes')
            prob.model.connect('ab:dPqP', 'DESIGN.ab.dPqP')
            prob.model.connect('nozz:Cv', 'DESIGN.nozz.Cv')
            prob.model.connect('shaft:Nmech', 'DESIGN.Nmech')

            prob.model.connect('inlet:MN_out', 'DESIGN.inlet.MN')
            prob.model.connect('duct1:MN_out', 'DESIGN.duct1.MN')
            prob.model.connect('comp:MN_out', 'DESIGN.comp.MN')
            prob.model.connect('burner:MN_out', 'DESIGN.burner.MN')
            prob.model.connect('turb:MN_out', 'DESIGN.turb.MN')
            prob.model.connect('ab:MN_out', 'DESIGN.ab.MN')
            prob.model.connect('ab:FAR', 'DESIGN.ab.Fl_I:FAR')

            prob.model.connect('comp:cool1:frac_W', 'DESIGN.comp.cool1:frac_W')
            prob.model.connect('comp:cool1:frac_P', 'DESIGN.comp.cool1:frac_P')
            prob.model.connect('comp:cool1:frac_work', 'DESIGN.comp.cool1:frac_work')

            prob.model.connect('comp:cool2:frac_W', 'DESIGN.comp.cool2:frac_W')
            prob.model.connect('comp:cool2:frac_P', 'DESIGN.comp.cool2:frac_P')
            prob.model.connect('comp:cool2:frac_work', 'DESIGN.comp.cool2:frac_work')

            prob.model.connect('turb:cool1:frac_P', 'DESIGN.turb.cool1:frac_P')
            prob.model.connect('turb:cool2:frac_P', 'DESIGN.turb.cool2:frac_P')

            # # DESIGN CASE (Fixed)
            # prob.root.add('DESIGN', Turbojet_Fixed_Design())

            # OFF DESIGN CASES
            pts = ['OD1']
            # pts = [] #'OD1','OD2','OD3','OD4','OD5','OD6','OD7','OD8'
            self.OD_statics = True

            for pt in pts:
                prob.model.add_subsystem(pt, Turbojet(design=False, statics=self.OD_statics))

                prob.model.connect('duct1:dPqP', pt+'.duct1.dPqP')
                prob.model.connect('burn:dPqP', pt+'.burner.dPqP')
                prob.model.connect('ab:dPqP', pt+'.ab.dPqP')
                prob.model.connect('nozz:Cv', pt+'.nozz.Cv')

                prob.model.connect('comp:cool1:frac_W', pt+'.comp.cool1:frac_W')
                prob.model.connect('comp:cool1:frac_P', pt+'.comp.cool1:frac_P')
                prob.model.connect('comp:cool1:frac_work', pt+'.comp.cool1:frac_work')

                prob.model.connect('comp:cool2:frac_W', pt+'.comp.cool2:frac_W')
                prob.model.connect('comp:cool2:frac_P', pt+'.comp.cool2:frac_P')
                prob.model.connect('comp:cool2:frac_work', pt+'.comp.cool2:frac_work')

                prob.model.connect('turb:cool1:frac_P', pt+'.turb.cool1:frac_P')
                prob.model.connect('turb:cool2:frac_P', pt+'.turb.cool2:frac_P')

                prob.model.connect(pt+'_alt', pt+'.fc.alt')
                prob.model.connect(pt+'_MN', pt+'.fc.MN')

                prob.model.connect('DESIGN.comp.s_PRdes', pt+'.comp.s_PRdes')
                prob.model.connect('DESIGN.comp.s_WcDes', pt+'.comp.s_WcDes')
                prob.model.connect('DESIGN.comp.s_effDes', pt+'.comp.s_effDes')
                prob.model.connect('DESIGN.comp.s_NcDes', pt+'.comp.s_NcDes')

                prob.model.connect('DESIGN.turb.s_PRdes', pt+'.turb.s_PRdes')
                prob.model.connect('DESIGN.turb.s_WpDes', pt+'.turb.s_WpDes')
                prob.model.connect('DESIGN.turb.s_effDes', pt+'.turb.s_effDes')
                prob.model.connect('DESIGN.turb.s_NpDes', pt+'.turb.s_NpDes')

                if self.OD_statics:
                    prob.model.connect('DESIGN.inlet.Fl_O:stat:area', pt+'.inlet.area')
                    prob.model.connect('DESIGN.duct1.Fl_O:stat:area', pt+'.duct1.area')
                    prob.model.connect('DESIGN.comp.Fl_O:stat:area', pt+'.comp.area')
                    prob.model.connect('DESIGN.burner.Fl_O:stat:area', pt+'.burner.area')
                    prob.model.connect('DESIGN.turb.Fl_O:stat:area', pt+'.turb.area')
                    prob.model.connect('DESIGN.ab.Fl_O:stat:area', pt+'.ab.area')

                prob.model.connect(pt+'_T4', pt+'.balance.rhs:FAR')
                #prob.model.connect(pt+'_Rline', pt+'.balance.rhs:W')
                prob.model.connect('W', pt+'.inlet.Fl_I:stat:W')

            prob.model.linear_solver = DirectSolver(assemble_jac=flag)

            prob.model.add_design_var('T4max')
            #prob.model.add_design_var('T4max')
            prob.model.add_design_var('W')
            prob.model.add_design_var('ab:FAR')
            prob.model.add_objective('DESIGN.perf.TSFC')
            prob.model.add_constraint('W', upper=150.0)
            for pt in pts:
                prob.model.add_constraint(pt + '.perf.Fn', equals=11800)

        # N+3 Model
        elif nstate == 29689:
            des_vars = prob.model.add_subsystem('des_vars', IndepVarComp(), promotes=["*"])

            des_vars.add_output('inlet:ram_recovery', 0.9980),
            des_vars.add_output('fan:PRdes', 1.300),
            des_vars.add_output('fan:effDes', 0.96888),
            des_vars.add_output('fan:effPoly', 0.97),
            des_vars.add_output('splitter:BPR', 23.7281), #23.9878
            des_vars.add_output('duct2:dPqP', 0.0100),
            des_vars.add_output('lpc:PRdes', 3.000),
            des_vars.add_output('lpc:effDes', 0.889513),
            des_vars.add_output('lpc:effPoly', 0.905),
            des_vars.add_output('duct25:dPqP', 0.0150),
            des_vars.add_output('hpc:PRdes', 14.103),
            des_vars.add_output('OPR', 53.6332) #53.635)
            des_vars.add_output('hpc:effDes', 0.847001),
            des_vars.add_output('hpc:effPoly', 0.89),
            des_vars.add_output('burner:dPqP', 0.0400),
            des_vars.add_output('hpt:effDes', 0.922649),
            des_vars.add_output('hpt:effPoly', 0.91),
            des_vars.add_output('duct45:dPqP', 0.0050),
            des_vars.add_output('lpt:effDes', 0.940104),
            des_vars.add_output('lpt:effPoly', 0.92),
            des_vars.add_output('duct5:dPqP', 0.0100),
            des_vars.add_output('core_nozz:Cv', 0.9999),
            des_vars.add_output('duct17:dPqP', 0.0150),
            des_vars.add_output('byp_nozz:Cv', 0.9975),
            des_vars.add_output('fan_shaft:Nmech', 2184.5, units='rpm'),
            des_vars.add_output('lp_shaft:Nmech', 6772.0, units='rpm'),
            des_vars.add_output('lp_shaft:fracLoss', 0.01)
            des_vars.add_output('hp_shaft:Nmech', 20871.0, units='rpm'),
            des_vars.add_output('hp_shaft:HPX', 350.0, units='hp'),

            des_vars.add_output('bld25:sbv:frac_W', 0.0),
            des_vars.add_output('hpc:bld_inlet:frac_W', 0.0),
            des_vars.add_output('hpc:bld_inlet:frac_P', 0.1465),
            des_vars.add_output('hpc:bld_inlet:frac_work', 0.5),
            des_vars.add_output('hpc:bld_exit:frac_W', 0.02),
            des_vars.add_output('hpc:bld_exit:frac_P', 0.1465),
            des_vars.add_output('hpc:bld_exit:frac_work', 0.5),
            des_vars.add_output('hpc:cust:frac_W', 0.0),
            des_vars.add_output('hpc:cust:frac_P', 0.1465),
            des_vars.add_output('hpc:cust:frac_work', 0.35),
            des_vars.add_output('bld3:bld_inlet:frac_W', 0.063660111), #different than NPSS due to Wref
            des_vars.add_output('bld3:bld_exit:frac_W', 0.07037185), #different than NPSS due to Wref
            des_vars.add_output('hpt:bld_inlet:frac_P', 1.0),
            des_vars.add_output('hpt:bld_exit:frac_P', 0.0),
            des_vars.add_output('lpt:bld_inlet:frac_P', 1.0),
            des_vars.add_output('lpt:bld_exit:frac_P', 0.0),
            des_vars.add_output('bypBld:frac_W', 0.0),

            des_vars.add_output('inlet:MN_out', 0.625),
            des_vars.add_output('fan:MN_out', 0.45)
            des_vars.add_output('splitter:MN_out1', 0.45)
            des_vars.add_output('splitter:MN_out2', 0.45)
            des_vars.add_output('duct2:MN_out', 0.45),
            des_vars.add_output('lpc:MN_out', 0.45),
            des_vars.add_output('bld25:MN_out', 0.45),
            des_vars.add_output('duct25:MN_out', 0.45),
            des_vars.add_output('hpc:MN_out', 0.30),
            des_vars.add_output('bld3:MN_out', 0.30)
            des_vars.add_output('burner:MN_out', 0.10),
            des_vars.add_output('hpt:MN_out', 0.30),
            des_vars.add_output('duct45:MN_out', 0.45),
            des_vars.add_output('lpt:MN_out', 0.35),
            des_vars.add_output('duct5:MN_out', 0.25),
            des_vars.add_output('bypBld:MN_out', 0.45),
            des_vars.add_output('duct17:MN_out', 0.45),

            # POINT 1: Top-of-climb (TOC)
            des_vars.add_output('TOC:alt', 35000., units='ft'),
            des_vars.add_output('TOC:MN', 0.8),
            des_vars.add_output('TOC:T4max', 3150.0, units='degR'),
            # des_vars.add_output('FAR', 0.02833)
            des_vars.add_output('TOC:Fn_des', 6073.4, units='lbf'),
            des_vars.add_output('TOC:ram_recovery', 0.9980),
            des_vars.add_output('TR', 0.926470588)

            # POINT 2: Rolling Takeoff (RTO)
            des_vars.add_output('RTO:MN', 0.25),
            des_vars.add_output('RTO:alt', 0.0, units='ft'),
            des_vars.add_output('RTO:Fn_target', 22800.0, units='lbf'), #8950.0
            des_vars.add_output('RTO:dTs', 27.0, units='degR')
            des_vars.add_output('RTO:Ath', 5532.3, units='inch**2')
            des_vars.add_output('RTO:RlineMap', 1.75)
            des_vars.add_output('RTO:T4max', 3400.0, units='degR')
            des_vars.add_output('RTO:W', 1916.13, units='lbm/s')
            des_vars.add_output('RTO:ram_recovery', 0.9970),
            des_vars.add_output('RTO:duct2:dPqP', 0.0073)
            des_vars.add_output('RTO:duct25:dPqP', 0.0138)
            des_vars.add_output('RTO:duct45:dPqP', 0.0051)
            des_vars.add_output('RTO:duct5:dPqP', 0.0058)
            des_vars.add_output('RTO:duct17:dPqP', 0.0132)

            # POINT 3: Sea-Level Static (SLS)
            des_vars.add_output('SLS:MN', 0.000001),
            des_vars.add_output('SLS:alt', 0.0, units='ft'),
            des_vars.add_output('SLS:Fn_target', 28620.9, units='lbf'), #8950.0
            des_vars.add_output('SLS:dTs', 27.0, units='degR')
            des_vars.add_output('SLS:Ath', 6315.6, units='inch**2')
            des_vars.add_output('SLS:RlineMap', 1.75)
            des_vars.add_output('SLS:ram_recovery', 0.9950),
            des_vars.add_output('SLS:duct2:dPqP', 0.0058)
            des_vars.add_output('SLS:duct25:dPqP', 0.0126)
            des_vars.add_output('SLS:duct45:dPqP', 0.0052)
            des_vars.add_output('SLS:duct5:dPqP', 0.0043)
            des_vars.add_output('SLS:duct17:dPqP', 0.0123)

            # POINT 4: Cruise (CRZ)
            des_vars.add_output('CRZ:MN', 0.8),
            des_vars.add_output('CRZ:alt', 35000.0, units='ft'),
            des_vars.add_output('CRZ:Fn_target', 5466.5, units='lbf'), #8950.0
            des_vars.add_output('CRZ:dTs', 0.0, units='degR')
            des_vars.add_output('CRZ:Ath', 4747.1, units='inch**2')
            des_vars.add_output('CRZ:RlineMap', 1.9397)
            des_vars.add_output('CRZ:ram_recovery', 0.9980),
            des_vars.add_output('CRZ:duct2:dPqP', 0.0092)
            des_vars.add_output('CRZ:duct25:dPqP', 0.0138)
            des_vars.add_output('CRZ:duct45:dPqP', 0.0050)
            des_vars.add_output('CRZ:duct5:dPqP', 0.0097)
            des_vars.add_output('CRZ:duct17:dPqP', 0.0148)
            des_vars.add_output('CRZ:VjetRatio', 1.41038)


            # TOC POINT (DESIGN)
            prob.model.add_subsystem('TOC', N3(statics=True))

            prob.model.connect('TOC:alt', 'TOC.fc.alt')
            prob.model.connect('TOC:MN', 'TOC.fc.MN')
            # prob.model.connect('TOC:Fn_des', 'TOC.balance.rhs:W')
            # prob.model.connect('TOC:T4max', 'TOC.balance.rhs:FAR')
            # prob.model.connect('FAR','TOC.burner.Fl_I:FAR')

            prob.model.connect('TOC:ram_recovery', 'TOC.inlet.ram_recovery')
            prob.model.connect('fan:PRdes', ['TOC.fan.map.PRdes', 'TOC.opr_calc.FPR'])
            # prob.model.connect('fan:effDes', 'TOC.fan.map.effDes')
            prob.model.connect('fan:effPoly', 'TOC.balance.rhs:fan_eff')
            # prob.model.connect('splitter:BPR', 'TOC.splitter.BPR')
            prob.model.connect('duct2:dPqP', 'TOC.duct2.dPqP')
            prob.model.connect('lpc:PRdes', ['TOC.lpc.map.PRdes', 'TOC.opr_calc.LPCPR'])
            # prob.model.connect('lpc:effDes', 'TOC.lpc.map.effDes')
            prob.model.connect('lpc:effPoly', 'TOC.balance.rhs:lpc_eff')
            prob.model.connect('duct25:dPqP', 'TOC.duct25.dPqP')
            # prob.model.connect('hpc:PRdes', 'TOC.hpc.map.PRdes')
            # prob.model.connect('OPR', 'TOC.balance.rhs:hpc_PR')
            prob.model.connect('OPR_simple', 'TOC.balance.rhs:hpc_PR')
            # prob.model.connect('hpc:effDes', 'TOC.hpc.map.effDes')
            # prob.model.connect('hpc:effPoly', 'TOC.balance.rhs:hpc_eff')
            prob.model.connect('burner:dPqP', 'TOC.burner.dPqP')
            # prob.model.connect('hpt:effDes', 'TOC.hpt.map.effDes')
            prob.model.connect('hpt:effPoly', 'TOC.balance.rhs:hpt_eff')
            prob.model.connect('duct45:dPqP', 'TOC.duct45.dPqP')
            # prob.model.connect('lpt:effDes', 'TOC.lpt.map.effDes')
            prob.model.connect('lpt:effPoly', 'TOC.balance.rhs:lpt_eff')
            prob.model.connect('duct5:dPqP', 'TOC.duct5.dPqP')
            prob.model.connect('core_nozz:Cv', ['TOC.core_nozz.Cv', 'TOC.ext_ratio.core_Cv'])
            prob.model.connect('duct17:dPqP', 'TOC.duct17.dPqP')
            prob.model.connect('byp_nozz:Cv', ['TOC.byp_nozz.Cv', 'TOC.ext_ratio.byp_Cv'])
            prob.model.connect('fan_shaft:Nmech', 'TOC.Fan_Nmech')
            prob.model.connect('lp_shaft:Nmech', 'TOC.LP_Nmech')
            prob.model.connect('lp_shaft:fracLoss', 'TOC.lp_shaft.fracLoss')
            prob.model.connect('hp_shaft:Nmech', 'TOC.HP_Nmech')
            prob.model.connect('hp_shaft:HPX', 'TOC.hp_shaft.HPX')

            prob.model.connect('bld25:sbv:frac_W', 'TOC.bld25.sbv:frac_W')
            prob.model.connect('hpc:bld_inlet:frac_W', 'TOC.hpc.bld_inlet:frac_W')
            prob.model.connect('hpc:bld_inlet:frac_P', 'TOC.hpc.bld_inlet:frac_P')
            prob.model.connect('hpc:bld_inlet:frac_work', 'TOC.hpc.bld_inlet:frac_work')
            prob.model.connect('hpc:bld_exit:frac_W', 'TOC.hpc.bld_exit:frac_W')
            prob.model.connect('hpc:bld_exit:frac_P', 'TOC.hpc.bld_exit:frac_P')
            prob.model.connect('hpc:bld_exit:frac_work', 'TOC.hpc.bld_exit:frac_work')
            # prob.model.connect('bld3:bld_inlet:frac_W', 'TOC.bld3.bld_inlet:frac_W')
            # prob.model.connect('bld3:bld_exit:frac_W', 'TOC.bld3.bld_exit:frac_W')
            prob.model.connect('hpc:cust:frac_W', 'TOC.hpc.cust:frac_W')
            prob.model.connect('hpc:cust:frac_P', 'TOC.hpc.cust:frac_P')
            prob.model.connect('hpc:cust:frac_work', 'TOC.hpc.cust:frac_work')
            prob.model.connect('hpt:bld_inlet:frac_P', 'TOC.hpt.bld_inlet:frac_P')
            prob.model.connect('hpt:bld_exit:frac_P', 'TOC.hpt.bld_exit:frac_P')
            prob.model.connect('lpt:bld_inlet:frac_P', 'TOC.lpt.bld_inlet:frac_P')
            prob.model.connect('lpt:bld_exit:frac_P', 'TOC.lpt.bld_exit:frac_P')
            prob.model.connect('bypBld:frac_W', 'TOC.byp_bld.bypBld:frac_W')

            prob.model.connect('inlet:MN_out', 'TOC.inlet.MN')
            prob.model.connect('fan:MN_out', 'TOC.fan.MN')
            prob.model.connect('splitter:MN_out1', 'TOC.splitter.MN1')
            prob.model.connect('splitter:MN_out2', 'TOC.splitter.MN2')
            prob.model.connect('duct2:MN_out', 'TOC.duct2.MN')
            prob.model.connect('lpc:MN_out', 'TOC.lpc.MN')
            prob.model.connect('bld25:MN_out', 'TOC.bld25.MN')
            prob.model.connect('duct25:MN_out', 'TOC.duct25.MN')
            prob.model.connect('hpc:MN_out', 'TOC.hpc.MN')
            prob.model.connect('bld3:MN_out', 'TOC.bld3.MN')
            prob.model.connect('burner:MN_out', 'TOC.burner.MN')
            prob.model.connect('hpt:MN_out', 'TOC.hpt.MN')
            prob.model.connect('duct45:MN_out', 'TOC.duct45.MN')
            prob.model.connect('lpt:MN_out', 'TOC.lpt.MN')
            prob.model.connect('duct5:MN_out', 'TOC.duct5.MN')
            prob.model.connect('bypBld:MN_out', 'TOC.byp_bld.MN')
            prob.model.connect('duct17:MN_out', 'TOC.duct17.MN')

            # OTHER POINTS (OFF-DESIGN)
            pts = ['RTO','SLS','CRZ']
            self.OD_statics = True

            prob.model.connect('RTO:Fn_target', 'RTO.balance.rhs:FAR')
            # prob.model.connect('SLS:Fn_target', 'SLS.balance.rhs:FAR')
            # prob.model.connect('CRZ:Fn_target', 'CRZ.balance.rhs:FAR')

            prob.model.add_subsystem('RTO', N3(design=False, statics=self.OD_statics, cooling=True))
            prob.model.add_subsystem('SLS', N3(design=False, statics=self.OD_statics))
            prob.model.add_subsystem('CRZ', N3(design=False, statics=self.OD_statics))


            for pt in pts:
                # ODpt.nonlinear_solver.options['maxiter'] = 0

                prob.model.connect(pt+':alt', pt+'.fc.alt')
                prob.model.connect(pt+':MN', pt+'.fc.MN')
                # prob.model.connect(pt+':Fn_target', pt+'.balance.rhs:FAR')
                prob.model.connect(pt+':dTs', pt+'.fc.dTs')
                # prob.model.connect(pt+':Ath',pt+'.balance.rhs:BPR')
                prob.model.connect(pt+':RlineMap',pt+'.balance.rhs:BPR')

                # prob.model.connect(pt+':cust_fracW', pt+'.hpc.cust:frac_W')

                prob.model.connect(pt+':ram_recovery', pt+'.inlet.ram_recovery')
                # prob.model.connect('splitter:BPR', pt+'.splitter.BPR')
                # prob.model.connect(pt+':duct2:dPqP', pt+'.duct2.dPqP')
                prob.model.connect('TOC.duct2.s_dPqP', pt+'.duct2.s_dPqP')
                # prob.model.connect(pt+':duct25:dPqP', pt+'.duct25.dPqP')
                prob.model.connect('TOC.duct25.s_dPqP', pt+'.duct25.s_dPqP')
                prob.model.connect('burner:dPqP', pt+'.burner.dPqP')
                # prob.model.connect(pt+':duct45:dPqP', pt+'.duct45.dPqP')
                prob.model.connect('TOC.duct45.s_dPqP', pt+'.duct45.s_dPqP')
                # prob.model.connect(pt+':duct5:dPqP', pt+'.duct5.dPqP')
                prob.model.connect('TOC.duct5.s_dPqP', pt+'.duct5.s_dPqP')
                prob.model.connect('core_nozz:Cv', [pt+'.core_nozz.Cv', pt+'.ext_ratio.core_Cv'])
                # prob.model.connect(pt+':duct17:dPqP', pt+'.duct17.dPqP')
                prob.model.connect('TOC.duct17.s_dPqP', pt+'.duct17.s_dPqP')
                prob.model.connect('byp_nozz:Cv', [pt+'.byp_nozz.Cv', pt+'.ext_ratio.byp_Cv'])
                prob.model.connect('lp_shaft:fracLoss', pt+'.lp_shaft.fracLoss')
                prob.model.connect('hp_shaft:HPX', pt+'.hp_shaft.HPX')

                prob.model.connect('bld25:sbv:frac_W', pt+'.bld25.sbv:frac_W')
                prob.model.connect('hpc:bld_inlet:frac_W', pt+'.hpc.bld_inlet:frac_W')
                prob.model.connect('hpc:bld_inlet:frac_P', pt+'.hpc.bld_inlet:frac_P')
                prob.model.connect('hpc:bld_inlet:frac_work', pt+'.hpc.bld_inlet:frac_work')
                prob.model.connect('hpc:bld_exit:frac_W', pt+'.hpc.bld_exit:frac_W')
                prob.model.connect('hpc:bld_exit:frac_P', pt+'.hpc.bld_exit:frac_P')
                prob.model.connect('hpc:bld_exit:frac_work', pt+'.hpc.bld_exit:frac_work')
                # prob.model.connect('bld3:bld_inlet:frac_W', pt+'.bld3.bld_inlet:frac_W')
                # prob.model.connect('bld3:bld_exit:frac_W', pt+'.bld3.bld_exit:frac_W')
                # prob.model.connect('TOC.balance.hpt_chrg_cool_frac', pt+'.bld3.bld_inlet:frac_W')
                # prob.model.connect('TOC.balance.hpt_nochrg_cool_frac', pt+'.bld3.bld_exit:frac_W')
                prob.model.connect('hpc:cust:frac_W', pt+'.hpc.cust:frac_W')
                prob.model.connect('hpc:cust:frac_P', pt+'.hpc.cust:frac_P')
                prob.model.connect('hpc:cust:frac_work', pt+'.hpc.cust:frac_work')
                prob.model.connect('hpt:bld_inlet:frac_P', pt+'.hpt.bld_inlet:frac_P')
                prob.model.connect('hpt:bld_exit:frac_P', pt+'.hpt.bld_exit:frac_P')
                prob.model.connect('lpt:bld_inlet:frac_P', pt+'.lpt.bld_inlet:frac_P')
                prob.model.connect('lpt:bld_exit:frac_P', pt+'.lpt.bld_exit:frac_P')
                prob.model.connect('bypBld:frac_W', pt+'.byp_bld.bypBld:frac_W')

                prob.model.connect('TOC.fan.s_PRdes', pt+'.fan.s_PRdes')
                prob.model.connect('TOC.fan.s_WcDes', pt+'.fan.s_WcDes')
                prob.model.connect('TOC.fan.s_effDes', pt+'.fan.s_effDes')
                prob.model.connect('TOC.fan.s_NcDes', pt+'.fan.s_NcDes')
                prob.model.connect('TOC.lpc.s_PRdes', pt+'.lpc.s_PRdes')
                prob.model.connect('TOC.lpc.s_WcDes', pt+'.lpc.s_WcDes')
                prob.model.connect('TOC.lpc.s_effDes', pt+'.lpc.s_effDes')
                prob.model.connect('TOC.lpc.s_NcDes', pt+'.lpc.s_NcDes')
                prob.model.connect('TOC.hpc.s_PRdes', pt+'.hpc.s_PRdes')
                prob.model.connect('TOC.hpc.s_WcDes', pt+'.hpc.s_WcDes')
                prob.model.connect('TOC.hpc.s_effDes', pt+'.hpc.s_effDes')
                prob.model.connect('TOC.hpc.s_NcDes', pt+'.hpc.s_NcDes')
                prob.model.connect('TOC.hpt.s_PRdes', pt+'.hpt.s_PRdes')
                prob.model.connect('TOC.hpt.s_WpDes', pt+'.hpt.s_WpDes')
                prob.model.connect('TOC.hpt.s_effDes', pt+'.hpt.s_effDes')
                prob.model.connect('TOC.hpt.s_NpDes', pt+'.hpt.s_NpDes')
                prob.model.connect('TOC.lpt.s_PRdes', pt+'.lpt.s_PRdes')
                prob.model.connect('TOC.lpt.s_WpDes', pt+'.lpt.s_WpDes')
                prob.model.connect('TOC.lpt.s_effDes', pt+'.lpt.s_effDes')
                prob.model.connect('TOC.lpt.s_NpDes', pt+'.lpt.s_NpDes')

                prob.model.connect('TOC.gearbox.gear_ratio', pt+'.gearbox.gear_ratio')
                prob.model.connect('TOC.core_nozz.Throat:stat:area',pt+'.balance.rhs:W')
                # prob.model.connect('TOC.byp_nozz.Throat:stat:area',pt+'.balance.rhs:BPR')

                if self.OD_statics:
                    prob.model.connect('TOC.inlet.Fl_O:stat:area', pt+'.inlet.area')
                    prob.model.connect('TOC.fan.Fl_O:stat:area', pt+'.fan.area')
                    prob.model.connect('TOC.splitter.Fl_O1:stat:area', pt+'.splitter.area1')
                    prob.model.connect('TOC.splitter.Fl_O2:stat:area', pt+'.splitter.area2')
                    prob.model.connect('TOC.duct2.Fl_O:stat:area', pt+'.duct2.area')
                    prob.model.connect('TOC.lpc.Fl_O:stat:area', pt+'.lpc.area')
                    prob.model.connect('TOC.bld25.Fl_O:stat:area', pt+'.bld25.area')
                    prob.model.connect('TOC.duct25.Fl_O:stat:area', pt+'.duct25.area')
                    prob.model.connect('TOC.hpc.Fl_O:stat:area', pt+'.hpc.area')
                    prob.model.connect('TOC.bld3.Fl_O:stat:area', pt+'.bld3.area')
                    prob.model.connect('TOC.burner.Fl_O:stat:area', pt+'.burner.area')
                    prob.model.connect('TOC.hpt.Fl_O:stat:area', pt+'.hpt.area')
                    prob.model.connect('TOC.duct45.Fl_O:stat:area', pt+'.duct45.area')
                    prob.model.connect('TOC.lpt.Fl_O:stat:area', pt+'.lpt.area')
                    prob.model.connect('TOC.duct5.Fl_O:stat:area', pt+'.duct5.area')
                    prob.model.connect('TOC.byp_bld.Fl_O:stat:area', pt+'.byp_bld.area')
                    prob.model.connect('TOC.duct17.Fl_O:stat:area', pt+'.duct17.area')


            prob.model.connect('RTO.balance.hpt_chrg_cool_frac', 'TOC.bld3.bld_exit:frac_W')
            prob.model.connect('RTO.balance.hpt_nochrg_cool_frac', 'TOC.bld3.bld_inlet:frac_W')

            prob.model.connect('RTO.balance.hpt_chrg_cool_frac', 'SLS.bld3.bld_exit:frac_W')
            prob.model.connect('RTO.balance.hpt_nochrg_cool_frac', 'SLS.bld3.bld_inlet:frac_W')

            prob.model.connect('RTO.balance.hpt_chrg_cool_frac', 'CRZ.bld3.bld_exit:frac_W')
            prob.model.connect('RTO.balance.hpt_nochrg_cool_frac', 'CRZ.bld3.bld_inlet:frac_W')


            bal = prob.model.add_subsystem('bal', BalanceComp())
            bal.add_balance('TOC_BPR', val=23.7281, units=None, eq_units='ft/s', use_mult=True)
            prob.model.connect('bal.TOC_BPR', 'TOC.splitter.BPR')
            prob.model.connect('CRZ.ext_ratio.ER', 'bal.lhs:TOC_BPR')
            prob.model.connect('CRZ:VjetRatio', 'bal.rhs:TOC_BPR')

            bal.add_balance('TOC_W', val=820.95, units='lbm/s', eq_units='degR')
            prob.model.connect('bal.TOC_W', 'TOC.fc.fs.W')
            prob.model.connect('RTO.burner.Fl_O:tot:T', 'bal.lhs:TOC_W')
            prob.model.connect('RTO:T4max','bal.rhs:TOC_W')

            bal.add_balance('CRZ_Fn_target', val=5514.4, units='lbf', eq_units='lbf', use_mult=True, mult_val=0.9, ref0=5000.0, ref=7000.0)
            prob.model.connect('bal.CRZ_Fn_target', 'CRZ.balance.rhs:FAR')
            prob.model.connect('TOC.perf.Fn', 'bal.lhs:CRZ_Fn_target')
            prob.model.connect('CRZ.perf.Fn','bal.rhs:CRZ_Fn_target')

            bal.add_balance('SLS_Fn_target', val=28620.8, units='lbf', eq_units='lbf', use_mult=True, mult_val=1.2553, ref0=28000.0, ref=30000.0)
            prob.model.connect('bal.SLS_Fn_target', 'SLS.balance.rhs:FAR')
            prob.model.connect('RTO.perf.Fn', 'bal.lhs:SLS_Fn_target')
            prob.model.connect('SLS.perf.Fn','bal.rhs:SLS_Fn_target')

            prob.model.add_subsystem('T4_ratio',
                    ExecComp('TOC_T4 = RTO_T4*TR',
                            RTO_T4={'value': 3400.0, 'units':'degR'},
                            TOC_T4={'value': 3150.0, 'units':'degR'},
                            TR={'value': 0.926470588, 'units': None}))
            prob.model.connect('RTO:T4max','T4_ratio.RTO_T4')
            prob.model.connect('T4_ratio.TOC_T4', 'TOC.balance.rhs:FAR')
            prob.model.connect('TR', 'T4_ratio.TR')
            prob.model.set_order(['des_vars', 'T4_ratio', 'TOC', 'RTO', 'SLS', 'CRZ', 'bal'])

            newton = prob.model.nonlinear_solver = NewtonSolver()
            newton.options['atol'] = 1e-6
            newton.options['rtol'] = 1e-6
            newton.options['iprint'] = 2
            newton.options['maxiter'] = 20
            newton.options['solve_subsystems'] = True
            newton.options['max_sub_solves'] = 10
            newton.options['err_on_maxiter'] = True
            # newton.linesearch =  ArmijoGoldsteinLS()
            newton.linesearch =  BoundsEnforceLS()
            # newton.linesearch.options['maxiter'] = 2
            newton.linesearch.options['bound_enforcement'] = 'scalar'
            # newton.linesearch.options['print_bound_enforce'] = True
            newton.linesearch.options['iprint'] = -1
            # newton.linesearch.options['print_bound_enforce'] = False
            # newton.linesearch.options['alpha'] = 0.5

            prob.model.linear_solver = DirectSolver(assemble_jac=flag)

            # setup the optimization
            prob.model.add_design_var('fan:PRdes', lower=1.20, upper=1.4)
            prob.model.add_design_var('lpc:PRdes', lower=2.0, upper=4.0)
            prob.model.add_design_var('OPR', lower=40.0, upper=70.0, ref0=40.0, ref=70.0)
            prob.model.add_design_var('RTO:T4max', lower=3000.0, upper=3600.0, ref0=3000.0, ref=3600.0)
            prob.model.add_design_var('CRZ:VjetRatio', lower=1.35, upper=1.45, ref0=1.35, ref=1.45)
            prob.model.add_design_var('TR', lower=0.5, upper=0.95, ref0=0.5, ref=0.95)

            # prob.model.add_objective('CRZ.perf.TSFC')
            prob.model.add_objective('TOC.perf.TSFC')

            # to add the constraint to the model
            prob.model.add_constraint('TOC.fan_dia.FanDia', upper=100.0, ref=100.0)
            prob.model.add_constraint('TOC.perf.Fn', lower=5800.0, ref=6000.0)

        prob.set_solver_print(level=-1)
        prob.set_solver_print(level=2, depth=1)

    def post_setup(self, prob, ndv, nstate, nproc, flag):

        if nstate == 6317:
            # initial guesses
            prob['DESIGN.balance.FAR'] = 0.0175506829934
            prob['W'] = 168.453135137
            prob['DESIGN.balance.turb_PR'] = 4.46138725662
            prob['DESIGN.fc.balance.Pt'] = 14.6955113159
            prob['DESIGN.fc.balance.Tt'] = 518.665288153

            pts = ['OD1']
            for pt in pts:

                # OD3 Guesses
                #prob[pt+'.balance.W'] = 166.073
                prob[pt+'.balance.FAR'] = 0.01680
                prob[pt+'.balance.Nmech'] = 8197.38
                prob[pt+'.fc.balance.Pt'] = 15.703
                prob[pt+'.fc.balance.Tt'] = 558.31
                prob[pt+'.turb.PR'] = 4.6690

        elif nstate == 29689:
            prob['RTO.hpt_cooling.x_factor'] = 0.9

            # initial guesses
            prob['TOC.balance.FAR'] = 0.02650
            prob['bal.TOC_W'] = 820.95
            prob['TOC.balance.lpt_PR'] = 10.937
            prob['TOC.balance.hpt_PR'] = 4.185
            prob['TOC.fc.balance.Pt'] = 5.272
            prob['TOC.fc.balance.Tt'] = 444.41

            pts = ['RTO','SLS','CRZ']
            for pt in pts:

                if pt == 'RTO':
                    prob[pt+'.balance.FAR'] = 0.02832
                    prob[pt+'.balance.W'] = 1916.13
                    prob[pt+'.balance.BPR'] = 25.5620
                    prob[pt+'.balance.fan_Nmech'] = 2132.6
                    prob[pt+'.balance.lp_Nmech'] = 6611.2
                    prob[pt+'.balance.hp_Nmech'] = 22288.2
                    prob[pt+'.fc.balance.Pt'] = 15.349
                    prob[pt+'.fc.balance.Tt'] = 552.49
                    prob[pt+'.hpt.PR'] = 4.210
                    prob[pt+'.lpt.PR'] = 8.161
                    prob[pt+'.fan.map.RlineMap'] = 1.7500
                    prob[pt+'.lpc.map.RlineMap'] = 2.0052
                    prob[pt+'.hpc.map.RlineMap'] = 2.0589
                    prob[pt+'.gearbox.trq_base'] = 52509.1

                if pt == 'SLS':
                    prob[pt+'.balance.FAR'] = 0.02541
                    prob[pt+'.balance.W'] = 1734.44
                    prob[pt+'.balance.BPR'] = 27.3467
                    prob[pt+'.balance.fan_Nmech'] = 1953.1
                    prob[pt+'.balance.lp_Nmech'] = 6054.5
                    prob[pt+'.balance.hp_Nmech'] = 21594.0
                    prob[pt+'.fc.balance.Pt'] = 14.696
                    prob[pt+'.fc.balance.Tt'] = 545.67
                    prob[pt+'.hpt.PR'] = 4.245
                    prob[pt+'.lpt.PR'] = 7.001
                    prob[pt+'.fan.map.RlineMap'] = 1.7500
                    prob[pt+'.lpc.map.RlineMap'] = 1.8632
                    prob[pt+'.hpc.map.RlineMap'] = 2.0281
                    prob[pt+'.gearbox.trq_base'] = 41779.4

                if pt == 'CRZ':
                    prob[pt+'.balance.FAR'] = 0.02510
                    prob[pt+'.balance.W'] = 802.79
                    prob[pt+'.balance.BPR'] = 24.3233
                    prob[pt+'.balance.fan_Nmech'] = 2118.7
                    prob[pt+'.balance.lp_Nmech'] = 6567.9
                    prob[pt+'.balance.hp_Nmech'] = 20574.1
                    prob[pt+'.fc.balance.Pt'] = 5.272
                    prob[pt+'.fc.balance.Tt'] = 444.41
                    prob[pt+'.hpt.PR'] = 4.197
                    prob[pt+'.lpt.PR'] = 10.803
                    prob[pt+'.fan.map.RlineMap'] = 1.9397
                    prob[pt+'.lpc.map.RlineMap'] = 2.1075
                    prob[pt+'.hpc.map.RlineMap'] = 1.9746
                    prob[pt+'.gearbox.trq_base'] = 22369.7

    def post_run(self, prob, ndv, nstate, nproc, flag):
        # Check stuff here.

        if nstate == 1143:

            print("foo FS Fl_O:tot:P", prob['design.fc.Fl_O:tot:P'])
            print("foo FS Fl_O:stat:P", prob['design.fc.Fl_O:stat:P'])


            print("design")

            print("shaft power (hp)", prob['design.fan.power'])
            print("W (lbm/s)", prob['design.pwr_balance.W'])
            print()
            print("MN", prob['MN'])
            print("FS Fl_O:stat:P", cu(prob['design.fc.Fl_O:stat:P'], 'lbf/inch**2', 'Pa'))
            print("FS Fl_O:stat:T", cu(prob['design.fc.Fl_O:stat:T'], 'degR', 'degK'))
            print("FS Fl_O:tot:P", cu(prob['design.fc.Fl_O:tot:P'], 'lbf/inch**2', 'Pa'))
            print("FS Fl_O:tot:T", cu(prob['design.fc.Fl_O:stat:T'], 'degR', 'degK'))
            print()
            print("Inlet Fl_O:stat:P", cu(prob['design.inlet.Fl_O:stat:P'], 'lbf/inch**2', 'Pa'))
            print("Inlet Fl_O:stat:area", prob['design.inlet.Fl_O:stat:area'])
            print("Fan Fl_O:stat:W", prob['design.inlet.Fl_O:stat:W'])

        if nstate == 6317:

            pts = ['OD1']
            for pt in ['DESIGN']+pts:

                if pt == 'DESIGN':
                    MN = prob['MN']
                    PR = prob['DESIGN.balance.turb_PR']
                    FAR = prob['DESIGN.balance.FAR']
                    AB_FAR = prob['ab:FAR']
                else:
                    MN = prob[pt+'_MN']
                    PR = prob[pt+'.turb.PR']
                    FAR = prob[pt+'.balance.FAR']
                    AB_FAR = prob[pt+'_ab_FAR']

                print("----------------------------------------------------------------------------")
                print("                              POINT:", pt)
                print("----------------------------------------------------------------------------")
                print("                       PERFORMANCE CHARACTERISTICS")
                print("    Mach      Alt       W      Fn      Fg    Fram     OPR     TSFC      Ath")
                print(" %7.5f  %7.1f %7.3f %7.1f %7.1f %7.1f %7.3f  %7.5f  %7.3f" %(MN, prob[pt+'.fc.alt'],prob[pt+'.inlet.Fl_O:stat:W'],prob[pt+'.perf.Fn'],prob[pt+'.perf.Fg'],prob[pt+'.inlet.F_ram'],prob[pt+'.perf.OPR'],prob[pt+'.perf.TSFC'], prob[pt+'.nozz.Throat:stat:area']))
                print("----------------------------------------------------------------------------")
                print("                         FLOW STATION PROPERTIES")
                print("Component        Pt      Tt      ht       S       W      MN       V        A")
                if pt == 'DESIGN' or self.OD_statics:
                    print("Start       %7.3f %7.2f %7.3f %7.4f %7.3f %7.4f %7.2f      " %(prob[pt+'.fc.Fl_O:tot:P'], prob[pt+'.fc.Fl_O:tot:T'], prob[pt+'.fc.Fl_O:tot:h'], prob[pt+'.fc.Fl_O:tot:S'], prob[pt+'.fc.Fl_O:stat:W'], prob[pt+'.fc.Fl_O:stat:MN'], prob[pt+'.fc.Fl_O:stat:V']))
                    print("Inlet       %7.3f %7.2f %7.3f %7.4f %7.3f %7.4f %7.2f %8.2f" %(prob[pt+'.inlet.Fl_O:tot:P'], prob[pt+'.inlet.Fl_O:tot:T'], prob[pt+'.inlet.Fl_O:tot:h'], prob[pt+'.inlet.Fl_O:tot:S'], prob[pt+'.inlet.Fl_O:stat:W'], prob[pt+'.inlet.Fl_O:stat:MN'], prob[pt+'.inlet.Fl_O:stat:V'], prob[pt+'.inlet.Fl_O:stat:area']))
                    print("Duct1       %7.3f %7.2f %7.3f %7.4f %7.3f %7.4f %7.2f %8.2f" %(prob[pt+'.duct1.Fl_O:tot:P'], prob[pt+'.duct1.Fl_O:tot:T'], prob[pt+'.duct1.Fl_O:tot:h'], prob[pt+'.duct1.Fl_O:tot:S'], prob[pt+'.duct1.Fl_O:stat:W'], prob[pt+'.duct1.Fl_O:stat:MN'], prob[pt+'.duct1.Fl_O:stat:V'], prob[pt+'.duct1.Fl_O:stat:area']))
                    print("Compressor  %7.3f %7.2f %7.3f %7.4f %7.3f %7.4f %7.2f %8.2f" %(prob[pt+'.comp.Fl_O:tot:P'], prob[pt+'.comp.Fl_O:tot:T'], prob[pt+'.comp.Fl_O:tot:h'], prob[pt+'.comp.Fl_O:tot:S'], prob[pt+'.comp.Fl_O:stat:W'], prob[pt+'.comp.Fl_O:stat:MN'], prob[pt+'.comp.Fl_O:stat:V'], prob[pt+'.comp.Fl_O:stat:area']))
                    print("Burner      %7.3f %7.2f %7.3f %7.4f %7.3f %7.4f %7.2f %8.2f" %(prob[pt+'.burner.Fl_O:tot:P'], prob[pt+'.burner.Fl_O:tot:T'], prob[pt+'.burner.Fl_O:tot:h'], prob[pt+'.burner.Fl_O:tot:S'], prob[pt+'.burner.Fl_O:stat:W'], prob[pt+'.burner.Fl_O:stat:MN'], prob[pt+'.burner.Fl_O:stat:V'], prob[pt+'.burner.Fl_O:stat:area']))
                    print("Turbine     %7.3f %7.2f %7.3f %7.4f %7.3f %7.4f %7.2f %8.2f" %(prob[pt+'.turb.Fl_O:tot:P'], prob[pt+'.turb.Fl_O:tot:T'], prob[pt+'.turb.Fl_O:tot:h'], prob[pt+'.turb.Fl_O:tot:S'], prob[pt+'.turb.Fl_O:stat:W'], prob[pt+'.turb.Fl_O:stat:MN'], prob[pt+'.turb.Fl_O:stat:V'], prob[pt+'.turb.Fl_O:stat:area']))
                    print("Afterburner %7.3f %7.2f %7.3f %7.4f %7.3f %7.4f %7.2f %8.2f" %(prob[pt+'.ab.Fl_O:tot:P'], prob[pt+'.ab.Fl_O:tot:T'], prob[pt+'.ab.Fl_O:tot:h'], prob[pt+'.ab.Fl_O:tot:S'], prob[pt+'.ab.Fl_O:stat:W'], prob[pt+'.ab.Fl_O:stat:MN'], prob[pt+'.ab.Fl_O:stat:V'], prob[pt+'.ab.Fl_O:stat:area']))
                    print("Nozzle      %7.3f %7.2f %7.3f %7.4f %7.3f %7.4f %7.2f %8.2f" %(prob[pt+'.nozz.Fl_O:tot:P'], prob[pt+'.nozz.Fl_O:tot:T'], prob[pt+'.nozz.Fl_O:tot:h'], prob[pt+'.nozz.Fl_O:tot:S'], prob[pt+'.nozz.Fl_O:stat:W'], prob[pt+'.nozz.Fl_O:stat:MN'], prob[pt+'.nozz.Fl_O:stat:V'], prob[pt+'.nozz.Fl_O:stat:area']))
                else:
                    print("Start       %7.3f %7.2f %7.3f %7.4f %7.3f " %(prob[pt+'.fc.Fl_O:tot:P'], prob[pt+'.fc.Fl_O:tot:T'], prob[pt+'.fc.Fl_O:tot:h'], prob[pt+'.fc.Fl_O:tot:S'], prob[pt+'.fc.Fl_O:stat:W']))
                    print("Inlet       %7.3f %7.2f %7.3f %7.4f %7.3f " %(prob[pt+'.inlet.Fl_O:tot:P'], prob[pt+'.inlet.Fl_O:tot:T'], prob[pt+'.inlet.Fl_O:tot:h'], prob[pt+'.inlet.Fl_O:tot:S'], prob[pt+'.inlet.Fl_O:stat:W']))
                    print("Duct1       %7.3f %7.2f %7.3f %7.4f %7.3f " %(prob[pt+'.duct1.Fl_O:tot:P'], prob[pt+'.duct1.Fl_O:tot:T'], prob[pt+'.duct1.Fl_O:tot:h'], prob[pt+'.duct1.Fl_O:tot:S'], prob[pt+'.duct1.Fl_O:stat:W']))
                    print("Compressor  %7.3f %7.2f %7.3f %7.4f %7.3f " %(prob[pt+'.comp.Fl_O:tot:P'], prob[pt+'.comp.Fl_O:tot:T'], prob[pt+'.comp.Fl_O:tot:h'], prob[pt+'.comp.Fl_O:tot:S'], prob[pt+'.comp.Fl_O:stat:W']))
                    print("Burner      %7.3f %7.2f %7.3f %7.4f %7.3f " %(prob[pt+'.burner.Fl_O:tot:P'], prob[pt+'.burner.Fl_O:tot:T'], prob[pt+'.burner.Fl_O:tot:h'], prob[pt+'.burner.Fl_O:tot:S'], prob[pt+'.burner.Fl_O:stat:W']))
                    print("Turbine     %7.3f %7.2f %7.3f %7.4f %7.3f " %(prob[pt+'.turb.Fl_O:tot:P'], prob[pt+'.turb.Fl_O:tot:T'], prob[pt+'.turb.Fl_O:tot:h'], prob[pt+'.turb.Fl_O:tot:S'], prob[pt+'.turb.Fl_O:stat:W']))
                    print("Afterburner %7.3f %7.2f %7.3f %7.4f %7.3f " %(prob[pt+'.ab.Fl_O:tot:P'], prob[pt+'.ab.Fl_O:tot:T'], prob[pt+'.ab.Fl_O:tot:h'], prob[pt+'.ab.Fl_O:tot:S'], prob[pt+'.ab.Fl_O:stat:W']))
                    print("Nozzle      %7.3f %7.2f %7.3f %7.4f %7.3f " %(prob[pt+'.nozz.Fl_O:tot:P'], prob[pt+'.nozz.Fl_O:tot:T'], prob[pt+'.nozz.Fl_O:tot:h'], prob[pt+'.nozz.Fl_O:tot:S'], prob[pt+'.nozz.Fl_O:stat:W']))
                print("----------------------------------------------------------------------------")
                print("                        TURBOMACHINERY PROPERTIES")
                print("Component      Wc/p      PR     eff    Nc/p      pwr")
                print("Compressor  %7.3f %7.4f %7.5f %7.2f %8.1f" %(prob[pt+'.comp.Wc'],prob[pt+'.comp.PR'],prob[pt+'.comp.eff'],prob[pt+'.comp.Nc'],prob[pt+'.shaft.pwr_out']))
                print("Turbine     %7.3f %7.4f %7.5f %7.2f %8.1f" %(prob[pt+'.turb.Wp'], PR, prob[pt+'.turb.eff'],prob[pt+'.turb.Np'],prob[pt+'.shaft.pwr_in']))
                print("----------------------------------------------------------------------------")
                print("                            BURNER PROPERTIES")
                print("Component      dPqP   TtOut   Wfuel      FAR")
                print("Burner      %7.2f %7.2f %7.4f  %7.5f" %(prob[pt+'.burner.dPqP'], prob[pt+'.burner.Fl_O:tot:T'],prob[pt+'.burner.Wfuel'], FAR))
                print("Afterburner %7.2f %7.2f %7.4f  %7.5f" %(prob[pt+'.ab.dPqP'], prob[pt+'.ab.Fl_O:tot:T'],prob[pt+'.ab.Wfuel'], AB_FAR))
                print("----------------------------------------------------------------------------")
                print("                            NOZZLE PROPERTIES")
                print("Component        PR      Cv     Ath    MNth   MNout       V      Fg")
                print("Nozzle      %7.4f %7.4f %7.3f %7.4f %7.4f %7.1f %7.1f" %(prob[pt+'.nozz.PR'],prob[pt+'.nozz.Cv'],prob[pt+'.nozz.Throat:stat:area'],prob[pt+'.nozz.Throat:stat:MN'],prob[pt+'.nozz.Fl_O:stat:MN'],prob[pt+'.nozz.Fl_O:stat:V'],prob[pt+'.nozz.Fg']))
                print("----------------------------------------------------------------------------")
                print("                            SHAFT PROPERTIES")
                print("Component     Nmech    trqin   trqout    pwrin   pwrout")
                print("Shaft       %7.2f %8.1f %8.1f %8.1f %8.1f" %(prob[pt+'.shaft.Nmech'],prob[pt+'.shaft.trq_in'],prob[pt+'.shaft.trq_out'],prob[pt+'.shaft.pwr_in'],prob[pt+'.shaft.pwr_out']))
                print("----------------------------------------------------------------------------")
                print("                            BLEED PROPERTIES")
                print("Bleed       Wb/Win   Pfrac Workfrac       W      Tt      ht      Pt")
                print("Cool1      %7.4f %7.4f %8.4f %7.4f %7.2f %7.3f %7.3f" %(prob[pt+'.comp.cool1:frac_W'],prob[pt+'.comp.cool1:frac_P'],prob[pt+'.comp.cool1:frac_work'],prob[pt+'.comp.cool1:stat:W'],prob[pt+'.comp.cool1:tot:T'],prob[pt+'.comp.cool1:tot:h'],prob[pt+'.comp.cool1:tot:P']))
                print("Cool2      %7.4f %7.4f %8.4f %7.4f %7.2f %7.3f %7.3f" %(prob[pt+'.comp.cool2:frac_W'],prob[pt+'.comp.cool2:frac_P'],prob[pt+'.comp.cool2:frac_work'],prob[pt+'.comp.cool2:stat:W'],prob[pt+'.comp.cool2:tot:T'],prob[pt+'.comp.cool2:tot:h'],prob[pt+'.comp.cool2:tot:P']))
                print("----------------------------------------------------------------------------")
                print()

                print(pt+'.comp.s_PRdes', prob[pt+'.comp.s_PRdes'][0])
                print(pt+'.comp.s_WcDes', prob[pt+'.comp.s_WcDes'][0])
                print(pt+'.comp.s_effDes', prob[pt+'.comp.s_effDes'][0])
                print(pt+'.comp.s_NcDes', prob[pt+'.comp.s_NcDes'][0])

                print(pt+'.turb.s_PRdes', prob[pt+'.turb.s_PRdes'][0])
                print(pt+'.turb.s_WpDes', prob[pt+'.turb.s_WpDes'][0])
                print(pt+'.turb.s_effDes', prob[pt+'.turb.s_effDes'][0])
                print(pt+'.turb.s_NpDes', prob[pt+'.turb.s_NpDes'][0])

                print('DESIGN.inlet.Fl_O:stat:area', prob['DESIGN.inlet.Fl_O:stat:area'][0])
                print('DESIGN.duct1.Fl_O:stat:area', prob['DESIGN.duct1.Fl_O:stat:area'][0])
                print('DESIGN.comp.Fl_O:stat:area', prob['DESIGN.comp.Fl_O:stat:area'][0])
                print('DESIGN.burner.Fl_O:stat:area', prob['DESIGN.burner.Fl_O:stat:area'][0])
                print('DESIGN.turb.Fl_O:stat:area', prob['DESIGN.turb.Fl_O:stat:area'][0])
                print('DESIGN.ab.Fl_O:stat:area', prob['DESIGN.ab.Fl_O:stat:area'][0])
                print()

        elif nstate == 29689:

            pts = ['RTO','SLS','CRZ']
            for pt in ['TOC']+pts:

                if pt == 'TOC':
                    MN = prob['TOC:MN']
                    LPT_PR = prob['TOC.balance.lpt_PR']
                    HPT_PR = prob['TOC.balance.hpt_PR']
                    FAR = prob['TOC.balance.FAR']
                    duct2_dPqP = prob['duct2:dPqP']
                    duct25_dPqP = prob['duct25:dPqP']
                    duct45_dPqP = prob['duct45:dPqP']
                    duct5_dPqP = prob['duct5:dPqP']
                    duct17_dPqP = prob['duct17:dPqP']
                    # FAR = prob['FAR']
                else:
                    MN = prob[pt+':MN']
                    LPT_PR = prob[pt+'.lpt.PR']
                    HPT_PR = prob[pt+'.hpt.PR']
                    FAR = prob[pt+'.balance.FAR']

                    duct2_dPqP = prob[pt+'.duct2.dPqP']
                    duct25_dPqP = prob[pt+'.duct25.dPqP']
                    duct45_dPqP = prob[pt+'.duct45.dPqP']
                    duct5_dPqP = prob[pt+'.duct5.dPqP']
                    duct17_dPqP = prob[pt+'.duct17.dPqP']


                print("----------------------------------------------------------------------------")
                print("                              POINT:", pt)
                print("----------------------------------------------------------------------------")
                print("                       PERFORMANCE CHARACTERISTICS")
                print("    Mach      Alt       W      Fn      Fg    Fram     OPR     TSFC      BPR ")
                print(" %7.5f  %7.1f %7.3f %7.1f %7.1f %7.1f %7.3f  %7.5f  %7.3f" %(MN, prob[pt+'.fc.alt'],prob[pt+'.inlet.Fl_O:stat:W'],prob[pt+'.perf.Fn'],prob[pt+'.perf.Fg'],prob[pt+'.inlet.F_ram'],prob[pt+'.perf.OPR'],prob[pt+'.perf.TSFC'], prob[pt+'.splitter.BPR']))
                print("----------------------------------------------------------------------------")
                print("                         FLOW STATION PROPERTIES")
                print("Component        Pt      Tt      ht       S       W      MN       V        A")
                if pt == 'TOC' or self.OD_statics:
                    print("Start       %7.3f %7.2f %7.2f %7.4f %7.3f %7.4f %7.2f      " %(prob[pt+'.fc.Fl_O:tot:P'], prob[pt+'.fc.Fl_O:tot:T'], prob[pt+'.fc.Fl_O:tot:h'], prob[pt+'.fc.Fl_O:tot:S'], prob[pt+'.fc.Fl_O:stat:W'], prob[pt+'.fc.Fl_O:stat:MN'], prob[pt+'.fc.Fl_O:stat:V']))
                    print("Inlet       %7.3f %7.2f %7.2f %7.4f %7.3f %7.4f %7.2f %8.2f" %(prob[pt+'.inlet.Fl_O:tot:P'], prob[pt+'.inlet.Fl_O:tot:T'], prob[pt+'.inlet.Fl_O:tot:h'], prob[pt+'.inlet.Fl_O:tot:S'], prob[pt+'.inlet.Fl_O:stat:W'], prob[pt+'.inlet.Fl_O:stat:MN'], prob[pt+'.inlet.Fl_O:stat:V'], prob[pt+'.inlet.Fl_O:stat:area']))
                    print("Fan         %7.3f %7.2f %7.2f %7.4f %7.3f %7.4f %7.2f %8.2f" %(prob[pt+'.fan.Fl_O:tot:P'], prob[pt+'.fan.Fl_O:tot:T'], prob[pt+'.fan.Fl_O:tot:h'], prob[pt+'.fan.Fl_O:tot:S'], prob[pt+'.fan.Fl_O:stat:W'], prob[pt+'.fan.Fl_O:stat:MN'], prob[pt+'.fan.Fl_O:stat:V'], prob[pt+'.fan.Fl_O:stat:area']))
                    print("Splitter1   %7.3f %7.2f %7.2f %7.4f %7.3f %7.4f %7.2f %8.2f" %(prob[pt+'.splitter.Fl_O1:tot:P'], prob[pt+'.splitter.Fl_O1:tot:T'], prob[pt+'.splitter.Fl_O1:tot:h'], prob[pt+'.splitter.Fl_O1:tot:S'], prob[pt+'.splitter.Fl_O1:stat:W'], prob[pt+'.splitter.Fl_O1:stat:MN'], prob[pt+'.splitter.Fl_O1:stat:V'], prob[pt+'.splitter.Fl_O1:stat:area']))
                    print("Splitter2   %7.3f %7.2f %7.2f %7.4f %7.3f %7.4f %7.2f %8.2f" %(prob[pt+'.splitter.Fl_O2:tot:P'], prob[pt+'.splitter.Fl_O2:tot:T'], prob[pt+'.splitter.Fl_O2:tot:h'], prob[pt+'.splitter.Fl_O2:tot:S'], prob[pt+'.splitter.Fl_O2:stat:W'], prob[pt+'.splitter.Fl_O2:stat:MN'], prob[pt+'.splitter.Fl_O2:stat:V'], prob[pt+'.splitter.Fl_O2:stat:area']))
                    print("Duct2       %7.3f %7.2f %7.2f %7.4f %7.3f %7.4f %7.2f %8.2f" %(prob[pt+'.duct2.Fl_O:tot:P'], prob[pt+'.duct2.Fl_O:tot:T'], prob[pt+'.duct2.Fl_O:tot:h'], prob[pt+'.duct2.Fl_O:tot:S'], prob[pt+'.duct2.Fl_O:stat:W'], prob[pt+'.duct2.Fl_O:stat:MN'], prob[pt+'.duct2.Fl_O:stat:V'], prob[pt+'.duct2.Fl_O:stat:area']))
                    print("LPC         %7.3f %7.2f %7.2f %7.4f %7.3f %7.4f %7.2f %8.2f" %(prob[pt+'.lpc.Fl_O:tot:P'], prob[pt+'.lpc.Fl_O:tot:T'], prob[pt+'.lpc.Fl_O:tot:h'], prob[pt+'.lpc.Fl_O:tot:S'], prob[pt+'.lpc.Fl_O:stat:W'], prob[pt+'.lpc.Fl_O:stat:MN'], prob[pt+'.lpc.Fl_O:stat:V'], prob[pt+'.lpc.Fl_O:stat:area']))
                    print("Duct25      %7.3f %7.2f %7.2f %7.4f %7.3f %7.4f %7.2f %8.2f" %(prob[pt+'.duct25.Fl_O:tot:P'], prob[pt+'.duct25.Fl_O:tot:T'], prob[pt+'.duct25.Fl_O:tot:h'], prob[pt+'.duct25.Fl_O:tot:S'], prob[pt+'.duct25.Fl_O:stat:W'], prob[pt+'.duct25.Fl_O:stat:MN'], prob[pt+'.duct25.Fl_O:stat:V'], prob[pt+'.duct25.Fl_O:stat:area']))
                    print("HPC         %7.3f %7.2f %7.2f %7.4f %7.3f %7.4f %7.2f %8.2f" %(prob[pt+'.hpc.Fl_O:tot:P'], prob[pt+'.hpc.Fl_O:tot:T'], prob[pt+'.hpc.Fl_O:tot:h'], prob[pt+'.hpc.Fl_O:tot:S'], prob[pt+'.hpc.Fl_O:stat:W'], prob[pt+'.hpc.Fl_O:stat:MN'], prob[pt+'.hpc.Fl_O:stat:V'], prob[pt+'.hpc.Fl_O:stat:area']))
                    print("Bld3        %7.3f %7.2f %7.2f %7.4f %7.3f %7.4f %7.2f %8.2f" %(prob[pt+'.bld3.Fl_O:tot:P'], prob[pt+'.bld3.Fl_O:tot:T'], prob[pt+'.bld3.Fl_O:tot:h'], prob[pt+'.bld3.Fl_O:tot:S'], prob[pt+'.bld3.Fl_O:stat:W'], prob[pt+'.bld3.Fl_O:stat:MN'], prob[pt+'.bld3.Fl_O:stat:V'], prob[pt+'.bld3.Fl_O:stat:area']))
                    print("Burner      %7.3f %7.2f %7.2f %7.4f %7.3f %7.4f %7.2f %8.2f" %(prob[pt+'.burner.Fl_O:tot:P'], prob[pt+'.burner.Fl_O:tot:T'], prob[pt+'.burner.Fl_O:tot:h'], prob[pt+'.burner.Fl_O:tot:S'], prob[pt+'.burner.Fl_O:stat:W'], prob[pt+'.burner.Fl_O:stat:MN'], prob[pt+'.burner.Fl_O:stat:V'], prob[pt+'.burner.Fl_O:stat:area']))
                    print("HPT         %7.3f %7.2f %7.2f %7.4f %7.3f %7.4f %7.2f %8.2f" %(prob[pt+'.hpt.Fl_O:tot:P'], prob[pt+'.hpt.Fl_O:tot:T'], prob[pt+'.hpt.Fl_O:tot:h'], prob[pt+'.hpt.Fl_O:tot:S'], prob[pt+'.hpt.Fl_O:stat:W'], prob[pt+'.hpt.Fl_O:stat:MN'], prob[pt+'.hpt.Fl_O:stat:V'], prob[pt+'.hpt.Fl_O:stat:area']))
                    print("Duct45      %7.3f %7.2f %7.2f %7.4f %7.3f %7.4f %7.2f %8.2f" %(prob[pt+'.duct45.Fl_O:tot:P'], prob[pt+'.duct45.Fl_O:tot:T'], prob[pt+'.duct45.Fl_O:tot:h'], prob[pt+'.duct45.Fl_O:tot:S'], prob[pt+'.duct45.Fl_O:stat:W'], prob[pt+'.duct45.Fl_O:stat:MN'], prob[pt+'.duct45.Fl_O:stat:V'], prob[pt+'.duct45.Fl_O:stat:area']))
                    print("LPT         %7.3f %7.2f %7.2f %7.4f %7.3f %7.4f %7.2f %8.2f" %(prob[pt+'.lpt.Fl_O:tot:P'], prob[pt+'.lpt.Fl_O:tot:T'], prob[pt+'.lpt.Fl_O:tot:h'], prob[pt+'.lpt.Fl_O:tot:S'], prob[pt+'.lpt.Fl_O:stat:W'], prob[pt+'.lpt.Fl_O:stat:MN'], prob[pt+'.lpt.Fl_O:stat:V'], prob[pt+'.lpt.Fl_O:stat:area']))
                    print("Duct5       %7.3f %7.2f %7.2f %7.4f %7.3f %7.4f %7.2f %8.2f" %(prob[pt+'.duct5.Fl_O:tot:P'], prob[pt+'.duct5.Fl_O:tot:T'], prob[pt+'.duct5.Fl_O:tot:h'], prob[pt+'.duct5.Fl_O:tot:S'], prob[pt+'.duct5.Fl_O:stat:W'], prob[pt+'.duct5.Fl_O:stat:MN'], prob[pt+'.duct5.Fl_O:stat:V'], prob[pt+'.duct5.Fl_O:stat:area']))
                    print("CoreNozz    %7.3f %7.2f %7.2f %7.4f %7.3f %7.4f %7.2f %8.2f" %(prob[pt+'.core_nozz.Fl_O:tot:P'], prob[pt+'.core_nozz.Fl_O:tot:T'], prob[pt+'.core_nozz.Fl_O:tot:h'], prob[pt+'.core_nozz.Fl_O:tot:S'], prob[pt+'.core_nozz.Fl_O:stat:W'], prob[pt+'.core_nozz.Fl_O:stat:MN'], prob[pt+'.core_nozz.Fl_O:stat:V'], prob[pt+'.core_nozz.Fl_O:stat:area']))
                    print("BypBld      %7.3f %7.2f %7.2f %7.4f %7.3f %7.4f %7.2f %8.2f" %(prob[pt+'.byp_bld.Fl_O:tot:P'], prob[pt+'.byp_bld.Fl_O:tot:T'], prob[pt+'.byp_bld.Fl_O:tot:h'], prob[pt+'.byp_bld.Fl_O:tot:S'], prob[pt+'.byp_bld.Fl_O:stat:W'], prob[pt+'.byp_bld.Fl_O:stat:MN'], prob[pt+'.byp_bld.Fl_O:stat:V'], prob[pt+'.byp_bld.Fl_O:stat:area']))
                    print("Duct17      %7.3f %7.2f %7.2f %7.4f %7.3f %7.4f %7.2f %8.2f" %(prob[pt+'.duct17.Fl_O:tot:P'], prob[pt+'.duct17.Fl_O:tot:T'], prob[pt+'.duct17.Fl_O:tot:h'], prob[pt+'.duct17.Fl_O:tot:S'], prob[pt+'.duct17.Fl_O:stat:W'], prob[pt+'.duct17.Fl_O:stat:MN'], prob[pt+'.duct17.Fl_O:stat:V'], prob[pt+'.duct17.Fl_O:stat:area']))
                    print("BypNozz     %7.3f %7.2f %7.2f %7.4f %7.3f %7.4f %7.2f %8.2f" %(prob[pt+'.byp_nozz.Fl_O:tot:P'], prob[pt+'.byp_nozz.Fl_O:tot:T'], prob[pt+'.byp_nozz.Fl_O:tot:h'], prob[pt+'.byp_nozz.Fl_O:tot:S'], prob[pt+'.byp_nozz.Fl_O:stat:W'], prob[pt+'.byp_nozz.Fl_O:stat:MN'], prob[pt+'.byp_nozz.Fl_O:stat:V'], prob[pt+'.byp_nozz.Fl_O:stat:area']))
                else:
                    print("Start       %7.3f %7.2f %7.2f %7.4f %7.3f " %(prob[pt+'.fc.Fl_O:tot:P'], prob[pt+'.fc.Fl_O:tot:T'], prob[pt+'.fc.Fl_O:tot:h'], prob[pt+'.fc.Fl_O:tot:S'], prob[pt+'.fc.Fl_O:stat:W']))
                    print("Inlet       %7.3f %7.2f %7.2f %7.4f %7.3f " %(prob[pt+'.inlet.Fl_O:tot:P'], prob[pt+'.inlet.Fl_O:tot:T'], prob[pt+'.inlet.Fl_O:tot:h'], prob[pt+'.inlet.Fl_O:tot:S'], prob[pt+'.inlet.Fl_O:stat:W']))
                    print("Fan         %7.3f %7.2f %7.2f %7.4f %7.3f " %(prob[pt+'.fan.Fl_O:tot:P'], prob[pt+'.fan.Fl_O:tot:T'], prob[pt+'.fan.Fl_O:tot:h'], prob[pt+'.fan.Fl_O:tot:S'], prob[pt+'.fan.Fl_O:stat:W']))
                    print("Splitter1   %7.3f %7.2f %7.2f %7.4f %7.3f " %(prob[pt+'.splitter.Fl_O1:tot:P'], prob[pt+'.splitter.Fl_O1:tot:T'], prob[pt+'.splitter.Fl_O1:tot:h'], prob[pt+'.splitter.Fl_O1:tot:S'], prob[pt+'.splitter.Fl_O1:stat:W']))
                    print("Splitter2   %7.3f %7.2f %7.2f %7.4f %7.3f " %(prob[pt+'.splitter.Fl_O2:tot:P'], prob[pt+'.splitter.Fl_O2:tot:T'], prob[pt+'.splitter.Fl_O2:tot:h'], prob[pt+'.splitter.Fl_O2:tot:S'], prob[pt+'.splitter.Fl_O2:stat:W']))
                    print("Duct2       %7.3f %7.2f %7.2f %7.4f %7.3f " %(prob[pt+'.duct2.Fl_O:tot:P'], prob[pt+'.duct2.Fl_O:tot:T'], prob[pt+'.duct2.Fl_O:tot:h'], prob[pt+'.duct2.Fl_O:tot:S'], prob[pt+'.duct2.Fl_O:stat:W']))
                    print("LPC         %7.3f %7.2f %7.2f %7.4f %7.3f " %(prob[pt+'.lpc.Fl_O:tot:P'], prob[pt+'.lpc.Fl_O:tot:T'], prob[pt+'.lpc.Fl_O:tot:h'], prob[pt+'.lpc.Fl_O:tot:S'], prob[pt+'.lpc.Fl_O:stat:W']))
                    print("Duct25      %7.3f %7.2f %7.2f %7.4f %7.3f " %(prob[pt+'.duct25.Fl_O:tot:P'], prob[pt+'.duct25.Fl_O:tot:T'], prob[pt+'.duct25.Fl_O:tot:h'], prob[pt+'.duct25.Fl_O:tot:S'], prob[pt+'.duct25.Fl_O:stat:W']))
                    print("HPC         %7.3f %7.2f %7.2f %7.4f %7.3f " %(prob[pt+'.hpc.Fl_O:tot:P'], prob[pt+'.hpc.Fl_O:tot:T'], prob[pt+'.hpc.Fl_O:tot:h'], prob[pt+'.hpc.Fl_O:tot:S'], prob[pt+'.hpc.Fl_O:stat:W']))
                    print("Bld3        %7.3f %7.2f %7.2f %7.4f %7.3f " %(prob[pt+'.bld3.Fl_O:tot:P'], prob[pt+'.bld3.Fl_O:tot:T'], prob[pt+'.bld3.Fl_O:tot:h'], prob[pt+'.bld3.Fl_O:tot:S'], prob[pt+'.bld3.Fl_O:stat:W']))
                    print("Burner      %7.3f %7.2f %7.2f %7.4f %7.3f " %(prob[pt+'.burner.Fl_O:tot:P'], prob[pt+'.burner.Fl_O:tot:T'], prob[pt+'.burner.Fl_O:tot:h'], prob[pt+'.burner.Fl_O:tot:S'], prob[pt+'.burner.Fl_O:stat:W']))
                    print("HPT         %7.3f %7.2f %7.2f %7.4f %7.3f " %(prob[pt+'.hpt.Fl_O:tot:P'], prob[pt+'.hpt.Fl_O:tot:T'], prob[pt+'.hpt.Fl_O:tot:h'], prob[pt+'.hpt.Fl_O:tot:S'], prob[pt+'.hpt.Fl_O:stat:W']))
                    print("Duct45      %7.3f %7.2f %7.2f %7.4f %7.3f " %(prob[pt+'.duct45.Fl_O:tot:P'], prob[pt+'.duct45.Fl_O:tot:T'], prob[pt+'.duct45.Fl_O:tot:h'], prob[pt+'.duct45.Fl_O:tot:S'], prob[pt+'.duct45.Fl_O:stat:W']))
                    print("LPT         %7.3f %7.2f %7.2f %7.4f %7.3f " %(prob[pt+'.lpt.Fl_O:tot:P'], prob[pt+'.lpt.Fl_O:tot:T'], prob[pt+'.lpt.Fl_O:tot:h'], prob[pt+'.lpt.Fl_O:tot:S'], prob[pt+'.lpt.Fl_O:stat:W']))
                    print("Duct5       %7.3f %7.2f %7.2f %7.4f %7.3f " %(prob[pt+'.duct5.Fl_O:tot:P'], prob[pt+'.duct5.Fl_O:tot:T'], prob[pt+'.duct5.Fl_O:tot:h'], prob[pt+'.duct5.Fl_O:tot:S'], prob[pt+'.duct5.Fl_O:stat:W']))
                    print("CoreNozz    %7.3f %7.2f %7.2f %7.4f %7.3f " %(prob[pt+'.core_nozz.Fl_O:tot:P'], prob[pt+'.core_nozz.Fl_O:tot:T'], prob[pt+'.core_nozz.Fl_O:tot:h'], prob[pt+'.core_nozz.Fl_O:tot:S'], prob[pt+'.core_nozz.Fl_O:stat:W']))
                    print("BypBld      %7.3f %7.2f %7.2f %7.4f %7.3f " %(prob[pt+'.byp_bld.Fl_O:tot:P'], prob[pt+'.byp_bld.Fl_O:tot:T'], prob[pt+'.byp_bld.Fl_O:tot:h'], prob[pt+'.byp_bld.Fl_O:tot:S'], prob[pt+'.byp_bld.Fl_O:stat:W']))
                    print("Duct17      %7.3f %7.2f %7.2f %7.4f %7.3f " %(prob[pt+'.duct17.Fl_O:tot:P'], prob[pt+'.duct17.Fl_O:tot:T'], prob[pt+'.duct17.Fl_O:tot:h'], prob[pt+'.duct17.Fl_O:tot:S'], prob[pt+'.duct17.Fl_O:stat:W']))
                    print("BypNozz     %7.3f %7.2f %7.2f %7.4f %7.3f " %(prob[pt+'.byp_nozz.Fl_O:tot:P'], prob[pt+'.byp_nozz.Fl_O:tot:T'], prob[pt+'.byp_nozz.Fl_O:tot:h'], prob[pt+'.byp_nozz.Fl_O:tot:S'], prob[pt+'.byp_nozz.Fl_O:stat:W']))
                print("----------------------------------------------------------------------------")
                print("                          COMPRESSOR PROPERTIES")
                print("Component        Wc      PR     eff effPoly     Nc      pwr   Rline   NcMap s_WcDes s_PRdes s_effDes  s_NcDes")
                print("Fan         %7.2f %7.4f %7.5f %7.5f %7.1f %8.1f %7.4f %7.4f %7.4f %7.4f %8.4f %8.2f" %(prob[pt+'.fan.Wc'],prob[pt+'.fan.PR'],prob[pt+'.fan.eff'],prob[pt+'.fan.eff_poly'],prob[pt+'.fan.Nc'],prob[pt+'.fan.power'],prob[pt+'.fan.map.RlineMap'],prob[pt+'.fan.map.readMap.NcMap'],prob[pt+'.fan.s_WcDes'],prob[pt+'.fan.s_PRdes'],prob[pt+'.fan.s_effDes'],prob[pt+'.fan.s_NcDes']))
                print("LPC         %7.2f %7.4f %7.5f %7.5f %7.1f %8.1f %7.4f %7.4f %7.4f %7.4f %8.4f %8.2f" %(prob[pt+'.lpc.Wc'],prob[pt+'.lpc.PR'],prob[pt+'.lpc.eff'],prob[pt+'.lpc.eff_poly'],prob[pt+'.lpc.Nc'],prob[pt+'.lpc.power'],prob[pt+'.lpc.map.RlineMap'],prob[pt+'.lpc.map.readMap.NcMap'],prob[pt+'.lpc.s_WcDes'],prob[pt+'.lpc.s_PRdes'],prob[pt+'.lpc.s_effDes'],prob[pt+'.lpc.s_NcDes']))
                print("HPC         %7.2f %7.4f %7.5f %7.5f %7.1f %8.1f %7.4f %7.4f %7.4f %7.4f %8.4f %8.2f" %(prob[pt+'.hpc.Wc'],prob[pt+'.hpc.PR'],prob[pt+'.hpc.eff'],prob[pt+'.hpc.eff_poly'],prob[pt+'.hpc.Nc'],prob[pt+'.hpc.power'],prob[pt+'.hpc.map.RlineMap'],prob[pt+'.hpc.map.readMap.NcMap'],prob[pt+'.hpc.s_WcDes'],prob[pt+'.hpc.s_PRdes'],prob[pt+'.hpc.s_effDes'],prob[pt+'.hpc.s_NcDes']))
                print("----------------------------------------------------------------------------")
                print("                            BURNER PROPERTIES")
                print("Component      dPqP   TtOut   Wfuel      FAR")
                print("Burner      %7.4f %7.2f %7.4f  %7.5f" %(prob[pt+'.burner.dPqP'], prob[pt+'.burner.Fl_O:tot:T'],prob[pt+'.burner.Wfuel'], FAR))
                print("----------------------------------------------------------------------------")
                print("                           TURBINE PROPERTIES")
                print("Component        Wp      PR     eff effPoly      Np      pwr   NpMap s_WpDes s_PRdes s_effDes s_NpDes")
                print("HPT         %7.3f %7.4f %7.5f %7.5f %7.1f %8.1f %7.3f %7.4f %7.4f %8.4f %7.4f" %(prob[pt+'.hpt.Wp'], HPT_PR, prob[pt+'.hpt.eff'],prob[pt+'.hpt.eff_poly'],prob[pt+'.hpt.Np'],prob[pt+'.hpt.power'],prob[pt+'.hpt.map.readMap.NpMap'],prob[pt+'.hpt.s_WpDes'],prob[pt+'.hpt.s_PRdes'],prob[pt+'.hpt.s_effDes'],prob[pt+'.hpt.s_NpDes']))
                print("LPT         %7.3f %7.4f %7.5f %7.5f %7.1f %8.1f %7.3f %7.4f %7.4f %8.4f %7.4f" %(prob[pt+'.lpt.Wp'], LPT_PR, prob[pt+'.lpt.eff'],prob[pt+'.lpt.eff_poly'],prob[pt+'.lpt.Np'],prob[pt+'.lpt.power'],prob[pt+'.lpt.map.readMap.NpMap'],prob[pt+'.lpt.s_WpDes'],prob[pt+'.lpt.s_PRdes'],prob[pt+'.lpt.s_effDes'],prob[pt+'.lpt.s_NpDes']))
                print("----------------------------------------------------------------------------")
                print("                            NOZZLE PROPERTIES")
                print("Component        PR      Cv     Ath    MNth   MNout       V      Fg")
                print("CoreNozz    %7.4f %7.4f %7.2f %7.4f %7.4f %7.1f %7.1f" %(prob[pt+'.core_nozz.PR'],prob[pt+'.core_nozz.Cv'],prob[pt+'.core_nozz.Throat:stat:area'],prob[pt+'.core_nozz.Throat:stat:MN'],prob[pt+'.core_nozz.Fl_O:stat:MN'],prob[pt+'.core_nozz.Fl_O:stat:V'],prob[pt+'.core_nozz.Fg']))
                print("BypNozz     %7.4f %7.4f %7.2f %7.4f %7.4f %7.1f %7.1f" %(prob[pt+'.byp_nozz.PR'],prob[pt+'.byp_nozz.Cv'],prob[pt+'.byp_nozz.Throat:stat:area'],prob[pt+'.byp_nozz.Throat:stat:MN'],prob[pt+'.byp_nozz.Fl_O:stat:MN'],prob[pt+'.byp_nozz.Fl_O:stat:V'],prob[pt+'.byp_nozz.Fg']))
                print("----------------------------------------------------------------------------")
                print("                             DUCT PROPERTIES")
                print("Component      dPqP      MN       A")
                print("Duct2       %7.4f %7.4f %7.2f" %(duct2_dPqP,prob[pt+'.duct2.Fl_O:stat:MN'],prob[pt+'.duct2.Fl_O:stat:area']))
                print("Duct25      %7.4f %7.4f %7.2f" %(duct25_dPqP,prob[pt+'.duct25.Fl_O:stat:MN'],prob[pt+'.duct25.Fl_O:stat:area']))
                print("Duct45      %7.4f %7.4f %7.2f" %(duct45_dPqP,prob[pt+'.duct45.Fl_O:stat:MN'],prob[pt+'.duct45.Fl_O:stat:area']))
                print("Duct5       %7.4f %7.4f %7.2f" %(duct5_dPqP,prob[pt+'.duct5.Fl_O:stat:MN'],prob[pt+'.duct5.Fl_O:stat:area']))
                print("Duct17      %7.4f %7.4f %7.2f" %(duct17_dPqP,prob[pt+'.duct17.Fl_O:stat:MN'],prob[pt+'.duct17.Fl_O:stat:area']))
                print("----------------------------------------------------------------------------")
                print("                            SHAFT PROPERTIES")
                print("Component     Nmech    trqin   trqout    pwrin   pwrout")
                print("HP_Shaft    %7.1f %8.1f %8.1f %8.1f %8.1f" %(prob[pt+'.hp_shaft.Nmech'],prob[pt+'.hp_shaft.trq_in'],prob[pt+'.hp_shaft.trq_out'],prob[pt+'.hp_shaft.pwr_in'],prob[pt+'.hp_shaft.pwr_out']))
                print("LP_Shaft    %7.1f %8.1f %8.1f %8.1f %8.1f" %(prob[pt+'.lp_shaft.Nmech'],prob[pt+'.lp_shaft.trq_in'],prob[pt+'.lp_shaft.trq_out'],prob[pt+'.lp_shaft.pwr_in'],prob[pt+'.lp_shaft.pwr_out']))
                print("Fan_Shaft   %7.1f %8.1f %8.1f %8.1f %8.1f" %(prob[pt+'.fan_shaft.Nmech'],prob[pt+'.fan_shaft.trq_in'],prob[pt+'.fan_shaft.trq_out'],prob[pt+'.fan_shaft.pwr_in'],prob[pt+'.fan_shaft.pwr_out']))
                print("----------------------------------------------------------------------------")
                print("                            BLEED PROPERTIES")
                print("Bleed       Wb/Win   Pfrac Workfrac       W      Tt      ht      Pt")
                print("SBV        %7.4f %7.4f %8.4f %7.4f %7.2f %7.3f %7.3f" %(prob[pt+'.bld25.sbv:frac_W'],1.0,1.0,prob[pt+'.bld25.sbv:stat:W'],prob[pt+'.bld25.sbv:tot:T'],prob[pt+'.bld25.sbv:tot:h'],prob[pt+'.bld25.sbv:tot:P']))
                print("LPT_inlet  %7.4f %7.4f %8.4f %7.4f %7.2f %7.3f %7.3f" %(prob[pt+'.hpc.bld_inlet:frac_W'],prob[pt+'.hpc.bld_inlet:frac_P'],prob[pt+'.hpc.bld_inlet:frac_work'],prob[pt+'.hpc.bld_inlet:stat:W'],prob[pt+'.hpc.bld_inlet:tot:T'],prob[pt+'.hpc.bld_inlet:tot:h'],prob[pt+'.hpc.bld_inlet:tot:P']))
                print("LPT_exit   %7.4f %7.4f %8.4f %7.4f %7.2f %7.3f %7.3f" %(prob[pt+'.hpc.bld_exit:frac_W'],prob[pt+'.hpc.bld_exit:frac_P'],prob[pt+'.hpc.bld_exit:frac_work'],prob[pt+'.hpc.bld_exit:stat:W'],prob[pt+'.hpc.bld_exit:tot:T'],prob[pt+'.hpc.bld_exit:tot:h'],prob[pt+'.hpc.bld_exit:tot:P']))
                print("HPT_inlet  %7.4f %7.4f %8.4f %7.4f %7.2f %7.3f %7.3f" %(prob[pt+'.bld3.bld_inlet:frac_W'],1.0,1.0,prob[pt+'.bld3.bld_inlet:stat:W'],prob[pt+'.bld3.bld_inlet:tot:T'],prob[pt+'.bld3.bld_inlet:tot:h'],prob[pt+'.bld3.bld_inlet:tot:P']))
                print("HPT_exit   %7.4f %7.4f %8.4f %7.4f %7.2f %7.3f %7.3f" %(prob[pt+'.bld3.bld_exit:frac_W'],1.0,1.0,prob[pt+'.bld3.bld_exit:stat:W'],prob[pt+'.bld3.bld_exit:tot:T'],prob[pt+'.bld3.bld_exit:tot:h'],prob[pt+'.bld3.bld_exit:tot:P']))
                print("Cust       %7.4f %7.4f %8.4f %7.4f %7.2f %7.3f %7.3f" %(prob[pt+'.hpc.cust:frac_W'],prob[pt+'.hpc.cust:frac_P'],prob[pt+'.hpc.cust:frac_work'],prob[pt+'.hpc.cust:stat:W'],prob[pt+'.hpc.cust:tot:T'],prob[pt+'.hpc.cust:tot:h'],prob[pt+'.hpc.cust:tot:P']))
                print("BypBld     %7.4f %7.4f %8.4f %7.4f %7.2f %7.3f %7.3f" %(prob[pt+'.byp_bld.bypBld:frac_W'],1.0,1.0,prob[pt+'.byp_bld.bypBld:stat:W'],prob[pt+'.byp_bld.bypBld:tot:T'],prob[pt+'.byp_bld.bypBld:tot:h'],prob[pt+'.byp_bld.bypBld:tot:P']))
                print("----------------------------------------------------------------------------")
                print()

if __name__ == "__main__":

    desvars = [1]

    # These are model numbers. Just generating a chart.
    states = [1143, 6317, 29689]
    states = [29689]
    procs = [1]

    bench = MyBench(desvars, states, procs, mode='fwd', name='pycycleMisc', use_flag=True)
    bench.num_averages = 1
    bench.time_linear = True
    bench.time_driver = False

    bench.run_benchmark()