"""
Create plots for the Beam problem benchmarks.
"""
from __future__ import print_function

from om_bench.post import post_process, BenchPost


filename = 'minTimeClimb_state_nl_ln_drv.dat'

bp = BenchPost('Min Time Climb')
bp.flagtxt="Simultaneous Derivatives"
bp.title_driver = "Compute Coloring"
bp.special_plot_driver_on_linear = True
bp.equal_axis = True

bp.post_process(filename)
