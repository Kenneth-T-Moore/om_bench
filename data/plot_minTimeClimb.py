"""
Create plots for the Beam problem benchmarks.
"""
from __future__ import print_function

from om_bench.post import post_process, BenchPost


filename = 'minTimeClimb_state_nl_ln_drv.dat'

bp = BenchPost()

bp.post_process(filename, 'Min Time Climb', flagtxt="Simultaneous Derivatives")
