"""
Create plots for the Beam problem benchmarks.
"""
from __future__ import print_function

from om_bench.post import post_process


filename = 'minTimeClimb_state_nl_drv.dat'

post_process(filename, 'Min Time Climb', flagtxt="Simultaneous Derivatives")
