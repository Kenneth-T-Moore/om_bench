"""
Create plots for the Beam problem benchmarks.
"""
from __future__ import print_function

from om_bench.post import post_process


filename = 'beam_state_nl_ln.dat'

post_process(filename, 'FEM Beam')
