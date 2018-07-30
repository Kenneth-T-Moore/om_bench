"""
Create plots for the Beam problem benchmarks.
"""
from __future__ import print_function

from om_bench.post import post_process, BenchPost


filename = 'AMD_state_nl_ln.dat'

bp = BenchPost('AMD 128 Routes')
bp.flagtxt = "Parallel Derivatives"

bp.post_process(filename)
