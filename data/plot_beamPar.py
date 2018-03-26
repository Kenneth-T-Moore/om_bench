"""
Create plots for the BeamPar problem benchmarks.
"""
from __future__ import print_function

from om_bench.post import post_process


filename = 'beamPar_proc_nl_ln.dat'

post_process(filename, 'FEM Beam with and without Parallel Derivatives', flagtxt='Parallel Derivatives')