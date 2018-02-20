"""
Templates for MPI submission."""


qsub_template = """
::::::::::::::
Auto generated batch file.
::::::::::::::
#PBS -S /bin/bash
#PBS -N <name>
#PBS -l select=6:ncpus=24:model=has
#PBS -l walltime=<walltime>:00:00
#PBS -j oe
#PBS -W group_list=a1607
#PBS -m bae
#PBS -o stdout_<name>.out
#PBS -e stderr_<name>.out
#PBS -q normal

source ~/.bashrc

cd <local>

mpiexec python -u <name>.py
"""


run_template = """
from openmdao.utils.mpi import MPI

from <module> import <classname>

bench = <classname>(<ndv>, <nstate>, <nproc>, name='<name>')
bench.time_linear = <time_linear>
bench.time_driver = <time_driver>

print('Running: dv=<ndv>, state=<nstate>, proc=<nproc>, av=<average>')

t1, t3, t5 = bench._run_nl_ln_drv(<ndv>, <nstate>, <nproc>, use_mpi=True)

if MPI and MPI.COMM_WORLD.rank == 0:
    outname = '_%s_%d_%d_%d_%d.dat' % ('<name>', <ndv>, <nstate>, <nproc>, <average>)
    outfile = open(outname, 'w')
    outfile.write('%f, %f, %f' % (t1, t3, t5))
    outfile.close()
"""
