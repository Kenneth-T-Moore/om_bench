"""
Create plots for the Beam problem benchmarks.
"""
from __future__ import print_function

import fnmatch
import os

import numpy as np

import matplotlib
import matplotlib.pyplot as plt

#matplotlib.use('Agg')

filename = 'brach_state_nl_ln_drv_detail.dat'

flagtxt = "Simultaneous Derivatives"
title_driver = "Compute Coloring"
special_plot_driver_on_linear = True
equal_axis = True

title = "Brachistocrone"

outfile = open(filename, 'r')
data = outfile.readlines()

name = data[0].strip()
mode = data[1].strip()
ops = data[2].strip().split(',')
nl = 'True' in ops[0]
ln = 'True' in ops[1]
drv = 'True' in ops[2]

data = data[3:]
npt = len(data)

t1u = np.empty((npt, ))
t3u = np.empty((npt, ))
t5u = np.empty((npt, ))
t3au = np.empty((npt, ))
t3bu = np.empty((npt, ))
t3cu = np.empty((npt, ))
t3du = np.empty((npt, ))
t3eu = np.empty((npt, ))
flag = np.empty((npt, ), dtype=np.bool)
x_dv = np.empty((npt, ))
x_state = np.empty((npt, ))
x_proc = np.empty((npt, ))

for j, line in enumerate(data):
    x_dv[j], x_state[j], x_proc[j], flag[j], t1u[j], t3u[j], t5u[j], t3au[j], t3bu[j], t3cu[j], t3du[j], t3eu[j] = line.strip().split(',')

if np.any(flag):
    use_flag = True
else:
    use_flag = False

# Times are all normalized.
t1 = t1u
t3 = t3u
t5 = t5u
t3a = t3au
t3b = t3bu
t3c = t3cu
t3d = t3du
t3e = t3eu

if mode == 'state':
    x = x_state
    xlab = "Number of states."
elif mode == 'desvar':
    xlab = "Number of design vars."
    x = x_dv
elif mode == 'proc':
    x = x_proc
    xlab = "Number of processors."

if use_flag:

    # Split them up. We know the pattern.
    t1F = t1[0::2]
    t1T = t1[1::2]
    t3F = t3[0::2]
    t3T = t3[1::2]
    t5F = t5[0::2]
    t5T = t5[1::2]

    t3aF = t3a[0::2]
    t3aT = t3a[1::2]
    t3bF = t3b[0::2]
    t3bT = t3b[1::2]
    t3cF = t3c[0::2]
    t3cT = t3c[1::2]
    t3dF = t3d[0::2]
    t3dT = t3d[1::2]
    t3eF = t3e[0::2]
    t3eT = t3e[1::2]

    xT = x[0::2]
    xF = x[1::2]

    # Generate plots

    plt.figure(3)
    plt.loglog(xF, t3aF, 'o-')
    plt.loglog(xT, t3aT, 'ro-')

    plt.xlabel(xlab)
    plt.ylabel('LU Factor: Time')
    plt.title(title)
    plt.grid(True)
    if equal_axis:
        plt.axis('equal')
    plt.legend(['Default', flagtxt], loc=0)
    plt.savefig("%s_%s_%s_LUfact.png" % (name, mode, 'ln'))

    plt.figure(4)
    plt.loglog(xF, t3bF, 'o-')
    plt.loglog(xT, t3bT, 'ro-')

    plt.xlabel(xlab)
    plt.ylabel('LU Solve: Time')
    plt.title(title)
    plt.grid(True)
    if equal_axis:
        plt.axis('equal')
    plt.legend(['Default', flagtxt], loc=0)
    plt.savefig("%s_%s_%s_LUsolve.png" % (name, mode, 'ln'))

    plt.figure(5)
    plt.loglog(xF, t3cF, 'o-')
    plt.loglog(xT, t3cT, 'ro-')

    plt.xlabel(xlab)
    plt.ylabel('Linearize System: Time')
    plt.title(title)
    plt.grid(True)
    if equal_axis:
        plt.axis('equal')
    plt.legend(['Default', flagtxt], loc=0)
    plt.savefig("%s_%s_%s_linsys.png" % (name, mode, 'ln'))

    plt.figure(6)
    plt.loglog(xF, t3dF, 'o-')
    plt.loglog(xT, t3dT, 'ro-')

    plt.xlabel(xlab)
    plt.ylabel('Linearize Solver: Time')
    plt.title(title)
    plt.grid(True)
    if equal_axis:
        plt.axis('equal')
    plt.legend(['Default', flagtxt], loc=0)
    plt.savefig("%s_%s_%s_linsolver.png" % (name, mode, 'ln'))

    plt.figure(7)
    plt.loglog(xF, t3eF, 'o-')
    plt.loglog(xT, t3eT, 'ro-')

    plt.xlabel(xlab)
    plt.ylabel('Solve: Time')
    plt.title(title)
    plt.grid(True)
    if equal_axis:
        plt.axis('equal')
    plt.legend(['Default', flagtxt], loc=0)
    plt.savefig("%s_%s_%s_solve.png" % (name, mode, 'ln'))

    plt.figure(8)
    plt.loglog(xF, t3bF, 'bo-')
    plt.loglog(xT, t3bT, 'ro-')
    plt.loglog(xF, t3eF, 'bo--')
    plt.loglog(xT, t3eT, 'ro--')

    plt.xlabel(xlab)
    plt.ylabel('Solve: Time')
    plt.title(title)
    plt.grid(True)
    if equal_axis:
        plt.axis('equal')
    plt.legend(['LU:Default', 'LU:' + flagtxt, 'LU+overhead:Default', 'LU+overhead:' + flagtxt], loc=0)
    plt.savefig("%s_%s_%s_solve_overhead.png" % (name, mode, 'ln'))

plt.show()
print('done')
