"""
Function that makes journal quality scaling plots from pre-generated scaling benchmark data.
"""
import numpy as np

import matplotlib.pyplot as plt


def post_process(filename, title):
    """
    Read benchmark data and make scaling plots.
    """
    outfile = open(filename, 'r')
    data = outfile.readlines()

    name = data[0].strip()
    mode = data[1].strip()
    ops = data[2].strip().split(',')
    nl, ln, drv = (bool(item) for item in ops)

    data = data[3:]
    npt = len(data)

    t1 = np.empty((npt, ))
    t3 = np.empty((npt, ))
    t5 = np.empty((npt, ))
    x_dv = np.empty((npt, ))
    x_state = np.empty((npt, ))
    x_proc = np.empty((npt, ))

    for j, line in enumerate(data):
        x_dv[j], x_state[j], x_proc[j], t1[j], t3[j], t5[j] = line.strip().split(',')

    t1 = t1/t1[0]
    t3 = t3/t3[0]
    t5 = t5/t5[0]

    if mode == 'state':
        x = x_state
        xlab = "Normalized number of states."
    elif mode == 'desvar':
        xlab = "Normalized number of design vars."
        x = x_dv
    elif mode == 'proc':
        x = x_proc
        xlab = "Normalized number of processors."

    # Generate plots

    if nl:
        plt.figure(1)
        plt.loglog(x, t1, 'o-')

        plt.xlabel(xlab)
        plt.ylabel('Nonlinear Solve: Normalized Time')
        plt.title(title)
        plt.grid(True)
        plt.savefig("%s_%s_%s.png" % (name, mode, 'nl'))

    if ln:
        plt.figure(2)
        plt.loglog(x, t3, 'o-')

        plt.xlabel(xlab)
        plt.ylabel('Linear Solve: Normalized Time')
        plt.title(title)
        plt.grid(True)
        plt.savefig("%s_%s_%s.png" % (name, mode, 'nl'))

    plt.show()
    print('done')