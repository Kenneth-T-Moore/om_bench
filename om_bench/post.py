"""
Function that makes journal quality scaling plots from pre-generated scaling benchmark data.
"""
import fnmatch
import os

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
        x_dv[j], x_state[j], x_proc[j], _, t1[j], t3[j], t5[j] = line.strip().split(',')

    # Times are all normalized.
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
        xlab = "Number of processors."

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
        plt.savefig("%s_%s_%s.png" % (name, mode, 'ln'))

        # For procs, we also view the time/proc as a function of number of procs.
        if mode == 'proc':
            plt.figure(3)
            plt.loglog(x, t3/x, 'o-')

            plt.xlabel(xlab)
            plt.ylabel('Linear Solve: Normalized Time per Processor')
            plt.title(title)
            plt.grid(True)
            plt.savefig("%s_%s_%s_per_proc.png" % (name, mode, 'ln'))

    plt.show()
    print('done')


def assemble_mpi_results():
    '''
    Scan current directly for mpi result output files and assemble them together.
    '''
    allfiles = os.listdir('.')
    files = [n for n in allfiles if fnmatch.fnmatch(n, '_*.dat')]

    stem_parts = files[0].split('_')[1:-5]
    stem = '_' + '_'.join(stem_parts)

    nl = 'nl' in stem_parts
    ln = 'ln' in stem_parts
    drv = 'drv' in stem_parts
    op = []
    if nl:
        op.append('nl')
    if ln:
        op.append('ln')
    if drv:
        op.append('drv')
    op = '_'.join(op)

    name = stem_parts[0]
    mode = stem_parts[1]

    # Find remaining parts
    dv = set()
    state = set()
    proc = set()
    av = set()
    flag = set()
    for fname in files:

        if not fname.startswith(stem):
            msg = 'Parsing failed because files from multiple independent runs found in the same directory.'
            raise RuntimeError(msg)

        fname = fname.strip('.dat').strip(stem)
        parts = fname.split('_')

        #  ndv, nstate, nproc, av
        dv.add(int(parts[0]))
        state.add(int(parts[1]))
        proc.add(int(parts[2]))
        flag.add(parts[3])
        av.add(int(parts[4]))

    data = []
    for idv in sorted(dv):
        for istate in sorted(state):
            for iproc in sorted(proc):
                for iflag in flag:

                    t1_sum = 0.0
                    t3_sum = 0.0
                    t5_sum = 0.0
                    for iav in av:
                        filename = stem + '_%s_%s_%s_%s_%s.dat' % (idv, istate, iproc, iflag, iav)
                        infile = open(filename, 'r')

                        line = infile.readline()
                        parts = line.split(',')
                        t1 = float(parts[0].strip())
                        t3 = float(parts[1].strip())
                        t5 = float(parts[2].strip())

                        t1_sum += t1
                        t3_sum += t3
                        t5_sum += t5

                    t1av = t1_sum / (iav + 1)
                    t3av = t3_sum / (iav + 1)
                    t5av = t5_sum / (iav + 1)

                    data.append((idv, istate, iproc, iflag, t1av, t3av, t5av))

    filename = '%s_%s_%s.dat' % (name, mode, op)

    outfile = open(filename, 'w')
    outfile.write(name)
    outfile.write('\n')
    outfile.write(mode)
    outfile.write('\n')
    outfile.write('%s, %s, %s' % (nl, ln, drv))
    outfile.write('\n')

    for ndv, nstate, nproc, nflag, t1, t3, t5 in data:
        outfile.write('%d, %d, %d, %s, %f, %f, %f' % (ndv, nstate, nproc, nflag, t1, t3, t5))
        outfile.write('\n')
    outfile.close()

    print("done")
