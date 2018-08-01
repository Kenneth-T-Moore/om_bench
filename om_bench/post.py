"""
Function that makes journal quality scaling plots from pre-generated scaling benchmark data.
"""
import fnmatch
import os

import numpy as np

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')

class BenchPost(object):

    def __init__(self, title):
        self.title = title

        self.flagtxt = "Insert Text Here"
        self.title_driver = "Driver Execution"

        # In this mode, we want to do a three way comparison.
        self.special_plot_driver_on_linear = False

    def post_process(self, filename):
        """
        Read benchmark data and make scaling plots.
        """
        title = self.title

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
        flag = np.empty((npt, ), dtype=np.bool)
        x_dv = np.empty((npt, ))
        x_state = np.empty((npt, ))
        x_proc = np.empty((npt, ))

        for j, line in enumerate(data):
            x_dv[j], x_state[j], x_proc[j], flag[j], t1u[j], t3u[j], t5u[j] = line.strip().split(',')

        if np.any(flag):
            use_flag = True
        else:
            use_flag = False

        # Times are all normalized.
        t1 = t1u/t1u[0]
        t3 = t3u/t3u[0]
        t5 = t5u/t5u[0]

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

            flagtxt = self.flagtxt

            # Split them up. We know the pattern.
            t1F = t1[0::2]
            t1T = t1[1::2]
            t3F = t3[0::2]
            t3T = t3[1::2]
            t5F = t5[0::2]
            t5T = t5[1::2]

            xT = x[0::2]
            xF = x[1::2]

            # Generate plots

            if nl:
                plt.figure(1)
                plt.loglog(xF, t1F, 'bo-')
                plt.loglog(xT, t1T, 'ro-')

                plt.xlabel(xlab)
                plt.ylabel('Nonlinear Solve: Normalized Time')
                plt.title(title)
                plt.grid(True)
                plt.legend(['Default', flagtxt], loc=0)
                plt.savefig("%s_%s_%s.png" % (name, mode, 'nl'))

            if ln:
                plt.figure(2)
                plt.loglog(xF, t3F, 'o-')
                plt.loglog(xT, t3T, 'ro-')

                plt.xlabel(xlab)
                plt.ylabel('Compute Totals: Normalized Time')
                plt.title(title)
                plt.grid(True)
                plt.legend(['Default', flagtxt], loc=0)
                plt.savefig("%s_%s_%s.png" % (name, mode, 'ln'))

            if drv:
                plt.figure(3)
                plt.loglog(xF, t5F, 'o-')
                plt.loglog(xT, t5T, 'ro-')

                plt.xlabel(xlab)
                plt.ylabel(self.title_driver + ': Normalized Time')
                plt.title(title)
                plt.grid(True)
                plt.legend(['Default', flagtxt], loc=0)
                plt.savefig("%s_%s_%s.png" % (name, mode, 'drv'))

            if self.special_plot_driver_on_linear:

                # Plot whatever driver does (e.g., coloring) on the same axis and normalization as linear time.
                t5 = t5u/t3u[0]
                t5F = t5[0::2]
                t5T = t5[1::2]

                plt.figure(4)
                plt.loglog(xF, t3F, 'o-')
                plt.loglog(xT, t3T, 'ro-')
                plt.loglog(xT, t5T, 'mo-')

                plt.xlabel(xlab)
                plt.ylabel('Normalized Time')
                plt.title(title)
                plt.grid(True)
                plt.legend(['Compute Totals', 'Compute Totals: ' + flagtxt, self.title_driver], loc=0)
                plt.savefig("%s_%s_%s.png" % (name, mode, 'spec1'))

        else:

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
                plt.ylabel('Compute Totals: Normalized Time')
                plt.title(title)
                plt.grid(True)
                plt.savefig("%s_%s_%s.png" % (name, mode, 'ln'))

                # For procs, we also view the time/proc as a function of number of procs.
                if mode == 'proc':
                    plt.figure(3)
                    plt.loglog(x, t3/x, 'o-')

                    plt.xlabel(xlab)
                    plt.ylabel('Compute Totals: Normalized Time per Processor')
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
                for iflag in sorted(flag):

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


# Legacy for older files.
def post_process(filename, title, flagtxt="Insert Text Here"):

    benchpost = BenchPost(title)
    benchpost.flagtxt = flagtxt
    benchpost.post_process(filename=filename)

