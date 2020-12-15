import os
import sys
import signal
import netCDF4
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from datetime import datetime
from git import Repo

from .eos import EquationOfState
from .solver import Solver


class Run:

    def __init__(self, options, disc, geometry, numerics, material, plot, reducedOut, out_dir, q_init):

        self.options = options
        self.disc = disc
        self.geometry = geometry
        self.numerics = numerics
        self.material = material

        self.writeInterval = int(options['writeInterval'])
        self.name = str(options['name'])

        tol = float(numerics['tol'])
        maxT = float(numerics['maxT'])

        dx = float(disc['dx'])
        dy = float(disc['dy'])
        self.Nx = int(disc['Nx'])
        self.Ny = int(disc['Ny'])

        self.Lx = dx * self.Nx
        self.Ly = dy * self.Ny

        disc['Lx'] = self.Lx
        disc['Ly'] = self.Ly

        self.tStart = time.time()

        self.sol = Solver(disc, geometry, numerics, material, q_init)

        self.run(plot, reducedOut, out_dir, tol, maxT)

    def receive_signal(self, signum, stack):
        total_time = time.time() - self.tStart
        print(f"python: PID: {os.getpid()} recieved {signum} at time {total_time}")
        sys.stdout.flush()                             # displays the message right now, othervise it would be buffered

        if signum == signal.SIGTERM:
            sys.exit()

        if signum == signal.SIGUSR1:
            timeString = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            self.nc.tEnd = timeString
            HH, MM, SS = self.time_to_HHMMSS(total_time)
            print(f"Total wall clock time: {HH:02d}:{MM:02d}:{SS:02d} (Performance: {self.sol.time * 1e12 / total_time:.2f} ps/s)")
            # print("python: finishing properly")
            sys.exit()

    def run(self, plot, reducedOut, out_dir, tol, maxT):

        if plot is False:
            self.file_tag = 1
            self.j = 0

            if not(os.path.exists(out_dir)):
                os.makedirs(out_dir)

            while str(self.name) + '_' + str(self.file_tag).zfill(4) + '.nc' in os.listdir(out_dir):
                self.file_tag += 1

            outfile = str(self.name) + '_' + str(self.file_tag).zfill(4) + '.nc'

            i = 0

            # initialize NetCDF file
            self.nc = netCDF4.Dataset(os.path.join(out_dir, outfile), 'w', format='NETCDF3_64BIT_OFFSET')
            self.nc.createDimension('x', self.Nx)
            self.nc.createDimension('y', self.Ny)
            self.nc.createDimension('step', None)

            # create conserved variables timeseries of fields
            self.rho = self.nc.createVariable('rho', 'f8', ('step', 'x', 'y'))
            self.jx = self.nc.createVariable('jx', 'f8', ('step', 'x', 'y'))
            self.jy = self.nc.createVariable('jy', 'f8', ('step', 'x', 'y'))
            if reducedOut is False:
                self.p = self.nc.createVariable('p', 'f8', ('step', 'x', 'y'))

            # create scalar variables
            self.time = self.nc.createVariable('time', 'f8', ('step'))
            self.mass = self.nc.createVariable('mass', 'f8', ('step'))
            self.vmax = self.nc.createVariable('vmax', 'f8', ('step'))
            self.vSound = self.nc.createVariable('vSound', 'f8', ('step'))
            self.dt = self.nc.createVariable('dt', 'f8', ('step'))
            self.eps = self.nc.createVariable('eps', 'f8', ('step'))

            print("{:10s}\t{:12s}\t{:12s}\t{:12s}".format("Step", "Timestep", "Time", "Epsilon"))
            while self.sol.time < maxT:

                self.sol.solve(i)
                self.write(i, 0, reducedOut)
                if i % self.writeInterval == 0:
                    print("{:10d}\t{:.6e}\t{:.6e}\t{:.6e}".format(i, self.sol.dt, self.sol.time, self.sol.eps))
                    sys.stdout.flush()
                i += 1

                signal.signal(signal.SIGTERM, self.receive_signal)
                signal.signal(signal.SIGHUP, self.receive_signal)
                signal.signal(signal.SIGUSR1, self.receive_signal)
                signal.signal(signal.SIGUSR2, self.receive_signal)

                tDiff = maxT - self.sol.time

                if self.sol.eps < tol:
                    self.write(i, 1, reducedOut)
                    print(f"{i:10d}\t{self.sol.dt:.6e}\t{self.sol.time:.6e}\t{self.sol.eps:.6e}")
                    print(f"\nSolution has converged after {i:d} steps, Output written to: {os.path.join(out_dir, outfile):s}")
                    timeString = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                    self.nc.tEnd = timeString
                    break
                elif tDiff < self.sol.dt:
                    self.sol.solve(i)
                    self.write(i, 1, reducedOut)
                    print(f"{i:10d}\t{self.sol.dt:.6e}\t{self.sol.time:.6e}\t{self.sol.eps:.6e}")
                    print(f"\nNo convergence within {i:d} steps. Stopping criterion: maximum time {maxT:.1e} s reached.")
                    print(f"Output written to: {os.path.join(out_dir, outfile):s}")
                    timeString = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                    self.nc.tEnd = timeString

        else:
            print("{:10s}\t{:12s}\t{:12s}\t{:12s}".format("Step", "Time step", "Time", "Epsilon"))
            self.plot()

        tDiff = time.time() - self.tStart
        HH, MM, SS = self.time_to_HHMMSS(tDiff)

        print(f"Total wall clock time: {HH:02d}:{MM:02d}:{SS:02d} (Performance: {self.sol.time * 1e12 / tDiff:.2f} ps/s)")

    def time_to_HHMMSS(self, t):

        MM = t // 60
        HH = int(MM // 60)
        MM = int(MM - HH * 60)
        SS = int(t - HH * 60 * 60 - MM * 60)

        return HH, MM, SS

    def write(self, i, last, reduced):

        # HDF5 output file
        if i % self.writeInterval == 0 or last == 1:

            if self.j == 0:
                now = datetime.now()
                timeString = now.strftime("%d/%m/%Y %H:%M:%S")
                if os.getcwd().split(os.sep)[-1] == "MD-FVM":
                    repo_path = "."
                else:
                    repo_path = os.environ["PYTHONPATH"]
                git_commit = str(Repo(path=repo_path, search_parent_directories=True).head.object.hexsha)
                self.nc.tStart = timeString
                self.nc.commit = git_commit

                categories = {"options": self.options,
                              "disc": self.disc,
                              "geometry": self.geometry,
                              "numerics": self.numerics,
                              "material": self.material}

                for cat_name, cat in categories.items():
                    for key, value in cat.items():
                        name = cat_name + "_" + key
                        self.nc.setncattr(name, value)

            self.rho[self.j] = self.sol.q.field[0]
            self.jx[self.j] = self.sol.q.field[1]
            self.jy[self.j] = self.sol.q.field[2]

            if reduced is False:
                self.p[self.j] = EquationOfState(self.material).isoT_pressure(self.sol.q.field[0])

            self.time[self.j] = self.sol.time
            self.mass[self.j] = self.sol.mass
            self.vmax[self.j] = self.sol.vmax
            self.vSound[self.j] = self.sol.vSound
            self.dt[self.j] = self.sol.dt
            self.eps[self.j] = self.sol.eps

            self.j += 1

    def plot(self):

        fig, ax = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
        x = np.linspace(0, self.Lx, self.Nx, endpoint=True)

        line0, = ax[0, 0].plot(x, self.sol.q.field[1][:, int(self.Ny / 2)])
        line1, = ax[0, 1].plot(x, self.sol.q.field[2][:, int(self.Ny / 2)])
        line2, = ax[1, 0].plot(x, self.sol.q.field[0][:, int(self.Ny / 2)])
        line3, = ax[1, 1].plot(x, EquationOfState(self.material).isoT_pressure(self.sol.q.field[0][:, int(self.Ny / 2)]))

        ax[0, 0].set_title(r'$j_x$')
        ax[0, 1].set_title(r'$j_y$')
        ax[1, 0].set_title(r'$\rho$')
        ax[1, 1].set_title(r'$p$')

        ax[1, 0].set_xlabel('distance x (m)')
        ax[1, 1].set_xlabel('distance x (m)')

        limits = np.zeros((4, 3))

        for j in range(3):
            limits[j, 0] = np.amin(self.sol.q.field[j][:, int(self.Ny / 2)])
            limits[j, 1] = np.amax(self.sol.q.field[j][:, int(self.Ny / 2)])

        limits[3, 0] = np.amin(EquationOfState(self.material).isoT_pressure(self.sol.q.field[0][:, int(self.Ny / 2)]))
        limits[3, 1] = np.amax(EquationOfState(self.material).isoT_pressure(self.sol.q.field[0][:, int(self.Ny / 2)]))

        _ = animation.FuncAnimation(fig, self.animate1D, 100000, fargs=(fig, ax,
                                                                        line0, line1, line2, line3, limits,), interval=1, repeat=False)

        plt.show()

    def animate1D(self, i, fig, ax, line0, line1, line2, line3, limits):

        limits = self.adaptiveLimits(limits)

        fig.suptitle('time = {:.2f} ns'.format(self.sol.time * 1e9))

        ax[0, 0].set_ylim(limits[1, 0] - limits[1, 2], limits[1, 1] + limits[1, 2])
        ax[0, 1].set_ylim(limits[2, 0] - limits[2, 2], limits[2, 1] + limits[2, 2])
        ax[1, 0].set_ylim(limits[0, 0] - limits[0, 2], limits[0, 1] + limits[0, 2])
        ax[1, 1].set_ylim(limits[3, 0] - limits[3, 2], limits[3, 1] + limits[3, 2])

        line0.set_ydata(self.sol.q.field[1][:, int(self.Ny / 2)])
        line1.set_ydata(self.sol.q.field[2][:, int(self.Ny / 2)])
        line2.set_ydata(self.sol.q.field[0][:, int(self.Ny / 2)])
        line3.set_ydata(EquationOfState(self.material).isoT_pressure(self.sol.q.field[0][:, int(self.Ny / 2)]))

        self.sol.solve(i)
        if i % self.writeInterval == 0:
            print("{:10d}\t{:.6e}\t{:.6e}\t{:.6e}".format(i, self.sol.dt, self.sol.time, self.sol.eps))

    def adaptiveLimits(self, limits):

        for j in range(3):
            if np.amin(self.sol.q.field[j]) < limits[j, 0]:
                limits[j, 0] = np.amin(self.sol.q.field[j])
            if np.amax(self.sol.q.field[j]) > limits[j, 1]:
                limits[j, 1] = np.amax(self.sol.q.field[j])

        if np.amin(EquationOfState(self.material).isoT_pressure(self.sol.q.field[0])) < limits[3, 0]:
            limits[3, 0] = np.amin(EquationOfState(self.material).isoT_pressure(self.sol.q.field[0]))
        if np.amax(EquationOfState(self.material).isoT_pressure(self.sol.q.field[0])) > limits[3, 1]:
            limits[3, 1] = np.amax(EquationOfState(self.material).isoT_pressure(self.sol.q.field[0]))

        for j in range(4):
            if limits[j, 1] == limits[j, 0] and limits[j, 0] != 0.:
                limits[j, 2] = 0.5 * limits[j, 1]
            elif limits[j, 1] == limits[j, 0] and limits[j, 0] == 0.:
                limits[j, 2] = 1.
            else:
                limits[j, 2] = 0.1 * (limits[j, 1] - limits[j, 0])

        return limits
