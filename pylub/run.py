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

    def __init__(self, options, disc, BC, geometry, numerics, material, plot, out_dir, q_init):

        self.options = options
        self.disc = disc
        self.BC = BC
        self.geometry = geometry
        self.numerics = numerics
        self.material = material

        self.writeInterval = int(options['writeInterval'])
        self.name = str(options['name'])

        tol = float(numerics['tol'])
        maxT = float(numerics['maxT'])

        self.Nx = int(disc['Nx'])
        self.Ny = int(disc['Ny'])

        if "dx" in disc.keys():
            dx = float(disc['dx'])
            self.Lx = dx * self.Nx
            disc['Lx'] = self.Lx
        else:
            self.Lx = float(disc["Lx"])
            disc["dx"] = self.Lx / self.Nx

        if "dy" in disc.keys():
            dy = float(disc['dy'])
            self.Ly = dy * self.Ny
            disc['Ly'] = self.Ly
        else:
            self.Ly = float(disc["Ly"])
            disc["dy"] = self.Ly / self.Ny

        self.tStart = time.time()

        self.sol = Solver(disc, BC, geometry, numerics, material, q_init)

        self.run(plot, out_dir, tol, maxT)

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
            print(f"Total wall clock time: {HH:02d}:{MM:02d}:{SS:02d} (Performance: {self.sol.time * 1e9 / total_time:.2f} ns/s)")
            sys.exit()

    def run(self, plot, out_dir, tol, maxT):

        if plot:
            print("{:10s}\t{:12s}\t{:12s}\t{:12s}".format("Step", "Time step", "Time", "Epsilon"))
            self.plot()
        else:
            if not(os.path.exists(out_dir)):
                os.makedirs(out_dir)

            file_tag = len([f for f in os.listdir(out_dir) if f.startswith(f"{self.name}_")]) + 1
            outfile = f"{self.name}_{str(file_tag).zfill(4)}.nc"
            self.relpath = os.path.join(out_dir, outfile)

            # initialize NetCDF file
            self.nc = netCDF4.Dataset(self.relpath, 'w', format='NETCDF3_64BIT_OFFSET')
            self.nc.createDimension('x', self.Nx)
            self.nc.createDimension('y', self.Ny)
            self.nc.createDimension('step', None)

            # create conserved variables timeseries of fields
            self.rho = self.nc.createVariable('rho', 'f8', ('step', 'x', 'y'))
            self.jx = self.nc.createVariable('jx', 'f8', ('step', 'x', 'y'))
            self.jy = self.nc.createVariable('jy', 'f8', ('step', 'x', 'y'))

            # create scalar variables
            self.time = self.nc.createVariable('time', 'f8', ('step'))
            self.mass = self.nc.createVariable('mass', 'f8', ('step'))
            self.vmax = self.nc.createVariable('vmax', 'f8', ('step'))
            self.vSound = self.nc.createVariable('vSound', 'f8', ('step'))
            self.dt = self.nc.createVariable('dt', 'f8', ('step'))
            self.eps = self.nc.createVariable('eps', 'f8', ('step'))

            i = 0
            while self.sol.time < maxT:

                self.sol.solve(i)
                tDiff = maxT - self.sol.time

                signal.signal(signal.SIGTERM, self.receive_signal)
                signal.signal(signal.SIGHUP, self.receive_signal)
                signal.signal(signal.SIGUSR1, self.receive_signal)
                signal.signal(signal.SIGUSR2, self.receive_signal)

                if i % self.writeInterval == 0:
                    self.write(i)
                elif self.sol.eps < tol:
                    self.write(i, mode="converged")
                    break
                elif tDiff < self.sol.dt:
                    i += 1
                    self.sol.solve(i)
                    self.write(i, mode="maxtime")

                i += 1

    def time_to_HHMMSS(self, t):

        MM = t // 60
        HH = int(MM // 60)
        MM = int(MM - HH * 60)
        SS = int(t - HH * 60 * 60 - MM * 60)

        return HH, MM, SS

    def write(self, i, mode="normal"):

        # netCDF4 output file
        if i == 0:
            print("{:10s}\t{:12s}\t{:12s}\t{:12s}".format("Step", "Timestep", "Time", "Epsilon"))

            timeString = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

            repo_path = [path for path in sys.path if path.endswith("pylub")][0]
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

        print(f"{i:10d}\t{self.sol.dt:.6e}\t{self.sol.time:.6e}\t{self.sol.eps:.6e}")
        sys.stdout.flush()

        k = self.rho.shape[0]

        self.rho[k] = self.sol.q.field[0, 1:-1, 1:-1]
        self.jx[k] = self.sol.q.field[1, 1:-1, 1:-1]
        self.jy[k] = self.sol.q.field[2, 1:-1, 1:-1]

        self.time[k] = self.sol.time
        self.mass[k] = self.sol.mass
        self.vmax[k] = self.sol.vmax
        self.vSound[k] = self.sol.vSound
        self.dt[k] = self.sol.dt
        self.eps[k] = self.sol.eps

        if mode == "converged":
            print(f"\nSolution has converged after {i:d} steps, Output written to: {self.relpath}")
        elif mode == "maxtime":
            print(f"\nNo convergence within {i:d} steps. Stopping criterion: maximum time {self.numerics['maxT']:.1e} s reached.")
            print(f"Output written to: {self.relpath}")

        if mode != "normal":
            self.nc.tEnd = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            walltime = time.time() - self.tStart
            HH, MM, SS = self.time_to_HHMMSS(walltime)

            print(f"Total wall clock time: {HH:02d}:{MM:02d}:{SS:02d} (Performance: {self.sol.time * 1e9 / walltime:.2f} ns/s)")

    def plot(self):

        fig, ax = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
        x = np.linspace(0, self.Lx, self.Nx, endpoint=True)

        ax[0, 0].plot(x, self.sol.q.field[1, 1:-1, self.Ny // 2])
        ax[0, 1].plot(x, self.sol.q.field[2, 1:-1, self.Ny // 2])
        ax[1, 0].plot(x, self.sol.q.field[0, 1:-1, self.Ny // 2])
        ax[1, 1].plot(x, EquationOfState(self.material).isoT_pressure(self.sol.q.field[0, 1:-1, self.Ny // 2]))

        ax[0, 0].set_title(r'$j_x$')
        ax[0, 1].set_title(r'$j_y$')
        ax[1, 0].set_title(r'$\rho$')
        ax[1, 1].set_title(r'$p$')

        ax[1, 0].set_xlabel('distance x (m)')
        ax[1, 1].set_xlabel('distance x (m)')

        def init():
            pass

        _ = animation.FuncAnimation(fig,
                                    self.animate1D,
                                    100000,
                                    fargs=(fig, ax),
                                    interval=1,
                                    init_func=init,
                                    repeat=False)

        plt.show()

    def animate1D(self, i, fig, ax):

        fig.suptitle('time = {:.2f} ns'.format(self.sol.time * 1e9))

        ax[0, 0].lines[0].set_ydata(self.sol.q.field[1, 1:-1, self.Ny // 2])
        ax[0, 1].lines[0].set_ydata(self.sol.q.field[2, 1:-1, self.Ny // 2])
        ax[1, 0].lines[0].set_ydata(self.sol.q.field[0, 1:-1, self.Ny // 2])
        ax[1, 1].lines[0].set_ydata(EquationOfState(self.material).isoT_pressure(self.sol.q.field[0, 1:-1, self.Ny // 2]))

        ax = self.adaptiveLimits(ax)

        self.sol.solve(i)
        if i % self.writeInterval == 0:
            print("{:10d}\t{:.6e}\t{:.6e}\t{:.6e}".format(i, self.sol.dt, self.sol.time, self.sol.eps))

    def adaptiveLimits(self, ax):

        a0y_min = np.amin(ax[0, 0].lines[0].get_ydata())
        a0y_max = np.amax(ax[0, 0].lines[0].get_ydata())
        a1y_min = np.amin(ax[0, 1].lines[0].get_ydata())
        a1y_max = np.amax(ax[0, 1].lines[0].get_ydata())
        a2y_min = np.amin(ax[1, 0].lines[0].get_ydata())
        a2y_max = np.amax(ax[1, 0].lines[0].get_ydata())
        a3y_min = np.amin(ax[1, 1].lines[0].get_ydata())
        a3y_max = np.amax(ax[1, 1].lines[0].get_ydata())

        def offset(x, y): return 0.05 * (x - y) if (x - y) != 0 else 1.

        ax[0, 0].set_ylim(a0y_min - offset(a0y_max, a0y_min), a0y_max + offset(a0y_max, a0y_min))
        ax[0, 1].set_ylim(a1y_min - offset(a1y_max, a1y_min), a1y_max + offset(a1y_max, a1y_min))
        ax[1, 0].set_ylim(a2y_min - offset(a2y_max, a2y_min), a2y_max + offset(a2y_max, a2y_min))
        ax[1, 1].set_ylim(a3y_min - offset(a3y_max, a3y_min), a3y_max + offset(a3y_max, a3y_min))

        return ax
