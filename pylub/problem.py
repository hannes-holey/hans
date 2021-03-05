import os
import sys
import signal
import netCDF4
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from datetime import datetime
from git import Repo
import numpy as np

from pylub.eos import EquationOfState
from pylub.plottools import adaptiveLimits
from pylub.integrate import ConservedField


class Problem:
    """Problem class, contains all information about a specific simulation

    Attributes
    ----------
    options : dict
        Contains IO options.
    disc : dict
        Contains discretization parameters.
    geometry : dict
        Contains geometry parameters.
    numerics : dict
        Contains parameters for the numeric scheme.
    material : dict
        Contains material parameters.

    """

    def __init__(self, options, disc, BC, geometry, numerics, material, q_init, t_init):
        """Constructor

        Parameters
        ----------
        options : dict
            Contains IO options.
        disc : dict
            Contains discretization parameters.
        geometry : dict
            Contains geometry parameters.
        numerics : dict
            Contains parameters for the numeric scheme.
        material : dict
            Contains material parameters.
        q_init : np.ndarray
            Solution field at last time step of restart file.
        t_init : tuple
            Contains total time and last time step of simulation from restart file.
        """

        self.options = options
        self.disc = disc
        self.BC = BC
        self.geometry = geometry
        self.numerics = numerics
        self.material = material
        self.q_init = q_init
        self.t_init = t_init

        self.writeInterval = int(options['writeInterval'])
        self.name = str(options['name'])

        self.Nx = self.disc["Nx"]
        self.Ny = self.disc["Ny"]

    def run(self, plot=False, out_dir="data"):
        self.q = ConservedField(self.disc,
                                self.BC,
                                self.geometry,
                                self.material,
                                self.numerics,
                                q_init=self.q_init,
                                t_init=self.t_init)

        self.tStart = time.time()

        if plot:
            print("{:10s}\t{:12s}\t{:12s}\t{:12s}".format("Step", "Time step", "Time", "Epsilon"))
            self.plot()
        else:
            self.init_netcdf(out_dir)

            maxT = float(self.numerics["maxT"])
            tol = float(self.numerics["tol"])

            i = 0
            while self.q.time < maxT:

                self.q.update(i)
                tDiff = maxT - self.q.time

                signal.signal(signal.SIGTERM, self.receive_signal)
                signal.signal(signal.SIGHUP, self.receive_signal)
                signal.signal(signal.SIGUSR1, self.receive_signal)
                signal.signal(signal.SIGUSR2, self.receive_signal)

                if i % self.writeInterval == 0:
                    self.write(i)
                elif self.q.eps < tol:
                    self.write(i, mode="converged")
                    break
                elif tDiff < self.q.dt:
                    i += 1
                    self.q.update(i)
                    self.write(i, mode="maxtime")

                i += 1

    def init_netcdf(self, out_dir):
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

    def write(self, i, mode=None):

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

        print(f"{i:10d}\t{self.q.dt:.6e}\t{self.q.time:.6e}\t{self.q.eps:.6e}")
        sys.stdout.flush()

        k = self.rho.shape[0]

        self.rho[k] = self.q.field[0, 1:-1, 1:-1]
        self.jx[k] = self.q.field[1, 1:-1, 1:-1]
        self.jy[k] = self.q.field[2, 1:-1, 1:-1]

        self.time[k] = self.q.time
        self.mass[k] = self.q.mass
        self.vmax[k] = self.q.vmax
        self.vSound[k] = self.q.vSound
        self.dt[k] = self.q.dt
        self.eps[k] = self.q.eps

        if mode == "converged":
            self.nc.tEnd = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            print(f"\nSolution has converged after {i:d} steps, Output written to: {self.relpath}")
        elif mode == "maxtime":
            self.nc.tEnd = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            print(f"\nNo convergence within {i:d} steps. Stopping criterion: maximum time {float(self.numerics['maxT']):.1e} s reached.")
            print(f"Output written to: {self.relpath}")
        elif mode == "abort":
            self.nc.tEnd = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            print("Walltime exceeded")

        if mode is not None:
            walltime = time.time() - self.tStart
            print(f"Total wall clock time: {time.strftime('%H:%M:%S', time.gmtime(walltime))} (Performance: {i / walltime:.2f} steps/s)")

    def receive_signal(self, i, signum, stack):
        walltime = time.time() - self.tStart
        print(f"python: PID: {os.getpid()} recieved {signum} at time {walltime}", flush=True)

        if signum == signal.SIGTERM:
            sys.exit()

        if signum == signal.SIGUSR1:
            self.write(i, mode="abort")
            sys.exit()

    def plot(self):

        fig, ax = plt.subplots(2, 2, figsize=(14, 9), sharex=True)

        x = self.q.xx[1:-1, self.Ny // 2]
        ax[0, 0].plot(x, self.q.field[1, 1:-1, self.Ny // 2])
        ax[0, 1].plot(x, self.q.field[2, 1:-1, self.Ny // 2])
        ax[1, 0].plot(x, self.q.field[0, 1:-1, self.Ny // 2])
        ax[1, 1].plot(x, EquationOfState(self.material).isoT_pressure(self.q.field[0, 1:-1, self.Ny // 2]))

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

        fig.suptitle('time = {:.2f} ns'.format(self.q.time * 1e9))

        ax[0, 0].lines[0].set_ydata(self.q.field[1, 1:-1, self.Ny // 2])
        ax[0, 1].lines[0].set_ydata(self.q.field[2, 1:-1, self.Ny // 2])
        ax[1, 0].lines[0].set_ydata(self.q.field[0, 1:-1, self.Ny // 2])
        ax[1, 1].lines[0].set_ydata(EquationOfState(self.material).isoT_pressure(self.q.field[0, 1:-1, self.Ny // 2]))

        ax = adaptiveLimits(ax)

        self.q.update(i)
        if i % self.writeInterval == 0:
            print("{:10d}\t{:.6e}\t{:.6e}\t{:.6e}".format(i, self.q.dt, self.q.time, self.q.eps))