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
import shutil
# from functools import partial

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

    def __init__(self, options, disc, BC, geometry, numerics, material, restart_file):
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
        restart_file : str
            Filename of the netCDF file, from which simulation is restarted
        """

        self.options = options
        self.disc = disc
        self.BC = BC
        self.geometry = geometry
        self.numerics = numerics
        self.material = material
        self.restart_file = restart_file
        self.writeInterval = int(options['writeInterval'])
        self.name = str(options['name'])
        self.q_init = None
        self.t_init = None

        if self.restart_file is not None:
            self.read_last_frame()

        self.Nx = self.disc["Nx"]
        self.Ny = self.disc["Ny"]

    def read_last_frame(self):
        """Read last frame from the restart file and use as initial values for new run
        """

        file = netCDF4.Dataset(self.restart_file, "r")

        rho = np.array(file.variables['rho'])[-1]
        jx = np.array(file.variables['jx'])[-1]
        jy = np.array(file.variables['jy'])[-1]
        dt = file.variables["dt"][-1]
        t = file.variables["time"][-1]

        q0 = np.zeros([3] + list(rho.shape))

        q0[0] = rho
        q0[1] = jx
        q0[2] = jy

        self.q_init = q0
        self.t_init = (t, dt)

    def run(self, plot=False, out_dir="data"):
        self.q = ConservedField(self.disc,
                                self.BC,
                                self.geometry,
                                self.material,
                                self.numerics,
                                q_init=self.q_init,
                                t_init=self.t_init)

        self.tStart = time.time()
        print("{:10s}\t{:12s}\t{:12s}\t{:12s}".format("Step", "Timestep", "Time", "Epsilon"))

        if plot:
            self.plot()
        else:

            self.init_netcdf(out_dir)

            maxT = self.numerics["maxT"]
            tol = self.numerics["tol"]
            self.write_mode = None

            i = 0
            while True:
                self.q.update(i)

                # catch signals and execute signal handler
                signal.signal(signal.SIGINT, self.receive_signal)
                signal.signal(signal.SIGTERM, self.receive_signal)
                signal.signal(signal.SIGHUP, self.receive_signal)
                signal.signal(signal.SIGUSR1, self.receive_signal)
                # signal.signal(signal.SIGUSR2, self.receive_signal)

                # convergence
                if self.q.eps < tol:
                    self.write_mode = "converged"

                # maximum time reached
                if (maxT - self.q.time) < self.q.dt:
                    i += 1
                    self.q.update(i)
                    self.write_mode = "maxtime"

                # write to file and stdout given the specified mode
                self.write(i, mode=self.write_mode)

                # increase time step
                i += 1

    def init_netcdf(self, out_dir):
        if not(os.path.exists(out_dir)):
            os.makedirs(out_dir)

        if self.restart_file is None:
            file_tag = 1
            existing_tags = sorted([int(os.path.splitext(f)[0].split("_")[-1].split("-")[0].lstrip("0"))
                                    for f in os.listdir(out_dir) if f.startswith(f"{self.name}_")])
            if len(existing_tags) > 0:
                file_tag = existing_tags[-1] + 1

            outfile = f"{self.name}_{str(file_tag).zfill(4)}.nc"
            self.outpath = os.path.join(out_dir, outfile)

            # initialize NetCDF file
            self.nc = netCDF4.Dataset(self.outpath, 'w', format='NETCDF3_64BIT_OFFSET')
            self.nc.restarts = 0
            self.nc.createDimension('x', self.Nx)
            self.nc.createDimension('y', self.Ny)
            self.nc.createDimension('step', None)

            # create conserved variables timeseries of fields
            self.nc.createVariable('rho', 'f8', ('step', 'x', 'y'))
            self.nc.createVariable('jx', 'f8', ('step', 'x', 'y'))
            self.nc.createVariable('jy', 'f8', ('step', 'x', 'y'))

            # create scalar variables
            self.nc.createVariable('time', 'f8', ('step'))
            self.nc.createVariable('mass', 'f8', ('step'))
            self.nc.createVariable('vmax', 'f8', ('step'))
            self.nc.createVariable('vSound', 'f8', ('step'))
            self.nc.createVariable('dt', 'f8', ('step'))
            self.nc.createVariable('eps', 'f8', ('step'))

            repo_path = [path for path in sys.path if path.endswith("pylub")][0]
            git_commit = str(Repo(path=repo_path, search_parent_directories=True).head.object.hexsha)

            self.nc.setncattr(f"tStart-{self.nc.restarts}", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
            self.nc.commit = git_commit

            categories = {"options": self.options,
                          "disc": self.disc,
                          "bc": self.BC,
                          "geometry": self.geometry,
                          "numerics": self.numerics,
                          "material": self.material}

            for cat_name, cat in categories.items():
                for key, value in cat.items():
                    name = cat_name + "_" + key
                    self.nc.setncattr(name, value)

        else:

            # append to existing netCDF file
            self.nc = netCDF4.Dataset(self.restart_file, 'a', format='NETCDF3_64BIT_OFFSET')
            self.outpath = os.path.relpath(self.restart_file)

            # create backup
            backup_file = f"{os.path.splitext(self.restart_file)[0]}-{self.nc.restarts}.nc"
            shutil.copy(self.restart_file, backup_file)

            # increase restart counter
            self.nc.restarts += 1

            # append modified attributes
            self.nc.setncattr(f"tStart-{self.nc.restarts}", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
            for key, value in self.numerics.items():
                name = "numerics_" + key + "-" + str(self.nc.restarts)
                self.nc.setncattr(name, value)

    def write(self, i, mode=None):

        def to_netcdf(i, last=False):
            print(f"{i:10d}\t{self.q.dt:.6e}\t{self.q.time:.6e}\t{self.q.eps:.6e}", flush=True)

            k = self.nc.variables["rho"].shape[0]

            self.nc.variables["rho"][k] = self.q.field[0, 1:-1, 1:-1]
            self.nc.variables["jx"][k] = self.q.field[1, 1:-1, 1:-1]
            self.nc.variables["jy"][k] = self.q.field[2, 1:-1, 1:-1]

            self.nc.variables["time"][k] = self.q.time
            self.nc.variables["mass"][k] = self.q.mass
            self.nc.variables["vmax"][k] = self.q.vmax
            self.nc.variables["vSound"][k] = self.q.vSound
            self.nc.variables["dt"][k] = self.q.dt
            self.nc.variables["eps"][k] = self.q.eps

            if last:
                self.nc.setncattr(f"tEnd-{self.nc.restarts}", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))

        if i % self.writeInterval == 0:
            to_netcdf(i)
        elif mode == "converged":
            to_netcdf(i, last=True)
            print(f"\nSolution has converged after {i:d} steps, Output written to: {self.outpath}")
        elif mode == "maxtime":
            to_netcdf(i, last=True)
            print(f"\nNo convergence within {i:d} steps. Stopping criterion: maximum time {self.numerics['maxT']:.1e} s reached.")
            print(f"Output written to: {self.outpath}")
        elif mode == "abort":
            to_netcdf(i, last=True)
            print(f"Execution stopped. Output written to: {self.outpath}")

        if mode is not None:
            walltime = time.time() - self.tStart
            print(f"Total wall clock time: {time.strftime('%H:%M:%S', time.gmtime(walltime))} (Performance: {i / walltime:.2f} steps/s)")
            sys.exit()

    def receive_signal(self, signum, frame):

        if signum in [signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGUSR1]:
            self.write_mode = "abort"

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
