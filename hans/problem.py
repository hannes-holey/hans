"""
MIT License

Copyright 2021 Hannes Holey

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


import os
import signal
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from datetime import datetime
from pkg_resources import get_distribution
import numpy as np
import shutil

from hans.material import Material
from hans.plottools import adaptiveLimits
from hans.integrate import ConservedField


class Problem:

    def __init__(self, options, disc, bc, geometry, numerics, material, surface, restart_file):
        """
        Collects all information about a single problem
        and contains the methods to run a simulation, based on the problem defintiion."

        Parameters
        ----------
        options : dict
            Contains IO options.
        disc : dict
            Contains discretization parameters.
        bc : dict
            Contains boundary condition parameters.
        geometry : dict
            Contains geometry parameters.
        numerics : dict
            Contains numerics parameters.
        material : dict
            Contains material parameters.
        surface : dict
            Contains surface parameters.
        restart_file : str
            Filename of the netCDF file, from which simulation is restarted.

        """

        self.options = options
        self.disc = disc
        self.bc = bc
        self.geometry = geometry
        self.numerics = numerics
        self.material = material
        self.surface = surface
        self.restart_file = restart_file

    def run(self, out_dir="data", out_name=None, plot=False):
        """
        Starts the simulation.

        Parameters
        ----------
        out_dir : str
            Output directory (default: data).
        out_name : str
            NetCDF output filename (default: None)
        plot : bool
            On-the-fly plotting flag (default: False).

        """

        # global write options
        writeInterval = self.options['writeInterval']
        maxT = self.numerics["maxT"]
        tol = self.numerics["tol"]

        # read last frame of restart file
        if self.restart_file is not None:
            q_init, t_init = self.read_last_frame()
        else:
            q_init = q_init = np.zeros((3, self.disc["Nx"], self.disc["Ny"]))
            q_init[0] += self.material["rho0"]
            t_init = (0., self.numerics["dt"])

        # intialize solution field
        self.q = ConservedField(self.disc,
                                self.bc,
                                self.geometry,
                                self.material,
                                self.numerics,
                                self.surface,
                                q_init,
                                t_init)

        rank = self.q.comm.Get_rank()

        # time stamp of simulation start time
        self.tStart = datetime.now()

        # Header line for screen output
        if rank == 0:
            print(f"{'Step':10s}\t{'Timestep':12s}\t{'Time':12s}\t{'Epsilon':12s}", flush=True)

        if plot:
            # on-the-fly plotting
            self.plot(writeInterval)
        else:
            nc = self.init_netcdf(out_dir, out_name, rank)

            i = 0
            self._write_mode = 0

            while self._write_mode == 0:

                # Perform time update
                self.q.update(i)

                # increase time step
                i += 1

                # catch signals and execute signal handler
                signal.signal(signal.SIGINT, self.receive_signal)
                signal.signal(signal.SIGTERM, self.receive_signal)
                signal.signal(signal.SIGHUP, self.receive_signal)
                signal.signal(signal.SIGUSR1, self.receive_signal)
                signal.signal(signal.SIGUSR2, self.receive_signal)

                # convergence
                if i > 1 and self.q.eps < tol:
                    self._write_mode = 1
                    break

                # maximum time reached
                if round(self.q.time, 15) >= maxT:
                    self._write_mode = 2
                    break

                if i % writeInterval == 0:
                    self.write_to_netcdf(i, nc, mode=self._write_mode)
                    if rank == 0:
                        self.write_to_stdout(i, mode=self._write_mode)

            self.write_to_netcdf(i, nc, mode=self._write_mode)
            if rank == 0:
                self.write_to_stdout(i, mode=self._write_mode)

    def read_last_frame(self):
        """
        Read last frame from restart file and use as initial values for new run.

        Returns
        -------
        np.array
            Solution field at last frame, used as inital field.
        tuple
            Total time and timestep of last frame.
        """

        file = Dataset(self.restart_file, "r")

        rho = np.array(file.variables['rho'])[-1]
        jx = np.array(file.variables['jx'])[-1]
        jy = np.array(file.variables['jy'])[-1]
        dt = float(file.variables["dt"][-1])
        t = float(file.variables["time"][-1])

        q0 = np.zeros([3] + list(rho.shape))

        q0[0] = rho
        q0[1] = jx
        q0[2] = jy

        return q0, (t, dt)

    def init_netcdf(self, out_dir, out_name, rank):
        """
        Initialize netCDF4 file, create dimensions, variables and metadata.

        Parameters
        ----------
        out_dir : str
            Output directoy.
        out_name : str
            Filename prefix.
        rank : int
            Rank of the MPI communicator
        Returns
        -------
        netCDF4.Dataset
            Initialized dataset.

        """

        if rank == 0:
            if not(os.path.exists(out_dir)):
                os.makedirs(out_dir)

        if self.restart_file is None:
            if rank == 0:
                if out_name is None:
                    # default unique filename with timestamp
                    timestamp = datetime.now().replace(microsecond=0).strftime("%Y-%m-%d_%H%M%S")
                    name = self.options["name"]
                    outfile = f"{timestamp}_{name}.nc"
                else:
                    # custom filename with zero padded number
                    tag = str(len([1 for f in os.listdir(out_dir) if f.startswith(out_name)]) + 1).zfill(4)
                    outfile = f"{out_name}-{tag}.nc"

                self.outpath = os.path.join(out_dir, outfile)
            else:
                self.outpath = None

            self.outpath = self.q.comm.bcast(self.outpath, root=0)

            # initialize NetCDF file
            parallel = False
            if self.q.comm.Get_size() > 1:
                parallel = True

            nc = Dataset(self.outpath, 'w', parallel=parallel, format='NETCDF3_64BIT_OFFSET')
            nc.restarts = 0
            nc.createDimension('x', self.disc["Nx"])
            nc.createDimension('y', self.disc["Ny"])
            nc.createDimension('step', None)

            # create unknown variable buffer as timeseries of 2D fields
            var0 = nc.createVariable('rho', 'f8', ('step', 'x', 'y'))
            var1 = nc.createVariable('jx', 'f8', ('step', 'x', 'y'))
            var2 = nc.createVariable('jy', 'f8', ('step', 'x', 'y'))
            var0.set_collective(True)
            var1.set_collective(True)
            var2.set_collective(True)

            # create scalar variables
            nc.createVariable('time', 'f8', ('step'))
            nc.createVariable('mass', 'f8', ('step'))
            nc.createVariable('vmax', 'f8', ('step'))
            nc.createVariable('vSound', 'f8', ('step'))
            nc.createVariable('dt', 'f8', ('step'))
            nc.createVariable('eps', 'f8', ('step'))

            # write metadata
            nc.setncattr(f"tStart-{nc.restarts}", self.tStart.strftime("%d/%m/%Y %H:%M:%S"))
            nc.setncattr("version", get_distribution('hans').version)

            disc = self.disc.copy()
            bc = self.bc.copy()

            categories = {"options": self.options,
                          "disc": disc,
                          "bc": bc,
                          "geometry": self.geometry,
                          "numerics": self.numerics,
                          "material": self.material}

            if self.surface is not None:
                categories["surface"] = self.surface

            # reset modified input dictionaries
            bc["x0"] = "".join(bc["x0"])
            bc["x1"] = "".join(bc["x1"])
            bc["y0"] = "".join(bc["y0"])
            bc["y1"] = "".join(bc["y1"])

            del disc["nghost"]
            del disc["pX"]
            del disc["pY"]

            for name, cat in categories.items():
                for key, value in cat.items():
                    nc.setncattr(f"{name}_{key}", value)

        else:
            # append to existing netCDF file
            parallel = False
            if self.q.comm.Get_size() > 1:
                parallel = True
            nc = Dataset(self.restart_file, 'a', parallel=parallel, format='NETCDF3_64BIT_OFFSET')
            self.outpath = os.path.relpath(self.restart_file)

            # create backup
            if rank == 0:
                backup_file = f"{os.path.splitext(self.restart_file)[0]}-{nc.restarts}.nc"
                shutil.copy(self.restart_file, backup_file)

            # increase restart counter
            nc.restarts += 1

            # append modified attributes
            nc.setncattr(f"tStart-{nc.restarts}", self.tStart.strftime("%d/%m/%Y %H:%M:%S"))
            for key, value in self.numerics.items():
                name = f"numerics_{key}-{nc.restarts}"
                nc.setncattr(name, value)

        return nc

    def write_to_stdout(self, i, mode):
        """
        Write information about the current time step to stdout.

        Parameters
        ----------
        i : int
            Current time step.
        mode : int
            Writing mode (0: normal, 1: converged, 2: max time, 3: execution stopped).

        """
        print(f"{i:10d}\t{self.q.dt:.6e}\t{self.q.time:.6e}\t{self.q.eps:.6e}", flush=True)

        if mode == 1:
            print(f"\nSolution has converged after {i:d} steps, Output written to: {self.outpath}", flush=True)
        elif mode == 2:
            print(f"\nNo convergence within {i: d} steps", end=" ", flush=True)
            print(f"Stopping criterion: maximum time {self.numerics['maxT']: .1e} s reached.", flush=True)
            print(f"Output written to: {self.outpath}", flush=True)
        elif mode == 3:
            print(f"Execution stopped. Output written to: {self.outpath}", flush=True)

        if mode > 0:
            walltime = datetime.now() - self.tStart
            print(f"Total wall clock time: {str(walltime).split('.')[0]}", end=" ", flush=True)
            print(f"(Performance: {i/walltime.total_seconds(): .2f} steps/s", end=" ", flush=True)
            print(f"on {self.q.comm.dims[0]} x {self.q.comm.dims[1]} MPI grid)", flush=True)

    def write_to_netcdf(self, i, nc, mode):
        """
        Append current solution field to netCDF file.

        Parameters
        ----------
        i : int
            Current time step.
        nc : netCDF4.Dataset
            NetCDF Dataset object.
        mode : int
            Writing mode (0: normal, 1: converged, 2: max time, 3: execution stopped).

        """

        step = nc.variables["rho"].shape[0]
        xrange, yrange = self.q.without_ghost

        nc.variables['rho'][step, xrange, yrange] = self.q.inner[0]
        nc.variables['jx'][step, xrange, yrange] = self.q.inner[1]
        nc.variables['jy'][step, xrange, yrange] = self.q.inner[2]

        nc.variables["time"][step] = self.q.time
        nc.variables["mass"][step] = self.q.mass
        nc.variables["vmax"][step] = self.q.vmax
        nc.variables["vSound"][step] = self.q.vSound
        nc.variables["dt"][step] = self.q.dt
        nc.variables["eps"][step] = self.q.eps

        if mode > 0:
            nc.setncattr(f"tEnd-{nc.restarts}", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
            nc.close()

    def receive_signal(self, signum, frame):
        """
        Signal handler. Catches signals send to the process and sets write mode to 3 (abort).

        Parameters
        ----------
        signum :
            signal code
        frame :
            Description of parameter `frame`.

        """

        if signum in [signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGUSR1]:
            self._write_mode = 3

    def plot(self, writeInterval):
        """
        Initialize on-the-fly plotting.

        Parameters
        ----------
        writeInterval : int
            Write interval for stdout in plotting mode.

        """

        fig, ax = plt.subplots(2, 2, figsize=(14, 9), sharex=True)

        Nx = self.disc["Nx"]
        dx = self.disc["dx"]
        Lx = self.disc["Lx"]
        x = np.arange(Nx) * Lx + dx / 2

        ax[0, 0].plot(x, self.q.centerline_x[1])
        ax[0, 1].plot(x, self.q.centerline_x[2])
        ax[1, 0].plot(x, self.q.centerline_x[0])
        ax[1, 1].plot(x, Material(self.material).eos_pressure(self.q.centerline_x[0]))

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
                                    fargs=(fig, ax, writeInterval),
                                    interval=1,
                                    init_func=init,
                                    repeat=False)

        plt.show()

    def animate1D(self, i, fig, ax, writeInterval):
        """
        Animator function. Update solution and plots.

        Parameters
        ----------
        i : type
            Current time step.
        fig : matplotlib.figure
            Figure object.
        ax :  np.array
            Array containing the axes of the figure.
        writeInterval : int
            Write interval for stdout in plotting mode.

        """

        self.q.update(i)
        fig.suptitle('time = {:.2f} ns'.format(self.q.time * 1e9))

        ax[0, 0].lines[0].set_ydata(self.q.centerline_x[1])
        ax[0, 1].lines[0].set_ydata(self.q.centerline_x[2])
        ax[1, 0].lines[0].set_ydata(self.q.centerline_x[0])
        ax[1, 1].lines[0].set_ydata(Material(self.material).eos_pressure(self.q.centerline_x[0]))

        ax = adaptiveLimits(ax)

        if i % writeInterval == 0:
            print(f"{i:10d}\t{self.q.dt:.6e}\t{self.q.time:.6e}\t{self.q.eps:.6e}", flush=True)
