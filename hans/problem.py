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

from hans.tools import abort
from hans.material import Material
from hans.plottools import adaptiveLimits
from hans.integrate import ConservedField


class Problem:

    def __init__(self, options, disc, bc, geometry, numerics, material, surface, ic):
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
        self.ic = ic

        self.sanity_checks()

    def sanity_checks(self):

        self.check_options()
        self.check_disc()
        self.check_geo()
        self.check_num()
        self.check_mat()
        self.check_bc()
        if self.ic is not None:
            self.check_ic()
        if self.surface is not None:
            self.check_surface()

        print("Sanity checks completed. Start simulation!")
        print(60 * "-")

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

        if "maxT" in self.numerics.keys():
            maxT = self.numerics["maxT"]
        else:
            maxT = np.inf

        if "maxIt" in self.numerics.keys():
            maxIt = self.numerics["maxIt"]
        else:
            maxIt = np.inf

        if "tol" in self.numerics.keys():
            tol = self.numerics["tol"]
        else:
            tol = 0.

        # Initial conditions
        q_init, t_init = self.get_initial_conditions()

        # intialize solution field
        self.q = ConservedField(self.disc,
                                self.bc,
                                self.geometry,
                                self.material,
                                self.numerics,
                                self.surface,
                                q_init=q_init,
                                t_init=t_init)

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

                # maximum number of iterations reached
                if i >= maxIt:
                    self._write_mode = 3
                    break

                # NaN detected
                if self.q.isnan > 0:
                    self._write_mode = 4
                    break

                if i % writeInterval == 0:
                    self.write_to_netcdf(i, nc, mode=self._write_mode)
                    if rank == 0:
                        self.write_to_stdout(i, mode=self._write_mode)

            self.write_to_netcdf(i, nc, mode=self._write_mode)
            if rank == 0:
                self.write_to_stdout(i, mode=self._write_mode)

    def get_initial_conditions(self):
        """
        Return the initial field given by last frame of restart file
        or as defined through inputs.

        Returns
        -------
        np.array
            Inital field of conserved variables.
        tuple
            Inital time and timestep
        """

        if self.ic is None:
            q_init = np.zeros((3, self.disc["Nx"], self.disc["Ny"]))
            q_init[0] += self.material["rho0"]
            t_init = (0., self.numerics["dt"])
        else:
            # read last frame of restart file
            if self.ic["type"] == "restart":
                q_init, t_init = self.read_last_frame()
            elif self.ic["type"] == "perturbation":
                q_init = np.zeros((3, self.disc["Nx"], self.disc["Ny"]))
                q_init[0] += self.material["rho0"]
                t_init = (0., self.numerics["dt"])
                q_init[0, self.disc["Nx"] // 2, self.disc["Ny"] // 2] *= self.ic["factor"]
            elif self.ic["type"] == "longitudinal_wave":
                x = np.linspace(0 + self.disc["dx"]/2, self.disc["Lx"] - self.disc["dx"]/2, self.disc["Nx"])
                y = np.linspace(0 + self.disc["dy"]/2, self.disc["Ly"] - self.disc["dy"]/2, self.disc["Ny"])
                xx, yy = np.meshgrid(x, y, indexing="ij")

                q_init = np.zeros((3, self.disc["Nx"], self.disc["Ny"]))
                q_init[0] += self.material["rho0"]
                k = 2. * np.pi / self.disc["Lx"] * self.ic["nwave"]
                q_init[1] += self.ic["amp"] * np.sin(k * xx)
                t_init = (0., self.numerics["dt"])
            elif self.ic["type"] == "shear_wave":
                x = np.linspace(0 + self.disc["dx"]/2, self.disc["Lx"] - self.disc["dx"]/2, self.disc["Nx"])
                y = np.linspace(0 + self.disc["dy"]/2, self.disc["Ly"] - self.disc["dy"]/2, self.disc["Ny"])
                xx, yy = np.meshgrid(x, y, indexing="ij")

                q_init = np.zeros((3, self.disc["Nx"], self.disc["Ny"]))
                q_init[0] += self.material["rho0"]
                k = 2. * np.pi / self.disc["Lx"] * self.ic["nwave"]
                q_init[2] += self.ic["amp"] * np.sin(k * xx)
                t_init = (0., self.numerics["dt"])

        return q_init, t_init

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

        file = Dataset(self.ic["file"], "r")

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

        if self.ic is None or self.ic["type"] != "restart":
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
            nc.createVariable('ekin', 'f8', ('step'))

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

            if self.ic is not None:
                categories["ic"] = self.ic

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
            nc = Dataset(self.ic["file"], 'a', parallel=parallel, format='NETCDF3_64BIT_OFFSET')
            self.outpath = os.path.relpath(self.ic["file"])

            backup_file = f"{os.path.splitext(self.ic['file'])[0]}-{nc.restarts}.nc"

            # create backup
            if rank == 0:
                shutil.copy(self.ic["file"], backup_file)

            # increase restart counter
            nc.restarts += 1

            # append modified attributes
            nc.setncattr(f"tStart-{nc.restarts}", self.tStart.strftime("%d/%m/%Y %H:%M:%S"))
            for key, value in self.numerics.items():
                name = f"numerics_{key}-{nc.restarts}"
                nc.setncattr(name, value)

            nc.setncattr(f"ic_type-{nc.restarts}", "restart")
            nc.setncattr(f"ic_file-{nc.restarts}", backup_file)

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
            print(f"\nSolution has converged after {i:d} steps.", flush=True)
        elif mode == 2:
            print(f"\nNo convergence within {i: d} steps. Stopping criterion: \
maximum time {self.numerics['maxT']: .1e} s reached.", flush=True)
        elif mode == 3:
            print(f"\nNo convergence within {i: d} steps. Stopping criterion: \
maximum number of iterations reached.", flush=True)
        elif mode == 4:
            print(f"Nan detetcted in solution. Execution stopped.", flush=True)
        elif mode == 5:
            print(f"Execution stopped.", flush=True)

        if mode > 0:
            walltime = datetime.now() - self.tStart
            print(f"Output written to: {self.outpath}", flush=True)
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
        nc.variables["ekin"][step] = self.q.ekin

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
            self._write_mode = 5

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
        x = np.arange(Nx) * dx + dx / 2

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

    def check_options(self):
        """
        Sanity check for I/O options input.
        """
        print("Checking I/O options... ")

        try:
            writeInterval = int(self.options["writeInterval"])
            assert writeInterval > 0
        except KeyError:
            print("***Output interval not given, fallback to 1000")
            self.options["writeInterval"] = 1000
        except AssertionError:
            try:
                assert writeInterval != 0
            except AssertionError:
                print("***Output interval is zero. fallback to 1000")
                self.options["writeInterval"] = 1000
            else:
                print("***Output interval is negative. Converting to positive value.")
                writeInterval *= -1
                self.options["writeInterval"] = writeInterval

    def check_disc(self):
        """
        Sanity check for discretization input.
        [Nx, Ny] are required, then look for [Lx, Ly] or [dx, dy] (in that order).
        """
        print("Checking discretization... ")

        try:
            self.disc["Nx"] = int(self.disc['Nx'])
            assert self.disc["Nx"] > 0
        except KeyError:
            print("***Number of grid cells Nx not specified. Abort.")
            abort()
        except AssertionError:
            print("***Number of grid cells Nx must be larger than zero. Abort")
            abort()

        try:
            self.disc["Ny"] = int(self.disc['Ny'])
            assert self.disc["Ny"] > 0
        except KeyError:
            print("***Number of grid cells 'Ny' not specified. Abort.")
            abort()
        except AssertionError:
            print("***Number of grid cells 'Ny' must be larger than zero. Abort")
            abort()

        try:
            self.disc["Lx"] = float(self.disc["Lx"])
        except KeyError:
            try:
                self.disc["dx"] = float(self.disc["dx"])
            except KeyError:
                print("At least two of 'Nx' 'Lx', 'dx' must be given. Abort.")
                abort()
            else:
                self.disc["Lx"] = self.disc["dx"] * self.disc["Nx"]
        else:
            self.disc["dx"] = self.disc["Lx"] / self.disc["Nx"]

        try:
            self.disc["Ly"] = float(self.disc["Ly"])
        except KeyError:
            try:
                self.disc["dy"] = float(self.disc["dy"])
            except KeyError:
                print("At least two of 'Ny' 'Ly', 'dy' must be given. Abort.")
                abort()
            else:
                self.disc["Ly"] = self.disc["dy"] * self.disc["Ny"]
        else:
            self.disc["dy"] = self.disc["Ly"] / self.disc["Ny"]

    def check_geo(self):
        """
        Sanity check for geometry input.
        """
        print("Checking geometry... ")

        if self.geometry["type"] in ["journal", "journal_x", "journal_y"]:
            self.geometry["CR"] = float(self.geometry["CR"])
            self.geometry["eps"] = float(self.geometry["eps"])
        elif self.geometry["type"] == "parabolic":
            self.geometry["hmin"] = float(self.geometry['hmin'])
            self.geometry["hmax"] = float(self.geometry['hmax'])
        elif self.geometry["type"] == "twin_parabolic":
            self.geometry["hmin"] = float(self.geometry['hmin'])
            self.geometry["hmax"] = float(self.geometry['hmax'])
        elif self.geometry["type"] in ["inclined", "inclined_x", "inclined_y"]:
            self.geometry["h1"] = float(self.geometry['h1'])
            self.geometry["h2"] = float(self.geometry['h2'])
        elif self.geometry["type"] == "inclined_pocket":
            self.geometry["h1"] = float(self.geometry['h1'])
            self.geometry["h2"] = float(self.geometry['h2'])
            self.geometry["hp"] = float(self.geometry['hp'])
            self.geometry["c"] = float(self.geometry['c'])
            self.geometry["l"] = float(self.geometry['l'])
            self.geometry["w"] = float(self.geometry['w'])
        elif self.geometry["type"] in ["half_sine", "half_sine_squared"]:
            self.geometry["h0"] = float(self.geometry['h0'])
            self.geometry["amp"] = float(self.geometry['amp'])
            self.geometry["num"] = float(self.geometry['num'])

    def check_num(self):
        """
        Sanity check for numerics options.
        """
        print("Checking numerics options... ")

        try:
            self.numerics["integrator"] = self.numerics["integrator"]
            assert self.numerics["integrator"] in ["MC", "MC_bf", "MC_fb", "MC_alt", "LW", "RK3"]
        except KeyError:
            print("***Integrator not specified. Use default (MacCormack).")
            self.numerics["integrator"] = "MC"
        except AssertionError:
            print(f'***Unknown integrator \'{self.numerics["integrator"]}\'. Abort.')
            abort()

        if self.numerics["integrator"].startswith("MC"):
            try:
                self.numerics["fluxLim"] = float(self.numerics["fluxLim"])
            except KeyError:
                pass

        try:
            self.numerics["stokes"] = int(self.numerics["stokes"])
        except KeyError:
            print("***Boolean parameter 'stokes' not given. Use default (True).")
            self.numerics["stokes"] = 1

        try:
            self.numerics["adaptive"] = int(self.numerics["adaptive"])
        except KeyError:
            print("***Boolean parameter 'adaptive' not given. Use default (False).")
            self.numerics["adaptive"] = 0

        if self.numerics["adaptive"] == 1:
            try:
                self.numerics["C"] = float(self.numerics["C"])
            except KeyError:
                print("***CFL number not given. Use default (0.5).")
                self.numerics["C"] = 0.5

        try:
            self.numerics["dt"] = float(self.numerics["dt"])
        except KeyError:
            print("***Timestep not given. Use default (1e-10).")
            self.numerics["dt"] = 1e-10

        stopping_criteria = 0

        try:
            self.numerics["tol"] = float(self.numerics["tol"])
            stopping_criteria += 1
        except KeyError:
            pass

        try:
            self.numerics["maxT"] = float(self.numerics["maxT"])
            stopping_criteria += 1
        except KeyError:
            pass

        try:
            self.numerics["maxIt"] = int(self.numerics["maxIt"])
            stopping_criteria += 1
        except KeyError:
            pass

        if stopping_criteria == 0:
            print("***No stopping criterion given. Abort.")
            abort()

        if self.numerics["integrator"] == "RK3":
            self.disc["nghost"] = 2
        else:
            self.disc["nghost"] = 1

    def check_mat(self):
        """
        Sanity check on material settings.
        """
        print("Checking material options... ")

        if self.material["EOS"] == "DH":
            self.material["rho0"] = float(self.material["rho0"])
            self.material["P0"] = float(self.material["P0"])
            self.material["C1"] = float(self.material["C1"])
            self.material["C2"] = float(self.material["C2"])
        elif self.material["EOS"] == "PL":
            self.material["rho0"] = float(self.material["rho0"])
            self.material["P0"] = float(self.material["P0"])
            self.material["alpha"] = float(self.material['alpha'])
        elif self.material["EOS"] == "vdW":
            self.material["M"] = float(self.material['M'])
            self.material["T"] = float(self.material['T0'])
            self.material["a"] = float(self.material['a'])
            self.material["b"] = float(self.material['b'])
        elif self.material["EOS"] == "Tait":
            self.material["rho0"] = float(self.material["rho0"])
            self.material["P0"] = float(self.material["P0"])
            self.material["K"] = float(self.material['K'])
            self.material["n"] = float(self.material['n'])
        elif self.material["EOS"] == "cubic":
            self.material["a"] = float(self.material['a'])
            self.material["b"] = float(self.material['b'])
            self.material["c"] = float(self.material['c'])
            self.material["d"] = float(self.material['d'])
        elif self.material["EOS"].startswith("Bayada"):
            self.material["cl"] = float(self.material["cl"])
            self.material["cv"] = float(self.material["cv"])
            self.material["rhol"] = float(self.material["rhol"])
            self.material["rhov"] = float(self.material["rhov"])
            self.material["shear"] = float(self.material["shear"])
            self.material["shearv"] = float(self.material["shearv"])
            self.material["rhov"] = float(self.material["rhov"])

        self.material["shear"] = float(self.material["shear"])
        self.material["bulk"] = float(self.material["bulk"])

        if "Pcav" in self.material.keys():
            self.material["Pcav"] = float(self.material["Pcav"])

        if "piezo" in self.material.keys():
            if self.material["piezo"] == "Barus":
                self.material["aB"] = float(self.material["aB"])
            elif self.material["piezo"] == "Vogel":
                self.material["rho0"] = float(self.material['rho0'])
                self.material["g"] = float(self.material["g"])
                self.material["mu_inf"] = float(self.material["mu_inf"])
                self.material["phi_inf"] = float(self.material["phi_inf"])
                self.material["BF"] = float(self.material["BF"])

        if "thinning" in self.material.keys():
            if self.material["thinning"] == "Eyring":
                self.material["tau0"] = float(self.material["tau0"])
            elif self.material["thinning"] == "Carreau":
                self.material["relax"] = float(self.material["relax"])
                self.material["a"] = float(self.material["a"])
                self.material["N"] = float(self.material["N"])
            elif self.material["thinning"] == "PL":
                self.material["shear"] = float(self.material["shear"])
                self.material["n"] = float(self.material["n"])

        if "PLindex" in self.material.keys():
            self.material["PLindex"] = float(self.material["PLindex"])

    def check_surface(self):
        """
        Sanity check for surface input.
        """
        print("Checking surface parameters... ")

        if "lslip" in self.surface.keys():
            self.surface["lslip"] = float(self.surface["lslip"])
        else:
            self.surface["lslip"] = 0.

        if self.surface["type"] in ["stripes", "stripes_x", "stripes_y"]:
            try:
                self.surface["num"] = int(self.surface["num"])
            except KeyError:
                self.surface["num"] = 1
            try:
                self.surface["sign"] = int(self.surface["sign"])
            except KeyError:
                self.surface["sign"] = -1

    def check_ic(self):
        """
        Sanity check for initial conditions input.
        """
        print("Checking initial conditions... ")

        if self.ic["type"] != "restart":
            if self.ic["type"] == "perturbation":
                self.ic["factor"] = float(self.ic["factor"])
            elif self.ic["type"] in ["longitudinal_wave", "shear_wave"]:
                self.ic["amp"] = float(self.ic["amp"])
                if "nwave" in self.ic.keys():
                    self.ic["nwave"] = int(self.ic["nwave"])
                else:
                    self.ic["nwave"] = 1

    def check_bc(self):
        """
        Sanity check for boundary condition input.

        Parameters
        ----------
        bc : dict
            Boundary condition parameters read from yaml input file.
        disc : dict
            Discretization parameters.
        material : dict
            Material parameters.

        Returns
        -------
        dict
            Boundary condition parameters.

        """
        print("Checking boundary conditions... ")

        self.bc["x0"] = np.array(list(self.bc["x0"]))
        self.bc["x1"] = np.array(list(self.bc["x1"]))
        self.bc["y0"] = np.array(list(self.bc["y0"]))
        self.bc["y1"] = np.array(list(self.bc["y1"]))

        assert len(self.bc["x0"]) == 3
        assert len(self.bc["x1"]) == 3
        assert len(self.bc["y0"]) == 3
        assert len(self.bc["y1"]) == 3

        if "P" in self.bc["x0"] and "P" in self.bc["x1"]:
            self.disc["pX"] = 1
        else:
            self.disc["pX"] = 0

        if "P" in self.bc["y0"] and "P" in self.bc["y1"]:
            self.disc["pY"] = 1
        else:
            self.disc["pY"] = 0

        if "D" in self.bc["x0"]:
            if "px0" in self.bc.keys():
                px0 = float(self.bc["px0"])
                self.bc["rhox0"] = Material(self.material).eos_density(px0)
            else:
                self.bc["rhox0"] = self.material["rho0"]

        if "D" in self.bc["x1"]:
            if "px1" in self.bc.keys():
                px1 = float(self.bc["px1"])
                self.bc["rhox1"] = Material(self.material).eos_density(px1)
            else:
                self.bc["rhox1"] = self.material["rho0"]

        if "D" in self.bc["y0"]:
            if "py0" in self.bc.keys():
                py0 = float(self.bc["py0"])
                self.bc["rhoy0"] = Material(self.material).eos_density(py0)
            else:
                self.bc["rhoy0"] = self.material["rho0"]

        if "D" in self.bc["y1"]:
            if "py1" in self.bc.keys():
                py1 = float(self.bc["py1"])
                self.bc["rhoy1"] = Material(self.material).eos_density(py1)
            else:
                self.bc["rhoy1"] = self.material["rho0"]

        assert np.all((self.bc["x0"] == "P") == (self.bc["x1"] == "P")), "Inconsistent boundary conditions (x)"
        assert np.all((self.bc["y0"] == "P") == (self.bc["y1"] == "P")), "Inconsistent boundary conditions (y)"
