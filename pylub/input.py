import yaml
import netCDF4
import numpy as np

from pylub.problem import Problem


class Input:
    """Reads the yaml input file to define a problem.

    Attributes
    ----------
    inputFile : str
        filename of the .yaml input file
    restartFile : str
        filename of the .nc data file
    """

    def __init__(self, inputFile, restartFile=None):
        """ Constructor

        Parameters
        ----------
        inputFile : str
            filename of the .yaml input file
        restartFile : str
            filename of the .nc data file
        """
        self.inputFile = inputFile
        self.restartFile = restartFile

    def getProblem(self):
        """Parses through yaml file and returns an instance of the Problem class.

        Returns
        -------
        Problem
            Object containing all information about the current job.

        """

        with open(self.inputFile, 'r') as ymlfile:
            inp = yaml.full_load(ymlfile)

            options = self.check_options(inp['options'])
            disc = self.check_disc(inp['disc'])
            BC = self.check_bc(inp['BC'])
            geometry = self.check_geo(inp['geometry'])
            numerics = self.check_num(inp['numerics'])
            material = self.check_mat(inp['material'])

        q_init = None
        t_init = None

        if self.restartFile is not None:
            q_init = self.getInitialField()
            time = self.getInitialTime()

        thisProblem = Problem(options,
                              disc,
                              BC,
                              geometry,
                              numerics,
                              material,
                              q_init,
                              t_init)

        return thisProblem

    # consistency checks for input dicts
    def check_disc(self, disc):
        print("Checking discretization... ", end="", flush=True)

        Nx = int(disc['Nx'])
        Ny = int(disc['Ny'])

        if "dx" in disc.keys():
            dx = float(disc['dx'])
            Lx = dx * Nx
            disc['Lx'] = Lx
        else:
            Lx = float(disc["Lx"])
            disc["dx"] = Lx / Nx

        if "dy" in disc.keys():
            dy = float(disc['dy'])
            Ly = dy * Ny
            disc['Ly'] = Ly
        else:
            Ly = float(disc["Ly"])
            disc["dy"] = Ly / Ny

        print("Done!")

        return disc

    def check_options(self, options):
        print("Checking I/O options... ", end="", flush=True)
        print("Done!")
        return options

    def check_bc(self, bc):
        print("Checking boundary conditions... ", end="", flush=True)

        x0 = np.array(list(bc["x0"]))
        x1 = np.array(list(bc["x1"]))
        y0 = np.array(list(bc["y0"]))
        y1 = np.array(list(bc["y1"]))

        assert len(x0) == 3
        assert len(x1) == 3
        assert len(y0) == 3
        assert len(y1) == 3

        assert np.all((x0 == "P") == (x1 == "P")), "Inconsistent boundary conditions (x)"
        assert np.all((y0 == "P") == (y1 == "P")), "Inconsistent boundary conditions (y)"

        print("Done!")
        return bc

    def check_geo(self, geo):
        print("Checking geometry... ", end="", flush=True)
        print("Done!")
        return geo

    def check_num(self, numerics):
        print("Checking numerics options... ", end="", flush=True)
        print("Done!")
        return numerics

    def check_mat(self, material):
        print("Checking material options... ", end="", flush=True)
        print("Done!")
        return material

    def getInitialField(self):
        """Read final field from NetCDF restart datafile.

        Returns
        -------
        q0 : numpy.ndarray
            Initial field to restart a simulation from a previous run.
        """

        file = netCDF4.Dataset(self.restartFile)

        rho = np.array(file.variables['rho'])[-1]
        jx = np.array(file.variables['jx'])[-1]
        jy = np.array(file.variables['jy'])[-1]

        q0 = np.zeros([3] + list(rho.shape))

        q0[0] = rho
        q0[1] = jx
        q0[2] = jy

        return q0

    def getInitialTime(self):

        file = netCDF4.Dataset(self.restartFile)

        dt = float(file.variables["dt"][-1])
        t = float(file.variables["time"][-1])

        return (t, dt)
