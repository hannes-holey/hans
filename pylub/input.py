import sys
import yaml
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

        thisProblem = Problem(options,
                              disc,
                              BC,
                              geometry,
                              numerics,
                              material,
                              self.restartFile)

        return thisProblem

    def check_disc(self, disc):
        """Check correctness of discretization input.

        Returns
        -------
        disc : dict
            Discretaization dictionary
        """
        print("Checking discretization... ", end="", flush=True)

        try:
            Nx = int(disc['Nx'])
            assert Nx > 0
        except KeyError:
            print("\nNumber of grid cells Nx not given. Abort.")
            sys.exit(1)
        except AssertionError:
            try:
                assert Nx != 0
            except AssertionError:
                print("\nNumber of grid cells Nx zero. Abort")
                sys.exit(1)
            else:
                print("\nNumber of grid cells Nx negative. Converting to positive value.")
                Nx *= -1
                disc["Nx"] = Nx

        try:
            Ny = int(disc['Ny'])
            assert Ny > 0
        except KeyError:
            print("\nNumber of grid cells Ny not given. Abort.")
            sys.exit(1)
        except AssertionError:
            try:
                assert Ny != 0
            except AssertionError:
                print("\nNumber of grid cells Ny zero. Abort")
                sys.exit(1)
            else:
                print("\nNumber of grid cells Ny negative. Converting to positive value.")
                Ny *= -1
                disc["Ny"] = Ny

        try:
            assert "dx" in disc.keys() or "Lx" in disc.keys()
        except AssertionError:
            print("\nNeither 'dx' nor 'Lx' given. Abort.")
            sys.exit(1)

        try:
            assert "dy" in disc.keys() or "Ly" in disc.keys()
        except AssertionError:
            print("\nNeither 'dy' nor 'Ly' given. Abort.")
            sys.exit(1)

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
