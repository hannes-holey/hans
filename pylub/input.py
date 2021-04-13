import sys
import yaml
import numpy as np

from pylub.problem import Problem
from pylub.eos import EquationOfState


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
            geometry = self.check_geo(inp['geometry'])
            numerics = self.check_num(inp['numerics'], disc)
            material = self.check_mat(inp['material'])
            BC = self.check_bc(inp['BC'], disc, material)

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

    def check_bc(self, bc, disc, material):
        print("Checking boundary conditions... ", end="", flush=True)

        x0 = np.array(list(bc["x0"]))
        x1 = np.array(list(bc["x1"]))
        y0 = np.array(list(bc["y0"]))
        y1 = np.array(list(bc["y1"]))

        assert len(x0) == 3
        assert len(x1) == 3
        assert len(y0) == 3
        assert len(y1) == 3

        if "P" in x0 and "P" in x1:
            disc["pX"] = 1
        else:
            disc["pX"] = 0

        if "P" in y0 and "P" in y1:
            disc["pY"] = 1
        else:
            disc["pY"] = 0

        if "D" in x0:
            if "px0" in bc.keys():
                px0 = float(bc["px0"])
                bc["rhox0"] = EquationOfState(material).isoT_density(px0)
            else:
                bc["rhox0"] = material["rho0"]

        if "D" in x1:
            if "px1" in bc.keys():
                px1 = float(bc["px1"])
                bc["rhox1"] = EquationOfState(material).isoT_density(px1)
            else:
                bc["rhox1"] = material["rho0"]

        if "D" in y0:
            if "py0" in bc.keys():
                py0 = float(bc["py0"])
                bc["rhoy0"] = EquationOfState(material).isoT_density(py0)
            else:
                bc["rhoy0"] = material["rho0"]

        if "D" in y1:
            if "py1" in bc.keys():
                py1 = float(bc["py1"])
                bc["rhoy1"] = EquationOfState(material).isoT_density(py1)
            else:
                bc["rhoy1"] = material["rho0"]

        assert np.all((x0 == "P") == (x1 == "P")), "Inconsistent boundary conditions (x)"
        assert np.all((y0 == "P") == (y1 == "P")), "Inconsistent boundary conditions (y)"

        print("Done!")
        return bc

    def check_geo(self, geo):
        print("Checking geometry... ", end="", flush=True)

        if geo["type"] == "journal":
            geo["CR"] = float(geo["CR"])
            geo["eps"] = float(geo["eps"])
        elif geo["type"] == "parabolic":
            geo["hmin"] = float(geo['hmin'])
            geo["hmax"] = float(geo['hmax'])
        elif geo["type"] == "twin_parabolic":
            geo["hmin"] = float(geo['hmin'])
            geo["hmax"] = float(geo['hmax'])
        elif geo["type"] == "inclined":
            geo["h1"] = float(geo['h1'])
            geo["h2"] = float(geo['h2'])
        elif geo["type"] == "inclined_pocket":
            geo["h1"] = float(geo['h1'])
            geo["h2"] = float(geo['h2'])
            geo["hp"] = float(geo['hp'])
            geo["c"] = float(geo['c'])
            geo["l"] = float(geo['l'])
            geo["w"] = float(geo['w'])
        elif geo["type"] == "half_sine" or geo["type"] == "half_sine_squared":
            geo["h0"] = float(geo['h0'])
            geo["amp"] = float(geo['amp'])
            geo["num"] = float(geo['num'])

        print("Done!")
        return geo

    def check_num(self, numerics, disc):
        print("Checking numerics options... ", end="", flush=True)

        if "fluctuating" in numerics.keys():
            numerics["fluctuating"] = int(numerics["fluctuating"])
        else:
            numerics["fluctuating"] = 0

        numerics["stokes"] = int(numerics["stokes"])
        numerics["adaptive"] = int(numerics["adaptive"])
        numerics["C"] = float(numerics["C"])
        numerics["tol"] = float(numerics["tol"])
        numerics["dt"] = float(numerics["dt"])
        numerics["maxT"] = float(numerics["maxT"])
        numerics["integrator"] = str(numerics["integrator"])

        if numerics["integrator"] == "RK3":
            disc["ngxl"] = 1
            disc["ngxr"] = 2
            disc["ngyb"] = 1
            disc["ngyt"] = 2
        else:
            disc["ngxl"] = 1
            disc["ngxr"] = 1
            disc["ngyb"] = 1
            disc["ngyt"] = 1

        print("Done!")
        return numerics

    def check_mat(self, material):
        print("Checking material options... ", end="", flush=True)

        if material["EOS"] == "DH":
            material["rho0"] = float(material["rho0"])
            material["P0"] = float(material["P0"])
            material["C1"] = float(material["C1"])
            material["C2"] = float(material["C2"])
        elif material["EOS"] == "PL":
            material["rho0"] = float(material["rho0"])
            material["P0"] = float(material["P0"])
            material["alpha"] = float(material['alpha'])
        elif material["EOS"] == "vdW":
            material["M"] = float(material['M'])
            material["T"] = float(material['T0'])
            material["a"] = float(material['a'])
            material["b"] = float(material['b'])
        elif material["EOS"] == "Murnaghan":
            material["rho0"] = float(material["rho0"])
            material["P0"] = float(material["P0"])
            material["K"] = float(material['K'])
            material["n"] = float(material['n'])
        elif material["EOS"] == "cubic":
            material["a"] = float(material['a'])
            material["b"] = float(material['b'])
            material["c"] = float(material['c'])
            material["d"] = float(material['d'])
        elif material["EOS"].startswith("Bayada"):
            material["cl"] = float(material["cl"])
            material["cv"] = float(material["cv"])
            material["rhol"] = float(material["rhol"])
            material["rhov"] = float(material["rhov"])
            material["shear"] = float(material["shear"])
            material["shearv"] = float(material["shearv"])
            material["rhov"] = float(material["rhov"])

        material["shear"] = float(material["shear"])
        material["bulk"] = float(material["bulk"])

        if "Pcav" in material.keys():
            material["Pcav"] = float(material["Pcav"])

        if "piezo" in material.keys():
            if material["piezo"] == "Barus":
                material["aB"] = float(material["aB"])
            elif material["piezo"] == "Vogel":
                material["rho0"] = float(material['rho0'])
                material["g"] = float(material["g"])
                material["mu_inf"] = float(material["mu_inf"])
                material["phi_inf"] = float(material["phi_inf"])
                material["BF"] = float(material["BF"])

        if "thinning" in material.keys():
            if material["thinning"] == "Eyring":
                material["tau0"] = float(material["tau0"])
            elif material["thinning"] == "Carreau":
                material["G"] = float(material["G"])
                material["a"] = float(material["a"])
                material["N"] = float(material["N"])

        print("Done!", flush=True)
        return material
