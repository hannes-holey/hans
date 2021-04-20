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


import sys
import yaml
import numpy as np

from pylub.problem import Problem
from pylub.material import Material


class Input:
    """Reads the yaml input file that defines a problem.

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
            filename of the .nc data file (default: None)
        """
        self.inputFile = inputFile
        self.restartFile = restartFile

    def getProblem(self):
        """Parses the yaml input file, performs sanity checks,
        and returns an instance of the Problem class.

        Returns
        -------
        Problem
            Problem Object containing all information about the current job.

        """

        with open(self.inputFile, 'r') as ymlfile:
            inp = yaml.full_load(ymlfile)

            # Perform sanity checks for input parameters
            options = self.check_options(inp['options'])
            disc = self.check_disc(inp['disc'])
            geometry = self.check_geo(inp['geometry'])
            numerics = self.check_num(inp['numerics'])
            material = self.check_mat(inp['material'])
            bc = self.check_bc(inp['BC'], disc, material)
            print("Sanity checks completed. Start simulation!")
            print(60 * "-")

        thisProblem = Problem(options,
                              disc,
                              bc,
                              geometry,
                              numerics,
                              material,
                              self.restartFile)

        return thisProblem

    def check_options(self, options):
        """Sanity check for I/O options input.

        Parameters
        ----------
        options : dict
            I/O options read from yaml file.

        Returns
        -------
        dict
            Contains I/O options.

        """
        print("Checking I/O options... ")

        try:
            writeInterval = int(options["writeInterval"])
            assert writeInterval > 0
        except KeyError:
            print("\nOutput interval not given, fallback to 1000", flush=True)
            options["writeInterval"] = 1000
        except AssertionError:
            try:
                assert writeInterval != 0
            except AssertionError:
                print("\nOutput interval is zero. fallback to 1000", flush=True)
                options["writeInterval"] = 1000
            else:
                print("\nOutput interval is negative. Converting to positive value.", flush=True)
                writeInterval *= -1
                options["writeInterval"] = writeInterval

        return options

    def check_disc(self, disc):
        """Sanity check for discretization input.

        Parameters
        ----------
        disc : dict
            Discretization parameters read from yaml file.

        Returns
        -------
        dict
            Discretization parameters.
        """
        print("Checking discretization... ")

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

        return disc

    def check_geo(self, geo):
        """Sanity check for geometry input.

        Parameters
        ----------
        geo : dict
            Discretization parameters read from yaml input file.

        Returns
        -------
        dict
            Geometry parameters.

        """
        print("Checking geometry... ")

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

        return geo

    def check_num(self, numerics):
        """Sanity check for numerics options.

        Parameters
        ----------
        numerics : dict
            Numerics options read from yaml input file.

        Returns
        -------
        dict
            Numerics options.

        """
        print("Checking numerics options... ")

        try:
            numerics["numFlux"] = numerics["numFlux"]
            assert numerics["numFlux"] in ["MC", "MC_bf", "MC_fb", "MC_alt", "LW", "RK3"]
        except KeyError:
            print("***Integrator not specified. Use default (MacCormack).")
            numerics["numFlux"] = "MC"
        except AssertionError:
            print(f'***Unknown integrator \'{numerics["numFlux"]}\'. Abort.')
            sys.exit(1)

        try:
            numerics["stokes"] = int(numerics["stokes"])
        except KeyError:
            print("***Boolean parameter 'stokes' not given. Use default (True).")
            numerics["stokes"] = 1

        try:
            numerics["adaptive"] = int(numerics["adaptive"])
        except KeyError:
            print("***Boolean parameter 'adaptive' not given. Use default (False).")
            numerics["adaptive"] = 0

        if numerics["adaptive"] == 1:
            try:
                numerics["C"] = float(numerics["C"])
            except KeyError:
                print("***CFL number not given. Use default (0.5).")
                numerics["C"] = 0.5

        try:
            numerics["tol"] = float(numerics["tol"])
        except KeyError:
            print("***Convergence tolerance not given. Use default (1e-9).")
            numerics["tol"] = 1e-9

        try:
            numerics["dt"] = float(numerics["dt"])
        except KeyError:
            print("***Timestep not given. Use default (1e-10).")
            numerics["dt"] = 1e-10

        try:
            numerics["maxT"] = float(numerics["maxT"])
        except KeyError:
            print("***Maximum time not given. Use default (1e-6).")
            numerics["maxT"] = 1e-6

        return numerics

    def check_mat(self, material):
        """Sanity check on material settings.

        Parameters
        ----------
        material : dict
            Material parameters read from yaml input file.

        Returns
        -------
        dict
            Material parameters

        """
        print("Checking material options... ")

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

        return material

    def check_bc(self, bc, disc, material):
        """Sanity check for boundary condition input.

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
                bc["rhox0"] = Material(material).eos_density(px0)
            else:
                bc["rhox0"] = material["rho0"]

        if "D" in x1:
            if "px1" in bc.keys():
                px1 = float(bc["px1"])
                bc["rhox1"] = Material(material).eos_density(px1)
            else:
                bc["rhox1"] = material["rho0"]

        if "D" in y0:
            if "py0" in bc.keys():
                py0 = float(bc["py0"])
                bc["rhoy0"] = Material(material).eos_density(py0)
            else:
                bc["rhoy0"] = material["rho0"]

        if "D" in y1:
            if "py1" in bc.keys():
                py1 = float(bc["py1"])
                bc["rhoy1"] = Material(material).eos_density(py1)
            else:
                bc["rhoy1"] = material["rho0"]

        assert np.all((x0 == "P") == (x1 == "P")), "Inconsistent boundary conditions (x)"
        assert np.all((y0 == "P") == (y1 == "P")), "Inconsistent boundary conditions (y)"

        return bc
