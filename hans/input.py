#
# Copyright 2019, 2023 Hannes Holey
#           2019 Andrea Codrignani
#
# ### MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#


import yaml
import numpy as np

from hans.problem import Problem
from hans.tools import abort
from hans.material import Material


class Input:

    def __init__(self, inputFile, restartFile=None):
        """
        Reads the yaml input file that defines a problem.

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
        """
        Parses the yaml input file, performs sanity checks,
        and returns an instance of the Problem class.

        Returns
        -------
        Problem
            Problem object containing all information about the current job.

        """

        with open(self.inputFile, 'r') as ymlfile:

            inp = yaml.full_load(ymlfile)

            print(f"Reading input from {self.inputFile}...")

            # Mandatory inputs
            options = self.sanitize_options(inp['options'])
            disc = self.sanitize_disc(inp['disc'])
            numerics, disc = self.sanitize_numerics(inp['numerics'], disc)
            geometry = self.sanitize_geometry(inp['geometry'], disc)
            material = self.sanitize_material(inp['material'])
            bc, disc = self.sanitize_BC(inp['BC'], disc, material)

            # Optional inputs
            if "surface" in inp.keys():
                surface = self.sanitize_surface(inp["surface"])
            else:
                surface = None

            if "roughness" in inp.keys():
                roughness = self.sanitize_roughness(inp["roughness"])
            else:
                roughness = None

            if "gp" in inp.keys():
                # TODO: sanitize GP input
                gp = self.sanitize_gp(inp["gp"])
            else:
                gp = None

            if self.restartFile is not None:
                ic = {"type": "restart", "file": self.restartFile}
            elif "IC" in inp.keys():
                ic = self.sanitize_IC(inp["IC"], disc)
            else:
                ic = None

            print("Sanity checks completed. Start simulation!")
            print(60 * "-")

        thisProblem = Problem(options,
                              disc,
                              bc,
                              geometry,
                              numerics,
                              material,
                              surface,
                              ic,
                              roughness,
                              gp)

        return thisProblem

    def sanitize_disc(self, disc):
        """Sanity check for discretization settings

        Parameters
        ----------
        disc : dict
            Discretization input dict

        Returns
        -------
        dict
            Sanitized discretization input dict
        """

        print("Checking discretization... ")

        try:
            disc["Nx"] = int(disc['Nx'])
            assert disc["Nx"] > 0
        except KeyError:
            print("***Number of grid cells Nx not specified. Abort.")
            abort()
        except AssertionError:
            print("***Number of grid cells Nx must be larger than zero. Abort")
            abort()

        try:
            disc["Ny"] = int(disc['Ny'])
            assert disc["Ny"] > 0
        except KeyError:
            print("***Number of grid cells 'Ny' not specified. Abort.")
            abort()
        except AssertionError:
            print("***Number of grid cells 'Ny' must be larger than zero. Abort")
            abort()

        try:
            disc["Lx"] = float(disc["Lx"])
        except KeyError:
            try:
                disc["dx"] = float(disc["dx"])
            except KeyError:
                print("At least two of 'Nx' 'Lx', 'dx' must be given. Abort.")
                abort()
            else:
                disc["Lx"] = disc["dx"] * disc["Nx"]
        else:
            disc["dx"] = disc["Lx"] / disc["Nx"]

        try:
            disc["Ly"] = float(disc["Ly"])
        except KeyError:
            try:
                disc["dy"] = float(disc["dy"])
            except KeyError:
                print("At least two of 'Ny' 'Ly', 'dy' must be given. Abort.")
                abort()
            else:
                disc["Ly"] = disc["dy"] * disc["Ny"]
        else:
            disc["dy"] = disc["Ly"] / disc["Ny"]

        return disc

    def sanitize_options(self, options):
        """Sanitize output settings


        Parameters
        ----------
        options : dict
            Output settings

        Returns
        -------
        dict
            Sanitized output settings
        """
        print("Checking I/O options... ")

        try:
            writeInterval = int(options["writeInterval"])
            assert writeInterval > 0
        except KeyError:
            print("***Output interval not given, fallback to 1000")
            options["writeInterval"] = 1000
        except AssertionError:
            try:
                assert writeInterval != 0
            except AssertionError:
                print("***Output interval is zero. fallback to 1000")
                options["writeInterval"] = 1000
            else:
                print("***Output interval is negative. Converting to positive value.")
                writeInterval *= -1
                options["writeInterval"] = writeInterval

        return options

    def sanitize_geometry(self, geometry, disc):
        """Sanitize geometry

        Parameters
        ----------
        geometry : dict
            Geometry settings
        disc : dict
            Discretization settings

        Returns
        -------
        dict
            Sanitized geometry settings
        """

        print("Checking geometry... ")

        if geometry["type"] in ["journal", "journal_x", "journal_y"]:
            geometry["CR"] = float(geometry["CR"])
            geometry["eps"] = float(geometry["eps"])
        elif geometry["type"] == "parabolic":
            geometry["hmin"] = float(geometry['hmin'])
            geometry["hmax"] = float(geometry['hmax'])
        elif geometry["type"] == "twin_parabolic":
            geometry["hmin"] = float(geometry['hmin'])
            geometry["hmax"] = float(geometry['hmax'])
        elif geometry["type"] in ["inclined", "inclined_x", "inclined_y"]:
            geometry["h1"] = float(geometry['h1'])
            geometry["h2"] = float(geometry['h2'])
        elif geometry["type"] == "inclined_pocket":
            geometry["h1"] = float(geometry['h1'])
            geometry["h2"] = float(geometry['h2'])
            geometry["hp"] = float(geometry['hp'])
            geometry["c"] = float(geometry['c'])
            geometry["l"] = float(geometry['l'])
            geometry["w"] = float(geometry['w'])

            try:
                geometry["t"] = float(geometry["t"])
            except KeyError:
                geometry["t"] = 0.

            try:
                assert 2 * geometry["t"] + geometry["l"] + geometry["c"] <= disc["Lx"]
                assert geometry["w"] <= disc["Ly"]
            except AssertionError:
                print("Size of the pocket is larger than the domain. Abort.")
                abort()

        elif geometry["type"] in ["half_sine", "half_sine_squared"]:
            geometry["h0"] = float(geometry['h0'])
            geometry["amp"] = float(geometry['amp'])
            geometry["num"] = float(geometry['num'])

        else:
            print(f"'{geometry['type']}' geometry not implemented")
            abort()

        return geometry

    def sanitize_numerics(self, numerics, disc):
        """Sanitize numerics settings

        Parameters
        ----------
        numerics : dict
            Numerics input settings
        disc : dict
            Discretization settings

        Returns
        ------- 
        dict
            Sanitized numerics settings
        dict
            Modified discretization settings
        """

        print("Checking numerics options... ")

        try:
            numerics["integrator"] = numerics["integrator"]
            assert numerics["integrator"] in ["MC", "MC_bf", "MC_fb", "MC_alt", "LW", "RK3"]
        except KeyError:
            print("***Integrator not specified. Use default (MacCormack).")
            numerics["integrator"] = "MC"
        except AssertionError:
            print(f'***Unknown integrator \'{numerics["integrator"]}\'. Abort.')
            abort()

        if numerics["integrator"].startswith("MC"):
            try:
                numerics["fluxLim"] = float(numerics["fluxLim"])
            except KeyError:
                pass

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
            numerics["dt"] = float(numerics["dt"])
        except KeyError:
            print("***Timestep not given. Use default (1e-10).")
            numerics["dt"] = 1e-10

        stopping_criteria = 0

        try:
            numerics["tol"] = float(numerics["tol"])
            stopping_criteria += 1
        except KeyError:
            pass

        try:
            numerics["maxT"] = float(numerics["maxT"])
            stopping_criteria += 1
        except KeyError:
            pass

        try:
            numerics["maxIt"] = int(numerics["maxIt"])
            stopping_criteria += 1
        except KeyError:
            pass

        if stopping_criteria == 0:
            print("***No stopping criterion given. Abort.")
            abort()

        if numerics["integrator"] == "RK3":
            disc["nghost"] = 2
        else:
            disc["nghost"] = 1

        return numerics, disc

    def sanitize_material(self, material):
        """Sanitize material settings

        Parameters
        ----------
        material : dict
            Material (fluid) properties

        Returns
        -------
        dict
            Sanitized material properties
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
        elif material["EOS"] == "Tait":
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
        elif material['EOS'] == 'BWR':
            material['T'] = float(material['T'])
            # Parameters: Johnson et al., Mol. Phys. 78 (1993)
            params = """
                    0.8623085097507421
                    2.976218765822098
                    -8.402230115796038
                    0.1054136629203555
                    -0.8564583828174598
                    1.582759470107601
                    0.7639421948305453
                    1.753173414312048
                    2.798291772190376e+03
                    -4.8394220260857657e-02
                    0.9963265197721935
                    -3.698000291272493e+01
                    2.084012299434647e+01
                    8.305402124717285e+01
                    -9.574799715203068e+02
                    -1.477746229234994e+02
                    6.398607852471505e+01
                    1.603993673294834e+01
                    6.805916615864377e+01
                    -2.791293578795945e+03
                    -6.245128304568454
                    -8.116836104958410e+03
                    1.488735559561229e+01
                    -1.059346754655084e+04
                    -1.131607632802822e+02
                    -8.867771540418822e+03
                    -3.986982844450543e+01
                    -4.689270299917261e+03
                    2.593535277438717e+02
                    -2.694523589434903e+03
                    -7.218487631550215e+02
                    1.721802063863269e+02
                    """

            x = [float(val) for val in params.split()]
            material['params'] = x

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
                material["relax"] = float(material["relax"])
                material["a"] = float(material["a"])
                material["N"] = float(material["N"])
            elif material["thinning"] == "PL":
                material["shear"] = float(material["shear"])
                material["n"] = float(material["n"])

        if "PLindex" in material.keys():
            material["PLindex"] = float(material["PLindex"])
            if "PLmethod" in material.keys():
                material["PLmethod"] = material["PLmethod"]
            else:
                material["PLmethod"] = "exact"

        return material

    def sanitize_BC(self, bc, disc, material):
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
        dict
            Modified discretization parameters.

        """
        print("Checking boundary conditions... ")

        bc["x0"] = np.array(list(bc["x0"]))
        bc["x1"] = np.array(list(bc["x1"]))
        bc["y0"] = np.array(list(bc["y0"]))
        bc["y1"] = np.array(list(bc["y1"]))

        assert len(bc["x0"]) == 3
        assert len(bc["x1"]) == 3
        assert len(bc["y0"]) == 3
        assert len(bc["y1"]) == 3

        if "P" in bc["x0"] and "P" in bc["x1"]:
            disc["pX"] = 1
        else:
            disc["pX"] = 0

        if "P" in bc["y0"] and "P" in bc["y1"]:
            disc["pY"] = 1
        else:
            disc["pY"] = 0

        if "D" in bc["x0"]:
            if "px0" in bc.keys():
                px0 = float(bc["px0"])
                bc["rhox0"] = Material(material).eos_density(px0)
            elif "rhox0" in bc.keys():
                rhox0 = float(bc["rhox0"])
                bc["rhox0"] = rhox0
            else:
                bc["rhox0"] = material["rho0"]

        if "D" in bc["x1"]:
            if "px1" in bc.keys():
                px1 = float(bc["px1"])
                bc["rhox1"] = Material(material).eos_density(px1)
            elif "rhox1" in bc.keys():
                rhox1 = float(bc["rhox1"])
                bc["rhox1"] = rhox1
            else:
                bc["rhox1"] = material["rho0"]

        if "D" in bc["y0"]:
            if "py0" in bc.keys():
                py0 = float(bc["py0"])
                bc["rhoy0"] = Material(material).eos_density(py0)
            elif "rhoy0" in bc.keys():
                rhoy0 = float(bc["rhoy0"])
                bc["rhoy0"] = rhoy0
            else:
                bc["rhoy0"] = material["rho0"]

        if "D" in bc["y1"]:
            if "py1" in bc.keys():
                py1 = float(bc["py1"])
                bc["rhoy1"] = Material(material).eos_density(py1)
            elif "rhoy1" in bc.keys():
                rhoy1 = float(bc["rhoy1"])
                bc["rhoy1"] = rhoy1
            else:
                bc["rhoy1"] = material["rho0"]

        assert np.all((bc["x0"] == "P") == (bc["x1"] == "P")), "Inconsistent boundary conditions (x)"
        assert np.all((bc["y0"] == "P") == (bc["y1"] == "P")), "Inconsistent boundary conditions (y)"

        return bc, disc

    def sanitize_surface(self, surface):
        """Sanitize surface parameters

        Parameters
        ----------
        surface : dict
            Input surface parameters

        Returns
        -------
        dict
            Sanitized surface parameters
        """
        print("Checking surface parameters... ")

        if "lslip" in surface.keys():
            surface["lslip"] = float(surface["lslip"])
        else:
            surface["lslip"] = 0.

        if surface["type"] in ["stripes", "stripes_x", "stripes_y"]:
            try:
                surface["num"] = int(surface["num"])
            except KeyError:
                surface["num"] = 1
            try:
                surface["sign"] = int(surface["sign"])
            except KeyError:
                surface["sign"] = -1

        return surface

    def check_roughness(self, roughness):
        """Sanitize roughness settings

        Parameters
        ----------
        roughness : dict
            Roughness settings

        Returns
        -------
        dict
            Sanitized roughness settings
        """

        print("Checking roughness parameters... ")

        if "file" not in roughness.keys():
            try:
                roughness["seed"] = int(roughness["seed"])
            except KeyError:
                roughness["seed"] = None
            try:
                roughness["Hurst"] = float(roughness["Hurst"])
            except KeyError:
                "Hurst exponent not given. Use default (0.8)."
                roughness["Hurst"] = 0.8
            try:
                roughness["rolloff"] = float(roughness["rolloff"])
            except KeyError:
                "Rollof not given. Use default (1.0)"
                roughness["rolloff"] = 1.0
            try:
                roughness["rmsHeight"] = float(roughness["rmsHeight"])
            except KeyError:
                roughness["rmsHeight"] = None
            try:
                roughness["rmsSlope"] = float(roughness["rmsSlope"])
            except KeyError:
                roughness["rmsSlope"] = None
            try:
                roughness["shortCutoff"] = float(roughness["shortCutoff"])
            except KeyError:
                roughness["shortCutoff"] = None
            try:
                roughness["longCutoff"] = float(roughness["longCutoff"])
            except KeyError:
                roughness["longCutoff"] = None

            if roughness["rmsHeight"] is None and roughness["rmsSlope"] is None:
                print('Neither rms height nor rms slope is defined! Abort.')
                abort()

        return roughness

    def sanitize_IC(self, ic, disc):
        """Sanitize initial conditions

        Parameters
        ----------
        ic : dict
            Initial conditions settings
        disc : dict
            Discretization settings

        Returns
        -------
        dict
            Sanitized initial conditions setting
        """
        print("Checking initial conditions... ")

        if ic["type"] != "restart":
            if ic["type"] == "perturbation":
                ic["factor"] = float(ic["factor"])
            elif ic["type"] in ["longitudinal_wave", "shear_wave"]:
                ic["amp"] = float(ic["amp"])
                if "nwave" in ic.keys():
                    ic["nwave"] = int(ic["nwave"])
                else:
                    ic["nwave"] = 1
            elif ic["type"] == "random":
                try:
                    ic["stdDens"] = float(ic["stdDens"])
                except KeyError:
                    ic["stdDens"] = 0.0
                try:
                    ic["stdFlux"] = float(ic["stdFlux"])
                except KeyError:
                    ic["stdFlux"] = 0.0
            elif ic["type"] == "gauss1D":
                try:
                    ic["mean"] = float(ic["mean"])
                except KeyError:
                    ic["mean"] = float(disc["Lx"]) / 2
                try:
                    ic["stdev"] = float(ic["stdev"])
                except KeyError:
                    ic["stdev"] = float(disc["Lx"]) / 40
                try:
                    ic["cutoff"] = float(ic["cutoff"])
                except KeyError:
                    ic["cutoff"] = 12 * ic['stdev']

        return ic

    def sanitize_gp(self, gp):
        print("Checking GP parameters... ")

        # lengthscales
        gp['lh'] = float(gp['lh'])
        gp['lrho'] = float(gp['lrho'])
        gp['lj'] = float(gp['lj'])

        # variances
        gp['var'] = float(gp['var'])
        gp['pvar'] = float(gp['pvar'])

        # tolerances (var)
        gp['tol'] = float(gp['tol'])
        gp['ptol'] = float(gp['ptol'])

        try:
            gp['remote'] = int(gp['remote'])
        except KeyError:
            gp['remote'] = 0

        try:
            gp['local'] = gp['local']
        except KeyError:
            gp['local'] = '/tmp/dtool/'

        # remote storage backend
        if gp['remote'] == 1:
            try:
                gp['storage'] = gp['storage']
            except KeyError:
                gp['storage'] = 's3://test-bucket'

        # TODO: may need new category for MD parameters
        try:
            gp['lmp'] = int(gp['lmp'])
        except KeyError:
            gp['lmp'] = 1

        try:
            gp['ncpu'] = int(gp['ncpu'])
        except KeyError:
            gp['ncpu'] = 2

        # TODO: query

        try:
            gp['sns'] = float(gp['sns'])
        except KeyError:
            gp['sns'] = 0.

        try:
            gp['snp'] = float(gp['snp'])
        except KeyError:
            gp['snp'] = 0.

        try:
            gp['fix'] = int(gp['fix'])
        except KeyError:
            gp['fix'] = 1

        try:
            gp['start'] = int(gp['start'])
        except KeyError:
            gp['start'] = 1

        try:
            gp['num_restarts'] = int(gp['num_restarts'])
        except KeyError:
            gp['num_restarts'] = 10

        try:
            gp['verbose'] = int(gp['verbose'])
        except KeyError:
            gp['verbose'] = 1

        try:
            assert gp['tol'] < gp['var']
        except AssertionError:
            print('Uncertainty threshold is larger than initial kernel variance. Setting var = 2 * tol.')
            gp['var'] = 2. * gp['tol']

        try:
            assert gp['ptol'] < gp['pvar']
        except AssertionError:
            print('Uncertainty threshold (pressure) is larger than initial kernel variance. Setting pvar = 2 * ptol.')
            gp['pvar'] = 2. * gp['ptol']

        return gp
