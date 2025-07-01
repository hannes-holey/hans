#
# Copyright 2020, 2025 Hannes Holey
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


import os
import time
import netCDF4
import numpy as np

from hans.material import Material
from hans.geometry import GapHeight


class DatasetSelector:

    def __init__(self, path, mode="select", fname=[], prefix="", silent=False):

        self._ds = self.get_files(path, prefix=prefix, mode=mode, fname=fname, silent=silent)

    def get_files(self, path, prefix="", mode="select", fname=[], silent=False):
        """
        Select netCDF data files for plotting.

        Parameters
        ----------
        path : str
            relative path below which is searched for files
        prefix : str
            filter files that start with prefix, default=""
        mode : str
            can be one of the following, default="select"
            - select: manually select files
            - single: manually select a single file
            - all: select all files found below path with prefix and suffix
            - name: select files by name through 'fname' keyword argument
        fname : list
            list of file names for mode=name, relative to 'path', defaul="[]"
        Returns
        ----------
        out : dict
            dictionary where keys are filenames and values are corresponding datasets.
            Datasets currently only implemented for suffices "nc" (netCDF4.Dataset) and "dat" (numpy.ndarray).
            Else, values are None.
        """

        assert mode in ["single", "select", "all", "name"], "mode must be 'single', select, 'all' or 'name'"

        silent = silent if mode == 'all' else False

        fileList = []

        for root, dirs, files in os.walk(path, topdown=False):
            for name in files:
                if name.startswith(prefix) and name.endswith(".nc"):
                    fileList.append(os.path.join(root, name))

        fileList = sorted(fileList)

        if mode != "name":
            if not silent:
                print("Available files:")
            for i, file in enumerate(fileList):
                date = time.strftime('%d/%m/%Y %H:%M', time.localtime(os.path.getmtime(file)))
                if not silent:
                    print(f"{i:3d}: {file:<50} {date}")

        if mode == "single":
            mask = [int(input("Enter file key: "))]
        elif mode == "all":
            mask = list(range(len(fileList)))
        elif mode == "select":
            inp = input("Enter file keys (space separated or range [start]-[end] or combination of both): ")

            mask = [int(i) for i in inp.split() if len(i.split("-")) < 2]
            mask_range = [i for i in inp.split() if len(i.split("-")) == 2]

            for j in mask_range:
                mask += list(range(int(j.split("-")[0]), int(j.split("-")[1]) + 1))
        elif mode == "name":
            fileList = [os.path.join(path, f) for f in fname]
            mask = np.arange(len(fileList))

        out = {f: netCDF4.Dataset(f) for i, f in enumerate(fileList) if i in mask}

        return out

    def get_filenames(self):
        return list(self._ds.keys())

    def get_NetCDF_Datasets(self):
        return list(self._ds.values())

    def get_metadata(self):
        category_prefix = ["options", "disc", "bc", "geometry", "roughness", "ic", "numerics", "material", "surface", "gp"]
        out = []
        for f in self._ds.values():
            metadict = {}
            for cat in category_prefix:
                catdict = self._generate_input_dict(f, cat)
                if catdict is not None:
                    metadict[cat] = catdict

            metadict["info"] = {}
            for info in ["restarts", "version"]:
                metadict["info"].update({info: f.getncattr(info)})

            out.append(metadict)

        return out

    def _generate_input_dict(self, data, category):
        out = {(k.split("_")[-1]): (v if str(v) != "None" else None) for k, v in dict(data.__dict__).items() if k.startswith(category)}

        if len(out) > 0:
            return out
        else:
            return None

    def _get_grid(self, data):

        Nx = int(data.disc_Nx)
        Ny = int(data.disc_Ny)
        Lx = float(data.disc_Lx)
        Ly = float(data.disc_Ly)

        return Nx, Ny, Lx, Ly

    def get_centerline(self, key=None, index=-1, dir='x'):

        out = []
        keys = ["rho", "p", "jx", "jy"]

        for data in self._ds.values():

            Nx, Ny, Lx, Ly = self._get_grid(data)

            if dir == "x":
                xdata = (np.arange(Nx) + 0.5) * Lx / Nx
            elif dir == "y":
                xdata = (np.arange(Ny) + 0.5) * Ly / Ny

            if key is None:
                ydata = {}
                for k in keys:

                    if k == "p":
                        rho = np.array(data.variables["rho"][index])
                        material = self._generate_input_dict(data, "material")
                        frame = Material(material).eos_pressure(rho)
                    else:
                        frame = np.array(data.variables[k][index])

                    if dir == "x":
                        ydata[k] = frame[:, Ny // 2]
                    elif dir == "y":
                        ydata[k] = frame[Nx // 2, :]

            else:
                assert key in keys

                if key == "p":
                    rho = np.array(data.variables["rho"][index])
                    material = self._generate_input_dict(data, "material")
                    frame = Material(material).eos_pressure(rho)
                else:
                    frame = np.array(data.variables[key][index])
                if dir == "x":
                    ydata = frame[:, Ny // 2]
                elif dir == "y":
                    ydata = frame[Nx // 2, :]

            out.append((xdata, ydata))

        return out

    def get_centerlines(self, key=None, freq=1, dir='x'):
        """returns variable values of specified key in either x- or y-direction. If key is pressure, it is calculated from rho using EOS

        Parameters
        ----------
        key : str
            specify variable to read
        freq : int, defaults to 1
            timestep sample step
        dir : char
            spacial sampling direction, defaults to ‘x’

        Returns
        ----------
        out : array of tuple of np.array(s)
            time : timestamps
            xdata : x coordinates
            ydata : variable values

        Usage
        ----------
        time, x, ydata = files.get_centerlines("p")[0] → np.array(s)
        """

        out = []
        keys = ["rho", "p", "jx", "jy"]

        for data in self._ds.values():

            Nx, Ny, Lx, Ly = self._get_grid(data)

            if dir == "x":
                xdata = (np.arange(Nx) + 0.5) * Lx / Nx
            elif dir == "y":
                xdata = (np.arange(Ny) + 0.5) * Ly / Ny

            time = np.array(data.variables["time"][::freq])

            if key is None:
                ydata = {}
                for k in keys:

                    if k == "p":
                        rho = np.array(data.variables["rho"][::freq])
                        material = self._generate_input_dict(data, "material")
                        frames = Material(material).eos_pressure(rho)
                    else:
                        frames = np.array(data.variables[k][::freq])

                    if dir == "x":
                        ydata[k] = frames[:, :, Ny // 2]
                    elif dir == "y":
                        ydata[k] = frames[:, Nx // 2, :]

            else:
                assert key in keys

                if key == "p":
                    rho = np.array(data.variables["rho"][::freq])
                    material = self._generate_input_dict(data, "material")
                    frames = Material(material).eos_pressure(rho)
                else:
                    frames = np.array(data.variables[key][::freq])

                if dir == "x":
                    ydata = frames[:, :, Ny // 2]
                elif dir == "y":
                    ydata = frames[:, Nx // 2, :]

            out.append((time, xdata, ydata))

        return out

    def get_scalar(self, key=None, freq=1):

        out = []

        keys = ["mass", "eps", "vSound", "vmax", "dt", "ekin"]

        for data in self._ds.values():

            time = np.array(data.variables['time'])[::freq]

            if key is None:
                ydata = {}
                for k in keys:
                    try:
                        ydata[k] = np.array(data.variables[k])[::freq]
                    except KeyError:
                        print(f"Scalar variable {k} not existing.")
                        pass

            else:
                assert key in keys
                time = np.array(data.variables['time'])[::freq]
                ydata = np.array(data.variables[key])[::freq]

            out.append((time, ydata))

        return out

    def get_field(self, key=None, index=-1):

        out = []

        keys = ["rho", "p", "jx", "jy"]

        for data in self._ds.values():

            if key is None:
                zdata = {}
                for k in keys:
                    if k == "p":
                        rho = np.array(data.variables["rho"][index])
                        material = self._generate_input_dict(data, "material")
                        zdata[k] = Material(material).eos_pressure(rho)
                    else:
                        zdata[k] = np.array(data.variables[k][index])

            else:
                assert key in keys
                if key == "p":
                    rho = np.array(data.variables["rho"][index])
                    material = self._generate_input_dict(data, "material")
                    zdata = Material(material).eos_pressure(rho)
                else:
                    zdata = np.array(data.variables[key][index])

            out.append(zdata)

        return out

    def get_height(self):

        out = []

        for data in self._ds.values():
            disc = self._generate_input_dict(data, "disc")
            disc['pX'] = True
            disc['pY'] = True
            disc['nghost'] = 1

            geometry = self._generate_input_dict(data, "geometry")
            roughness = self._generate_input_dict(data, "roughness")
            frame = GapHeight(disc, geometry, roughness)

            out.append(frame.inner[0])

        return out

    def get_centerline_height(self, dir='x'):

        out = []

        for data in self._ds.values():

            Nx, Ny, Lx, Ly = self._get_grid(data)

            if dir == "x":
                xdata = (np.arange(Nx) + 0.5) * Lx / Nx
            elif dir == "y":
                xdata = (np.arange(Ny) + 0.5) * Ly / Ny

            disc = self._generate_input_dict(data, "disc")
            disc['pX'] = True
            disc['pY'] = True
            disc['nghost'] = 1

            geometry = self._generate_input_dict(data, "geometry")
            roughness = self._generate_input_dict(data, "roughness")
            frame = GapHeight(disc, geometry, roughness)

            if dir == "x":
                ydata = frame.centerline_x[0]
            elif dir == "y":
                ydata = frame.centerline_y[0]

            out.append((xdata, ydata))

        return out

    def get_fields(self, key=None, freq=1):

        out = []

        keys = ["rho", "p", "jx", "jy"]

        for data in self._ds.values():

            time = np.array(data.variables["time"][::freq])

            if key is None:
                zdata = {}
                for k in keys:
                    if k == "p":
                        rho = np.array(data.variables["rho"][::freq])
                        material = self._generate_input_dict(data, "material")
                        zdata[k] = Material(material).eos_pressure(rho)
                    else:
                        zdata[k] = np.array(data.variables[k][::freq])
            else:
                assert key in keys

                if key == "p":
                    rho = np.array(data.variables["rho"][::freq])
                    material = self._generate_input_dict(data, "material")
                    zdata = Material(material).eos_pressure(rho)
                else:
                    zdata = np.array(data.variables[key][::freq])

            out.append((time, zdata))

        return out


def adaptiveLimits(ax):

    def offset(x, y): return 0.05 * (x - y) if (x - y) != 0 else 1.

    try:
        axes = ax.flat
    except AttributeError:
        axes = [ax]

    for a in axes:

        y_min = np.amin(a.lines[0].get_ydata())
        y_max = np.amax(a.lines[0].get_ydata())

        a.set_ylim(y_min - offset(y_max, y_min), y_max + offset(y_max, y_min))

    return ax
