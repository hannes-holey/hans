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
import time
import netCDF4
import numpy as np

from hans.material import Material


class DatasetSelector:

    def __init__(self, path, mode="select", fname=[]):

        self.ds = self.get_files(path, mode=mode, fname=fname)

    def get_files(self, path, prefix="", mode="select", fname=[]):
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

        assert mode in ["single", "select", "all", "name"], f"mode must be 'single', select, 'all' or 'name'"

        fileList = []

        for root, dirs, files in os.walk(path, topdown=False):
            for name in files:
                if name.startswith(prefix) and name.endswith(".nc"):
                    fileList.append(os.path.join(root, name))

        fileList = sorted(fileList)

        if mode != "name":
            print("Available files:")
            for i, file in enumerate(fileList):
                date = time.strftime('%d/%m/%Y %H:%M', time.localtime(os.path.getmtime(file)))
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

    def get_centerline(self, key=None, index=-1, dir='x'):

        out = {}

        keys = ["rho", "p", "jx", "jy"]

        for filename, data in self.ds.items():

            Lx = float(data.disc_Lx)
            Ly = float(data.disc_Ly)
            Nx = int(data.disc_Nx)
            Ny = int(data.disc_Ny)

            if dir == "x":
                xdata = (np.arange(Nx) + 0.5) * Lx / Nx
            elif dir == "y":
                xdata = (np.arange(Ny) + 0.5) * Ly / Ny

            out[filename] = {}

            if key is None:
                for k in keys:

                    if k == "p":
                        rho = np.array(data.variables["rho"][index])
                        material = get_material_dict(data)
                        frame = Material(material).eos_pressure(rho)
                    else:
                        frame = np.array(data.variables[k][index])

                    if dir == "x":
                        ydata = frame[:, Ny // 2]
                    elif dir == "y":
                        ydata = frame[Nx // 2, :]

                    out[filename][k] = (xdata, ydata)

            else:

                assert key in keys

                if key == "p":
                    rho = np.array(data.variables["rho"][index])
                    material = get_material_dict(data)
                    frame = Material(material).eos_pressure(rho)
                else:
                    frame = np.array(data.variables[key][index])
                if dir == "x":
                    ydata = frame[:, Ny // 2]
                elif dir == "y":
                    ydata = frame[Nx // 2, :]

                out[filename][key] = (xdata, ydata)

        return out

    def get_centerlines(self, key=None, freq=1, dir='x'):

        out = {}
        keys = ["rho", "p", "jx", "jy"]

        for filename, data in self.ds.items():

            Lx = float(data.disc_Lx)
            Ly = float(data.disc_Ly)
            Nx = int(data.disc_Nx)
            Ny = int(data.disc_Ny)

            if dir == "x":
                xdata = (np.arange(Nx) + 0.5) * Lx / Nx
            elif dir == "y":
                xdata = (np.arange(Ny) + 0.5) * Ly / Ny

            time = np.array(data.variables["time"][::freq])

            out[filename] = {}

            if key is None:

                for k in keys:
                    out[filename][k] = {}

                    if k == "p":
                        rho = np.array(data.variables["rho"][::freq])
                        material = get_material_dict(data)
                        frames = Material(material).eos_pressure(rho)
                    else:
                        frames = np.array(data.variables[k][::freq])

                    for i, t in enumerate(time):
                        if dir == "x":
                            ydata = frames[i, :, Ny // 2]
                        elif dir == "y":
                            ydata = frames[i, Nx // 2, :]

                        out[filename][k][t] = (xdata, ydata)

            else:
                assert key in keys
                out[filename][key] = {}

                if key == "p":
                    rho = np.array(data.variables["rho"][::freq])
                    material = get_material_dict(data)
                    frames = Material(material).eos_pressure(rho)
                else:
                    frames = np.array(data.variables[key][::freq])

                for i, t in enumerate(time):
                    if dir == "x":
                        ydata = frames[i, :, Ny // 2]
                    elif dir == "y":
                        ydata = frames[i, Nx // 2, :]

                    out[filename][key][t] = (xdata, ydata)

        return out

    def get_scalar(self, key=None, freq=1):

        out = {}

        keys = ["mass", "eps", "vSound", "vmax", "dt", "ekin"]

        for filename, data in self.ds.items():
            out[filename] = {}
            if key is None:
                for k in keys:
                    time = np.array(data.variables['time'])[::freq]
                    try:
                        ydata = np.array(data.variables[k])[::freq]
                    except KeyError:
                        print(f"Scalar variable {k} not in {filename}.")
                        pass
                    else:
                        out[filename][k] = (time, ydata)

            else:
                assert key in keys
                time = np.array(data.variables['time'])[::freq]
                ydata = np.array(data.variables[key])[::freq]

                out[filename][key] = (time, ydata)

        return out

    def get_field(self, key=None, index=-1):

        out = {}

        keys = ["rho", "p", "jx", "jy"]

        for filename, data in self.ds.items():

            out[filename] = {}

            if key is None:
                for k in keys:
                    if k == "p":
                        rho = np.array(data.variables["rho"][index])
                        material = get_material_dict(data)
                        frame = Material(material).eos_pressure(rho)
                    else:
                        frame = np.array(data.variables[k][index])
                    out[filename][k] = frame

            else:
                assert key in keys
                if key == "p":
                    rho = np.array(data.variables["rho"][index])
                    material = get_material_dict(data)
                    frame = Material(material).eos_pressure(rho)
                else:
                    frame = np.array(data.variables[key][index])

                out[filename][key] = frame

        return out

    def get_fields(self, key=None, freq=1):

        out = {}

        keys = ["rho", "p", "jx", "jy"]

        for filename, data in self.ds.items():

            time = np.array(data.variables["time"][::freq])

            out[filename] = {}

            if key is None:
                for k in keys:
                    out[filename][k] = {}
                    if key == "p":
                        rho = np.array(data.variables["rho"][::freq])
                        material = get_material_dict(data)
                        frames = Material(material).eos_pressure(rho)
                    else:
                        frames = np.array(data.variables[k][::freq])

                    for i, t in enumerate(time):
                        out[filename][k][t] = frames[i]

            else:
                assert key in keys

                out[filename][key] = {}

                if key == "p":
                    rho = np.array(data.variables["rho"][::freq])
                    material = get_material_dict(data)
                    frames = Material(material).eos_pressure(rho)
                else:
                    frames = np.array(data.variables[key][::freq])

                for i, t in enumerate(time):
                    out[filename][key][t] = frames[i]

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


def get_material_dict(data):
    return {k.split("_")[-1]: v for k, v in dict(data.__dict__).items() if k.startswith("material")}
