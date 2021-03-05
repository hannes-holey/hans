import os
import time
import netCDF4
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.ticker as tk
from mpl_toolkits.axes_grid1 import make_axes_locatable

from pylub.eos import EquationOfState


class Plot:

    def __init__(self, path, mode="select", fname=[]):

        self.ds = self.select_nc_files(path, mode=mode, fname=fname)

        self.ylabels = {"rho": r"Density $\rho$",
                        "p": r"Pressure $p$",
                        "jx": r"Momentum density $j_x$",
                        "jy": r"Momentum denisty $j_y$"}

    def select_nc_files(self, path, prefix="", mode="select", fname=[]):
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

    def plot_cut(self, choice="all", dir='x', figsize=(6.4, 4.8), xscale=1., yscale=1.):
        if choice == "all":
            fig, ax = plt.subplots(2, 2, sharex=True, figsize=figsize, tight_layout=True)
        else:
            fig, ax = plt.subplots(1, figsize=figsize, tight_layout=True)

        for filename, data in self.ds.items():

            # reconstruct input dicts
            material = {k.split("_")[-1]: v for k, v in dict(data.__dict__).items() if k.startswith("material")}

            Lx = float(data.disc_Lx)
            Ly = float(data.disc_Ly)
            Nx = int(data.disc_Nx)
            Ny = int(data.disc_Ny)

            rho = np.array(data.variables["rho"])[-1]
            p = EquationOfState(material).isoT_pressure(rho)
            jx = np.array(data.variables["jx"])[-1]
            jy = np.array(data.variables["jy"])[-1]

            unknowns = {"rho": rho, "p": p, "jx": jx, "jy": jy}

            if dir == "x":
                x = (np.arange(Nx) + 0.5) * Lx / Nx
            elif dir == "y":
                x = (np.arange(Ny) + 0.5) * Ly / Ny

            if choice == "all":
                for count, (key, a) in enumerate(zip(unknowns.keys(), ax.flat)):
                    if dir == "x":
                        var = unknowns[key][:, Ny // 2]
                    elif dir == "y":
                        var = unknowns[key][Nx // 2, :]
                    a.plot(x * xscale, var * yscale)
                    a.set_ylabel(self.ylabels[key])
                    if count > 1:
                        a.set_xlabel(rf"Distance ${dir}$")
            else:
                if dir == "x":
                    var = unknowns[choice][:, Ny // 2]
                elif dir == "y":
                    var = unknowns[choice][Nx // 2, :]
                ax.plot(x * xscale, var * yscale)
                ax.set_ylabel(self.ylabels[choice])
                ax.set_xlabel(rf"Distance ${dir}$")

        return fig, ax

    def plot_2D(self, choice="all", figsize=(6.4, 4.8), xyscale=1., zscale=1., contour_levels=[], **kwargs):

        if "aspect" not in kwargs:
            kwargs["aspect"] = "equal"
        if "cmap" not in kwargs:
            kwargs["cmap"] = "viridis"
        if "interpolation" not in kwargs:
            kwargs["interpolation"] = "none"
        if "position" not in kwargs:
            kwargs["position"] = "right"
        if "orientation" not in kwargs:
            kwargs["orientation"] = "vertical"
        if "ticklocation" not in kwargs:
            kwargs["ticklocation"] = "right"
        if "color" not in kwargs:
            kwargs["color"] = "red"
        if "linewidth" not in kwargs:
            kwargs["linewidth"] = 1.

        if choice == "all":
            fig, ax = plt.subplots(2, 2, figsize=figsize, sharex=True, tight_layout=True)
        else:
            fig, ax = plt.subplots(1, figsize=figsize, tight_layout=True)

        for filename, data in self.ds.items():

            # reconstruct input dicts
            material = {k.split("_")[-1]: v for k, v in dict(data.__dict__).items() if k.startswith("material")}

            Nx = int(data.disc_Nx)
            Ny = int(data.disc_Ny)

            try:
                Lx = float(data.disc_Lx)
            except AttributeError:
                dx = float(data.disc_dx)
                Lx = dx * Nx

            try:
                Ly = float(data.disc_Ly)
            except AttributeError:
                dy = float(data.disc_dy)
                Ly = dy * Ny

            rho = np.array(data.variables["rho"])[-1]
            p = EquationOfState(material).isoT_pressure(rho)
            jx = np.array(data.variables["jx"])[-1]
            jy = np.array(data.variables["jy"])[-1]

            unknowns = {"rho": rho, "p": p, "jx": jx, "jy": jy}

            out = {}

            if choice == "all":
                for count, (key, a) in enumerate(zip(unknowns.keys(), ax.flat)):
                    im = a.imshow(unknowns[key].T * zscale, extent=(0, Lx*xyscale, 0, Ly*xyscale),
                                  interpolation=kwargs["interpolation"], aspect=kwargs["aspect"], cmap=kwargs['cmap'])

                    divider = make_axes_locatable(a)
                    cax = divider.append_axes(kwargs["position"], size="5%", pad=0.1)

                    fmt = tk.ScalarFormatter(useMathText=True)
                    fmt.set_powerlimits((0, 0))

                    cbar = plt.colorbar(im, cax=cax, format=fmt, orientation=kwargs["orientation"])
                    cbar.set_label(self.ylabels[key])

                    # Adjust ticks
                    a.set_xlabel(r'$L_x$')
                    a.set_ylabel(r'$L_y$')
            else:
                im = ax.imshow(unknowns[choice].T * zscale, extent=(0, Lx*xyscale, 0, Ly*xyscale),
                               interpolation=kwargs["interpolation"], aspect=kwargs["aspect"], cmap=kwargs['cmap'])

                if len(contour_levels) > 0:
                    CS = ax.contour(unknowns[choice].T * zscale, contour_levels, colors=kwargs["color"],
                                    extent=(0, Lx*xyscale, 0, Ly*xyscale), linewidths=kwargs["linewidth"])

                    out["contour"] = CS

                divider = make_axes_locatable(ax)
                cax = divider.append_axes(kwargs["position"], size="5%", pad=0.1)

                fmt = tk.ScalarFormatter(useMathText=True)
                fmt.set_powerlimits((0, 0))

                cbar = plt.colorbar(im, cax=cax, format=fmt, orientation=kwargs["orientation"], ticklocation=kwargs["ticklocation"])
                cbar.set_label(self.ylabels[choice])

                # Adjust ticks
                ax.set_xlabel(r'$L_x$')
                ax.set_ylabel(r'$L_y$')

            out["fig"] = fig
            out["ax"] = ax
            out["cbar"] = cbar

        return out

    def plot_cut_evolution(self, choice="all", dir="x", freq=1, figsize=(6.4, 4.8), xscale=1., yscale=1., tscale=1., colormap=True):
        for filename, data in self.ds.items():

            # reconstruct input dicts
            material = {k.split("_")[-1]: v for k, v in dict(data.__dict__).items() if k.startswith("material")}

            Lx = float(data.disc_Lx)
            Ly = float(data.disc_Ly)
            Nx = int(data.disc_Nx)
            Ny = int(data.disc_Ny)

            time = np.array(data.variables["time"])
            maxT = time[-1]

            rho = np.array(data.variables["rho"])
            p = EquationOfState(material).isoT_pressure(rho)
            jx = np.array(data.variables["jx"])
            jy = np.array(data.variables["jy"])

            if dir == "x":
                x = (np.arange(Nx) + 0.5) * Lx / Nx
            elif dir == "y":
                x = (np.arange(Ny) + 0.5) * Ly / Ny

            unknowns = {"rho": rho, "p": p, "jx": jx, "jy": jy}

            if colormap:
                cmap = plt.cm.coolwarm
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=maxT*tscale))

            if choice == "all":
                fig, ax = plt.subplots(2, 2, sharex=True, figsize=figsize)
                for count, (key, a) in enumerate(zip(unknowns.keys(), ax.flat)):
                    for it, t in enumerate(time[::freq]):
                        if dir == "x":
                            var = unknowns[key][it * freq, :, Ny // 2]
                        elif dir == "y":
                            var = unknowns[key][it * freq, Nx // 2, :]
                        if colormap:
                            a.plot(x * xscale, var * yscale, '-', color=cmap(t / maxT))
                        else:
                            a.plot(x * xscale, var * yscale, '-', label=f"{t*tscale}")
                        a.set_ylabel(self.ylabels[key])
                        if count > 1:
                            a.set_xlabel(rf"Distance ${dir}$")
                if colormap:
                    cbar = fig.colorbar(sm, ax=ax.ravel().tolist(), label='time $t$', extend='max')
            else:
                fig, ax = plt.subplots(1, figsize=figsize)
                for it, t in enumerate(time[::freq]):
                    if dir == "x":
                        var = unknowns[choice][it * freq, :, Ny // 2]
                    elif dir == "y":
                        var = unknowns[choice][it * freq, Nx // 2, :]
                    if colormap:
                        ax.plot(x * xscale, var * yscale, '-', color=cmap(t / maxT))
                    else:
                        ax.plot(x * xscale, var * yscale, '-', label=f"{t*tscale:.0f}")
                    ax.set_ylabel(self.ylabels[choice])
                    ax.set_xlabel(rf"Distance ${dir}$")

                if colormap:
                    cbar = fig.colorbar(sm, ax=ax, label='time $t$', extend='max')

        if colormap:
            return fig, ax, cbar
        else:
            return fig, ax

    def plot_timeseries(self, attr, figsize=(6.4, 4.8), xscale=1., yscale=1.):

        fig, ax = plt.subplots(1, figsize=figsize)

        for filename, data in self.ds.items():

            time = np.array(data.variables['time'])
            val = np.array(data.variables[attr])

            ylabels = {"mass": r"Mass $m$",
                       "vmax": r"Max. velocity $v_\mathrm{max}$",
                       "vSound": r"Velocity of sound $c$",
                       "dt": r"Time step $\Delta t$",
                       "eps": r"$\Vert\rho_{n+1} -\rho_n \Vert /(\Vert\rho_n\Vert\,CFL)$"}

            ax.plot(time * xscale, val * yscale, '-')
            ax.set_xlabel(r"Time $t$")
            ax.set_ylabel(ylabels[attr])

            if attr == "eps":
                ax.set_yscale("log")

        return fig, ax

    def animate2D(self, choice="rho", aspect="equal"):
        for filename, data in self.ds.items():

            # reconstruct input dicts
            material = {k.split("_")[-1]: v for k, v in dict(data.__dict__).items() if k.startswith("material")}

            rho = np.array(data.variables["rho"])
            p = EquationOfState(material).isoT_pressure(rho)
            jx = np.array(data.variables["jx"])
            jy = np.array(data.variables["jy"])

            unknowns = {"rho": rho, "p": p, "jx": jx, "jy": jy}

            A = unknowns[choice]
            t = np.array(data.variables['time'])
            Nx = int(data.disc_Nx)
            Ny = int(data.disc_Ny)
            Lx = float(data.disc_Lx)
            Ly = float(data.disc_Ly)

            fig, ax = plt.subplots(figsize=(Nx / Ny * 7, 7))

            # Initial plotting
            self.im = ax.imshow(A[0].T, extent=(0, Lx, 0, Ly), interpolation='none', aspect=aspect, cmap='viridis')

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.3)
            plt.colorbar(self.im, cax=cax)
            ax.invert_yaxis()

            # Adjust ticks
            ax.set_xlabel(r'$L_x$ (nm)')
            ax.set_ylabel(r'$L_y$ (nm)')

            # Create animation
            ani = animation.FuncAnimation(fig, self.update_grid, frames=len(A), fargs=(A, t, fig), interval=100, repeat=True)

        return fig, ax, ani

    def update_grid(self, i, A, t, fig):
        """
        Updates the plot in animation

        Parameters
        ----------
        i : int
            iterator
        A : np.ndarray
            array containing field variables at each time step
        t : np.ndarray
            array containing physical time at each time step
        """

        self.im.set_array(A[i].T)
        if i > 0:
            self.im.set_clim(vmin=np.amin(A[:i]), vmax=np.amax(A[:i]))
        fig.suptitle("Time: {:.1f} s".format(t[i]))


def adaptiveLimits(ax):

    def offset(x, y): return 0.05 * (x - y) if (x - y) != 0 else 1.

    for a in ax.flat:

        y_min = np.amin(a.lines[0].get_ydata())
        y_max = np.amax(a.lines[0].get_ydata())

        a.set_ylim(y_min - offset(y_max, y_min), y_max + offset(y_max, y_min))

    return ax