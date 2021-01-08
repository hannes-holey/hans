import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable

from pylub.eos import EquationOfState
from pylub.helper.data_parser import getData


class Plot:

    def __init__(self, path, mode="select"):

        self.ds = getData(path, mode=mode)

        self.ylabels = {"rho": r"Density $\rho$",
                        "p": r"Pressure $p$",
                        "jx": r"Momentum density $j_x$",
                        "jy": r"Momentum denisty $j_y$"}

    def plot_cut(self, choice="all", dir='x'):
        for filename, data in self.ds.items():

            # reconstruct input dicts
            material = {k.split("_")[-1]: v for k, v in dict(data.__dict__).items() if k.startswith("material")}

            Lx = data.disc_Lx
            Ly = data.disc_Ly
            Nx = data.disc_Nx
            Ny = data.disc_Ny

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
                fig, ax = plt.subplots(2, 2, sharex=True, tight_layout=True)
                for count, (key, a) in enumerate(zip(unknowns.keys(), ax.flat)):
                    if dir == "x":
                        var = unknowns[key][:, Ny // 2]
                    elif dir == "y":
                        var = unknowns[key][Nx // 2, :]
                    a.plot(x, var)
                    a.set_ylabel(self.ylabels[key])
                    if count > 1:
                        a.set_xlabel(rf"Distance ${dir}$")
            else:
                fig, ax = plt.subplots(1, tight_layout=True)
                if dir == "x":
                    var = unknowns[choice][:, Ny // 2]
                elif dir == "y":
                    var = unknowns[choice][Nx // 2, :]
                ax.plot(x, var)
                ax.set_ylabel(self.ylabels[choice])
                ax.set_xlabel(rf"Distance ${dir}$")

        return fig, ax

    def plot_cut_evolution(self, choice="all", dir="x", freq=1,):
        for filename, data in self.ds.items():

            # reconstruct input dicts
            material = {k.split("_")[-1]: v for k, v in dict(data.__dict__).items() if k.startswith("material")}

            Lx = data.disc_Lx
            Ly = data.disc_Ly
            Nx = data.disc_Nx
            Ny = data.disc_Ny

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

            cmap = plt.cm.coolwarm
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=maxT))

            if choice == "all":
                fig, ax = plt.subplots(2, 2, sharex=True)
                for count, (key, a) in enumerate(zip(unknowns.keys(), ax.flat)):
                    for it, t in enumerate(time[::freq]):
                        if dir == "x":
                            var = unknowns[key][it, :, Ny // 2]
                        elif dir == "y":
                            var = unknowns[key][it, Nx // 2, :]
                        a.plot(x, var, '-', color=cmap(t / maxT))
                        a.set_ylabel(self.ylabels[key])
                        if count > 1:
                            a.set_xlabel(rf"Distance ${dir}$")

                fig.colorbar(sm, ax=ax.ravel().tolist(), label='time (s)', extend='max')
            else:
                fig, ax = plt.subplots(1)
                for it, t in enumerate(time[::freq]):
                    if dir == "x":
                        var = unknowns[choice][it, :, Ny // 2]
                    elif dir == "y":
                        var = unknowns[choice][it, Nx // 2, :]
                    ax.plot(x, var, '-', color=cmap(t / maxT))
                    ax.set_ylabel(self.ylabels[choice])
                    ax.set_xlabel(rf"Distance ${dir}$")

                fig.colorbar(sm, ax=ax, label='time (s)', extend='max')

        return fig, ax

    def plot_timeseries(self, attr):

        fig, ax = plt.subplots(1)

        for filename, data in self.ds.items():

            time = np.array(data.variables['time'])
            val = np.array(data.variables[attr])

            ylabels = {"mass": r"Mass $m$",
                       "vmax": r"Max. velocity $v_\mathrm{max}$",
                       "vSound": r"Velocity of sound $c$",
                       "dt": r"Time step $\Delta t$",
                       "eps": r"Residual $\epsilon$"}

            ax.plot(time, val, '-')
            ax.set_xlabel(r"Time $t$")
            ax.set_ylabel(ylabels[attr])

        return fig, ax

    def animate2D(self, choice="rho"):
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
            Nx = data.disc_Nx
            Ny = data.disc_Ny
            Lx = data.disc_Lx
            Ly = data.disc_Ly

            fig, ax = plt.subplots(figsize=(Nx / Ny * 7, 7))

            # Initial plotting
            self.im = ax.imshow(A[0].T, extent=(0, Lx, 0, Ly), interpolation='none', aspect='equal', cmap='viridis')

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
