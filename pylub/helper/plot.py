import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable

from pylub.eos import EquationOfState
from pylub.helper.data_parser import getData


class Plot:

    def __init__(self, path, mode="select"):

        self.ds = getData(path, mode=mode)

    def plot_cut(self, choice=None, dir='x'):

        if choice is None:
            fig, ax = plt.subplots(2, 2)
        else:
            fig, ax = plt.subplots(1)

        for filename, data in self.ds.items():

            # reconstruct input dicts
            material = {k.split("_")[-1]: v for k, v in dict(data.__dict__).items() if k.startswith("material")}

            L = data.disc_Lx
            N = data.disc_Nx
            center = data.disc_Ny // 2

            rho = np.array(data.variables["rho"])[-1]
            p = EquationOfState(material).isoT_pressure(rho)
            jx = np.array(data.variables["jx"])[-1]
            jy = np.array(data.variables["jy"])[-1]

            if dir == "y":
                rho = rho.T
                p = p.T
                jx = jx.T
                jy = jy.T
                L = data.disc_Ly
                N = data.disc_Ny
                center = data.disc_Nx // 2

            x = (np.arange(N) + 0.5) * L / N
            unknowns = [rho, p, jx, jy]

            if choice is None:
                for i, a in enumerate(ax.flat):
                    a.plot(x, unknowns[i][:, center])
            else:
                ax.plot(x, unknowns[choice][:, center])

        return fig, ax

    def plot_cut_evolution(self, choice=None, dir="x", every=1,):
        for filename, data in self.ds.items():

            # reconstruct input dicts
            material = {k.split("_")[-1]: v for k, v in dict(data.__dict__).items() if k.startswith("material")}

            L = data.disc_Lx
            N = data.disc_Nx

            time = np.array(data.variables["time"])
            maxT = time[-1]

            rho = np.array(data.variables["rho"])
            p = EquationOfState(material).isoT_pressure(rho)
            jx = np.array(data.variables["jx"])
            jy = np.array(data.variables["jy"])

            if dir == "y":
                rho = rho.T
                p = p.T
                jx = jx.T
                jy = jy.T
                L = data.disc_Ly
                N = data.disc_Ny

            x = (np.arange(N) + 0.5) * L / N
            unknowns = [rho, p, jx, jy]

            cmap = plt.cm.coolwarm
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=maxT))

            if choice is None:
                fig, ax = plt.subplots(2, 2)
                for i, a in enumerate(ax.flat):
                    for it, t in enumerate(time[::every]):
                        a.plot(x, unknowns[i][it, :, N // 2], '-', color=cmap(t / maxT))

                fig.colorbar(sm, ax=ax.ravel().tolist(), label='time (s)', extend='max')

            else:
                fig, ax = plt.subplots(1)
                for it, t in enumerate(time[::every]):
                    ax.plot(x, unknowns[choice][it, :, N // 2], '-', color=cmap(t / maxT))

                fig.colorbar(sm, ax=ax, label='time (s)', extend='max')

        return fig, ax

    def plot_timeseries(self, attr):

        assert attr in ["mass", "vmax", "vSound", "dt", "eps"]

        fig, ax = plt.subplots(1)

        for filename, data in self.ds.items():

            time = np.array(data.variables['time']) * 1e9
            val = np.array(data.variables[attr])

            ax.plot(time, val, '-')

        return fig, ax

    def animate2D(self, choice=1):
        for filename, data in self.ds.items():

            # reconstruct input dicts
            material = {k.split("_")[-1]: v for k, v in dict(data.__dict__).items() if k.startswith("material")}

            rho = np.array(data.variables["rho"])
            p = EquationOfState(material).isoT_pressure(rho)
            jx = np.array(data.variables["jx"])
            jy = np.array(data.variables["jy"])

            unknowns = [rho, p, jx, jy]

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
