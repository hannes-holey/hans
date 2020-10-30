#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from fhl2d.helper.data_parser import getData


def plot_update(i, A, t):
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
    global glob_min
    global glob_max

    if np.amin(A[i]) < glob_min:
        glob_min = np.amin(A[i])
    if np.amax(A[i]) > glob_max:
        glob_max = np.amax(A[i])

    im.set_array(A[i].T)
    im.set_clim(vmin=glob_min, vmax=glob_max)
    # fig.suptitle("Time: {:.1f} µs".format(t[i]))


if __name__ == "__main__":

    plt.style.use('presentation')

    for filename, file in getData(".", mode="singl").items():

        # User input
        toPlot = {0: ['rho', r'mass density (kg/m$^3$)', 1.],
                  1: ['jx', r'mass flux $x$ [kg/(m$^2$s)]', 1.],
                  2: ['jy', r'mass flux $y$ [kg/(m$^2$s)]', 1.],
                  3: ['p', r'pressure (MPa)', 1e-6]}

        reduced = not('p' in file.variables)

        if reduced is True:
            choice = int(input("Choose field variable to plot:\n0:\tdensity\n1:\tmass flux x\n2:\tmass flux y\n"))
        else:
            choice = int(input("Choose field variable to plot:\n0:\tdensity\n1:\tmass flux x\n2:\tmass flux y\n3:\tpressure\n"))

        save = int(input("Show (0) or save (1) animation? "))

        # A, t, Nx, Ny, name = assembleArrays(file, toPlot[choice])
        A = np.array(file.variables[toPlot[choice][0]]) * toPlot[choice][2]
        t = np.array(file.variables['time']) * 1e6
        Nx = file.disc_Nx
        Ny = file.disc_Ny
        Lx = file.disc_Lx
        Ly = file.disc_Ly
        rho0 = file.material_rho0
        name = file.options_name

        ratio = Nx / Ny
        fig, ax = plt.subplots(figsize=(ratio * 5, 5))

        # Global colorbar limits
        glob_min = np.amin(A[0])
        glob_max = np.amax(A[0])

        # Initial plotting
        im = ax.imshow(np.empty((Ny,Nx)), interpolation='gaussian', aspect='equal', cmap='viridis')
        cbar = plt.colorbar(im, ax=ax, label=toPlot[choice][1])

        ax.invert_yaxis()

        # Adjust ticks
        ticksx = np.linspace(0, Lx, 3)
        ticksy = np.linspace(0, Ly, 3)
        ax.set_xticks(ticksx / Lx * Nx - 0.5)
        ax.set_xticklabels([f"{v*1e9:.0f}" for v in ticksx])
        ax.set_yticks(ticksy / Ly * Ny - 0.5)
        ax.set_yticklabels([f"{v*1e9:.0f}" for v in ticksy])
        ax.set_xlabel(r'$L_x$ (nm)')
        ax.set_ylabel(r'$L_y$ (nm)')

        # Create animation
        anim = animation.FuncAnimation(fig, plot_update, frames=len(A), fargs=(A, t,), interval=100, repeat=True)

        if save == 1:
            anim.save(name + '_' + toPlot[choice][0] + '_2D.mp4', fps=30)
        elif save == 0:
            plt.show()
