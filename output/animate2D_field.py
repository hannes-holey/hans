#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from helper import getFile

def assembleArrays(file, database):
    """
    Assemble list of arrays from hdf5-file to loop over in animation

    Parameters
    ----------
    file : h5py File object
        hdf5 input file
    database :  str
        name of the database to read from

    Returns
    -------
    A :  list
        list of result arrays
    t :  list
        list of snapshot times
    Nx : int
        number of grid cells in x
    Ny : int
        number of grid cells in y
    filename: str
        name of the output file
    """

    conf_disc = file.get('/config/disc')
    conf_opt = file.get('config/options')

    Nx = conf_disc.attrs['Nx']
    Ny = conf_disc.attrs['Ny']
    filename = conf_opt.attrs['name']

    A = []
    t = []
    for i in file.keys():
        if str(i) != 'config':
            g = file.get(i)
            A.append(np.array(g.get(database)))
            t.append(g.attrs['time']*1e9)

    return A, t, Nx, Ny, filename

def plot_update(i, A, t):
    """
    Updates the plot in animation

    Parameters
    ----------
    i : int
        iterator
    A : list
        list of result arrays
    t : list
        list of snapshot times
    """
    global glob_min
    global glob_max

    if np.amin(A[i]) < glob_min:
        glob_min = np.amin(A[i])
    if np.amax(A[i]) > glob_max:
        glob_max = np.amax(A[i])

    im.set_array(A[i].T)
    im.set_clim(vmin=glob_min, vmax=glob_max)
    fig.suptitle("Time: {:.1f}".format(t[i]))

if __name__ == "__main__":

    plt.style.use('presentation')
    fig, ax= plt.subplots(figsize=(8,6))

    file = getFile()

    # User input
    toPlot = {0:'j_x', 1:'j_y', 2:'rho', 3:'press'}
    choice = int(input("What to plot? (0: maxx flux x | 1: mass flux y | 2: density | 3: pressure) "))
    save = int(input("Show (0) or save (1) animation? "))

    A, t, Nx, Ny, name = assembleArrays(file, toPlot[choice])

    # Global colorbar limits
    glob_min = np.amin(A[0])
    glob_max = np.amax(A[0])

    # Initial plotting
    clab = {0: 'mass flux x', 1: 'mass flux y', 2: 'density', 3: 'pressure (Pa)'}
    im = ax.imshow(np.empty((Nx,Ny)), interpolation='nearest', cmap='viridis')
    cbar = plt.colorbar(im, ax = ax, label = clab[choice])
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    # Create animation
    anim = animation.FuncAnimation(fig, plot_update, frames=len(A), fargs=(A, t,), interval=100, repeat=True)

    if save == 1:
        anim.save(name + '_' + toPlot[choice] + '_2D.mp4' , fps=30)
    elif save == 0:
        plt.show()
