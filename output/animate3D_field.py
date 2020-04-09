#!/usr/bin/env python3

from helper import getFile
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

def assembleArrays(file, database):
    """
    Assemble list of arrays from hdf5-file to loop over in animation

    Parameters
    ----------
    file :
        h5py File object
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
    Lx : float
        length x
    Ly : float
        length y
    filename: str
        outpur filename
    """
    conf_disc = file.get('/config/disc')
    conf_opt = file.get('config/options')
    Nx = conf_disc.attrs['Nx']
    Ny = conf_disc.attrs['Ny']
    Lx = conf_disc.attrs['Lx']
    Ly = conf_disc.attrs['Ly']
    filename = conf_opt.attrs['name']
    A = []
    t = []
    for i in file.keys():
        if str(i) != 'config':
            g = file.get(i)
            A.append(np.array(g.get(database)))
            t.append(g.attrs['time']*1e9)
    file.close()

    return A, t, Nx, Ny, Lx, Ly, filename

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
    ax.cla()

    global glob_min
    global glob_max

    if np.amin(A[i]) < glob_min:
        glob_min = np.amin(A[i])
    if np.amax(A[i]) > glob_max:
        glob_max = np.amax(A[i])

    ax.plot_surface(xx,yy, A[i].T, rstride=1, cstride=1, linewidth=0, antialiased=False, cmap='viridis')
    ax.grid(False)
    ax.set_zlim(glob_min, glob_max)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    fig.suptitle("Time: {:.1f}".format(t[i]))

if __name__ == "__main__":

    plt.style.use('presentation')
    fig = plt.figure(figsize=(8,6))
    ax = Axes3D(fig)

    file = getFile()

    # User input
    toPlot = {0:'j_x', 1:'j_y', 2:'rho', 3:'press'}
    choice = int(input("What to plot? (0: maxx flux x | 1: mass flux y | 2: density | 3: pressure) "))
    save = int(input("Show (0) or save (1) animation? "))

    A, t, Nx, Ny, Lx, Ly, name = assembleArrays(file, toPlot[choice])

    # Global colorbar limits
    glob_min = np.amin(A[0])
    glob_max = np.amax(A[0])

    # Initial plotting
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    xx, yy = np.meshgrid(x,y)
    ax.plot_surface(xx,yy, A[0].T, rstride=1, cstride=1, linewidth=0, antialiased=False, cmap='viridis')
    ax.grid(False)
    ax.set_zlim(glob_min, glob_max)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    # Create animation
    anim = animation.FuncAnimation(fig, plot_update, frames=len(A), fargs=(A, t,), interval=500, repeat=True)

    if save == 1:
        anim.save(name + '_' + toPlot[choice] + '_3D.mp4' , fps=30)
    elif save == 0:
        plt.show()
