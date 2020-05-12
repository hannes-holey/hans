#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from helper import getFile
from scipy.optimize import curve_fit
from mpi4py import MPI

# plt.style.use('presentation')
# plt.figure(figsize=(12,7))

comm = MPI.COMM_WORLD

rank = comm.Get_rank()
size = comm.Get_size()


def gaussian(x, a, mean, sigma):
    return a * np.exp(-((x - mean)**2 / (2 * sigma**2)))


toPlot = {0: ['j_x', r'mass flux $x$ [kg/(m$^2$s)]', 10],
          1: ['j_y', r'mass flux $y$ [kg/(m$^2$s)]', 10],
          2: ['rho', r'density [kg/m$^3$]', 1e3],
          3: ['press', r'pressure (MPa)', 1e-6]}


if rank == 0:
    # TODO parallel getFile method required
    file = getFile()

    conf_disc = file.get('config/disc')

    Nx = int(conf_disc.attrs['Nx'])
    Ny = int(conf_disc.attrs['Ny'])

    keys = list(file.keys())[:-1]       # list with all timesteps ('str')
    last = keys[-1]                     # total number of samples
    global_size = len(keys)
    length = len(file.get(last))        # 'length' of the group(2: w/ pressure, 1: w/o pressure)

    if rank == 0:
        if length == 2:
            choice = int(input("What to plot? (0: mass flux x | 1: mass flux y | 2: density | 3: pressure) "))
        else:
            choice = int(input("What to plot? (0: mass flux x | 1: mass flux y | 2: density) "))

        assert choice in np.arange(0, length + 2), "Invalid choice"
else:
    Nx = None
    Ny = None
    keys = None
    last = None
    length = None
    choice = None
    global_size = None
    files = None

Nx = comm.bcast(Nx, root=0)
Ny = comm.bcast(Ny, root=0)
keys = comm.bcast(keys, root=0)
last = comm.bcast(last, root=0)
length = comm.bcast(length, root=0)
choice = comm.bcast(choice, root=0)
global_size = comm.bcast(global_size, root=0)
file = comm.bcast(file, root=0)

# intitalize and fill array with all sampled time steps
full_array = np.empty([global_size, Nx, Ny])
countFaults = 0

# Partitioning of timesteps
loc_size = global_size // size
if rank == size - 1:
    loc_size += global_size % loc_size

local_keys = keys[rank * loc_size:(rank + 1) * loc_size]
if rank == size - 1:
    local_keys = keys[global_size - loc_size:global_size]


for i in local_keys:
    # TODO: parallel reading of hdf5 requires parallel build of h5py package
    g = file.get(i)

    # SI units
    if g is not None:
        if choice == 3:
            d = np.array(g.get('p'))
        else:
            d = np.array(g.get('q')[choice])

        if np.isfinite(d).all() is False:
            countFaults += 1
        else:
            full_array[int(i)] = d

    else:
        countFaults += 1
#
print("Rank {:d}: read {:d} from {:d} samples".format(rank, loc_size - countFaults, loc_size))

# cellVariance = np.var(full_array, axis=0)
# variance = np.mean(cellVariance)
#
# print(variance)

# --- Loop over different sampling numbers --- #

# var = np.zeros((5,2))

# for i in range(5):
#     countFaults = int(i * 1e6 / 5)

    # if countFaults > 1:
    #     cellVariance = np.var(full_array[:-countFaults], axis=0)
    # else:
    #     cellVariance = np.var(full_array, axis=0)

# mean cell-variance
    # variance = np.mean(cellVariance)
    # var[i,0] = len(keys) - countFaults
    # var[i,1] = variance
    # print(variance)


# plt.plot(var[:,0], var[:,1], '-o')
# plt.yscale('log')
# plt.xscale('log')
# plt.show()


# global mean
# mean = np.mean(full_array)

# plot histogram of fluctuating quantity
# yhist, xhist, _ = plt.hist(np.mean(full_array, axis=(1,2)), bins=50)

# fit un-normalized Gaussian to the data and plot
# popt, pcov = curve_fit(gaussian, xhist[:-1], yhist, [1, mean, np.sqrt(variance)])
# x = np.linspace(xhist[0],xhist[-1], 100)
# plt.plot(x, gaussian(x, *popt), '--', color='red')
# plt.show()
