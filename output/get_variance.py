#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from helper import getFile
# from scipy.optimize import curve_fit

plt.style.use('presentation')
fig, ax = plt.subplots(3,2, figsize=(12,16))


def gaussian(x, a, mean, sigma):
    return a * np.exp(-((x - mean)**2 / (2 * sigma**2)))


filename, file = getFile()

Nx = file.Nx
Ny = file.Ny

toPlot = {0: ['jx', r'mass flux $x$ [kg/(m$^2$s)]', 1e1],       # last item in list: conversion to cgs units
          1: ['jy', r'mass flux $y$ [kg/(m$^2$s)]', 1e1],
          2: ['rho', r'density [kg/m$^3$]', 1e3],
          3: ['p', r'pressure (MPa)', 1e-6]}

choices = np.arange(3)

# table header
print("{:^10s}| {:^17s} | {:^17s} ".format("","mean","variance"))
print(50 * '-')

for choice in choices:
    # check if enough memory available, load data
    try:
        full_array = np.array(file.variables[toPlot[choice][0]], copy=False) / toPlot[choice][2]
    except MemoryError:
        print("Not enough memory, using single precision!")
        full_array = np.array(file.variables[toPlot[choice][0]], copy=False).astype(np.float32) / toPlot[choice][2]

    mask = np.all(np.isfinite(full_array), axis=(1,2))
    full_array = full_array[mask]

    cellVariance = np.var(full_array, axis=0)
    variance = np.mean(cellVariance)
    mean = np.mean(full_array)

# --- Loop over n different sampling numbers --- #
    n = 10
    var = np.zeros((n,3))
    # size = file.dimensions['step'].size
    size = int(full_array.size / Nx / Ny)

    for i in range(n):
        samples = (i + 1) * size // n

        cellVariance = np.var(full_array[0:samples + 1], axis=0)
        mean = np.mean(full_array[0:samples + 1])
        variance = np.mean(cellVariance)

        var[i,0] = samples
        var[i,1] = mean
        var[i,2] = variance

    col = 'C' + str(choice)
    # np.savetxt('var_' + toPlot[choice][0] + '-vs-N.dat', var, header="samples variance(cgs units)", comments="")
    ax[choice, 1].plot(var[:,0], var[:,2], '-o', color=col)
    # ax[choice, 0].set_xlim(mean - 0.5 * np.sqrt(variance), mean + 0.5 * np.sqrt(variance))

    # plot histogram of mean value time series
    # mean_t = np.mean(full_array, axis=(1,2))
    # yhist, xhist, _ = ax[choice, 0].hist(mean_t, bins=25, color=col)

    # plot histogram of fluctuating quantity in the center cell
    yhist, xhist, _ = ax[choice,0].hist(full_array[:,Nx // 2, Ny // 2], bins=25, color=col)

    # print variance to screen
    if not(np.isfinite(variance)):
        mean = var[np.isfinite(var[:,1])][-1,1]
        variance = var[np.isfinite(var[:,2])][-1,2]
    print("{:^10s}| {: .10e} | {: .10e} ".format(toPlot[choice][0], mean, variance))

plotVar = int(input("Show (0) or save (1) figure? "))

if plotVar == 1:
    plt.savefig(filename.split(".")[0] + "_var_" + toPlot[choice][0] + '.pdf')
elif plotVar == 0:
    plt.show()
