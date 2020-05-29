#!/usr/bin/env python3

import psutil
import numpy as np
import matplotlib.pyplot as plt
from helper import getFile
from scipy.optimize import curve_fit

plt.style.use('presentation')
fig, ax = plt.subplots(2,1, figsize=(12,16))


def gaussian(x, a, mean, sigma):
    return a * np.exp(-((x - mean)**2 / (2 * sigma**2)))


filename, file = getFile()

Nx = file.Nx
Ny = file.Ny

toPlot = {0: ['jx', r'mass flux $x$ [kg/(m$^2$s)]', 1e1],       # last item in list: conversion to cgs units
          1: ['jy', r'mass flux $y$ [kg/(m$^2$s)]', 1e1],
          2: ['rho', r'density [kg/m$^3$]', 1e3],
          3: ['p', r'pressure (MPa)', 1e-6]}

reduced = not('p' in file.variables)

if reduced is True:
    choice = int(input("Choose field variable to plot:\n0:\tmass flux x\n1:\tmass flux y\n2:\tdensity\n"))
else:
    choice = int(input("Choose field variable to plot:\n0:\tmass flux x\n1:\tmass flux y\n2:\tdensity\n3:\tpressure\n"))

# check if enough memory available, load data
try:
    full_array = np.array(file.variables[toPlot[choice][0]], copy=False) / toPlot[choice][2]
except MemoryError:
    print("Not enough memory, using single precision!")
    full_array = np.array(file.variables[toPlot[choice][0]], copy=False).astype(np.float32) / toPlot[choice][2]

cellVariance = np.var(full_array, axis=0)
variance = np.mean(cellVariance)
mean = np.mean(full_array)
print("mean : {:}, variance : {:}".format(mean, variance))

# --- Loop over n different sampling numbers --- #
n = 10
var = np.zeros((n,2))
size = file.dimensions['step'].size

for i in range(n):
    samples = (i + 1) * size // n

    cellVariance = np.var(full_array[0:samples + 1], axis=0)
    variance = np.mean(cellVariance)

    var[i,0] = samples
    var[i,1] = variance

# np.savetxt('var_' + toPlot[choice][0] + '-vs-N.dat', var, header="samples variance(cgs units)", comments="")
ax[0].plot(var[:,0], var[:,1], '-o')

# plot histogram of fluctuating quantity in the center cell
yhist, xhist, _ = plt.hist(full_array[:,Nx // 2, Ny // 2], bins=25)

bin_width = xhist[-1] - xhist[-2]

# fit un-normalized Gaussian to the data and plot
popt, pcov = curve_fit(gaussian, xhist[:-1] + bin_width / 2, yhist, [np.amax(xhist), mean, np.sqrt(variance)])
x = np.linspace(xhist[0],xhist[-1], 100)
ax[1].plot(x, gaussian(x, *popt), '--', color='red')

plotVar = int(input("Show (0) or save (1) figure? "))

if plotVar == 1:
    plt.savefig(filename.split(".")[0] + "_var_" + toPlot[choice][0] + '.pdf')
elif plotVar == 0:
    plt.show()
