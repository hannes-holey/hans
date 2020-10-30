#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from fhl2d.helper.data_parser import getData
from fhl2d.helper.mean_var import getMean_field, getVariance_field


plt.style.use('presentation')
fig, ax = plt.subplots(1,1)


def gaussian(x, a, mean, sigma):
    return a * np.exp(-((x - mean)**2 / (2 * sigma**2)))


files = getData(".")

toPlot = {0: ['rho', r'density (kg/m³)]'],
          1: ['jx', r'mass flux $x$ (kg/(m²s))'],
          2: ['jy', r'mass flux $y$ (kg/(m²s))'],
          3: ['p', r'pressure (MPa)']}

choice = int(input("Which field variable? "))

for filename, data in files.items():

    Nx = data.disc_Nx
    Ny = data.disc_Ny
    Lx = data.disc_Lx
    mean = getMean_field(data)
    var = getVariance_field(data)

    x = (np.arange(Nx) + 0.5) * Lx / Nx

    ax.plot(x, mean[choice,:, Ny // 2])
    ax.fill_between(x, mean[choice, :, Ny // 2] - np.sqrt(var[choice,:, Ny // 2]), mean[choice,:, Ny // 2] + np.sqrt(var[choice, :, Ny // 2]), alpha=0.5)
    ax.set_xlabel(r'distance (m)')
    ax.set_ylabel(toPlot[choice][1])

    plotVar = int(input("Show (0) or save (1) figure? "))

    if plotVar == 1:
        plt.savefig(filename.split(".")[0] + "_var_" + toPlot[choice][0] + '.pdf')
    elif plotVar == 0:
        plt.show()
