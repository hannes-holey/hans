#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from fhl2d.helper.data_parser import getData

plt.style.use('presentation')
plt.figure(figsize=(10,6), tight_layout=True)

files = getData(".", single=True)

for filename, data in files.items():

    Lx = data.disc_Lx
    Nx = data.disc_Nx

    toPlot = {0: ['rho', r'mass density (kg/m$^3$)', 1.],
              1: ['jx', r'mass flux $x$ [kg/(m$^2$s)]', 1.],
              2: ['jy', r'mass flux $y$ [kg/(m$^2$s)]', 1.],
              3: ['p', r'pressure (MPa)', 1e-6]}

    reduced = not('p' in data.variables)

    if reduced is True:
        choice = int(input("Choose field variable to plot:\n0:\tdensity\n1:\tmass flux x\n2:\tmass flux y\n"))
    else:
        choice = int(input("Choose field variable to plot:\n0:\tdensity\n1:\tmass flux x\n2:\tmass flux y\n3:\tpressure\n"))

    every = int(input("Plot every n-th written time step: "))

    time = np.array(data.variables['time']) * 1e6
    maxT = time[-1]

    cmap = plt.cm.coolwarm
    x = (np.arange(Nx) + 0.5) * Lx / Nx

    for i in range(0, len(time), every):

        d = np.array(data.variables[toPlot[choice][0]])[i]

        t = time[i]
        c = t / maxT

        plt.plot(x * 1.e3, d[:,int(d.shape[1] / 2)] * toPlot[choice][2], '-', color=cmap(c))

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=maxT))

    plt.xlabel('distance [mm]')
    plt.ylabel(toPlot[choice][1])

    plt.colorbar(sm, label='time [µs]', extend='max')
    plt.show()
