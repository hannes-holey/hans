#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from helper import getFile

plt.style.use('presentation')
plt.figure(figsize=(10,6), tight_layout=True)

filename, file = getFile()

Lx = file.disc_Lx
Nx = file.disc_Nx

toPlot = {0: ['jx', r'mass flux $x$ [kg/(m$^2$s)]', 1.],
          1: ['jy', r'mass flux $y$ [kg/(m$^2$s)]', 1.],
          2: ['rho', r'density [kg/m$^3$]', 1.],
          3: ['p', r'pressure (MPa)', 1e-6]}

reduced = not('p' in file.variables)

if reduced is True:
    choice = int(input("Choose field variable to plot:\n0:\tmass flux x\n1:\tmass flux y\n2:\tdensity\n"))
else:
    choice = int(input("Choose field variable to plot:\n0:\tmass flux x\n1:\tmass flux y\n2:\tdensity\n3:\tpressure\n"))

every = int(input("Plot every n-th written time step: "))

time = np.array(file.variables['time']) * 1e6
maxT = time[-1]

cmap = plt.cm.coolwarm
x = (np.arange(Nx) + 0.5) * Lx / Nx

for i in range(0, len(time), every):

    d = np.array(file.variables[toPlot[choice][0]])[i]

    t = time[i]
    c = t / maxT

    plt.plot(x * 1.e3, d[:,int(d.shape[1] / 2)] * toPlot[choice][2], '-', color=cmap(c))

sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=maxT))

plt.xlabel('distance [mm]')
plt.ylabel(toPlot[choice][1])

plt.colorbar(sm, label='time [µs]', extend='max')
plt.show()
