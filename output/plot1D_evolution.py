#!/usr/bin/env python3

import h5py
import numpy as np
import matplotlib.pyplot as plt
from helper import getFile

plt.style.use('presentation')
plt.figure(figsize=(10,6), tight_layout=True)

file = getFile()

conf_disc = file.get('config/disc')
Lx = conf_disc.attrs['Lx']

toPlot = {  0: ['j_x', r'mass flux $x$ [kg/(m$^2$s)]', 1.],
            1: ['j_y', r'mass flux $y$ [kg/(m$^2$s)]', 1.],
            2: ['rho', r'density [kg/m$^3$]', 1.],
            3: ['press', r'pressure (MPa)', 1e-6]}

choice      = int(input("What to plot? (0: maxx flux x | 1: mass flux y | 2: density | 3: pressure) "))
every       = int(input("Plot every n-th written time step: "))

keys = list(file.keys())[:-1]
maxT = file.get(keys[-1]).attrs['time']*1e6

cmap = plt.cm.coolwarm

for i in keys[::every]:

    g = file.get(i)
    d = np.array(g.get(toPlot[choice][0]))
    x = np.linspace(0, Lx, d.shape[0])

    t = g.attrs['time']*1e6
    c = t/maxT

    plt.plot(x * 1.e3, d[:,int(d.shape[1]/2)] * toPlot[choice][2], '-', color = cmap(c))

sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin = 0, vmax = maxT))
file.close()

plt.xlabel('distance [mm]')
plt.ylabel(toPlot[choice][1])

plt.colorbar(sm, label = 'time [µs]', extend='max')
plt.show()
