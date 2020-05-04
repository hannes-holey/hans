#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from helper import getFiles, getReference

plt.style.use('presentation')
fig, ax = plt.subplots(figsize=(12,9), tight_layout=False)

files = getFiles()

toPlot = {0: ['j_x', r'mass flux $x$ [kg/(m$^2$s)]', 1.],
          1: ['j_y', r'mass flux $y$ [kg/(m$^2$s)]', 1.],
          2: ['rho', r'density [kg/m$^3$]', 1.],
          3: ['press', r'pressure (MPa)', 1e-6]}

choice = int(input("What to plot? (0: maxx flux x | 1: mass flux y | 2: density | 3: pressure) "))

for file in files.values():

    conf_geo = file[1].get('config/geometry')
    conf_num = file[1].get('config/numerics')
    conf_disc = file[1].get('config/disc')
    conf_mat = file[1].get('config/material')

    config = [conf_disc, conf_geo, conf_num, conf_mat]

    print(file[0] + ": \n" + 40 * "-")
    for k in config:
        for l in list(k.attrs):
            print("{:s}: {:>}".format(l, k.attrs[l]))
        print(40 * "-")

    label = input("Enter Legend for " + file[0] + ": ")

    last = list(file[1].keys())[-2]

    Nx = int(conf_disc.attrs['Nx'])
    Lx = float(conf_disc.attrs['Lx'])

    g = file[1].get(last)
    d = np.array(g.get(toPlot[choice][0]))
    x = np.linspace(0, Lx, Nx)
    t = g.attrs['time'] * 1e9
    print("Actual time for \'{:s}\': {:.2f} ns".format(file[0], t))
    ax.plot(x * 1.e3, (d[:,int(d.shape[1] / 2)]) * toPlot[choice][2], '-', label=label)

    file[1].close()

    if choice > 1:
        ref = getReference()
    else:
        ref = None

    if ref is not None:
        ref_label = input("Enter Legend for reference: ")
        x_ref = np.linspace(0, 1., ref.shape[0])
        scalef = {2: 1., 3:1e-6}
        ax.plot(x_ref, ref[:,choice - 2] * scalef[choice], '--', label=ref_label)

ax.set_xlabel('distance (mm)')
plt.ylabel(toPlot[choice][1])

ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.legend(loc='best')
plt.show()
