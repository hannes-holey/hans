#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from helper import getFiles, getReference

plt.style.use('presentation')
fig, ax = plt.subplots(figsize=(12,9), tight_layout=False)

files = getFiles()

toPlot = {0: ['jx', r'mass flux $x$ [kg/(m$^2$s)]', 1.],
          1: ['jy', r'mass flux $y$ [kg/(m$^2$s)]', 1.],
          2: ['rho', r'density [kg/m$^3$]', 1.],
          3: ['p', r'pressure (MPa)', 1e-6]}

reduced = False
for file in files.values():
    if not('p' in file[1].variables):
        reduced = True
        break

if reduced is True:
    choice = int(input("Choose field variable to plot:\n0:\tmass flux x\n1:\tmass flux y\n2:\tdensity\n"))
else:
    choice = int(input("Choose field variable to plot:\n0:\tmass flux x\n1:\tmass flux y\n2:\tdensity\n3:\tpressure\n"))

for file in files.values():

    print(file[0] + ": \n" + 40 * "-")
    for name in file[1].ncattrs():
        print("{:20s}: {:>}".format(name, getattr(file[1], name)))
    print(40 * "-")

    label = input("Enter Legend for " + file[0] + ": ")

    Lx = file[1].Lx
    Nx = file[1].Nx

    d = np.array(file[1].variables[toPlot[choice][0]])[-1]
    x = np.linspace(0, Lx, Nx)
    t = np.array(file[1].variables['time'])[-1] * 1e9
    print("Actual time for \'{:s}\': {:.2f} ns".format(file[0], t))
    ax.plot(x * 1.e3, (d[:,int(d.shape[1] / 2)]) * toPlot[choice][2], '-', label=label)

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
