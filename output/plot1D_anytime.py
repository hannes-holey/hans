#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from helper import getFiles, getReference

plt.style.use('presentation')
fig, ax = plt.subplots(figsize=(12,9), tight_layout=False)

files = getFiles()

toPlot = {  0: ['j_x', r'mass flux $x$ [kg/(m$^2$s)]', 1.],
            1: ['j_y', r'mass flux $y$ [kg/(m$^2$s)]', 1.],
            2: ['rho', r'density [kg/m$^3$]', 1.],
            3: ['press', r'pressure (MPa)', 1e-6]}

choice = int(input("What to plot? (0: maxx flux x | 1: mass flux y | 2: density | 3: pressure) "))

for file in files.values():

    conf = file[1].get('/config')
    conf_opt =  file[1].get('/config/options')
    conf_num =  file[1].get('config/numerics')
    conf_disc = file[1].get('config/disc')

    Nx = conf_disc.attrs['Nx']
    Lx = conf_disc.attrs['Lx']
    maxT = conf_num.attrs['maxT']

    time = float(input("Approximate time in ns for \'{:s}\' (max = {:.1f} ns): ".format(file[0], maxT)))


    step_list = list(file[1].keys())[:-1]
    A = {}
    for i in step_list:
        A[i] = abs(file[1].get(i).attrs['time']*1e9 // 1. - time)

    flag = min(A, key=A.get)

    g = file[1].get(flag)
    d = np.array(g.get(toPlot[choice][0]))
    x = np.linspace(0, Lx, Nx)
    t = g.attrs['time']*1e9
    print("Closest available snapshot for \'{:s}\' at t = {:.2f} ns".format(file[0], t))
    ax.plot(x * 1.e3, (d[:,int(d.shape[1]/2)]) * toPlot[choice][2], '-', label = r'$t = ${:.1f} ns'.format(t))

    file[1].close()

ax.set_xlabel('distance [mm]')
ax.set_ylabel(toPlot[choice][1])

ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.legend(loc = 'best')
plt.show()
