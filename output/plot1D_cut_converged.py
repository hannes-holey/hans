#!/usr/bin/env python3
import os
import h5py
import numpy as np
import math as m
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

plt.style.use('presentation')
fig, ax = plt.subplots(figsize=(12,9), tight_layout=False)

availFiles = {}
i = 0
for file in sorted(os.listdir()):
    if file.endswith('h5'):
        availFiles.update({i: file})
        i +=1

print("Available files:")
for key, val in availFiles.items():
    print("{:3d}: {:20s}".format(key, val))

files = {}
ask = True
j = 0
while ask == True:
    userInput = input("Enter file key (any other key to exit): ")

    if userInput in np.arange(0, i).astype(str):
        files.update({j: [availFiles[int(userInput)], h5py.File(availFiles[int(userInput)], 'r')]})
        j += 1
    else:
        ask = False

toPlot = {  0: ['j_x', r'mass flux $j_x$ $(kg\, m^{-2}s^{-1})$', 1.],
            1: ['j_y', r'mass flux $j_x$ $(kg\, m^{-2}s^{-1})$', 1.],
            2: ['rho', r'density $\rho$ $(kg\,m^{-3})$', 1.],
            3: ['press', r'pressure $P$ $(MPa)$', 1e-6]}

choice = int(input("What to plot? (0: maxx flux x | 1: mass flux y | 2: density | 3: pressure) "))

for file in files.values():

    last = list(file[1].keys())[-2]

    conf = file[1].get('/config')
    conf_opt =  file[1].get('/config/options')
    conf_num =  file[1].get('config/numerics')
    conf_disc = file[1].get('config/disc')

    Nx = conf_disc.attrs['Nx']
    Lx = conf_disc.attrs['Lx']
    maxT = conf_num.attrs['maxT']

    g = file[1].get(last)
    d = np.array(g.get(toPlot[choice][0]))
    x = np.linspace(0, Lx, Nx)
    t = g.attrs['time']*1e9
    print("Actual time for \'{:s}\': {:.2f} ns".format(file[0], t))
    ax.plot(x * 1.e3, (d[:,int(d.shape[1]/2)]) * toPlot[choice][2], '-', label = r'$N_x = {:d}$'.format(Nx))

    file[1].close()

ax.set_xlabel('distance (mm)')
plt.ylabel(toPlot[choice][1])

# ref = np.loadtxt('p_500.dat')
# x_ref = np.linspace(0, 1., ref.shape[0])
# ax.plot(x_ref, ref*1e-6, '--', label = r'steady-state')

ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.legend(loc = 'best')
plt.show()
