#!/usr/bin/env python3

import h5py
import numpy as np
import matplotlib.pyplot as plt
import sys

plt.style.use('presentation')
fig, ax= plt.subplots(figsize=(8,6))

file = h5py.File(sys.argv[1], 'r')

conf_opt =  file.get('/config/options')
conf_num =  file.get('config/numerics')
conf_disc = file.get('config/disc')

Nx = conf_disc.attrs['Nx']
Ny = conf_disc.attrs['Ny']

toPlot = {0:'j_x', 1:'j_y', 2:'rho', 3:'press'}
choice = int(input("What to plot? (0: maxx flux x | 1: mass flux y | 2: density | 3: pressure) "))


im = ax.imshow(np.empty((Nx,Ny)), interpolation='nearest', cmap='viridis')
cbar = plt.colorbar(im, ax = ax)

for i in file.keys():
    if str(i) != 'config':

        g = file.get(i)
        d = np.array(g.get(toPlot[choice]))

        t = g.attrs['time']*1e9

        im.set_array(d)
        im.set_clim(vmin=np.amin(d),vmax=np.amax(d))

    plt.pause(0.001)

plt.show()
file.close()

# plt.title()

# if choice == 0:
#     ylab =  'mass flux x'
# elif  choice == 1:
#     ylab =  'mass flux y'
# elif  choice == 2:
#     ylab =  'density'
# else:
#     ylab = 'pressure (Pa)'
#
# plt.xlabel('distance (mm)')
# plt.ylabel(ylab)

#plt.legend(loc = 'best')
