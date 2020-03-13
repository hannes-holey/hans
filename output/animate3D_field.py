#!/usr/bin/env python3

import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LightSource
from matplotlib import cm

plt.style.use('presentation')
fig = plt.figure(figsize=(8,6))

file = h5py.File(sys.argv[1], 'r')

conf_opt =  file.get('/config/options')
conf_num =  file.get('config/numerics')
conf_disc = file.get('config/disc')

Nx = conf_disc.attrs['Nx']
Ny = conf_disc.attrs['Ny']
Lx = conf_disc.attrs['Lx']
Ly = conf_disc.attrs['Ly']

x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)

xx, yy = np.meshgrid(x,y)

toPlot = {0:'j_x', 1:'j_y', 2:'rho', 3:'press'}
choice = int(input("What to plot? (0: maxx flux x | 1: mass flux y | 2: density | 3: pressure) "))


# im = ax.imshow(np.empty((Nx,Ny)), interpolation='nearest', cmap='viridis')
# cbar = plt.colorbar(im, ax = ax)

ax = fig.add_subplot(111, projection='3d')

light = LightSource(90, 45)


for i in file.keys():
    if str(i) != 'config':

        g = file.get(i)
        d = np.array(g.get(toPlot[choice]))

        t = g.attrs['time']*1e9
        ax.clear()
        # illuminated_surface = light.shade(d, cmap=cm.coolwarm)
        ax.plot_surface(xx,yy,d, rstride=1, cstride=1, linewidth=0, antialiased=False, cmap='viridis')

        # im.set_array(d)
        # im.set_clim(vmin=np.amin(d),vmax=np.amax(d))

    plt.pause(0.0001)

# plt.show()
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
