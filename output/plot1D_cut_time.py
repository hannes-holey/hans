#!/usr/bin/env python3

import h5py
import numpy as np
import matplotlib.pyplot as plt
import sys

plt.style.use('presentation')
plt.figure(figsize=(10,6), tight_layout=True)

file = h5py.File(sys.argv[1], 'r')

conf_opt =  file.get('/config/options')
conf_num =  file.get('config/numerics')
conf_disc = file.get('config/disc')

writeInterval = conf_opt.attrs['writeInterval']
maxT = conf_num.attrs['maxT']
Lx = conf_disc.attrs['Lx']

toPlot = {0:'j_x', 1:'j_y', 2:'rho', 3:'press'}

choice      = int(input("What to plot? (0: maxx flux x | 1: mass flux y | 2: density | 3: pressure) "))
every       = int(input("Plot every n-th written time step: "))
plot_freq   = writeInterval


cmap = plt.cm.coolwarm

#for file in files:
for i in file.keys():
    if str(i) != 'config' and int(i) % (plot_freq*every) == 0:

        g = file.get(i)
        d = np.array(g.get(toPlot[choice]))
        x = np.linspace(0, 1e-3, d.shape[0])

        t = g.attrs['time']*1e9
        # t = int(i) * dt * 1.e9
        c = t/maxT

        plt.plot(x * 1.e3, d[:,int(d.shape[1]/2)], '-', label = 't = %2.f ns' % (t), color =cmap(c))

sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin = 0, vmax = t))
file.close()

if choice == 0:
    ylab =  'mass flux x'
elif  choice == 1:
    ylab =  'mass flux y'
elif  choice == 2:
    ylab =  'density'
else:
    ylab = 'pressure (Pa)'

plt.xlabel('distance (mm)')
plt.ylabel(ylab)

ref = np.loadtxt('p_500.dat')
x_ref = np.linspace(0, 1., ref.shape[0])
plt.plot(x_ref, ref, '--', color = 'black', label = r'steady-state')

plt.colorbar(sm, label = 'time (ns)', extend='max')
plt.show()
