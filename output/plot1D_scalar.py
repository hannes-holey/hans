#!/usr/bin/env python3

import h5py
import numpy as np
import math as m
import matplotlib.pyplot as plt
import os

plt.style.use('presentation')
plt.figure(figsize=(12,7))

availFiles = {}
i = 0
for file in sorted(os.listdir()):
    if file.endswith('h5'):
        availFiles.update({i: file})
        i +=1

print("Available files:")
for key, val in availFiles.items():
    print("{:3d}: {:20s}".format(key, val))


files = []
ask = True
while ask == True:
    userInput = input("Enter file key (any other key to exit): ")
    if userInput in np.arange(0, len(availFiles)).astype(str):
        files.append(h5py.File(availFiles[int(userInput)], 'r'))
    else:
        ask = False


toPlot = {0:['mass', ' (ng)', 1e12],
        1:['vmax',' (m/s)', 1.],
        2: ['vSound', ' (m/s)', 1.],
        3: ['dt', ' (ns)', 1e9],
        4: ['eps',r' ($10^{-9}$)', 1e9]}

choice = int(input("What to plot? (0: mass x | 1: vmax | 2: speed of sound | 3: time step | 4: epsilon) "))

for file in files:

    conf_opt =  file.get('/config/options')
    conf_num =  file.get('config/numerics')
    conf_disc = file.get('config/disc')

    Nx = conf_disc.attrs['Nx']
    Lx = conf_disc.attrs['Lx']

    A = np.empty(0)
    for i in file.keys():
        if str(i) != 'config':

            g = file.get(i)
            time = g.attrs['time']
            ydata = g.attrs[toPlot[choice][0]]

            A = np.append(A ,[time ,ydata])
            A = np.reshape(A, (int(A.shape[0]/2),2))

    plt.plot(A[:,0] * 1e9, A[:,1] * toPlot[choice][2], '-', label = '')
    # plt.plot(A[:,0] * 1e9, A[:,1] * toPlot[choice][2], '-', label = 'dt = %1.e, Nx = %d' % (dt, Nx))
    # plt.plot(A[:,0] * 1e9, A[:,1] * toPlot[choice][2], '-', label = r"$\Delta$t = %s, $N_x = %d$" % (sci_notation(dt), Nx))

plt.xlabel('time (ns)')
plt.ylabel(toPlot[choice][0] + toPlot[choice][1])
plt.legend(loc='best', ncol = 2)
plt.show()
