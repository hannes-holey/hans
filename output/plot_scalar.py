#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from helper import getFiles, copyTemp

plt.style.use('presentation')
plt.figure(figsize=(12,7))

files = getFiles()

toPlot = {  0: ['mass', r'$\Delta m/m_0$ [-]', 1.],
            1: ['vmax',r'$v_\mathsf{max}$ [m/s]', 1.],
            2: ['vSound', 'c [m/s]', 1.],
            3: ['dt', r'$\Delta t$ [ns]', 1e9],
            4: ['eps',r'$\epsilon$ [-]', 1.]}

choice = int(input("What to plot? (0: mass x | 1: vmax | 2: speed of sound | 3: time step | 4: epsilon) "))

for file in files.values():

    conf_geo = file[1].get('config/geometry')
    conf_num =  file[1].get('config/numerics')
    conf_disc = file[1].get('config/disc')
    conf_mat = file[1].get('config/material')

    config = [conf_disc, conf_geo, conf_num, conf_mat]

    print(file[0] + ": \n" + 40 * "-")
    for k in config:
        for l in list(k.attrs):
            print("{:s}: {:>}".format(l, k.attrs[l]))
        print(40 * "-")

    label = input("Enter Legend for " + file[0] + ": ")

    file_keys = list(file[1].keys())[:-1]

    A = np.empty(0)
    for i in file_keys:
        # if str(i) != 'config':

        g = file[1].get(i)
        time = g.attrs['time']
        ydata = g.attrs[toPlot[choice][0]]

        A = np.append(A ,[time ,ydata])
        A = np.reshape(A, (int(A.shape[0]/2),2))

    if choice == 0:
        plt.plot(A[:,0] * 1e9, (A[:,1] - A[0,1])/A[0,1], '-', label = label)
    else:
        plt.plot(A[:,0] * 1e9, A[:,1] * toPlot[choice][2], '-', label = label)

    file[1].close()

plt.xlabel('time (ns)')
plt.ylabel(toPlot[choice][1])
plt.legend(loc='best')
plt.show()
