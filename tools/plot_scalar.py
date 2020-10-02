#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from helper import getData

plt.style.use('presentation')
plt.figure(figsize=(12,7))

files = getData("../data")

toPlot = {0: ['mass', r'$\Delta m/m_0$ [-]', 1.],
          1: ['vmax',r'$v_\mathsf{max}$ [m/s]', 1.],
          2: ['vSound', 'c [m/s]', 1.],
          3: ['dt', r'$\Delta t$ [ns]', 1e9],
          4: ['eps',r'$\epsilon$ [-]', 1.]}

choice = int(input("Choose scalar quantity to plot as time series:\n0:\tmass\n1:\tvmax\n2:\tvSound\n3:\ttime step\n4:\tepsilon\n"))

for filename, data in files.items():

    print(filename + ": \n" + 40 * "-")
    for name in data.ncattrs():
        print("{:20s}: {:>}".format(name, getattr(data, name)))
    print(40 * "-")

    label = input("Enter Legend for " + filename + ": ")

    time = np.array(data.variables['time']) * 1e9
    scalar = np.array(data.variables[toPlot[choice][0]])
    scalefactor = toPlot[choice][2]

    if choice == 0:
        plt.plot(time[:], (scalar[:] / scalar[0] - 1) * scalefactor, '-', label=label)
    else:
        plt.plot(time[:], scalar[:] * scalefactor, '-', label=label)

plt.xlabel('time (ns)')
plt.ylabel(toPlot[choice][1])
plt.legend(loc='best')
plt.show()
