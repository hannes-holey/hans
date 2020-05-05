#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from helper import getFile
from scipy.optimize import curve_fit

plt.style.use('presentation')
plt.figure(figsize=(12,7))


def gaussian(x, a, mean, sigma):
    return a * np.exp(-((x - mean)**2 / (2 * sigma**2)))


file = getFile()

conf_disc = file.get('config/disc')

Lx = float(conf_disc.attrs['Lx'])
Nx = int(conf_disc.attrs['Nx'])
Ny = int(conf_disc.attrs['Ny'])

toPlot = {0: ['j_x', r'mass flux $x$ [kg/(m$^2$s)]', 10],
          1: ['j_y', r'mass flux $y$ [kg/(m$^2$s)]', 10],
          2: ['rho', r'density [kg/m$^3$]', 1e3],
          3: ['press', r'pressure (MPa)', 1e-6]}

choice = int(input("What to plot? (0: mass flux x | 1: mass flux y | 2: density | 3: pressure) "))


keys = list(file.keys())[:-1]

# intitalize and fill array with all sampled time steps
full_array = np.empty([len(keys), Nx, Ny])
countFaults = 0

for i in keys:

    g = file.get(i)

    # cgs units
    # d = np.array(g.get(toPlot[choice][0])) / toPlot[choice][2]

    # SI units
    if g is not None:
        d = np.array(g.get(toPlot[choice][0]))
        full_array[int(i)] = d
    else:
        countFaults += 1

print("Read {:d} from {:d} samples".format(len(keys) - countFaults, len(keys)))
# array of cell variances
cellVariance = np.var(full_array[:-countFaults], axis=0)

# mean cell-variance
variance = np.mean(cellVariance)
mean = np.mean(full_array)
print(variance)

# plot histogram of fluctuating quantity
yhist, xhist, _ = plt.hist(np.mean(full_array, axis=(1,2)), bins=50)

# fit un-normalized Gaussian to the data and plot
popt, pcov = curve_fit(gaussian, xhist[:-1], yhist, [1, mean, np.sqrt(variance)])
x = np.linspace(xhist[0],xhist[-1], 100)
plt.plot(x, gaussian(x, *popt), '--', color='red')
plt.show()
