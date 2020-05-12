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
Nx = int(conf_disc.attrs['Nx'])
Ny = int(conf_disc.attrs['Ny'])

toPlot = {0: ['jx', r'mass flux $x$ [kg/(m$^2$s)]', 1e1],
          1: ['jy', r'mass flux $y$ [kg/(m$^2$s)]', 1e1],
          2: ['rho', r'density [kg/m$^3$]', 1e3],
          3: ['press', r'pressure (MPa)', 1e-6]}


keys = list(file.keys())[:-1]
size = len(keys)
last = keys[-1]
members = len(file.get(last))

if members == 2:
    choice = int(input("What to plot? (0: mass flux x | 1: mass flux y | 2: density | 3: pressure) "))
else:
    choice = int(input("What to plot? (0: mass flux x | 1: mass flux y | 2: density) "))

assert choice in np.arange(0, members + 2), "Invalid choice"

# intitalize and fill array with all sampled time steps
full_array = np.empty([size, Nx, Ny])
countFaults = 0

for i in keys:

    g = file.get(i)

    # cgs units
    # d = np.array(g.get(toPlot[choice][0])) / toPlot[choice][2]

    # cgs units
    if g is not None:
        if choice == 3:
            d = np.array(g.get('p'))
        else:
            d = np.array(g.get('q')[choice]) / toPlot[choice][2]

        if np.isfinite(d).all() is False:
            countFaults += 1
        else:
            full_array[int(i)] = d

    else:
        countFaults += 1

print("Read {:d} from {:d} samples".format(size - countFaults, size))

# cellVariance = np.var(full_array, axis=0)
# variance = np.mean(cellVariance)
# mean = np.mean(full_array)
# print(mean, variance)

# --- Loop over n different sampling numbers --- #
n = 10

var = np.zeros((n,2))

for i in range(n):
    countFaults = i * size // n

    if countFaults > 1:
        cellVariance = np.var(full_array[:-countFaults], axis=0)
    else:
        cellVariance = np.var(full_array, axis=0)

    # mean cell-variance
    variance = np.mean(cellVariance)
    var[i,0] = size - countFaults
    var[i,1] = variance
    print(variance)

np.savetxt('var_' + toPlot[choice][0] + '-vs-N.dat', var, header="samples variance(SI units)", comments="")
# plt.plot(var[:,0], var[:,1] * 1e4, '-o')
# plt.yscale('log')
# plt.xscale('log')
# plt.show()


# global mean
# mean = np.mean(full_array)

# plot histogram of fluctuating quantity
# yhist, xhist, _ = plt.hist(np.mean(full_array, axis=(1,2)), bins=50)

# fit un-normalized Gaussian to the data and plot
# popt, pcov = curve_fit(gaussian, xhist[:-1], yhist, [1, mean, np.sqrt(variance)])
# x = np.linspace(xhist[0],xhist[-1], 100)
# plt.plot(x, gaussian(x, *popt), '--', color='red')
# plt.show()
