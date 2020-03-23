#!/usr/bin/env python3
import os
import h5py
import numpy as np
import math as m
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

def sci_notation(num, decimal_digits=1, precision=None, exponent=None):
    """
    Returns a string representation of the scientific
    notation of the given number formatted for use with
    LaTeX or Mathtext, with specified number of significant
    decimal digits and precision (number of decimal digits
    to show). The exponent to be used can also be specified
    explicitly.
    """
    if exponent is None:
        exponent = int(m.floor(m.log10(abs(num))))
    coeff = round(num / float(10**exponent), decimal_digits)
    if precision is None:
        precision = decimal_digits

    return r"${0:.{2}f}\cdot10^{{{1:d}}}$".format(coeff, exponent, precision)

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
check = 'y'
i = 0
while check not in ['n', 'N', 'Nope']:
    userInput = input("Enter file key: ")
    files.update({i: [availFiles[int(userInput)], h5py.File(availFiles[int(userInput)], 'r')]})
    check = input("Add another file? (y/n) " )
    i += 1

# f1 = h5py.File('wavefront_0002.h5', 'r')
# f2 = h5py.File('wavefront_0005.h5', 'r')
# f3 = h5py.File('wavefront_0006.h5', 'r')
#
# files = [f1, f2, f3]

toPlot = {0:'j_x', 1:'j_y', 2:'rho', 3:'press'}
choice = int(input("What to plot? (0: maxx flux x | 1: mass flux y | 2: density | 3: pressure) "))

for file in files.values():


    # for time in times:

    conf = file[1].get('/config')
    conf_opt =  file[1].get('/config/options')
    conf_num =  file[1].get('config/numerics')
    conf_disc = file[1].get('config/disc')

    Nx = conf_disc.attrs['Nx']
    Lx = conf_disc.attrs['Lx']
    maxT = conf_num.attrs['maxT']

    time = float(input("Approximate time in ns for \'{:s}\' (max = {:.1f} ns): ".format(file[0], maxT)))

    A = {}
    for i in file[1].keys():
        if str(i) != 'config':
            A[i] = abs(file[1].get(i).attrs['time']*1e9 // 1. - time)

    flag = min(A, key=A.get)

    g = file[1].get(flag)
    d = np.array(g.get(toPlot[choice]))
    x = np.linspace(0, Lx, Nx)
    t = g.attrs['time']*1e9
    print("Actual time for \'{:s}\': {:.2f} ns".format(file[0], t))
    ax.plot(x * 1.e3, (d[:,int(d.shape[1]/2)]) * 1e-6, '-', label = r'$t = ${:.0f} µs'.format(t/1e3))

    file[1].close()

if choice == 0:
    ylab =  'mass flux x'
elif  choice ==1:
    ylab =  'mass flux y'
elif  choice ==2:
    ylab =  'density'
else:
    ylab = r'pressure (MPa)'

ax.set_xlabel('distance (mm)')
ax.set_ylabel(ylab)

ref = np.loadtxt('p_500.dat')
x_ref = np.linspace(0, 1., ref.shape[0])
ax.plot(x_ref, ref*1e-6, '--', label = r'steady-state')

ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.legend(loc = 'best')
#ax.set_title(r'$t \approx {}$ µs'.format(time*1e-3))
plt.show()
