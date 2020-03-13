#!/usr/bin/env python3

import h5py
import numpy as np
import math as m
import matplotlib.pyplot as plt
import os

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
plt.figure(figsize=(12,7))

availFiles = {}
for file in os.listdir():
    i = 0
    if file.endswith('h5'):
        availFiles.update({i: file})
        i +=1

print("Available files:")
for key, val in availFiles.items():
    print("{:3d}: {:20s}".format(key, val))

files = []
check = 'y'
while check not in ['n', 'N', 'Nope']:
    userInput = input("Enter file key: ")
    files.append(h5py.File(availFiles[int(userInput)], 'r'))
    check = input("Add another file? (y/n) " )

toPlot = {0:['mass', ' (ng)', 1e12], 1:['vmax',' (m/s)', 1.], 2: ['CFL', '', 1.], 3: ['vSound', ' (m/s)', 1.], 4: ['dt', ' (ns)', 1e9]}
choice = int(input("What to plot? (0: mass x | 1: vmax | 2: CFL number | 3: speed of sound | 4: time step) "))

# f1 = h5py.File('perturb_0006.h5', 'r')
# f2 = h5py.File('perturb_0007.h5', 'r')
# f3 = h5py.File('perturb_0008.h5', 'r')
# f4 = h5py.File('journal_0004.h5', 'r')
# f5 = h5py.File('journal_0005.h5', 'r')

# files = [f1, f2]

for file in files:

    conf_opt =  file.get('/config/options')
    conf_num =  file.get('config/numerics')
    conf_disc = file.get('config/disc')

    writeInterval = conf_opt.attrs['writeInterval']
    Nx = conf_disc.attrs['Nx']
    Lx = conf_disc.attrs['Lx']

#    aVisc = conf.attrs['Lx']/conf.attrs['Nx']/conf.attrs['dt']
#    dt = conf.attrs['dt']

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
