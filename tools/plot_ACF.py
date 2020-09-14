#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from helper import getFile
from autocorrelation import getTimeACF


def ac_exp_cos(t, a, b):
    return np.exp(-a * t) * np.cos(b * t)


def ac_exp(t, a):
    return np.exp(-a * t)


plt.style.use('presentation')
fig, ax = plt.subplots(3,1, sharex=True)

file = getFile()[1]

# read parameters from NetCDF file
Lx = file.disc_Lx
Ly = file.disc_Ly
rho0 = file.material_rho0
eta = file.material_shear
ceta = file.material_bulk

# mean velocity of sound (internally calculated from EOS)
c = np.mean(np.array(file.variables['vSound']))

# isothermal sound absorption and kinematic viscosity
Gamma_T = 1 / (2 * rho0) * (eta + (eta / 3 + ceta))
nu = eta / rho0

# smallest wave number
q = 2 * np.pi / Lx

# Time for one pass through periodic box
tcross = Lx / c

C = getTimeACF(file)

# ??? fudge factor in time
# C[:,0] *= 2

ax[0].set_ylabel(r'$C_\rho$')
ax[0].plot(C[:,0] * 1e12, C[:,1])
ax[0].plot(C[:,0] * 1e12, C[:,2])
ax[1].set_ylabel(r'$C_{j\parallel}$')
ax[1].plot(C[:,0] * 1e12, C[:,3])
ax[1].plot(C[:,0] * 1e12, C[:,4])
ax[2].set_ylabel(r'$C_{j\perp}$')
ax[2].plot(C[:,0] * 1e12, C[:,5])
ax[2].plot(C[:,0] * 1e12, C[:,6])


C_long_ana = ac_exp_cos(C[:,0], Gamma_T * q**2, c * q)
C_trans_ana = ac_exp(C[:,0], nu * q**2)
ax[0].plot(C[:,0] * 1e12, C_long_ana, '--', color='0.7')
ax[1].plot(C[:,0] * 1e12, C_long_ana, '--', color='0.7')
ax[2].plot(C[:,0] * 1e12, C_trans_ana, '--', color='0.7')

ax[0].set_xlim(0, tcross * 1e12)
ax[2].set_xlabel(r"time (ps)")
plt.show()
