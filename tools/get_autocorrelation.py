#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from helper import getFile
from scipy.optimize import curve_fit

# Plot settings
plt.style.use('presentation')
fig, ax = plt.subplots(2,2, figsize=(16,12))

# Fitting function definitions


def ac_exp_cos(t, a, b):
    return np.exp(-a * t) * np.cos(b * t)


def ac_exp(t, a):
    return np.exp(-a * t)


# Load file
filename, file = getFile()

# Read simulation parameters from file
Nx = int(file.Nx)
Ny = int(file.Ny)
dx = float(file.dx)
dy = float(file.dy)
dt = float(file.dt)
mu = float(file.shear)
ceta = float(file.bulk)
# lam = float(file.lam)
rho0 = float(file.rho0)
P0 = float(file.P0)
time = np.array(file.variables['time'])

# Choose field variable for analysis
toPlot = {0: ['jx', r'mass flux $x$ [kg/(m$^2$s)]', 1e1],       # last item in list: conversion to cgs units
          1: ['jy', r'mass flux $y$ [kg/(m$^2$s)]', 1e1],
          2: ['rho', r'density [kg/m$^3$]', 1e3],
          3: ['p', r'pressure (MPa)', 1e-6]}

# choice = 2
# dir = 0

choice = int(input("Momentum density x (0) / momentum density y (1) / mass density (2): "))
dir = int(input("Direction (0 / 1): "))

# check if enough memory available, load data
try:
    full_array = np.array(file.variables[toPlot[choice][0]], copy=False)
except MemoryError:
    print("Not enough memory, using single precision!")
    full_array = np.array(file.variables[toPlot[choice][0]], copy=False).astype(np.float32)

# get full time series of arrays, discard NaN
mask = np.all(np.isfinite(full_array), axis=(1,2))
full_array = full_array[mask]
time = time[:full_array.shape[0]]

# take only steps smaller 100 ps
# full_array = full_array[:threshold]
# time = time[:threshold]

# plot cut through initial field and time series of one cell in real space
x = (np.arange(Nx) + 0.5) * dx
ax[0, 0].plot(x, full_array[0, :, Ny // 2])
ax[0, 1].plot(time, full_array[:, Nx // 2, Ny // 2])

# 2D FFT
field_fft = np.fft.fft2(full_array)
# field_fft = np.fft.fftshift(field_fft, axes=(1,2))
field_fft = np.real(field_fft)

# Calculate wave vector components
# wave_num = {0: [Nx // 2 + 1, Ny // 2], 1: [Nx // 2, Ny // 2 + 1]}
wave_num = {0: [1, 0], 1: [0, 1]}
ikx, iky = wave_num[dir]

kx = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)
ky = 2 * np.pi * np.fft.fftfreq(Ny, d=dy)
# kx = np.fft.fftshift(kx)
# ky = np.fft.fftshift(ky)

# Plot time series for a single wave number
ax[1, 0].plot(time, field_fft[:, ikx, iky])

# Compute autocorrelation function
# var = np.var(field_fft[:, ikx, iky])
# denom = np.absolute(field_fft[0, ikx, iky])**2
C = np.correlate(field_fft[:, ikx, iky], field_fft[:, ikx, iky], mode="full") / field_fft[:, ikx, iky].size
C = C[C.size // 2:]
C /= C[0]
# C /= var

# magnitude of wave vector
# k = np.sqrt((ky[iky] - ky[Ny // 2])**2 + (kx[ikx] - kx[Nx // 2])**2)
k = np.sqrt(ky[iky]**2 + kx[ikx]**2)

# Conversion to some parameters (sth wrong ???)
# ceta = 2 / 3 * mu + lam
Gamma_T = 1 / (2 * rho0) * (mu + (mu / 3 + ceta))
nu = mu / rho0

c = np.array(file.variables['vSound'])
c = c[np.isfinite(c)]
c = np.mean(c)
# print(c)
max_time = max(Nx * dx / c, Ny * dy / c)
imax_time = int(max_time / dt)


# Choose fitting function
if dir == choice or choice == 2:
    func = ac_exp_cos
    p0 = [Gamma_T * k**2, c * k]
    flag = 0
else:
    func = ac_exp
    p0 = nu * k**2
    flag = 1

# fit
popt, pcov = curve_fit(func, time[:imax_time], C[:imax_time], p0=p0)
# popt, pcov = curve_fit(func, time[:threshold], C[:threshold],p0=p0)

# Plot AC function and fit
ax[1,1].plot(time, func(time, *popt), '--', label='fit')
ax[1,1].plot(time, C, color='C0')
# ax[1,1].vlines(max_time, -0.5, 1, linestyles='dashed', color='0.9')

# out = np.concatenate((time, C)).reshape(C.size, 2)
# np.savetxt("out.dat", out)

# c = np.sqrt(3000)
# Print fitting results
if flag == 0:
    print(Gamma_T, c)
    print(popt[0] / k**2, popt[1] / k)
    print(popt[0] / k**2 / Gamma_T - 1, popt[1] / k / c - 1)
    ac_long_ana = np.exp(-Gamma_T * k**2 * time) * np.cos(c * k * time)
    ax[1,1].plot(time, ac_long_ana, '--', label='theory')
if flag == 1:
    print(nu)
    print(popt[0] / k**2)
    print(popt[0] / k**2 / nu - 1)
    ac_trans_ana = np.exp(-nu * k**2 * time)
    ax[1,1].plot(time, ac_trans_ana, '--', label='theory')


fig_out, ax_out = plt.subplots(1,1, figsize=(8,6))
ax_out.plot(time * 1e12, func(time, *popt), '--', label='fit')
ax_out.plot(time[::5] * 1e12, C[::5], 's', color='C0')
if flag == 0:
    ax_out.plot(time * 1e12, ac_long_ana, '--', label='theory')
    ax_out.set_xlabel(r'$t$ (ps)')
    ax_out.set_ylabel(r'$C_{j\parallel}$')
elif flag == 1:
    ax_out.plot(time * 1e12, ac_trans_ana, '--', label='theory')

plt.legend()
plt.show()
