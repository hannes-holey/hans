#!/usr/bin/env python3

import os
import numpy as np
from helper import getFiles

toPlot = {0: ['jx', r'mass flux $x$ [kg/(m$^2$s)]'],
          1: ['jy', r'mass flux $y$ [kg/(m$^2$s)]'],
          2: ['rho', r'density [kg/m$^3$]'],
          3: ['p', r'pressure (MPa)']}

files = getFiles()

for f in files.values():

    filename = f[0]
    file = f[1]

    Nx = file.Nx
    Ny = file.Ny
    dx = float(file.dx)
    dy = float(file.dy)
    dz = float(file.h1)
    dt = float(file.dt)

    f_header_disc = f"{dx:.2g} {dy:.2g} {dz:.2g} {dt:.2g}"
    f_var_cols = "time, var(jx), var(jy), var(rho)"
    f_ac_cols = "time, ac_l(jx) ac_t(jx) ac_l(jy) ac_t(jy) ac(rho) ac(rho)"

    time = np.array(file.variables["time"])
    out_var = np.empty([10, 4])
    out_ac = np.empty([len(time), 7])

    wave_num = {0: [1, 0], 1: [0, 1]}

    kx = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(Ny, d=dy)
    out_ac[:,0] = time

    for choice in np.arange(3):

        try:
            full_array = np.array(file.variables[toPlot[choice][0]], copy=False)
        except MemoryError:
            print("Not enough memory, using single precision!")
            full_array = np.array(file.variables[toPlot[choice][0]], copy=False).astype(np.float32)

        mask = np.all(np.isfinite(full_array), axis=(1,2))
        full_array = full_array[mask]
        size = len(time)

        n = 10
        for i in range(n):
            samples = (i + 1) * size // n

            cellVariance = np.var(full_array[0:samples], axis=0)
            # mean = np.mean(full_array[0:samples + 1])
            variance = np.mean(cellVariance)

            out_var[i, choice + 1] = variance
            out_var[i, 0] = time[samples - 1]
            # out_var[i,1] = mean
            # out_var[i,2] = variance

        field_fft = np.real(np.fft.fft2(full_array))

        for d in [0, 1]:
            if choice == 1:
                iky, ikx = wave_num[d]
            else:
                ikx, iky = wave_num[d]

            var = np.var(field_fft[:, ikx, iky])
            mean = np.mean(field_fft[:, ikx, iky])
            C = np.correlate(field_fft[:, ikx, iky] - mean, field_fft[:, ikx, iky] - mean, mode="full")
            C = C[C.size // 2:]
            C /= len(C)
            C /= var

            out_ac[:, 2 * choice + d + 1] = C

    np.savetxt(os.path.splitext(filename)[0] + "_stats_var.dat", out_var, header=f_header_disc + "\n" + f_var_cols)
    np.savetxt(os.path.splitext(filename)[0] + "_stats_ac.dat", out_ac, header=f_header_disc + "\n" + f_ac_cols)
