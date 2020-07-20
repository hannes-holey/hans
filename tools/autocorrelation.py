#!/usr/bin/env python3

import numpy as np


def getTimeACF(f):

    unknowns = {0: "jx", 1: "jy", 2: "rho"}
    time = np.array(f.variables['time'])
    wave_num = {0: [1, 0], 1: [0, 1]}

    out_ac = np.empty([len(time), len(wave_num) * 3 + 1])
    out_ac[:,0] = time

    for key, val in unknowns.items():

        try:
            full_array = np.array(f.variables[val], copy=False)
        except MemoryError:
            print("Not enough memory, using single precision!")
            full_array = np.array(f.variables[val], copy=False).astype(np.float32)

        field_fft = np.real(np.fft.fft2(full_array))

        for dir in wave_num.keys():
            if key == 1:
                iky, ikx = wave_num[dir]
            else:
                ikx, iky = wave_num[dir]

            var = np.var(field_fft[:, ikx, iky])
            mean = np.mean(field_fft[:, ikx, iky])
            C = np.correlate(field_fft[:, ikx, iky] - mean, field_fft[:, ikx, iky] - mean, mode="full")
            C = C[C.size // 2:]
            C /= len(C)
            C /= var

            out_ac[:, 2 * key + dir + 1] = C

    return out_ac
