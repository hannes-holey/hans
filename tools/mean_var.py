#!/usr/bin/env python3

import numpy as np


def getVariance_vs_time(f, n):

    unknowns = {0: "jx", 1: "jy", 2: "rho"}
    out_var = np.empty([n, len(unknowns) + 1])

    for key, val in unknowns.items():

        try:
            full_array = np.array(f.variables[val], copy=False)
        except MemoryError:
            print("Not enough memory, using single precision!")
            full_array = np.array(f.variables[val], copy=False).astype(np.float32)

        mask = np.all(np.isfinite(full_array), axis=(1,2))
        full_array = full_array[mask]
        time = np.array(f.variables["time"])
        size = len(time)

        for i in range(n):
            samples = (i + 1) * size // n

            variance = np.mean(np.var(full_array[0:samples], axis=0))

            out_var[i, key + 1] = variance
            out_var[i, 0] = time[samples - 1]

    return out_var


def getVariance_field(f):
    unknowns = {0: "jx", 1: "jy", 2: "rho"}
    out_field = np.empty([3, f.Nx, f.Ny])

    for key, val in unknowns.items():

        try:
            full_array = np.array(f.variables[val], copy=False)
        except MemoryError:
            print("Not enough memory, using single precision!")
            full_array = np.array(f.variables[val], copy=False).astype(np.float32)

        mask = np.all(np.isfinite(full_array), axis=(1,2))
        full_array = full_array[mask]

        out_field[key] = np.var(full_array, axis=0)

    return out_field


def getMean_field(f):
    unknowns = {0: "jx", 1: "jy", 2: "rho"}
    out_field = np.empty([3, f.Nx, f.Ny])

    for key, val in unknowns.items():

        try:
            full_array = np.array(f.variables[val], copy=False)
        except MemoryError:
            print("Not enough memory, using single precision!")
            full_array = np.array(f.variables[val], copy=False).astype(np.float32)

        mask = np.all(np.isfinite(full_array), axis=(1,2))
        full_array = full_array[mask]

        out_field[key] = np.mean(full_array, axis=0)

    return out_field
