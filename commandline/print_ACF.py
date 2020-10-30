#!/usr/bin/env python3

import numpy as np
from fhl2d.helper.data_parser import getData
from fhl2d.helper.autocorrelation import getTimeACF

for filename, data in getData(".").items():

    print(f"Processing {filename}...")
    outname = filename.rsplit(".", 1)[0]

    dx = float(data.disc_dx)
    dy = float(data.disc_dy)
    dz = float(data.geometry_h1)
    Nx = int(data.disc_Nx)
    Ny = int(data.disc_Ny)
    Nz = 1
    dt = float(data.numerics_dt)

    lengths = np.array([Nx * dx, Ny * dy])

    dim = np.greater(np.array([Nx, Ny, Nz]), 1)
    ndim = np.sum(dim)

    k_dirs = np.array(["x", "y", "z"])

    head = "time "
    for i in ["rho", "jx", "jy"]:
        for j in k_dirs[dim]:
            head += f"C_{i}_{j} "

    unknowns = {0: "rho", 1: "jx", 2: "jy"}
    time = np.array(data.variables['time'])

    out = np.empty([len(time), 1 + 3 * ndim])
    out[:,0] = time

    for i in range(1, min(Nx, Ny) // 2):
        for j, name in unknowns.items():
            out_ac = getTimeACF(time, np.array(data.variables[name])[:,:,:,np.newaxis], dim, i)[0][:,1:]
            out[:, 1 + j * ndim: 1 + (j + 1) * ndim] = out_ac

        head2 = f"\nabs(k): {2. * np.pi / lengths * i}"
        np.savetxt(f"{outname}_{Nx}x{Ny}x{Nz}_acf{str(i).zfill(3)}.dat", out, header=head + head2)
