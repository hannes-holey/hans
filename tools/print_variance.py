#!/usr/bin/env python3

import os
import numpy as np
from helper import getFiles
from mean_var import getVariance_vs_time

files = getFiles()

for f in files.values():

    filename, file = f
    print(f"Processing {filename}...")
    # filename = f[0]
    # file = f[1]

    dx = float(file.disc_dx)
    dy = float(file.disc_dy)
    Nx = float(file.disc_Nx)
    Ny = float(file.disc_Ny)
    dz = float(file.geometry_h1)
    dt = float(file.numerics_dt)

    f_header_disc = f"{dx:.2g} {dy:.2g} {dz:.2g} {Nx:.2g} {Ny:.2g} {dt:.2g}"
    f_var_cols = "time, var(jx), var(jy), var(rho)"

    out_var = getVariance_vs_time(file, 10)

    np.savetxt(os.path.splitext(filename)[0] + "_stats_var.dat", out_var, header=f_header_disc + "\n" + f_var_cols)
