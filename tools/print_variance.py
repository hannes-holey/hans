#!/usr/bin/env python3

import os
import numpy as np
from helper import getData
from mean_var import getVariance_vs_time

for filename, data in getData("../data").items():

    print(f"Processing {filename}...")

    dx = float(data.disc_dx)
    dy = float(data.disc_dy)
    Nx = float(data.disc_Nx)
    Ny = float(data.disc_Ny)
    dz = float(data.geometry_h1)
    dt = float(data.numerics_dt)

    f_header_disc = f"{dx:.2g} {dy:.2g} {dz:.2g} {Nx:.2g} {Ny:.2g} {dt:.2g}"
    f_var_cols = "time, var(rho), var(jx), var(jy)"

    out_var = getVariance_vs_time(data, 10)

    np.savetxt(os.path.splitext(filename)[0] + "_var.dat", out_var, header=f_header_disc + "\n" + f_var_cols)
