#!/usr/bin/env python3

import os
import numpy as np
from helper import getFiles
from autocorrelation import getTimeACF

files = getFiles()

for f in files.values():

    filename, file = f
    print(f"Processing {filename}...")
    # filename = f[0]
    # file = f[1]

    dx = float(file.disc_dx)
    dy = float(file.disc_dy)
    dz = float(file.geometry_h1)
    Nx = float(file.disc_Nx)
    Ny = float(file.disc_Ny)
    dt = float(file.numerics_dt)

    f_header_disc = f"{dx:.2g} {dy:.2g} {dz:.2g} {Nx:.2g} {Ny:.2g} {dt:.2g}"
    f_ac_cols = "time ac(rho) ac(rho) ac_l(jx) ac_t(jx) ac_l(jy) ac_t(jy)"

    out_ac = getTimeACF(file)

    np.savetxt(os.path.splitext(filename)[0] + "_stats_ac.dat", out_ac, header=f_header_disc + "\n" + f_ac_cols)
