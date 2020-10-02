#!/usr/bin/env python3

import os
import numpy as np
from helper import getData
from autocorrelation import getTimeACF

for filename, data in getData("../data").items():

    print(f"Processing {filename}...")

    dx = float(data.disc_dx)
    dy = float(data.disc_dy)
    dz = float(data.geometry_h1)
    Nx = float(data.disc_Nx)
    Ny = float(data.disc_Ny)
    dt = float(data.numerics_dt)

    f_header_disc = f"{dx:.2g} {dy:.2g} {dz:.2g} {Nx:.2g} {Ny:.2g} {dt:.2g}"
    f_ac_cols = "time ac_rho_x ac_rho_y ac_jx_x ac_jx_y ac_jy_x ac_jy_y"

    out_ac = getTimeACF(data)

    np.savetxt(os.path.splitext(filename)[0] + "_acf.dat", out_ac, header=f_header_disc + "\n" + f_ac_cols)
