#!/usr/bin/env python3

import os
import numpy as np
from helper import getFiles
from autocorrelation import getTimeACF

files = getFiles()

for f in files.values():

    filename = f[0]
    file = f[1]

    dx = float(file.dx)
    dy = float(file.dy)
    dz = float(file.h1)
    dt = float(file.dt)

    f_header_disc = f"{dx:.2g} {dy:.2g} {dz:.2g} {dt:.2g}"
    f_ac_cols = "time, ac_l(jx) ac_t(jx) ac_l(jy) ac_t(jy) ac(rho) ac(rho)"

    out_ac = getTimeACF(file)

    np.savetxt(os.path.splitext(filename)[0] + "_stats_ac.dat", out_ac, header=f_header_disc + "\n" + f_ac_cols)
