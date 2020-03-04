#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

class Analytic:

    def __init__(self, disc, geometry):

        self.geometry = geometry

        self.Lx = float(disc['Lx'])
        self.Ly = float(disc['Ly'])

    def linearSlider(self, x , y):
        "Linear height profile"

        h1 = float(self.geometry['h1'])
        h2 = float(self.geometry['h2'])

        sx = (h2 - h1)/(self.Lx)
        sy = 0.
        return h1 + sx * x + sy * y

    def journalBearing(self, x, y):

        CR = float(self.geometry['CR'])
        eps = float(self.geometry['eps'])

        Rb = self.Lx/(2*np.pi)
<<<<<<< HEAD
        c = CR * Rb
        e = eps * c
=======
        c = 1.e-2 * Rb
        e = ex * c
>>>>>>> 5e8ea53f54da68fc03f4851ce7bcc83c7699c72e

        return e + (c - e) * np.cos(2. * np.pi * x / self.Lx)
