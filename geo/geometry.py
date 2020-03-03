#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

class Analytic:

    def __init__(self, disc, geometry):

        self.geometry = geometry

        self.Lx = float(disc['Lx'])
        self.Ly = float(disc['Ly'])
        self.h1 = float(geometry['h1'])
        self.h2 = float(geometry['h2'])



    def linearSlider(self, x , y):
        "Linear height profile"
        sx = (self.h2 - self.h1)/(self.Lx)
        sy = 0.
        return self.h1 + sx * x + sy * y

    def journalBearing(self, x, y):

        CR = float(self.geometry['CR'])
        eps = float(self.geometry['eps'])

        Rb = self.Lx/(2*np.pi)
        c = CR * Rb
        e = eps * c

        return c + e * np.cos(2. * np.pi * x / self.Lx)
