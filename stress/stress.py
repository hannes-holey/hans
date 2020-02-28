#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from eos.eos import DowsonHigginson
from field.field import VectorField

class Newtonian:

    def __init__(self, disc):

        self.out = VectorField(disc)

    def reynolds(self, q, material):

        self.out.fromFunctionField(DowsonHigginson(material).isoT_pressure, q.field[2], 0)
        self.out.fromFunctionField(DowsonHigginson(material).isoT_pressure, q.field[2], 1)

        # if i < 3:
        #     self.out.fromFunctionField(DowsonHigginson(material).isoT_pressure, q.field[2], 0)
        #     self.out.field[0] *= -1

        return self.out

    def average_w4(self, q, h, geo, material, i):
        pass
