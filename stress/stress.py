#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from eos.eos import DowsonHigginson
from field.field import ScalarField

class Newtonian:

    def __init__(self, disc):

        self.out = ScalarField(disc)

    def Reynolds(self, q, material, i):

        if i < 3:
            self.out.fromFunctionField(DowsonHigginson(material).isoT_pressure, q.field[2], 0)
            self.out.field[0] *= -1

        return self.out
