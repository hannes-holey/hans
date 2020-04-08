#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from eos.eos import DowsonHigginson, PowerLaw
from field.field import VectorField

class Newtonian:

    def __init__(self, disc):

        self.out = VectorField(disc)

    def reynolds(self, q, material):

        if material['EOS'] == 'DH':
            eqOfState = DowsonHigginson(material)
        elif material['EOS'] == 'PL':
            eqOfState = PowerLaw(material)
        frac = float(material['frac'])

        self.out.fromFunctionField(eqOfState.isoT_pressure, q.field[2], 0)
        self.out.fromFunctionField(eqOfState.isoT_pressure, q.field[2], 1)

        self.out.field[0] *= -1.
        self.out.field[1] *= -1.

        self.out.addNoise(frac)

        return self.out

    def full(self, q, h, geo, material):

        U = float(geo['U'])
        V = float(geo['V'])
        mu = float(material['mu'])
        lam = float(material['lambda'])
        frac = float(material['frac'])

        if material['EOS'] == 'DH':
            eqOfState = DowsonHigginson(material)
        elif material['EOS'] == 'PL':
            eqOfState = PowerLaw(material)

        self.out.fromFunctionField(eqOfState.isoT_pressure, q.field[2], 0)
        self.out.fromFunctionField(eqOfState.isoT_pressure, q.field[2], 1)

        self.out.field[0] *= -1.
        self.out.field[1] *= -1.

        j_x = q.field[0]
        j_y = q.field[1]
        rho = q.field[2]

        h0 = h.field[0]
        hx = h.field[1]
        hy = h.field[2]

        self.out.field[0] +=  (-3*(lam + 2*mu)*(U*rho - 2*j_x)*hx - 3*lam*(V*rho - 2*j_y)*hy)/(2*h0*rho)
        self.out.field[1] +=  (-3*(lam + 2*mu)*(V*rho - 2*j_y)*hy - 3*lam*(U*rho - 2*j_x)*hx)/(2*h0*rho)
        self.out.field[2] += -3*((V*rho - 2*j_y)*hx + hy*(U*rho - 2*j_x))*mu/(2*h0*rho)

        self.out.addNoise(frac)

        return self.out
