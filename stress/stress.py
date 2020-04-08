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

        self.out.field[0] -= (4.*(U*q.field[2]-1.5*q.field[0])*(mu+lam/2.)*h.field[1] + 2.*lam*(V*q.field[2]-1.5*q.field[1])*h.field[2])/(q.field[2]*h.field[0])
        self.out.field[1] -= (4.*(V*q.field[2]-1.5*q.field[1])*(mu+lam/2.)*h.field[2] + 2.*lam*(U*q.field[2]-1.5*q.field[1])*h.field[1])/(q.field[2]*h.field[0])
        self.out.field[2] = -2.*mu*(h.field[1] * (V*q.field[2]-1.5*q.field[1]) + h.field[2] * (U*q.field[2]-1.5*q.field[0]))/(q.field[2]*h.field[0])

        return self.out
