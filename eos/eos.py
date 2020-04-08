#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

class DowsonHigginson:

    def __init__(self, material):
        self.rho0 = float(material['rho0'])
        self.P0 = float(material['P0'])
        self.C1 = float(material['C1'])
        self.C2 = float(material['C2'])

    def isoT_pressure(self, rho):
        return self.P0 + (self.C1 * (rho/self.rho0 - 1.))/(self.C2 - rho/self.rho0)

    def isoT_density(self, P):
        return self.rho0 * (self.C1 + self.C2 * (P - self.P0))/(self.C1 + P - self.P0)

    def soundSpeed(self, rho):
        return np.sqrt(self.C1 * self.rho0 *(self.C2 -1.)*(1/rho)**2 /((self.C2 * self.rho0/rho -1.)**2))

class PowerLaw:
    def __init__(self, material):
        self.rho0 = float(material['rho0'])
        self.P0 = float(material['P0'])
        self.alpha = float(material['alpha'])

    def isoT_pressure(self, rho):
        return self.P0 *(rho/self.rho0)**(1./(1. - 0.5 * self.alpha))

    def isoT_density(self, P):
        return self.rho0 * (P/self.P0)**(1. - 0.5 * self.alpha)

    def soundSpeed(self, rho):
        if self.alpha < 2:
            return np.sqrt(-2.*self.P0*(rho/self.rho0)**(-2./(self.alpha - 2.))/((self.alpha - 2) * rho))
