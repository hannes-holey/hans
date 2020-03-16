#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

class DowsonHigginson:

    def __init__(self, material):
        self.rho0 = float(material['rho0'])
        self.P0 = float(material['P0'])
        self.B1 = float(material['B1'])
        self.B2 = float(material['B2'])

    def isoT_pressure(self, rho):
        # B1 = 6.6009e-10
        # B2 = 2.8225e-9

        return (rho - self.rho0)/(self.B1*self.rho0 - self.B2*(rho - self.rho0)) + self.P0

    def isoT_density(self, P):
        # B1 = 6.6009e-10
        # B2 = 2.8225e-9

        return self.rho0 * (1. + self.B1*(P - self.P0) /(1. + self.B2*(P - self.P0)))

    def soundSpeed(self, rho):
        # B1 = 6.6009e-10
        # B2 = 2.8225e-9

        return np.sqrt(self.B1*self.rho0/((self.B1 + self.B2)*self.rho0 - self.B2 * rho)**2)

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
            return np.sqrt(2./(2. - self.alpha)*(rho/self.rho0)**(self.alpha/(2. - self.alpha)))
