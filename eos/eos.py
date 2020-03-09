#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

class DowsonHigginson:

    def __init__(self, material):
        self.rho0 = float(material['rho0'])
        self.P0 = float(material['P0'])

    def isoT_pressure(self, rho):
        B1 = 6.6009e-10
        B2 = 2.8225e-9

        return (rho - self.rho0)/(B1*self.rho0 - B2*(rho - self.rho0)) + self.P0

    def isoT_density(self, P):
        B1 = 6.6009e-10
        B2 = 2.8225e-9

        return self.rho0 * (1. + B1*(P - self.P0) /(1.+B2*(P - self.P0)))

    def soundSpeed(self, rho):
        B1 = 6.6009e-10
        B2 = 2.8225e-9

        return np.sqrt(B1*self.rho0/((B1 + B2)*self.rho0 - B2*rho)**2)
        #return -1./rho * np.sqrt(-B1*self.rho0/((self.rho0/rho * (B1 + B2) - B2)**2))
