#!/usr/bin/env python3
# -*- coding: utf-8 -*-

class DowsonHigginson:

    def __init__(self, rho0, P0):
        self.rho0 = rho0
        self.P0 = rho0

    def isoT_pressure(self, rho):
        B1 = 6.6009e-10
        B2 = 2.8225e-9
        P0= 101325.
        return (rho - self.rho0)/(B1*self.rho0 - B2*(rho - self.rho0)) + P0

    def isoT_density(self, P):
        B1 = 6.6009e-10
        B2 = 2.8225e-9
        P0= 101325.
        return self.rho0 * (1. + B1*(P - P0) /(1.+B2*(P-P0)))
