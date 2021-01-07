import numpy as np


class EquationOfState:

    def __init__(self, material):
        self.material = material

    def isoT_pressure(self, rho):

        # Dowson-Higginson
        if self.material['EOS'] == "DH":
            rho0 = float(self.material['rho0'])
            P0 = float(self.material['P0'])
            C1 = float(self.material['C1'])
            C2 = float(self.material['C2'])

            return P0 + (C1 * (rho / rho0 - 1.)) / (C2 - rho / rho0)

        # Power law, (alpha = 0: ideal gas)
        elif self.material['EOS'] == "PL":
            rho0 = float(self.material['rho0'])
            P0 = float(self.material['P0'])
            alpha = float(self.material['alpha'])

            return P0 * (rho / rho0)**(1. / (1. - 0.5 * alpha))

        # Tait equation (Murnaghan)
        elif self.material['EOS'] == "Tait":
            rho0 = float(self.material['rho0'])
            p0 = float(self.material['P0'])
            K = float(self.material['K'])
            n = float(self.material['n'])

            return K / n * ((rho / rho0)**n - 1) + p0

        # Cubic polynomial
        elif self.material['EOS'] == "cubic":
            a = float(self.material['a'])
            b = float(self.material['b'])
            c = float(self.material['c'])
            d = float(self.material['d'])

            return a * rho**3 + b * rho**2 + c * rho + d

        elif self.material['EOS'] == "bayada":
            return Cavitation(self.material).pressureBayada(rho)

        elif self.material['EOS'] == "bayada_nocav":
            return Cavitation(self.material).pressureLiquid(rho)

    # def isoT_density(self, p):
    #
    #     # Dowson-Higginson
    #     if self.material['EOS'] == "DH":
    #         rho0 = float(self.material['rho0'])
    #         p0 = float(self.material['P0'])
    #         C1 = float(self.material['C1'])
    #         C2 = float(self.material['C2'])
    #         return rho0 * (C1 + C2 * (p - p0)) / (C1 + p - p0)
    #
    #     # Power law, (alpha = 0: ideal gas)
    #     elif self.material['EOS'] == "PL":
    #         rho0 = float(self.material['rho0'])
    #         p0 = float(self.material['P0'])
    #         alpha = float(self.material['alpha'])
    #         return rho0 * (p / p0)**(1. - 0.5 * alpha)
    #
    #     # Tait equation (Murnaghan)
    #     elif self.material['EOS'] == "Tait":
    #         rho0 = float(self.material['rho0'])
    #         p0 = float(self.material['P0'])
    #         K = float(self.material['K'])
    #         n = float(self.material['n'])
    #
    #         return rho0 * ((p - p0) * n / K + 1.)**(1 / n)

    def soundSpeed(self, rho):

        # Dowson-Higginson
        if self.material['EOS'] == "DH":
            rho0 = float(self.material['rho0'])
            C1 = float(self.material['C1'])
            C2 = float(self.material['C2'])
            c_squared = C1 * rho0 * (C2 - 1.) * (1 / rho)**2 / ((C2 * rho0 / rho - 1.)**2)

        # Power law, (alpha = 0: ideal gas)
        elif self.material['EOS'] == "PL":
            rho0 = float(self.material['rho0'])
            p0 = float(self.material['P0'])
            alpha = float(self.material['alpha'])
            c_squared = -2. * p0 * (rho / rho0)**(-2. / (alpha - 2.)) / ((alpha - 2) * rho)

        # Tait equation (Murnaghan)
        elif self.material['EOS'] == "Tait":
            rho0 = float(self.material['rho0'])
            p0 = float(self.material['P0'])
            K = float(self.material['K'])
            n = float(self.material['n'])

            c_squared = K / rho0**n * rho**(n - 1)

        # Cubic polynomial
        elif self.material['EOS'] == "cubic":
            a = float(self.material['a'])
            b = float(self.material['b'])
            c = float(self.material['c'])

            c_squared = 3 * a * rho**2 + 2 * b * rho + c

        elif self.material['EOS'] == "bayada":
            c_squared = Cavitation(self.material).csquaredBayada(rho)

        elif self.material['EOS'] == "bayada_nocav":
            c_l = Cavitation(self.material).c_l

            if np.isscalar(rho):
                c_squared = c_l**2
            else:
                c_squared = np.ones_like(rho) * c_l**2

        return np.sqrt(np.amax(abs(c_squared)))


class Cavitation:

    def __init__(self, material):
        self.c_l = float(material["cl"])
        self.c_v = float(material["cv"])
        self.rho_l = float(material["rhol"])
        self.rho_v = float(material["rhov"])

        self.N = self.rho_v * self.c_v**2 * self.rho_l * self.c_l**2 * \
            (self.rho_v - self.rho_l) / (self.rho_v**2 * self.c_v**2 - self.rho_l**2 * self.c_l**2)
        self.Pcav = self.rho_v * self.c_v**2 - self.N * np.log10(self.rho_v**2 * self.c_v**2 / (self.rho_l**2 * self.c_l**2))

    def pressureBayada(self, rho):
        alpha = self.alphaOfRho(rho)

        p = self.pressureVapor(rho)
        p[alpha < 0] = self.pressureLiquid(rho[alpha < 0])
        p[np.logical_and(alpha <= 1, alpha >= 0)] = self.pressureMixture(rho[np.logical_and(alpha <= 1, alpha >= 0)])

        return p

    def csquaredBayada(self, rho):
        alpha = self.alphaOfRho(rho)

        if np.isscalar(rho):
            if alpha > 1:
                csquared = self.c_v**2
            elif alpha < 0:
                csquared = self.c_l**2
            else:
                csquared = self.csquaredMixture(rho)
        else:
            csquared = np.ones_like(rho) * self.c_v**2
            csquared[alpha < 0] = self.c_l**2
            csquared[np.logical_and(alpha <= 1, alpha >= 0)] = self.csquaredMixture(rho[np.logical_and(alpha <= 1, alpha >= 0)])

        return csquared

    def alphaOfRho(self, rho):
        return (rho - self.rho_l) / (self.rho_v - self.rho_l)

    def pressureLiquid(self, rho):
        return self.Pcav + (rho - self.rho_l) * self.c_l**2

    def pressureVapor(self, rho):
        return self.c_v**2 * rho

    def pressureMixture(self, rho):
        alpha = self.alphaOfRho(rho)

        return self.Pcav + self.N * np.log10(self.rho_v * self.c_v**2 * rho /
                                             (self.rho_l * (self.rho_v * self.c_v**2 * (1 - alpha) + self.rho_l * self.c_l**2 * alpha)))

    def csquaredMixture(self, rho):
        alpha = self.alphaOfRho(rho)
        return self.c_v**2 * self.rho_v * self.c_l**2 * self.rho_l / (rho * (alpha * self.c_l**2 * self.rho_l
                                                                             + (1 - alpha) * self.c_v**2 * self.rho_v))
