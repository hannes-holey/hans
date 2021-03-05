import numpy as np

# global constants
R = 8.314462618


class EquationOfState:

    def __init__(self, material):
        self.material = material

    def isoT_pressure(self, rho):

        # Dowson-Higginson (with cavitation)
        if self.material['EOS'] == "DH":
            rho0 = float(self.material['rho0'])
            P0 = float(self.material['P0'])
            C1 = float(self.material['C1'])
            C2 = float(self.material['C2'])

            p = P0 + (C1 * (rho / rho0 - 1.)) / (C2 - rho / rho0)
            if 'Pcav' in self.material.keys():
                Pcav = float(self.material['Pcav'])
                rho_cav = self.isoT_density(Pcav)
                p[rho < rho_cav] = Pcav

            return p

        # Power law, (alpha = 0: ideal gas)
        elif self.material['EOS'] == "PL":
            rho0 = float(self.material['rho0'])
            P0 = float(self.material['P0'])
            alpha = float(self.material['alpha'])

            return P0 * (rho / rho0)**(1. / (1. - 0.5 * alpha))

        # Van der Waals equation
        elif self.material['EOS'] == "vdW":
            M = float(self.material['M'])
            T = float(self.material['T0'])
            a = float(self.material['a'])
            b = float(self.material['b'])
            return R * T * rho / (M - b * rho) - a * rho**2 / M**2

        # Murnaghan equation (modified Tait eq.)
        elif self.material['EOS'] == "Murnaghan":
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

        # Cavitation model Bayada and Chupin, J. Trib. 135, 2013
        elif self.material['EOS'].startswith("Bayada"):
            c_l = float(self.material["cl"])
            c_v = float(self.material["cv"])
            rho_l = float(self.material["rhol"])
            rho_v = float(self.material["rhov"])
            N = rho_v * c_v**2 * rho_l * c_l**2 * (rho_v - rho_l) / (rho_v**2 * c_v**2 - rho_l**2 * c_l**2)
            Pcav = rho_v * c_v**2 - N * np.log(rho_v**2 * c_v**2 / (rho_l**2 * c_l**2))

            alpha = self.alphaOfRho(rho)

            if np.isscalar(rho):
                if alpha < 0:
                    p = Pcav + (rho - rho_l) * c_l**2
                elif alpha >= 0 and alpha <= 1:
                    p = Pcav + N * np.log(rho_v * c_v**2 * rho / (rho_l * (rho_v * c_v**2 * (1 - alpha) + rho_l * c_l**2 * alpha)))
                else:
                    p = c_v**2 * rho

            else:
                rho_mix = rho[np.logical_and(alpha <= 1, alpha >= 0)]
                alpha_mix = alpha[np.logical_and(alpha <= 1, alpha >= 0)]

                p = c_v**2 * rho
                p[alpha < 0] = Pcav + (rho[alpha < 0] - rho_l) * c_l**2
                p[np.logical_and(alpha <= 1, alpha >= 0)] = Pcav + \
                    N * np.log(rho_v * c_v**2 * rho_mix / (rho_l * (rho_v * c_v**2 * (1 - alpha_mix) + rho_l * c_l**2 * alpha_mix)))

            return p

    def isoT_density(self, p):

        # Dowson-Higginson
        if self.material['EOS'] == "DH":
            rho0 = float(self.material['rho0'])
            P0 = float(self.material['P0'])
            C1 = float(self.material['C1'])
            C2 = float(self.material['C2'])

            return rho0 * (C1 + C2 * (p - P0)) / (C1 + p - P0)

        # Power law, (alpha = 0: ideal gas)
        elif self.material['EOS'] == "PL":
            rho0 = float(self.material['rho0'])
            P0 = float(self.material['P0'])
            alpha = float(self.material['alpha'])

            return rho0 * (p / P0)**(1. - alpha / 2.)

        elif self.material['EOS'].startswith("Bayada"):
            c_l = float(self.material["cl"])
            c_v = float(self.material["cv"])
            rho_l = float(self.material["rhol"])
            rho_v = float(self.material["rhov"])
            N = rho_v * c_v**2 * rho_l * c_l**2 * (rho_v - rho_l) / (rho_v**2 * c_v**2 - rho_l**2 * c_l**2)
            Pcav = rho_v * c_v**2 - N * np.log(rho_v**2 * c_v**2 / (rho_l**2 * c_l**2))

            if np.isscalar(p):
                if p > Pcav:
                    rho = (p - Pcav) / c_l**2 + rho_l
                elif p < c_v**2 * rho_v:
                    rho = p / c_v**2
                else:
                    rho = (c_l**2 * rho_l**3 - c_v**2 * rho_l * rho_v**2) * np.exp((p - Pcav) / N) \
                        / ((c_l**2 * rho_l**2 - c_v**2 * rho_l * rho_v) * np.exp((p - Pcav) / N) + c_v**2 * rho_v * (-rho_v + rho_l))
            else:
                p_mix = np.logical_and(p <= Pcav, p >= c_v**2 * rho_v)
                rho = p / c_v**2
                rho[p > Pcav] = (p[p > Pcav] - Pcav) / c_l**2 + rho_l
                rho[p_mix] = (c_l**2 * rho_l**3 - c_v**2 * rho_l * rho_v**2) * np.exp((p[p_mix] - Pcav) / N) \
                    / ((c_l**2 * rho_l**2 - c_v**2 * rho_l * rho_v) * np.exp((p[p_mix] - Pcav) / N) + c_v**2 * rho_v * (-rho_v + rho_l))

            return rho

        elif self.material["EOS"] == "Murnaghan":
            rho0 = float(self.material['rho0'])
            p0 = float(self.material['P0'])
            K = float(self.material['K'])
            n = float(self.material['n'])

            return rho0 * (1 + n / K * (p - p0))**(1 / n)

    def alphaOfRho(self, rho):
        rho_l = float(self.material["rhol"])
        rho_v = float(self.material["rhov"])

        return (rho - rho_l) / (rho_v - rho_l)

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

        elif self.material['EOS'].startswith("Bayada"):
            # TODO: return also for mixture and vapor
            c_squared = float(self.material['cl'])**2

        return np.sqrt(np.amax(abs(c_squared)))

    def viscosity(self, U, V, rho, height):

        # Dukler viscosity in mixture (default)
        if str(self.material["EOS"]) == "Bayada" or str(self.material["EOS"]) == "Bayada_D":
            eta_l = float(self.material["shear"])
            eta_v = float(self.material["shearv"])
            rho_v = float(self.material["rhov"])
            alpha = self.alphaOfRho(rho)
            return alpha * eta_v + (1 - alpha) * eta_l

        # McAdams viscosity in mixture
        elif str(self.material["EOS"]) == "Bayada_MA":
            eta_l = float(self.material["shear"])
            eta_v = float(self.material["shearv"])
            rho_v = float(self.material["rhov"])
            alpha = self.alphaOfRho(rho)
            M = alpha * rho_v / rho
            return eta_v * eta_l / (eta_l * M + eta_v * (1 - M))

        else:
            if "piezo" in self.material.keys():

                if str(self.material["piezo"]) == "Barus":
                    mu0 = float(self.material["shear"])
                    aB = float(self.material["aB"])
                    p = self.isoT_pressure(rho)
                    mu0 *= np.exp(aB * p)

                elif str(self.material["piezo"]) == "Vogel":
                    rho0 = float(self.material['rho0'])
                    g = float(self.material["g"])
                    mu_inf = float(self.material["mu_inf"])
                    phi_inf = float(self.material["phi_inf"])
                    BF = float(self.material["BF"])

                    phi = (rho0 / rho)**g
                    mu0 = mu_inf * np.exp(BF * phi_inf / (phi - phi_inf))
                else:
                    mu0 = float(self.material["shear"])
            else:
                mu0 = float(self.material["shear"])

            if "thinning" in self.material.keys():
                if str(self.material["thinning"]) == "Eyring":
                    tau0 = float(self.material["tau0"])
                    shear_rate = np.sqrt(U**2 + V**2) / height

                    return tau0 / shear_rate * np.arcsinh(mu0 * shear_rate / tau0)

                elif str(self.material["thinning"]) == "Carreau":

                    shear_rate = np.sqrt(U**2 + V**2) / height
                    G = float(self.material["G"])
                    a = float(self.material["a"])
                    N = float(self.material["N"])

                    return mu0 / (1 + (mu0 / G * shear_rate)**a)**(((1 / N) - 1) / a)
                else:
                    return mu0
            else:
                return mu0
