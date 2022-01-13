"""
MIT License

Copyright 2021 Hannes Holey

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


import numpy as np

# global constants
R = 8.314462618


class Material:

    def __init__(self, material):
        self.material = material

    def eos_pressure(self, rho):

        # Dowson-Higginson (with cavitation)
        if self.material['EOS'] == "DH":
            rho0 = self.material['rho0']
            P0 = self.material['P0']
            C1 = self.material['C1']
            C2 = self.material['C2']

            p = P0 + (C1 * (rho / rho0 - 1.)) / (C2 - rho / rho0)
            if 'Pcav' in self.material.keys():
                Pcav = self.material['Pcav']
                return np.maximum(p, Pcav)
            else:
                return p

        # Power law, (alpha = 0: ideal gas)
        elif self.material['EOS'] == "PL":
            rho0 = self.material['rho0']
            P0 = self.material['P0']
            alpha = self.material['alpha']

            p = P0 * (rho / rho0)**(1. / (1. - 0.5 * alpha))

            if 'Pcav' in self.material.keys():
                Pcav = self.material['Pcav']
                return np.maximum(p, Pcav)
            else:
                return p

        # Van der Waals equation
        elif self.material['EOS'] == "vdW":
            M = self.material['M']
            T = self.material['T0']
            a = self.material['a']
            b = self.material['b']

            p = R * T * rho / (M - b * rho) - a * rho**2 / M**2
            if 'Pcav' in self.material.keys():
                Pcav = self.material['Pcav']
                return np.maximum(p, Pcav)
            else:
                return p

        # Murnaghan-Tait equation (modified Tait eq.)
        elif self.material['EOS'] == "Tait":
            rho0 = self.material['rho0']
            P0 = self.material['P0']
            K = self.material['K']
            n = self.material['n']

            return K / n * ((rho / rho0)**n - 1) + P0

        # Cubic polynomial
        elif self.material['EOS'] == "cubic":
            a = self.material['a']
            b = self.material['b']
            c = self.material['c']
            d = self.material['d']

            return a * rho**3 + b * rho**2 + c * rho + d

        # Cavitation model Bayada and Chupin, J. Trib. 135, 2013
        elif self.material['EOS'].startswith("Bayada"):
            c_l = self.material["cl"]
            c_v = self.material["cv"]
            rho_l = self.material["rhol"]
            rho_v = self.material["rhov"]
            N = rho_v * c_v**2 * rho_l * c_l**2 * (rho_v - rho_l) / (rho_v**2 * c_v**2 - rho_l**2 * c_l**2)
            Pcav = rho_v * c_v**2 - N * np.log(rho_v**2 * c_v**2 / (rho_l**2 * c_l**2))

            alpha = (rho - rho_l) / (rho_v - rho_l)

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

    def eos_density(self, p):

        # Dowson-Higginson
        if self.material['EOS'] == "DH":
            rho0 = self.material['rho0']
            P0 = self.material['P0']
            C1 = self.material['C1']
            C2 = self.material['C2']

            return rho0 * (C1 + C2 * (p - P0)) / (C1 + p - P0)

        # Power law, (alpha = 0: ideal gas)
        elif self.material['EOS'] == "PL":
            rho0 = self.material['rho0']
            P0 = self.material['P0']
            alpha = self.material['alpha']

            return rho0 * (p / P0)**(1. - alpha / 2.)

        # Cavitation model Bayada and Chupin, J. Trib. 135, 2013
        elif self.material['EOS'].startswith("Bayada"):
            c_l = self.material["cl"]
            c_v = self.material["cv"]
            rho_l = self.material["rhol"]
            rho_v = self.material["rhov"]
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

        elif self.material["EOS"] == "Tait":
            rho0 = self.material['rho0']
            P0 = self.material['P0']
            K = self.material['K']
            n = self.material['n']

            return rho0 * (1 + n / K * (p - P0))**(1 / n)

    # def alphaOfRho(self, rho):
    #     rho_l = float(self.material["rhol"])
    #     rho_v = float(self.material["rhov"])
    #
    #     return (rho - rho_l) / (rho_v - rho_l)

    def eos_sound_speed(self, rho):

        # Dowson-Higginson
        if self.material['EOS'] == "DH":
            rho0 = self.material['rho0']
            C1 = self.material['C1']
            C2 = self.material['C2']
            c_squared = C1 * rho0 * (C2 - 1.) * (1 / rho)**2 / ((C2 * rho0 / rho - 1.)**2)

        # Power law, (alpha = 0: ideal gas)
        elif self.material['EOS'] == "PL":
            rho0 = self.material['rho0']
            p0 = self.material['P0']
            alpha = self.material['alpha']
            c_squared = -2. * p0 * (rho / rho0)**(-2. / (alpha - 2.)) / ((alpha - 2) * rho)

        # Van der Waals equation
        elif self.material['EOS'] == "vdW":
            M = self.material['M']
            T = self.material['T0']
            a = self.material['a']
            b = self.material['b']
            c_squared = R * T * M / (M - b * rho)**2 - 2 * a * rho / M**2

        # Tait equation (Murnaghan)
        elif self.material['EOS'] == "Tait":
            rho0 = self.material['rho0']
            p0 = self.material['P0']
            K = self.material['K']
            n = self.material['n']

            c_squared = K / rho0**n * rho**(n - 1)

        # Cubic polynomial
        elif self.material['EOS'] == "cubic":
            a = self.material['a']
            b = self.material['b']
            c = self.material['c']

            c_squared = 3 * a * rho**2 + 2 * b * rho + c

        # Cavitation model Bayada and Chupin, J. Trib. 135, 2013
        elif self.material['EOS'].startswith("Bayada"):
            rho_v = self.material["rhov"]
            rho_l = self.material["rhol"]
            c_l = self.material['cl']
            c_v = self.material['cv']
            alpha = (rho - rho_l) / (rho_v - rho_l)

            if np.isscalar(rho):
                if alpha < 0:
                    c_squared = c_l**2
                elif alpha >= 0 and alpha <= 1:
                    c_squared = rho_v * rho_l * (c_v * c_l)**2 / (alpha * rho_l * c_l**2 + (1 - alpha) * rho_v * c_v**2) / rho
                else:
                    c_squared = c_v**2

            else:
                mix = np.logical_and(alpha <= 1, alpha >= 0)
                c_squared = np.ones_like(rho) * c_v**2
                c_squared[alpha < 0] = c_l**2
                c_squared[mix] = rho_v * rho_l * (c_v * c_l)**2 / (alpha[mix] * rho_l * c_l**2 +
                                                                   (1 - alpha[mix]) * rho_v * c_v**2) / rho[mix]

        return np.sqrt(np.amax(abs(c_squared)))

    def viscosity(self, U, V, rho, height):

        # Dukler viscosity in mixture (default)
        if self.material["EOS"] == "Bayada" or self.material["EOS"] == "Bayada_D":
            eta_l = self.material["shear"]
            eta_v = self.material["shearv"]
            rho_v = self.material["rhov"]
            rho_l = self.material["rhol"]
            alpha = (rho - rho_l) / (rho_v - rho_l)

            return alpha * eta_v + (1 - alpha) * eta_l

        # McAdams viscosity in mixture
        elif self.material["EOS"] == "Bayada_MA":
            eta_l = self.material["shear"]
            eta_v = self.material["shearv"]
            rho_v = self.material["rhov"]
            rho_l = self.material["rhol"]
            alpha = (rho - rho_l) / (rho_v - rho_l)
            M = alpha * rho_v / rho

            return eta_v * eta_l / (eta_l * M + eta_v * (1 - M))

        else:
            if "piezo" in self.material.keys():

                if self.material["piezo"] == "Barus":
                    mu0 = self.material["shear"]
                    aB = self.material["aB"]
                    p = self.eos_pressure(rho)
                    mu0 *= np.exp(aB * p)

                elif self.material["piezo"] == "Vogel":
                    rho0 = self.material['rho0']
                    g = self.material["g"]
                    mu_inf = self.material["mu_inf"]
                    phi_inf = self.material["phi_inf"]
                    BF = self.material["BF"]

                    phi = (rho0 / rho)**g
                    mu0 = mu_inf * np.exp(BF * phi_inf / (phi - phi_inf))
                else:
                    mu0 = self.material["shear"]
            else:
                mu0 = self.material["shear"]

            if "thinning" in self.material.keys():
                if self.material["thinning"] == "Eyring":
                    tau0 = self.material["tau0"]
                    shear_rate = np.sqrt(U**2 + V**2) / height

                    return tau0 / shear_rate * np.arcsinh(mu0 * shear_rate / tau0)

                elif self.material["thinning"] == "Carreau":

                    shear_rate = np.sqrt(U**2 + V**2) / height
                    lam = self.material["relax"]
                    a = self.material["a"]
                    N = self.material["N"]

                    if "shearinf" in self.material.keys():
                        mu_inf = self.material["shearinf"]
                    else:
                        mu_inf = 0.

                    return mu_inf + (mu0 - mu_inf) * (1 + (lam * shear_rate)**a)**((N - 1) / a)

                elif self.material["thinning"] == "PL":
                    shear_rate = np.sqrt(U**2 + V**2) / height
                    flow_consistency_index = self.material["shear"]
                    flow_behavior_index = self.material["n"]

                    return flow_consistency_index * shear_rate**(flow_behavior_index - 1)

                else:
                    return mu0
            else:
                return mu0
