import numpy as np


def dukler_mixture(rho, rho_l, rho_v, eta_l, eta_v):
    """[summary]

    [description]

    Parameters
    ----------
    rho : [type]
        [description]
    rho_l : [type]
        [description]
    rho_v : [type]
        [description]
    eta_l : [type]
        [description]
    eta_v : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """

    alpha = (rho - rho_l) / (rho_v - rho_l)
    return alpha * eta_v + (1 - alpha) * eta_l


def mc_adams_mixture(rho, rho_l, rho_v, eta_l, eta_v):
    """[summary]

    [description]

    Parameters
    ----------
    rho : [type]
        [description]
    rho_l : [type]
        [description]
    rho_v : [type]
        [description]
    eta_l : [type]
        [description]
    eta_v : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    alpha = (rho - rho_l) / (rho_v - rho_l)
    M = alpha * rho_v / rho

    return eta_v * eta_l / (eta_l * M + eta_v * (1 - M))


def barus_piezo(p, mu0, aB):
    """[summary]

    [description]

    Parameters
    ----------
    p : [type]
        [description]
    mu0 : [type]
        [description]
    aB : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    return mu0 * np.exp(aB * p)


def vogel_piezo(rho, rho0, mu_inf, phi_inf, BF, g):
    """[summary]

    [description]

    Parameters
    ----------
    rho : [type]
        [description]
    rho0 : [type]
        [description]
    mu_inf : [type]
        [description]
    phi_inf : [type]
        [description]
    BF : [type]
        [description]
    g : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    phi = (rho0 / rho)**g
    return mu_inf * np.exp(BF * phi_inf / (phi - phi_inf))


def eyring_shear(shear_rate, mu0, tau0):
    """[summary]

    [description]

    Parameters
    ----------
    shear_rate : [type]
        [description]
    mu0 : [type]
        [description]
    tau0 : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    return tau0 / shear_rate * np.arcsinh(mu0 * shear_rate / tau0)


def carreau_shear(shear_rate, mu0, mu_inf, lam, a, N):
    """[summary]

    [description]

    Parameters
    ----------
    shear_rate : [type]
        [description]
    mu0 : [type]
        [description]
    mu_inf : [type]
        [description]
    lam : [type]
        [description]
    a : [type]
        [description]
    N : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """

    return mu_inf + (mu0 - mu_inf) * (1 + (lam * shear_rate)**a)**((N - 1) / a)


def power_law_shear(shear_rate, mu0, N):
    """[summary]

    [description]

    Parameters
    ----------
    shear_rate : [type]
        [description]
    a : [type]
        [description]
    N : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """

    return mu0 * shear_rate**(N - 1)
