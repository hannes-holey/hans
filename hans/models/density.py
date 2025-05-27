import numpy as np


def dowson_higginson(p, rho0, P0, C1, C2):
    """[summary]

    [description]

    Parameters
    ----------
    p : [type]
        [description]
    rho0 : [type]
        [description]
    P0 : [type]
        [description]
    C1 : [type]
        [description]
    C2 : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """

    return rho0 * (C1 + C2 * (p - P0)) / (C1 + p - P0)


def power_law(p, rho0, P0, alpha):
    """[summary]

    [description]

    Parameters
    ----------
    p : [type]
        [description]
    rho0 : [type]
        [description]
    P0 : [type]
        [description]
    alpha : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    # Power law, (alpha = 0: ideal gas)
    return rho0 * (p / P0)**(1. - alpha / 2.)


def bayada_chupin(p, rho_l, rho_v, c_l, c_v):
    """[summary]

    Cavitation model Bayada and Chupin, J. Trib. 135, 2013

    Parameters
    ----------
    p : [type]
        [description]
    rho_l : [type]
        [description]
    rho_v : [type]
        [description]
    c_l : [type]
        [description]
    c_v : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """

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


def murnaghan_tait(p, rho0, P0, K, n):
    """[summary]

    [description]

    Parameters
    ----------
    p : [type]
        [description]
    rho0 : [type]
        [description]
    P0 : [type]
        [description]
    K : [type]
        [description]
    n : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """

    return rho0 * (1 + n / K * (p - P0))**(1 / n)
