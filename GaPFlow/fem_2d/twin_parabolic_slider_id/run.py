"""
Twin identical parabolic slider — Giacopini et al. (2010) benchmark.

Run generate_topography.py first to create the height field file:
    python generate_topography.py

Then run this script:
    python run.py
"""

import os
import jax.numpy as jnp
from jax import grad
import GaPFlow

os.chdir(os.path.dirname(os.path.abspath(__file__)))


def _p_mix(dens, rho_l, rho_v, c_l, c_v):
    N = (
        rho_v * c_v**2 * rho_l * c_l**2 * (rho_v - rho_l)
        / (rho_v**2 * c_v**2 - rho_l**2 * c_l**2)
    )
    Pcav = rho_v * c_v**2 - N * jnp.log(rho_v**2 * c_v**2 / (rho_l**2 * c_l**2))
    alpha = (dens - rho_l) / (rho_v - rho_l)
    denom = rho_l * (rho_v * c_v**2 * (1 - alpha) + rho_l * c_l**2 * alpha)
    return Pcav + N * jnp.log(rho_v * c_v**2 * dens / denom)


def _blend_bcs(rho_l, rho_v, c_l, c_v, eps_l, eps_r):
    args = (rho_l, rho_v, c_l, c_v)
    rho_left = rho_l - eps_l
    d1 = grad(_p_mix)
    d2 = grad(d1)
    d3 = grad(d2)
    f0 = _p_mix(rho_left, *args)
    f0p = d1(rho_left, *args)
    f0pp = d2(rho_left, *args)
    f0ppp = d3(rho_left, *args)

    N = (
        rho_v * c_v**2 * rho_l * c_l**2 * (rho_v - rho_l)
        / (rho_v**2 * c_v**2 - rho_l**2 * c_l**2)
    )
    Pcav = rho_v * c_v**2 - N * jnp.log(rho_v**2 * c_v**2 / (rho_l**2 * c_l**2))
    f1 = Pcav + eps_r * c_l**2
    f1p = c_l**2
    f1pp = 0.0
    f1ppp = 0.0

    return f0, f0p, f0pp, f0ppp, f1, f1p, f1pp, f1ppp


def bayada_chupin_c3(dens, rho_l, rho_v, c_l, c_v, eps_l=1.0, eps_r=1.0):
    """Bayada-Chupin with septic Hermite blend (C3) around rho_l."""
    N = (
        rho_v * c_v**2 * rho_l * c_l**2 * (rho_v - rho_l)
        / (rho_v**2 * c_v**2 - rho_l**2 * c_l**2)
    )
    Pcav = rho_v * c_v**2 - N * jnp.log(rho_v**2 * c_v**2 / (rho_l**2 * c_l**2))

    alpha = (dens - rho_l) / (rho_v - rho_l)
    p_mix = Pcav + N * jnp.log(
        rho_v * c_v**2 * dens
        / (rho_l * (rho_v * c_v**2 * (1 - alpha) + rho_l * c_l**2 * alpha))
    )
    p_liq = Pcav + (dens - rho_l) * c_l**2
    p_vap = c_v**2 * dens

    f0, f0p, f0pp, f0ppp, f1, f1p, f1pp, f1ppp = _blend_bcs(
        rho_l, rho_v, c_l, c_v, eps_l, eps_r
    )

    h = eps_l + eps_r
    t = (dens - (rho_l - eps_l)) / h

    H00 = 20*t**7 - 70*t**6 + 84*t**5 - 35*t**4 + 1
    H10 = 10*t**7 - 36*t**6 + 45*t**5 - 20*t**4 + t
    H20 = 2*t**7 - 7.5*t**6 + 10*t**5 - 5*t**4 + 0.5*t**2
    H30 = t**7/6 - (2/3)*t**6 + t**5 - (2/3)*t**4 + t**3/6
    H01 = -20*t**7 + 70*t**6 - 84*t**5 + 35*t**4
    H11 = 10*t**7 - 34*t**6 + 39*t**5 - 15*t**4
    H21 = -2*t**7 + 6.5*t**6 - 7*t**5 + 2.5*t**4
    H31 = t**7/6 - 0.5*t**6 + 0.5*t**5 - t**4/6

    p_blend = (H00 * f0
               + H10 * (h * f0p)
               + H20 * (h**2 * f0pp)
               + H30 * (h**3 * f0ppp)
               + H01 * f1
               + H11 * (h * f1p)
               + H21 * (h**2 * f1pp)
               + H31 * (h**3 * f1ppp))

    p = jnp.where(
        dens < rho_l - eps_l,
        p_mix,
        jnp.where(dens < rho_l + eps_r, p_blend, p_liq),
    )
    return jnp.where(dens < rho_v, p_vap, p)


problem = GaPFlow.Problem.from_yaml('twin_parabolic_slider_id.yaml')

rho_l = problem.prop['rho_l']
rho_v = problem.prop['rho_v']
c_l = problem.prop['c_l']
c_v = problem.prop['c_v']

#problem.set_eos_function(
#    lambda rho: bayada_chupin_c3(rho, rho_l, rho_v, c_l, c_v, eps_l=0.01, eps_r=0.01)
#)

problem.run()
