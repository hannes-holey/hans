from dataclasses import dataclass


@dataclass
class FieldSpec:
    """Single source of truth for one active variable or residual."""
    name: str
    grid: str
    idx: int


VAR_GRID = {
    'jx':    'v',
    'jy':    'v',
    'p':     'p',
    'E':     'p',
    'theta': 'p',
    'xi':    'p',
}

RES_GRID = {
    'momentum_x': 'v',
    'momentum_y': 'v',
    'mass':       'p',
    'energy':     'p',
    'fb':         'p',
    'R_oss':      'p',
}

# Nodal field name lists (P1 may be extended by theta/xi)
NODAL_P1 = ['rho', 'p', 'h', 'dh_dx', 'dh_dy']
NODAL_P2 = ['jx', 'jy']

# Quadrature field name sets
BASE_FIELDS = {
    'rho', 'jx', 'jy',
    'p', 'h', 'dh_dx', 'dh_dy', 'eta',
    'U_bot', 'V_bot', 'U_top', 'V_top', 'Ls',
    'dp_drho', 'drho_dp', 'd2p_drho2',
    'd_dx_jx', 'd_dy_jy',
    'p_prev', 'jx_prev', 'jy_prev',
    'force_x', 'force_y',
}

STRESS_FIELDS = {
    'tau_xz', 'dtau_xz_drho', 'dtau_xz_djx',
    'tau_xz_bot', 'dtau_xz_bot_drho', 'dtau_xz_bot_djx',
    'tau_yz', 'dtau_yz_drho', 'dtau_yz_djy',
    'tau_yz_bot', 'dtau_yz_bot_drho', 'dtau_yz_bot_djy',
}

ENERGY_FIELDS = {
    'E', 'Tb_top', 'Tb_bot',
    'T', 'dT_drho', 'dT_djx', 'dT_djy', 'dT_dE',
    'S', 'dS_drho', 'dS_djx', 'dS_djy', 'dS_dE',
    'E_prev',
}

CAVITATION_FIELDS = {
    'theta',
    'dtau_xz_dtheta', 'dtau_xz_bot_dtheta',
    'dtau_yz_dtheta', 'dtau_yz_bot_dtheta',
}

OSS_FIELDS = {'xi', 'a_vec_x', 'a_vec_y', 'tau_a_x', 'tau_a_y', 'one_minus_theta'}
