from dataclasses import dataclass


def resolve_source(problem, source: str):
    obj = problem
    for part in source.split('.'):
        if '[' in part:
            attr, key = part.split('[')
            obj = getattr(obj, attr)[key.strip('"]')]
        else:
            obj = getattr(obj, part)
    return obj


def categorize_registry_fields():
    """Split QUAD_FIELD_REGISTRY into nodal, scalar, and computed lists."""
    nodal, scalar, computed = [], [], []
    for name, entry in QUAD_FIELD_REGISTRY.items():
        t = entry['type']
        if t == 'nodal':
            nodal.append(name)
        elif t == 'scalar':
            scalar.append(name)
        elif t == 'computed' and entry['source'] is not None:
            computed.append(name)
    return nodal, scalar, computed


@dataclass
class FieldSpec:
    """Single source of truth for one active variable or residual."""
    name: str
    grid: str
    idx: int


VAR_GRID = {
    'jx':    'P2',
    'jy':    'P2',
    'p':     'P1',
    'E':     'P1',
    'theta': 'P1',
    'xi':    'P1',
}

RES_GRID = {
    'momentum_x': 'P2',
    'momentum_y': 'P2',
    'mass':       'P1',
    'energy':     'P1',
    'fb':         'P1',
    'R_oss':      'P1',
}

NODAL_P1 = ['rho', 'p']
NODAL_P2 = ['jx', 'jy']


# Shared arg lists for groups of computed fields with identical inputs
_ARGS_XZ  = ['rho', 'jx', 'jy', 'h', 'dh_dx', 'U_bot', 'V_bot', 'U_top', 'V_top', 'Ls', 'theta', 'd_dx_p', 'd_dy_p']
_ARGS_YZ  = ['rho', 'jx', 'jy', 'h', 'dh_dy', 'U_bot', 'V_bot', 'U_top', 'V_top', 'Ls', 'theta', 'd_dx_p', 'd_dy_p']
_ARGS_T   = ['rho', 'jx', 'jy', 'E']
_ARGS_S   = ['h', 'eta', 'rho', 'E', 'jx', 'jy', 'U_bot', 'V_bot', 'Tb_top', 'Tb_bot']

QUAD_FIELD_REGISTRY = {
    # ------------------------------------------------------------------
    # nodal: interpolated from nodal field to quad points
    # ------------------------------------------------------------------
    'h':      {'type': 'nodal', 'source': 'topo.h'},
    'dh_dx':  {'type': 'nodal', 'source': 'topo.dh_dx'},
    'dh_dy':  {'type': 'nodal', 'source': 'topo.dh_dy'},
    'eta':    {'type': 'nodal', 'source': 'viscosity.eta'},
    'Tb_top': {'type': 'nodal', 'source': 'energy.Tb_top'},
    'Tb_bot': {'type': 'nodal', 'source': 'energy.Tb_bot'},

    # ------------------------------------------------------------------
    # scalar: broadcast scalar value to all quad points
    # ------------------------------------------------------------------
    'U_bot':   {'type': 'scalar', 'source': 'geo["U_bot"]'},
    'V_bot':   {'type': 'scalar', 'source': 'geo["V_bot"]'},
    'U_top':   {'type': 'scalar', 'source': 'geo["U_top"]'},
    'V_top':   {'type': 'scalar', 'source': 'geo["V_top"]'},
    'Ls':      {'type': 'scalar', 'source': 'prop["slip_length"]'},
    'force_x': {'type': 'scalar', 'source': 'prop["force_x"]'},
    'force_y': {'type': 'scalar', 'source': 'prop["force_y"]'},

    # ------------------------------------------------------------------
    # computed: physics method called with quad field arguments
    # ------------------------------------------------------------------
    'drho_dp':   {'type': 'computed', 'source': 'pressure.drho_dp',   'args': ['p']},
    'dp_drho':   {'type': 'computed', 'source': 'pressure.dp_drho',   'args': ['rho']},
    'd2p_drho2': {'type': 'computed', 'source': 'pressure.d2p_drho2', 'args': ['rho']},

    # wall stress xz
    'tau_xz':             {'type': 'computed', 'source': 'wall_stress_xz.tau_xz',             'args': _ARGS_XZ},
    'dtau_xz_drho':       {'type': 'computed', 'source': 'wall_stress_xz.dtau_xz_drho',       'args': _ARGS_XZ},
    'dtau_xz_djx':        {'type': 'computed', 'source': 'wall_stress_xz.dtau_xz_djx',        'args': _ARGS_XZ},
    'dtau_xz_dtheta':     {'type': 'computed', 'source': 'wall_stress_xz.dtau_xz_dtheta',     'args': _ARGS_XZ},
    'tau_xz_bot':         {'type': 'computed', 'source': 'wall_stress_xz.tau_xz_bot',         'args': _ARGS_XZ},
    'dtau_xz_bot_drho':   {'type': 'computed', 'source': 'wall_stress_xz.dtau_xz_bot_drho',   'args': _ARGS_XZ},
    'dtau_xz_bot_djx':    {'type': 'computed', 'source': 'wall_stress_xz.dtau_xz_bot_djx',    'args': _ARGS_XZ},
    'dtau_xz_bot_dtheta': {'type': 'computed', 'source': 'wall_stress_xz.dtau_xz_bot_dtheta', 'args': _ARGS_XZ},

    # wall stress yz
    'tau_yz':             {'type': 'computed', 'source': 'wall_stress_yz.tau_yz',             'args': _ARGS_YZ},
    'dtau_yz_drho':       {'type': 'computed', 'source': 'wall_stress_yz.dtau_yz_drho',       'args': _ARGS_YZ},
    'dtau_yz_djy':        {'type': 'computed', 'source': 'wall_stress_yz.dtau_yz_djy',        'args': _ARGS_YZ},
    'dtau_yz_dtheta':     {'type': 'computed', 'source': 'wall_stress_yz.dtau_yz_dtheta',     'args': _ARGS_YZ},
    'tau_yz_bot':         {'type': 'computed', 'source': 'wall_stress_yz.tau_yz_bot',         'args': _ARGS_YZ},
    'dtau_yz_bot_drho':   {'type': 'computed', 'source': 'wall_stress_yz.dtau_yz_bot_drho',   'args': _ARGS_YZ},
    'dtau_yz_bot_djy':    {'type': 'computed', 'source': 'wall_stress_yz.dtau_yz_bot_djy',    'args': _ARGS_YZ},
    'dtau_yz_bot_dtheta': {'type': 'computed', 'source': 'wall_stress_yz.dtau_yz_bot_dtheta', 'args': _ARGS_YZ},

    # temperature and derivatives
    'T':       {'type': 'computed', 'source': 'energy.T_func',    'args': _ARGS_T},
    'dT_drho': {'type': 'computed', 'source': 'energy.T_grad_rho', 'args': _ARGS_T},
    'dT_djx':  {'type': 'computed', 'source': 'energy.T_grad_jx',  'args': _ARGS_T},
    'dT_djy':  {'type': 'computed', 'source': 'energy.T_grad_jy',  'args': _ARGS_T},
    'dT_dE':   {'type': 'computed', 'source': 'energy.T_grad_E',   'args': _ARGS_T},

    # wall heat flux and derivatives
    'S':       {'type': 'computed', 'source': 'energy.q_wall_sum',      'args': _ARGS_S},
    'dS_drho': {'type': 'computed', 'source': 'energy.q_wall_grad_rho', 'args': _ARGS_S},
    'dS_djx':  {'type': 'computed', 'source': 'energy.q_wall_grad_jx',  'args': _ARGS_S},
    'dS_djy':  {'type': 'computed', 'source': 'energy.q_wall_grad_jy',  'args': _ARGS_S},
    'dS_dE':   {'type': 'computed', 'source': 'energy.q_wall_grad_E',   'args': _ARGS_S},

    # OSS stabilization (computed inline in _update_oss_quad_fields, source TBD)
    'a_vec_x':         {'type': 'computed', 'source': None, 'args': ['dp_drho', 'jx']},
    'a_vec_y':         {'type': 'computed', 'source': None, 'args': ['dp_drho', 'jy']},
    'tau_a_x':         {'type': 'computed', 'source': None, 'args': ['dp_drho', 'jx', 'jy']},
    'tau_a_y':         {'type': 'computed', 'source': None, 'args': ['dp_drho', 'jx', 'jy']},
}
