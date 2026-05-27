"""
Generate explicit solver reference data for test_fem_2d_integration.py.

Run once (takes ~10 minutes) and commit the resulting .npz files:
    python tests/generate_explicit_reference.py
"""

import numpy as np
from pathlib import Path
from GaPFlow.problem import Problem

CONFIG_DIR = Path(__file__).parent / "configs"
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

EXPLICIT_DEFAULTS = {
    "Ny": 1,
    "solver": "explicit",
    "max_it": 50000,
    "pressure_stab_alpha": 0,
    "momentum_stab_alpha": 0,
}

CASES = {
    "inclined_slider": {
        "template": "inclined_slider",
        "params": {"dt": 1e-6, "adaptive": 0, "CFL": 0.5, "rho_bc": 1.0},
    },
    "journal_bearing": {
        "template": "journal_bearing",
        "params": {"dt": 1e-10, "adaptive": 1, "CFL": 0.1, "rho_bc": 1.0},
    },
    "parabolic_slider": {
        "template": "parabolic_slider",
        "params": {"dt": 1e-10, "adaptive": 1, "CFL": 0.45, "rho_bc": 850.0},
    },
}

for name, cfg in CASES.items():
    print(f"\nRunning explicit solver: {name} ...")
    with open(CONFIG_DIR / f"{cfg['template']}.yaml") as f:
        template = f.read()
    params = {**EXPLICIT_DEFAULTS, **cfg["params"]}
    problem = Problem.from_string(template.format(**params))
    problem.run()
    rho = problem.q[0][1:-1, 0].copy()
    jx = problem.q[1][1:-1, 0].copy()
    out = DATA_DIR / f"explicit_{name}.npz"
    np.savez(out, rho=rho, jx=jx)
    print(f"  Saved {out}  rho.shape={rho.shape}")

print("\nDone.")
