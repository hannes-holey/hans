"""Run explicit and fem_2d solvers on three benchmark cases, save results to .npz."""

import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from GaPFlow.problem import Problem

CONFIG_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = Path(__file__).resolve().parent

SOLVER_DEFAULTS = {
    'explicit': {'Ny': 1, 'solver': 'explicit', 'max_it': 50000, 'output_plots': False},
    'fem_2d': {'Ny': 4, 'solver': 'fem', 'dt': 0.1, 'max_it': 100, 'output_plots': False},
}

CASES = {
    'inclined_slider': {
        'explicit_overrides': {'dt': 1e-6},
    },
    'journal_bearing': {
        'explicit_overrides': {'dt': 1e-10, 'adaptive': 1, 'CFL': 0.1},
    },
    'parabolic_slider': {
        'explicit_overrides': {'dt': 1e-10, 'adaptive': 1, 'CFL': 0.45},
        'common_overrides': {'rho_bc': 850.0},
    },
}


def load_template(name):
    with open(CONFIG_DIR / f'{name}.yaml', 'r') as f:
        return f.read()


def build_config(template, solver, **overrides):
    params = {'adaptive': 0, 'CFL': 0.5, 'rho_bc': 1.0}
    params.update(SOLVER_DEFAULTS[solver])
    params.update(overrides)
    return template.format(**params)


def run_solver(template, solver, **overrides):
    yaml_str = build_config(template, solver, **overrides)
    problem = Problem.from_string(yaml_str)
    t0 = time.perf_counter()
    problem.run()
    elapsed = time.perf_counter() - t0
    rho = problem.q[0][1:-1, 0].copy()
    jx = problem.q[1][1:-1, 0].copy()
    return rho, jx, elapsed


def main():
    for case_name, case_cfg in CASES.items():
        print(f"\n{'='*60}")
        print(f"Case: {case_name}")
        print(f"{'='*60}")

        template = load_template(case_name)
        common = case_cfg.get('common_overrides', {})
        explicit_ov = case_cfg.get('explicit_overrides', {})

        results = {}
        for solver in ['explicit', 'fem_2d']:
            overrides = {**common, **(explicit_ov if solver == 'explicit' else {})}
            print(f"  Running {solver}...", end=' ', flush=True)
            rho, jx, elapsed = run_solver(template, solver, **overrides)
            results[solver] = {'rho': rho, 'jx': jx}
            print(f"done ({elapsed:.2f} s)")

        Nx = len(results['explicit']['rho'])
        x_norm = np.linspace(0, 1, Nx)

        out_path = OUTPUT_DIR / f'{case_name}.npz'
        np.savez(
            out_path,
            x_norm=x_norm,
            rho_explicit=results['explicit']['rho'],
            jx_explicit=results['explicit']['jx'],
            rho_fem_2d=results['fem_2d']['rho'],
            jx_fem_2d=results['fem_2d']['jx'],
        )
        print(f"  Saved: {out_path}")


if __name__ == '__main__':
    main()
