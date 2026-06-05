import time
from pathlib import Path
from GaPFlow.problem import Problem

CONFIG_DIR = Path('../../tests/configs')

SOLVER_DEFAULTS = {
    'explicit': {'Ny': 1, 'solver': 'explicit', 'max_it': 50000, 'output_plots': False},
    'solver_fem': {'Ny': 4, 'solver': 'fem', 'dt': 1e-02, 'max_it': 100, 'output_plots': True},
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


def run_all_solvers(template, explicit_overrides=None, **common_overrides):
    explicit_overrides = explicit_overrides or {}
    results = {}
    for solver in ['explicit', 'solver_fem']:
        overrides = {**common_overrides, **(explicit_overrides if solver == 'explicit' else {})}
        rho, jx, elapsed = run_solver(template, solver, **overrides)
        results[solver] = {'rho': rho, 'jx': jx, 'time': elapsed}
    return results


def print_timing(results):
    print("Timing:")
    for solver in ['explicit', 'solver_fem']:
        print(f"  {solver:<10} {results[solver]['time']:>6.2f} s")
