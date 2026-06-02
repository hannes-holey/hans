"""
Finite difference Jacobian verification with Bayada EoS near cavitation.

Initialises density so that some nodes are in the liquid region (rho > rho_l)
and others in the mixture region (rho < rho_l), exercising the EoS kink.
"""
import numpy as np
from GaPFlow.problem import Problem
from GaPFlow.solver_fem import FEMSolver

# =============================================================================
# Config: Bayada EoS, small grid, periodic y, Dirichlet rho x
# =============================================================================

_CONFIG = """
options:
    output: /tmp/bayada_fd_test
    write_freq: 1000
    silent: True

grid:
    Lx: 0.09
    Ly: 0.004
    Nx: {Nx}
    Ny: {Ny}
    xE: ['D', 'N', 'N']
    xW: ['D', 'N', 'N']
    yS: ['P', 'P', 'P']
    yN: ['P', 'P', 'P']
    xW_D: 850.01
    xE_D: 850.015

geometry:
    type: inclined
    hmax: 6e-5
    hmin: 3e-5
    U: 2.0
    V: 0.0

numerics:
    solver: fem
    dt: 0.1
    tol: 1e-11
    max_it: 5

properties:
    rho0: 850.
    EOS: Bayada
    rho_l: 850.0
    rho_v: 0.019
    c_l: 1600.0
    c_v: 352.0
    shear: 0.039
    bulk: 0.0

fem_solver:
    type: newton_alpha
    equations:
        energy: False
        {term_list_entry}
"""


def make_problem(Nx=4, Ny=3, term_list=None):
    if term_list is not None:
        entry = f"term_list: {term_list}"
    else:
        entry = ""
    config = _CONFIG.format(Nx=Nx, Ny=Ny, term_list_entry=entry)
    problem = Problem.from_string(config)
    solver = problem.solver
    solver.pre_run()
    return problem, solver


def perturb_rho_across_cavitation(solver, problem):
    """Set density so that roughly half the nodes are below rho_l (cavitating)."""
    rho_l = problem.prop['rho_l']
    q = solver.get_q_nodal().copy()
    rho_sl = solver._sol_slices['rho']
    n_rho = rho_sl.stop - rho_sl.start
    # Linear ramp from 0.95*rho_l to 1.005*rho_l
    q[rho_sl] = np.linspace(0.95 * rho_l, 1.005 * rho_l, n_rho)
    solver.set_q_nodal(q)
    solver.exchange_ghosts()
    solver.update_quad()
    return q


def compute_fd_jacobian(solver, eps=1e-7):
    """Central FD Jacobian of get_R_ w.r.t. nodal DOFs."""
    q0 = solver.get_q_nodal().copy()
    n = len(q0)
    J_fd = np.zeros((n, n))

    for j in range(n):
        eps_j = max(eps, eps * abs(q0[j]))

        q_plus = q0.copy()
        q_plus[j] += eps_j
        solver.set_q_nodal(q_plus)
        solver.exchange_ghosts()
        solver.update_quad()
        R_plus = solver.get_R_().copy()

        q_minus = q0.copy()
        q_minus[j] -= eps_j
        solver.set_q_nodal(q_minus)
        solver.exchange_ghosts()
        solver.update_quad()
        R_minus = solver.get_R_().copy()

        J_fd[:, j] = (R_plus - R_minus) / (2.0 * eps_j)

    solver.set_q_nodal(q0)
    solver.exchange_ghosts()
    solver.update_quad()
    return J_fd


def rel_err(A, B):
    return np.linalg.norm(A - B) / (np.linalg.norm(B) + 1e-15)


def run_test(term_list=None, label="full"):
    problem, solver = make_problem(Nx=4, Ny=3, term_list=term_list)
    perturb_rho_across_cavitation(solver, problem)

    M = solver.get_M_dense()
    J_fd = compute_fd_jacobian(solver)

    err = rel_err(M, J_fd)
    print(f"  {label:30s}  rel_err = {err:.4e}", end="")

    # Per-block breakdown
    worst_block = ""
    worst_err = 0.0
    for res in solver.residuals:
        for var in solver.variables:
            M_b = M[solver._res_slices[res], solver._sol_slices[var]]
            J_b = J_fd[solver._res_slices[res], solver._sol_slices[var]]
            n_fd = np.linalg.norm(J_b)
            if n_fd > 1e-10:
                be = np.linalg.norm(M_b - J_b) / n_fd
                if be > worst_err:
                    worst_err = be
                    worst_block = f"{res}/{var}"

    print(f"  (worst block: {worst_block} = {worst_err:.4e})")
    return err


def run_test_liquid_only(term_list=None, label="full"):
    """Same test but with all density in pure liquid (no cavitation)."""
    problem, solver = make_problem(Nx=4, Ny=3, term_list=term_list)
    rho_l = problem.prop['rho_l']
    q = solver.get_q_nodal().copy()
    rho_sl = solver._sol_slices['rho']
    n_rho = rho_sl.stop - rho_sl.start
    # All above rho_l — safely in liquid
    q[rho_sl] = np.linspace(1.001 * rho_l, 1.01 * rho_l, n_rho)
    solver.set_q_nodal(q)
    solver.exchange_ghosts()
    solver.update_quad()

    M = solver.get_M_dense()
    J_fd = compute_fd_jacobian(solver)
    err = rel_err(M, J_fd)
    print(f"  {label:30s}  rel_err = {err:.4e}")
    return err


if __name__ == '__main__':
    print("Bayada EoS Jacobian FD test (density across cavitation boundary)")
    print("=" * 70)

    # Individual terms
    print("\nIndividual terms (across cavitation):")
    run_test(['R21x'], 'R21x (pressure grad x)')
    run_test(['R21y'], 'R21y (pressure grad y)')
    run_test(['R24x'], 'R24x (wall stress x)')
    run_test(['R24y'], 'R24y (wall stress y)')
    run_test(['R11x', 'R11y', 'R1T'], 'mass equation')

    # Full system
    print("\nFull system (across cavitation):")
    run_test(None, 'all active terms')

    # Control: pure liquid (no kink)
    print("\nControl — pure liquid (no cavitation kink):")
    run_test_liquid_only(['R21x'], 'R21x liquid only')
    run_test_liquid_only(['R21y'], 'R21y liquid only')
    run_test_liquid_only(None, 'all terms liquid only')

    # Convergence study: does FD error decrease with eps?
    print("\nFD eps convergence for R21x (across cavitation):")
    problem, solver = make_problem(Nx=4, Ny=3, term_list=['R21x'])
    perturb_rho_across_cavitation(solver, problem)
    M = solver.get_M_dense()
    for eps in [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]:
        J_fd = compute_fd_jacobian(solver, eps=eps)
        err = rel_err(M, J_fd)
        print(f"  eps={eps:.0e}  rel_err={err:.4e}")

    # Detailed: which columns (rho DOFs) have the error?
    print("\nPer-column error in momentum_x/rho block (R21x, eps=1e-7):")
    J_fd = compute_fd_jacobian(solver, eps=1e-7)
    res_sl = solver._res_slices['momentum_x']
    var_sl = solver._sol_slices['rho']
    M_b = M[res_sl, var_sl]
    J_b = J_fd[res_sl, var_sl]

    q = solver.get_q_nodal()
    rho_vals = q[var_sl]
    rho_l = problem.prop['rho_l']

    for j in range(M_b.shape[1]):
        col_err = np.linalg.norm(M_b[:, j] - J_b[:, j])
        col_norm = np.linalg.norm(J_b[:, j])
        col_rel = col_err / (col_norm + 1e-15)
        alpha_j = (rho_vals[j] - rho_l) / (problem.prop['rho_v'] - rho_l)
        if col_rel > 1e-6:
            print(f"  col {j}: rho={rho_vals[j]:.4f}  alpha={alpha_j:.6f}  "
                  f"rel_err={col_rel:.4e}  abs_err={col_err:.4e}")
