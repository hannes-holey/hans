"""FD Jacobian check for P2P1 solver."""
import sys
import numpy as np
sys.path.insert(0, '/home/qd5728/fem_taylor_hood/GaPFlow')

from GaPFlow.problem import Problem
from GaPFlow.fem_2d.solver import TaylorHoodFEMSolver
from scipy.sparse import csr_matrix

config = '/home/qd5728/fem_taylor_hood/GaPFlow/GaPFlow/fem_2d/tests/configs/poiseuille_p2p1.yaml'
problem = Problem.from_yaml(config)
problem._pre_run()
solver = problem.solver

# Get initial state
q0 = solver.get_q_nodal().copy()
n = len(q0)
print(f"System size: {n}")
print(f"nb_inner_v={solver.nb_inner_v}, nb_inner_p={solver.nb_inner_p}")
print(f"res_size={solver.res_size}")

# Assemble analytical Jacobian
M_coo, R0 = solver.solver_step_fun(q0)
print(f"R0 norm: {np.linalg.norm(R0):.6e}")
print(f"nnz: {len(M_coo)}, nonzero entries: {np.count_nonzero(M_coo)}")

# Get global rows/cols for Jacobian
from GaPFlow.fem_2d.assembly import P2P1AssemblyInfo
info = solver.assembly.get_petsc_info(solver.res_size)
rows = info.mat_global_rows
cols = info.mat_global_cols
rhs_rows = info.rhs_global_rows

print(f"Global rows range: [{rows.min()}, {rows.max()}]")
print(f"Global cols range: [{cols.min()}, {cols.max()}]")
print(f"RHS rows range: [{rhs_rows.min()}, {rhs_rows.max()}]")

# Build dense analytical Jacobian
J_anal = np.zeros((n, n))
for r, c, v in zip(rows, cols, M_coo):
    J_anal[r, c] += v

print(f"J_anal rank: {np.linalg.matrix_rank(J_anal)} (expected {n})")

# Build RHS vector
rhs_anal = np.zeros(n)
np.add.at(rhs_anal, rhs_rows, -R0)

# FD Jacobian
eps = 1e-6
J_fd = np.zeros((n, n))
for j in range(n):
    q_p = q0.copy(); q_p[j] += eps
    q_m = q0.copy(); q_m[j] -= eps
    solver.set_q_nodal(q_p); solver.update_quad()
    R_p = solver.get_R()
    solver.set_q_nodal(q_m); solver.update_quad()
    R_m = solver.get_R()
    J_fd[:, j] = (R_p - R_m) / (2 * eps)

# The FD Jacobian is dR/dq; the analytical Jacobian should also be dR/dq
# (before sign flip for RHS)
diff = J_anal - J_fd
print(f"\nJ_anal - J_fd: max abs = {np.max(np.abs(diff)):.3e}, "
      f"rel = {np.max(np.abs(diff))/(np.max(np.abs(J_fd))+1e-16):.3e}")
print(f"J_fd rank: {np.linalg.matrix_rank(J_fd)}")

# Check which rows differ most
row_errs = np.max(np.abs(diff), axis=1)
bad_rows = np.argsort(row_errs)[-5:]
print(f"\nWorst 5 rows (max abs diff): {list(zip(bad_rows, row_errs[bad_rows].round(6)))}")

# Check which columns differ most
col_errs = np.max(np.abs(diff), axis=0)
bad_cols = np.argsort(col_errs)[-5:]
print(f"Worst 5 cols (max abs diff): {list(zip(bad_cols, col_errs[bad_cols].round(6)))}")

# Reset
solver.set_q_nodal(q0)
solver.update_quad()
