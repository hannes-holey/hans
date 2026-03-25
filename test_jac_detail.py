"""Detailed Jacobian mismatch analysis."""
import sys
import numpy as np
sys.path.insert(0, '/home/qd5728/fem_taylor_hood/GaPFlow')

from GaPFlow.problem import Problem
from GaPFlow.fem_2d.global_matrix import global_to_field

config = '/home/qd5728/fem_taylor_hood/GaPFlow/GaPFlow/fem_2d/tests/configs/poiseuille_p2p1.yaml'
problem = Problem.from_yaml(config)
problem._pre_run()
solver = problem.solver

q0 = solver.get_q_nodal().copy()
n = len(q0)
cols_v = solver.grid_idx.Nx_v_inner
M = solver.grid_idx.Nx_p_inner

print(f"cols_v={cols_v}, M={M}, n={n}")
print(f"Nx_v_inner={solver.grid_idx.Nx_v_inner}, Ny_v_inner={solver.grid_idx.Ny_v_inner}")

# Decode global index to (field_idx, res_type, grid_x, grid_y)
def decode(g):
    fi, rt = global_to_field(g, cols_v, M)
    var = ['jx','jy','rho','e'][rt]
    if rt in (0, 1):
        x = fi // solver.grid_idx.Ny_v_inner
        y = fi % solver.grid_idx.Ny_v_inner
    else:
        x = fi // solver.grid_idx.Ny_p_inner
        y = fi % solver.grid_idx.Ny_p_inner
    return var, x, y

# Analytical Jacobian
M_coo, R0 = solver.solver_step_fun(q0)
info = solver.assembly.get_petsc_info(solver.res_size)
rows = info.mat_global_rows
cols = info.mat_global_cols

J_anal = np.zeros((n, n))
for r, c, v in zip(rows, cols, M_coo):
    J_anal[r, c] += v

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

diff = J_anal - J_fd

print("\nTop 15 largest absolute differences (row, col, anal, fd, diff):")
flat_diff = np.abs(diff).ravel()
top_idx = np.argsort(flat_diff)[-15:][::-1]
for idx in top_idx:
    r, c = divmod(int(idx), n)
    rv, rx, ry = decode(r)
    cv, cx, cy = decode(c)
    print(f"  ({r:3d},{c:3d}) res={rv}[{rx},{ry}] var={cv}[{cx},{cy}]: "
          f"anal={J_anal[r,c]:+.6f}  fd={J_fd[r,c]:+.6f}  diff={diff[r,c]:+.6f}")

solver.set_q_nodal(q0)
solver.update_quad()
