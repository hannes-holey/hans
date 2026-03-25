"""Check l2g mappings."""
import sys
import numpy as np
sys.path.insert(0, '/home/qd5728/fem_taylor_hood/GaPFlow')

from GaPFlow.problem import Problem
from GaPFlow.fem_2d.global_matrix import field_to_global, global_to_field

config = '/home/qd5728/fem_taylor_hood/GaPFlow/GaPFlow/fem_2d/tests/configs/poiseuille_p2p1.yaml'
problem = Problem.from_yaml(config)
problem._pre_run()
solver = problem.solver
gi = solver.grid_idx

cols_v = solver.assembly._cols_v
M = solver.assembly._M
print(f"Assembly cols_v={cols_v}, M={M}")
print(f"GridIdx Nx_v_inner={gi.Nx_v_inner}, Ny_v_inner={gi.Ny_v_inner}")
print(f"GridIdx Nx_p_inner={gi.Nx_p_inner}, Ny_p_inner={gi.Ny_p_inner}")

l2g_v = gi.l2g_list_v
l2g_p = gi.l2g_list_p
nb_inner_v = gi.Nx_v_inner * gi.Ny_v_inner
nb_inner_p = gi.Nx_p_inner * gi.Ny_p_inner

print(f"\nl2g_v[:nb_inner_v] range: [{l2g_v[:nb_inner_v].min()}, {l2g_v[:nb_inner_v].max()}]")
print(f"l2g_p[:nb_inner_p] range: [{l2g_p[:nb_inner_p].min()}, {l2g_p[:nb_inner_p].max()}]")

print(f"\nFirst 10 l2g_v inner entries: {l2g_v[:10]}")
print(f"First 10 l2g_p inner entries: {l2g_p[:10]}")

# Check: for inner v node k, l2g_v[k] should equal k (F-order within inner grid)
# because index_mask_padded_global for inner nodes = their F-order global index
print(f"\nAre l2g_v[:nb_inner_v] == [0..nb_inner_v-1]? {np.all(l2g_v[:nb_inner_v] == np.arange(nb_inner_v))}")
print(f"Are l2g_p[:nb_inner_p] == [0..nb_inner_p-1]? {np.all(l2g_p[:nb_inner_p] == np.arange(nb_inner_p))}")

# Check field_to_global for jx, inner node 6 (x=6,y=0 in 7x7 inner grid F-order)
k6 = field_to_global(6, 0, cols_v, M)
print(f"\nfield_to_global(6, jx=0, cols_v={cols_v}, M={M}) = {k6}")
k15 = field_to_global(15, 2, cols_v, M)
print(f"field_to_global(15, rho=2, cols_v={cols_v}, M={M}) = {k15}")

# Now check global rows/cols in the COO pattern
from GaPFlow.fem_2d.assembly import P2P1AssemblyInfo
info = solver.assembly.get_petsc_info(solver.res_size)
rows = info.mat_global_rows
cols = info.mat_global_cols

# Find entries where col==96 or row==113
mask_c96 = cols == 96
mask_r113 = rows == 113
print(f"\nEntries with col=96: {np.sum(mask_c96)}")
print(f"Entries with row=113: {np.sum(mask_r113)}")
print(f"Entry (113,96) value: {solver.assembly.global_rows[mask_r113 & (cols==96)]}")

# Decode 96 and 113
fi96, rt96 = global_to_field(96, cols_v, M)
fi113, rt113 = global_to_field(113, cols_v, M)
print(f"\nglobal_to_field(96) = field_idx={fi96}, res_type={rt96} ({'jx jy rho e'.split()[rt96]})")
print(f"global_to_field(113) = field_idx={fi113}, res_type={rt113} ({'jx jy rho e'.split()[rt113]})")
