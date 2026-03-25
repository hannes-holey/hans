from tests.test_fem_2d_assembly_fd import make_problem
import numpy as np

_, solver = make_problem(4, 3, bc='dirichlet_rho')

# Get the dense Jacobian and the mass-jx block
M = solver.get_M_dense()
block = M[solver._res_slices['mass'], solver._sol_slices['jx']]

# Print rows with more than 6 nonzeros
for i in range(block.shape[0]):
    nnz_cols = np.where(np.abs(block[i]) > 1e-12)[0]
    if len(nnz_cols) > 6:
        print(f"Row {i}: {len(nnz_cols)} nnz, cols = {nnz_cols}, vals = {block[i,nnz_cols]}")

# Also print the nnz template for this block
asm = solver.assembly
# Find templates for (mass, jx, ...) keys
for key, tmpl in asm.assembly_templates.items():
    if len(key) == 4 and key[0] == 'mass' and key[1] == 'jx':
        print(f"\nTemplate key: {key}")
        nnz = tmpl['nnz']
        print(f"  nnz shape: {nnz.shape}, unique targets: {len(np.unique(nnz[nnz >=0]))}")