"""
Quick inspection script for the P2 interpolation operator.

Set the 9 square node values below and read off the 6 quadrature point
values produced by the convolution at cell (0,0).

P2 square node layout (fine grid indices):

    (0,2) --- (1,2) --- (2,2)
      2          7         3
      |                    |
    (0,1)     (1,1)     (2,1)
      4          6         8
      |                    |
    (0,0) --- (1,0) --- (2,0)
      0          5         1

Internal triangle local node layout:

    2
    |
    3     5
    |
    0 --- 4 --- 1

  t=0 (lower-left):  local [0,1,2,3,4,5] -> square [0,1,2,4,5,6]
  t=1 (upper-right): local [0,1,2,3,4,5] -> square [3,2,1,8,7,6]

Quadrature points (barycentric on reference triangle):
  q0 = (1/6, 1/6),  q1 = (2/3, 1/6),  q2 = (1/6, 2/3)
"""

import numpy as np
import muGrid
from GaPFlow.fem_2d_new.elements import TaylorHoodP2P1

# ---------------------------------------------------------------
# SET NODAL VALUES HERE  (square node index -> value)
# ---------------------------------------------------------------
nodal_values = {
    0: 1.0,   # BL corner    (0,0)
    1: 1.0,   # BR corner    (2,0)
    2: 1.0,   # TL corner    (0,2)
    3: 1.0,   # TR corner    (2,2)
    4: 1.0,   # left mid     (0,1)
    5: 1.0,   # bottom mid   (1,0)
    6: 1.0,   # center       (1,1)
    7: 1.0,   # top mid      (1,2)
    8: 1.0,   # right mid    (2,1)
}
# ---------------------------------------------------------------

elem = TaylorHoodP2P1(dx=1.0, dy=1.0)
P2 = elem.P2

# Build fine-grid field (3x3 pixels + 2 ghost cells on right)
fc = muGrid.GlobalFieldCollection([3, 3], nb_sub_pts={'quad': 6},
                                   nb_ghosts_left=[0, 0], nb_ghosts_right=[2, 2])
nf = fc.real_field('nodal', sub_pt='pixel')
qf = fc.real_field('quad',  sub_pt='quad')

# Map square node index -> fine grid (row, col) via square_node_offsets
nf.p[:] = 0.0
for sq_node, val in nodal_values.items():
    col, row = P2.square_node_offsets[sq_node]   # offsets are (x, y) = (col, row)
    nf.p[row, col] = val

elem.P2.interpolation_operator.apply(nf, qf)

result = qf.p[:, 0, 0]   # cell (0,0) is the only meaningful even cell here

print("Nodal values (square node -> value):")
for sq_node, val in nodal_values.items():
    col, row = P2.square_node_offsets[sq_node]
    print(f"  node {sq_node} ({col},{row}): {val}")

print()
print("Quad point values at cell (0,0):")
coords = elem.Quadrature.coordinates
for i in range(3):
    print(f"  t=0, q{i} {tuple(coords[i])}: {result[i]:.6f}")
for i in range(3):
    print(f"  t=1, q{i} {tuple(coords[i])}: {result[3+i]:.6f}")
