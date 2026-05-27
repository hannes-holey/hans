"""
Compare the two assembly options for R21x on a single reference triangle:

  Option A: ∫ Ni^P2 · (∂Nj^P1/∂x) dΩ   (d_dx_resfun=True,  der_testfun=False)
  Option B: ∫ (∂Ni^P2/∂x) · Nj^P1  dΩ   (d_dx_resfun=False, der_testfun='x')

Both represent ∫ Ni · (-∂p/∂x) dΩ after IBP (with zero boundary term on periodic domain).
Uses 7-point Gaussian quadrature on the reference triangle.
"""
import numpy as np

# ---------------------------------------------------------------------------
# 7-point quadrature (same as Quadrature7Points in elements.py)
# ---------------------------------------------------------------------------
_a1, _b1 = 0.059715871789770, 0.470142064105115
_a2, _b2 = 0.797426985353087, 0.101286507323456
quad_pts = np.array([[1/3, 1/3],
                     [_a1, _b1], [_b1, _a1], [_b1, _b1],
                     [_a2, _b2], [_b2, _a2], [_b2, _b2]])
quad_w   = np.array([9.0/80.0,
                     0.066197076394253, 0.066197076394253, 0.066197076394253,
                     0.062969590272414, 0.062969590272414, 0.062969590272414])

# Element dimensions
dx, dy = 0.025, 1/30   # representative values

# ---------------------------------------------------------------------------
# P1 shape functions and x-derivatives on reference triangle
# N0 = 1-x-y,  N1 = x,  N2 = y
# ---------------------------------------------------------------------------
def N_P1(x, y):
    return np.array([1-x-y, x, y])

def dN_dx_P1(x, y):
    return np.array([-1.0, 1.0, 0.0]) / dx   # physical derivative

def dN_dy_P1(x, y):
    return np.array([-1.0, 0.0, 1.0]) / dy

# ---------------------------------------------------------------------------
# P2 shape functions and x-derivatives on reference triangle
# Standard 6-node ordering matching elements.py idx_to_std[0] = [0,1,2,4,5,6]
# node 0: corner (0,0), node 1: corner (1,0), node 2: corner (0,1)
# node 3: midpoint (0,1)-(0,0) -> not on tri0; node 4: mid (0,0)-(1,0), node 5: mid (1,0)-(0,1)
# Using standard serendipity labeling for tri0 nodes: 0,1,2,4,5,6 in square numbering
# but on the reference triangle the 6 nodes are:
#   0:(0,0), 1:(1,0), 2:(0,1), 3: mid(0,1)-(0,0)=(0,0.5), 4:mid(0,0)-(1,0)=(0.5,0), 5:mid(1,0)-(0,1)=(0.5,0.5)
# ---------------------------------------------------------------------------
def N_P2(x, y):
    N = np.empty(6)
    N[0] = (1-x-y)*(1-2*x-2*y)
    N[1] = x*(2*x-1)
    N[2] = y*(2*y-1)
    N[3] = 4*y*(1-x-y)
    N[4] = 4*x*(1-x-y)
    N[5] = 4*x*y
    return N

def dN_dx_P2(x, y):
    dN = np.empty(6)
    dN[0] = (-3 + 4*x + 4*y) / dx
    dN[1] = (4*x - 1) / dx
    dN[2] = 0.0
    dN[3] = -4*y / dx
    dN[4] = (4 - 8*x - 4*y) / dx
    dN[5] = 4*y / dx
    return dN

# ---------------------------------------------------------------------------
# Numerical integration over the reference triangle (area = 0.5, scaled by dx*dy)
# Physical area element = dx * dy * (area of reference triangle) = dx*dy*0.5
# But quadrature weights already sum to 0.5 for the unit triangle, so:
#   ∫ f dΩ ≈ dx * dy * Σ w_q * f(x_q, y_q)
# ---------------------------------------------------------------------------
area_scale = dx * dy   # reference triangle has area 0.5, weights sum to 0.5

n_P1 = 3
n_P2 = 6

# Option A: M_A[i,j] = ∫ Ni^P2 * (∂Nj^P1/∂x) dΩ
M_A = np.zeros((n_P2, n_P1))
for w, (x, y) in zip(quad_w, quad_pts):
    Ni = N_P2(x, y)          # shape (6,)
    dNj = dN_dx_P1(x, y)     # shape (3,)
    M_A += w * area_scale * np.outer(Ni, dNj)

# Option B: M_B[i,j] = ∫ (∂Ni^P2/∂x) * Nj^P1 dΩ
M_B = np.zeros((n_P2, n_P1))
for w, (x, y) in zip(quad_w, quad_pts):
    dNi = dN_dx_P2(x, y)     # shape (6,)
    Nj  = N_P1(x, y)         # shape (3,)
    M_B += w * area_scale * np.outer(dNi, Nj)

print("Option A  ∫ Ni^P2 · (∂Nj^P1/∂x) dΩ  — shape (6, 3):")
np.set_printoptions(precision=6, suppress=True, linewidth=120)
print(M_A)

print()
print("Option B  ∫ (∂Ni^P2/∂x) · Nj^P1 dΩ  — shape (6, 3):")
print(M_B)

print()
print("M_A - M_B:")
print(M_A - M_B)

print()
print(f"||M_A||        = {np.linalg.norm(M_A):.6e}")
print(f"||M_B||        = {np.linalg.norm(M_B):.6e}")
print(f"||M_A - M_B||  = {np.linalg.norm(M_A - M_B):.6e}")
print("Note: M_A and M_B are both (6x3); they are NOT transposes of each other (that would require square matrices).")
