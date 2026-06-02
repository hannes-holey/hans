"""Analyze shape function integrals on the reference triangle.

Computes ∫ φ_i · ψ_j dΩ over the left standard triangle (0,0)-(1,0)-(0,1)
for all (i,j) node pairs, where φ and ψ are shape functions or their
derivatives from P1 or P2 elements.

Prints both the exact symbolic result and the 3-point quadrature approximation.

Usage:
    /home/qd5728/fem_taylor_hood/venv/bin/python \
        GaPFlow/solver_fem/tests/analyze_shape_int.py

Edit the CONFIG section to change element/derivative combinations.
"""

import numpy as np

# =============================================================================
# Config — edit these
# =============================================================================

# Element type for each factor: 'P1' or 'P2'
element_a = 'P2'
element_b = 'P2'

# Derivative: 'none', 'dx', or 'dy'
deriv_a = 'dx'
deriv_b = 'dx'

# =============================================================================
# Shape functions on reference triangle (0,0)-(1,0)-(0,1)
# =============================================================================

# P1: 3 nodes
P1_N = [
    lambda x, y: 1 - x - y,
    lambda x, y: x,
    lambda x, y: y,
]
P1_dNdx = [
    lambda x, y: -1.0,
    lambda x, y: 1.0,
    lambda x, y: 0.0,
]
P1_dNdy = [
    lambda x, y: -1.0,
    lambda x, y: 0.0,
    lambda x, y: 1.0,
]
P1_names = ['N0(1-x-y)', 'N1(x)', 'N2(y)']

# P2: 6 nodes
P2_N = [
    lambda x, y: (1 - x - y) * (1 - 2*x - 2*y),
    lambda x, y: x * (2*x - 1),
    lambda x, y: y * (2*y - 1),
    lambda x, y: 4 * y * (1 - x - y),
    lambda x, y: 4 * x * (1 - x - y),
    lambda x, y: 4 * x * y,
]
P2_dNdx = [
    lambda x, y: -3 + 4*x + 4*y,
    lambda x, y: 4*x - 1,
    lambda x, y: 0.0,
    lambda x, y: -4 * y,
    lambda x, y: 4 - 8*x - 4*y,
    lambda x, y: 4 * y,
]
P2_dNdy = [
    lambda x, y: -3 + 4*x + 4*y,
    lambda x, y: 0.0,
    lambda x, y: 4*y - 1,
    lambda x, y: 4 - 4*x - 8*y,
    lambda x, y: -4 * x,
    lambda x, y: 4 * x,
]
P2_names = ['N0(bl)', 'N1(br)', 'N2(tl)', 'N3(ml)', 'N4(bm)', 'N5(mm)']

# Quadrature schemes on reference triangle (0,0)-(1,0)-(0,1).
# Weights include the triangle area factor (area = 1/2), i.e. sum(w) = 1/2.
#
# Sources:
#   Dunavant, D.A. "High degree efficient symmetrical Gaussian quadrature
#   rules for the triangle", IJNME 21 (1985), 1129-1148.
#   Strang & Fix, "An Analysis of the Finite Element Method", Table 4.1.

QUAD_SCHEMES = {
    1: {  # 1 point, exact for degree 1
        'pts': np.array([[1/3, 1/3]]),
        'w':   np.array([1/2]),
    },
    3: {  # 3 points, exact for degree 2
        'pts': np.array([[1/6, 1/6],
                         [2/3, 1/6],
                         [1/6, 2/3]]),
        'w':   np.array([1/6, 1/6, 1/6]),
    },
    4: {  # 4 points, exact for degree 3
        'pts': np.array([[1/3, 1/3],
                         [1/5, 1/5],
                         [3/5, 1/5],
                         [1/5, 3/5]]),
        'w':   np.array([-27/96,
                          25/96,
                          25/96,
                          25/96]),
    },
    6: {  # 6 points, exact for degree 4
        'pts': np.array([[0.091576213509771, 0.091576213509771],
                         [0.816847572980459, 0.091576213509771],
                         [0.091576213509771, 0.816847572980459],
                         [0.445948490915965, 0.445948490915965],
                         [0.108103018168070, 0.445948490915965],
                         [0.445948490915965, 0.108103018168070]]),
        'w':   np.array([0.054975871827661,
                         0.054975871827661,
                         0.054975871827661,
                         0.111690794839006,
                         0.111690794839006,
                         0.111690794839006]),
    },
    7: {  # 7 points, exact for degree 5
        'pts': np.array([[1/3, 1/3],
                         [0.059715871789770, 0.470142064105115],
                         [0.470142064105115, 0.059715871789770],
                         [0.470142064105115, 0.470142064105115],
                         [0.797426985353087, 0.101286507323456],
                         [0.101286507323456, 0.797426985353087],
                         [0.101286507323456, 0.101286507323456]]),
        'w':   np.array([0.1125,
                         0.066197076394253,
                         0.066197076394253,
                         0.066197076394253,
                         0.062969590272414,
                         0.062969590272414,
                         0.062969590272414]),
    },
}

# Select quadrature scheme: edit this
n_quad = 4

# =============================================================================
# Helpers
# =============================================================================

def get_funcs(element, deriv):
    """Return (list_of_callables, list_of_names, n_nodes)."""
    if element == 'P1':
        names = P1_names
        if deriv == 'none':
            return P1_N, names, 3
        elif deriv == 'dx':
            return P1_dNdx, [f'd/dx {n}' for n in names], 3
        elif deriv == 'dy':
            return P1_dNdy, [f'd/dy {n}' for n in names], 3
    elif element == 'P2':
        names = P2_names
        if deriv == 'none':
            return P2_N, names, 6
        elif deriv == 'dx':
            return P2_dNdx, [f'd/dx {n}' for n in names], 6
        elif deriv == 'dy':
            return P2_dNdy, [f'd/dy {n}' for n in names], 6
    raise ValueError(f'Unknown: element={element}, deriv={deriv}')


def quad_integrate(fa, fb, scheme):
    """Quadrature of fa * fb over reference triangle using given scheme."""
    pts, w = scheme['pts'], scheme['w']
    val = 0.0
    for q in range(len(w)):
        x, y = pts[q]
        val += w[q] * fa(x, y) * fb(x, y)
    return val


def exact_integrate(fa, fb):
    """Exact symbolic integration over reference triangle using sympy."""
    from sympy import symbols, integrate, nsimplify
    x, y = symbols('x y', real=True)
    # Evaluate lambdas at symbolic args
    integrand = fa(x, y) * fb(x, y)
    result = integrate(integrate(integrand, (y, 0, 1 - x)), (x, 0, 1))
    return nsimplify(result)


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    scheme = QUAD_SCHEMES[n_quad]
    funcs_a, names_a, n_a = get_funcs(element_a, deriv_a)
    funcs_b, names_b, n_b = get_funcs(element_b, deriv_b)

    label_a = f'{element_a}' + (f' d{deriv_a}' if deriv_a != 'none' else '')
    label_b = f'{element_b}' + (f' d{deriv_b}' if deriv_b != 'none' else '')

    print(f'Integration: ∫ φ_i · ψ_j dΩ  over left triangle (0,0)-(1,0)-(0,1)')
    print(f'  φ (rows): {label_a}  ({n_a} nodes)')
    print(f'  ψ (cols): {label_b}  ({n_b} nodes)')
    print(f'  Quadrature: {n_quad} points')
    print()

    # Compute quadrature results
    Q = np.zeros((n_a, n_b))
    for i in range(n_a):
        for j in range(n_b):
            Q[i, j] = quad_integrate(funcs_a[i], funcs_b[j], scheme)

    # Print quadrature result as table
    col_w = 10
    header = ' ' * 14 + ''.join(f'{j:^{col_w}d}' for j in range(n_b))
    print(f'Quadrature ({n_quad}-point):')
    print(header)
    for i in range(n_a):
        row = ''.join(f'{Q[i,j]:>{col_w}.6f}' for j in range(n_b))
        print(f'  {i:>2d}          {row}')

    # Compute exact symbolic results
    print(f'\nExact (sympy):')
    print(header)
    for i in range(n_a):
        parts = []
        for j in range(n_b):
            val = exact_integrate(funcs_a[i], funcs_b[j])
            parts.append(f'{str(val):>{col_w}s}')
        print(f'  {i:>2d}          ' + ''.join(parts))

    # Check quadrature error
    print(f'\nQuadrature error (quad - exact):')
    print(header)
    for i in range(n_a):
        parts = []
        for j in range(n_b):
            exact = float(exact_integrate(funcs_a[i], funcs_b[j]))
            err = Q[i, j] - exact
            if abs(err) < 1e-15:
                parts.append(f'{"0":>{col_w}s}')
            else:
                parts.append(f'{err:>{col_w}.2e}')
        print(f'  {i:>2d}          ' + ''.join(parts))

    # Row/column sums (useful for checking partition of unity)
    print(f'\nRow sums (sum over ψ_j for each φ_i):')
    for i in range(n_a):
        print(f'  {i}: {Q[i,:].sum():.6f}')
    print(f'\nCol sums (sum over φ_i for each ψ_j):')
    for j in range(n_b):
        print(f'  {j}: {Q[:,j].sum():.6f}')
