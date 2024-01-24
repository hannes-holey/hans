import numpy as np
from scipy.optimize import fsolve
from hans.tools import power


def solve_zmax(self, U, V, um, vm, n):
    """Calculate the location of maximum velocity in z-direction to compute the power law stresses.

    Parameters
    ----------
    U : float
        Sliding velocity of the lower surface in x-direction.
    V : float
        Sliding velocity of the lower surface in y-direction.
    Um : numpy.ndarray
        Array of mean velocities in x-direction (jx/rho).
    Vm : numpy.ndarray
        Array of mean velocities in y-direction (jy/rho).
    n : float
        Exponent

    Returns
    -------
    numpy.ndarray
        Nondimensional z-coordinate of velocity maximum in x-direction
    numpy.ndarray
        Nondimensional z-coordinate of velocity maximum in y-direction
    """

    Nx, Ny = vm.shape

    if V == 0:
        zmaxv = np.ones_like(vm) * 0.5
    else:
        vn = vm / V
        zmaxv = np.ones_like(vm)
        init = 0.5 - 1 / (12*n*(vn - 0.5))

        for i in range(Nx):
            for j in range(Ny):
                if vn[i, j] >= (1 + n) / (1 + 2 * n):
                    zmaxv[i, j] = fsolve(zmax_nleq_case1, init[i, j], args=(n, vn[i, j]))[0]
                else:
                    zmaxv[i, j] = fsolve(zmax_nleq_case2, init[i, j], args=(n, vn[i, j]))[0]

    if U == 0:
        zmaxu = np.ones_like(um) * 0.5
    else:
        un = um / U
        zmaxu = np.ones_like(um)
        init = 0.5 - 1 / (12*n*(un - 0.5))

        for i in range(Nx):
            for j in range(Ny):
                if un[i, j] >= (1 + n) / (1 + 2 * n):
                    zmaxu[i, j] = fsolve(zmax_nleq_case1, init[i, j], args=(n, un[i, j]))[0]
                else:
                    zmaxu[i, j] = fsolve(zmax_nleq_case2, init[i, j], args=(n, un[i, j]))[0]

    # upper and lower bound for asymptotic behavior
    zmaxu = np.minimum(zmaxu, 1e5)
    zmaxu = np.maximum(zmaxu, -1e5)
    zmaxv = np.minimum(zmaxv, 1e5)
    zmaxv = np.maximum(zmaxv, -1e5)

    return zmaxu, zmaxv

def approximate_zmax(self, U, V, Um, Vm, n):
    """Approximate the location of maximum velocity in z-direction to compute the power law stresses.
    Exact values have been found numerically for both cases, but a good approximation
    is given by a hyperbola with vertical asymptote at jx/rho = U/2 (Couette profile).

    Parameters
    ----------
    U : float
        Sliding velocity of the lower surface in x-direction.
    V : float
        Sliding velocity of the lower surface in y-direction.
    Um : numpy.ndarray
        Array of mean velocities in x-direction (jx/rho).
    Vm : numpy.ndarray
        Array of mean velocities in y-direction (jy/rho).

    Returns
    -------
    numpy.ndarray
        Nondimensional z-coordinate of velocity maximum in x-direction
    numpy.ndarray
        Nondimensional z-coordinate of velocity maximum in y-direction
    """

    a = 0.5
    bu = -U / (12 * n)
    cu = U / 2

    bv = -V / (12 * n)
    cv = V / 2

    if V == 0:
        zmaxv = np.ones_like(Vm) * 0.5
    else:
        zmaxv = a + bv / (Vm - cv)

    if U == 0:
        zmaxu = np.ones_like(Um) * 0.5
    else:
        zmaxu = a + bu / (Um - cu)

    # upper and lower bound for asymptotic behavior
    zmaxu = np.minimum(zmaxu, 1e5)
    zmaxu = np.maximum(zmaxu, -1e5)
    zmaxv = np.minimum(zmaxv, 1e5)
    zmaxv = np.maximum(zmaxv, -1e5)

    return zmaxu, zmaxv


def zmax_nleq_case1(zmax, n, un):
    """Definition of nonlinear equation (Case 1) to be solved for zmax with scipy.optimize.fsolve.

    Parameters
    ----------
    zmax : float
        z-location of velocity maximum
    n : float
        Power-law exponent
    un : float
        non-dimensional height-averaged velocity.

    Returns
    -------
    float
        Function value f(zmax) = 0 for root-finding.
    """

    # Case1
    return power((1+n)/(n * (-power(zmax, 1+1/n)
                             + power(1. - zmax, 1+1/n))), n) - power(((1+2*n)*(un - zmax))/(n*(power(zmax, 2+1/n)
                                                                                               + power(1. - zmax, 2+1/n))), n)


def zmax_nleq_case2(zmax, n, un):
    """Definition of nonlinear equation (Case 2) to be solved for zmax with scipy.optimize.fsolve.

    Parameters
    ----------
    zmax : float
        z-location of velocity maximum
    n : float
        Power-law exponent
    un : float
        non-dimensional height-averaged velocity.

    Returns
    -------
    float
        Function value f(zmax) = 0 for root-finding.
    """

    return 1. - (un*(1+2*n)*(power(zmax, 1+1/n)-power(zmax-1, 1+1/n))) / (power(zmax, 2+1/n)*n - power(zmax-1, 1+1/n)*(zmax*n+(1+n)))