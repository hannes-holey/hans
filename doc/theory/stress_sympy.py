# %% [markdown]
# # Viscous stress tensor calculation

# %%
import numpy as np
import matplotlib.pyplot as plt

from sympy import symbols, diff, Function, simplify, Eq, integrate, solveset, Rational, python, Derivative


# %%
def stress_sympy_old(silent=True):

    x, y, z = symbols('x y z')
    Ut, Ub, Vt, Vb = symbols('Ut Ub Vt Vb')
    zeta, eta = symbols('zeta eta')
    a, b, c = symbols('a b c')

    h = Function('h')

    # Velocity profiles
    u = (Ut - Ub) * z /  h(x, y) + Ub + a * z * (h(x, y) - z)
    v = (Vt - Vb) * z /  h(x, y) + Vb + b * z * (h(x, y) - z)
    w = c * z**2 * (h(x, y) - z)**2

    # Solve for a, b, c (Poiseuille terms)
    jx, jy, jz, rho = symbols('jx jy jz rho')

    eq1 = Eq(jx, rho / h(x, y) * integrate(u, (z, 0, h(x, y))))
    eq2 = Eq(jy, rho / h(x, y) * integrate(v, (z, 0, h(x, y))))
    eq3 = Eq(jz, rho / h(x, y) * integrate(w, (z, 0, h(x, y))))
    
    _a, = solveset(eq1, a)
    _b, = solveset(eq2, b)
    _c, = solveset(eq3, c)
    
    # Strain gradient
    du_dx = diff(u.subs(a, _a), x)
    du_dy = diff(u.subs(a, _a), y)
    du_dz = diff(u.subs(a, _a), z)

    dv_dx = diff(v.subs(b, _b), x)
    dv_dy = diff(v.subs(b, _b), y)
    dv_dz = diff(v.subs(b, _b), z)

    dw_dx = diff(w.subs(c, _c), x)
    dw_dy = diff(w.subs(c, _c), y)
    dw_dz = diff(w.subs(c, _c), z)

    Dxx = du_dx
    Dyy = dv_dy
    Dzz = dw_dz
    Dxy = (du_dy + dv_dx) / 2
    Dxz = (du_dz + dw_dx) / 2 
    Dyz = (dv_dz + dw_dy) / 2

    # Viscous Stress
    trD = Dxx + Dyy + Dzz

    Txx = (zeta - Rational(2, 3) * eta) * trD + 2 * eta * Dxx
    Tyy = (zeta - Rational(2, 3) * eta) * trD + 2 * eta * Dyy
    Tzz = (zeta - Rational(2, 3) * eta) * trD + 2 * eta * Dzz
    Txy = 2 * eta * Dxy
    Txz = 2 * eta * Dxz
    Tyz = 2 * eta * Dyz

    Txx = Txx.subs({a: _a, b: _b, c: _c})
    Tyy = Tyy.subs({a: _a, b: _b, c: _c})
    Tzz = Tzz.subs({a: _a, b: _b, c: _c})
    Txy = Txy.subs({a: _a, b: _b, c: _c})
    Txz = Txz.subs({a: _a, b: _b, c: _c})
    Tyz = Tyz.subs({a: _a, b: _b, c: _c})

    # Average stress
    Txx_avg = integrate(Txx, (z, 0, h(x,y))) / h(x, y)
    Tyy_avg = integrate(Tyy, (z, 0, h(x,y))) / h(x, y)
    Txy_avg = integrate(Txy, (z, 0, h(x,y))) / h(x, y)

    # Some replacements for printing
    H, Hx, Hy = symbols('h[0] h[1] h[2]')
    q0, q1, q2 = symbols('q[0] q[1] q[2]')
    U, V = symbols('U V')

    replacements = {
        Derivative(h(x, y), x): Hx,
        Derivative(h(x, y), y): Hy,
        h(x, y): H, 
        rho: q0,
        jx: q1,
        jy: q2,
        jz: 0,
        Ub: U, 
        Vb: V,
        Vt: 0,
        Ut: 0,
        }

    if not silent:
        print_all(u, v, Txx_avg, Tyy_avg, Txy_avg, Txx, Tyy, Tzz, Tyz, Txz, Txy, replacements)

# %% [markdown]
# This is a markdown cell with LaTeX $u(x)=ax^2 + bx + c$


# %%
def stress_sympy_new(silent=True):

    x, y, z = symbols('x y z')
    Ut, Ub, Vt, Vb = symbols('Ut Ub Vt Vb')
    zeta, eta = symbols('zeta eta')
    a, b, c = symbols('a b c')

    h = Function('h')

    # Velocity profiles
    u = (Ut - Ub) * z /  h(x, y) + Ub + a * z * (h(x, y) - z)
    v = (Vt - Vb) * z /  h(x, y) + Vb + b * z * (h(x, y) - z)
    w = c * z**2 * (h(x, y) - z)**2

    # Solve for a, b, c (Poiseuille terms)
    # jx, jy, jz, rho = symbols('jx jy jz rho')

    jx = Function('jx')
    jy = Function('jy')
    jz = Function('jz')
    rho = Function('rho')

    eq1 = Eq(jx(x, y), rho(x, y) / h(x, y) * integrate(u, (z, 0, h(x, y))))
    eq2 = Eq(jy(x, y), rho(x, y) / h(x, y) * integrate(v, (z, 0, h(x, y))))
    eq3 = Eq(jz(x, y), rho(x, y) / h(x, y) * integrate(w, (z, 0, h(x, y))))
    
    _a, = solveset(eq1, a)
    _b, = solveset(eq2, b)
    _c, = solveset(eq3, c)
    
    # Strain gradient
    du_dx = diff(u.subs(a, _a), x)
    du_dy = diff(u.subs(a, _a), y)
    du_dz = diff(u.subs(a, _a), z)

    dv_dx = diff(v.subs(b, _b), x)
    dv_dy = diff(v.subs(b, _b), y)
    dv_dz = diff(v.subs(b, _b), z)

    dw_dx = diff(w.subs(c, _c), x)
    dw_dy = diff(w.subs(c, _c), y)
    dw_dz = diff(w.subs(c, _c), z)

    Dxx = du_dx
    Dyy = dv_dy
    Dzz = dw_dz
    Dxy = (du_dy + dv_dx) / 2
    Dxz = (du_dz + dw_dx) / 2 
    Dyz = (dv_dz + dw_dy) / 2

    # Viscous Stress
    trD = Dxx + Dyy + Dzz

    Txx = (zeta - Rational(2, 3) * eta) * trD + 2 * eta * Dxx
    Tyy = (zeta - Rational(2, 3) * eta) * trD + 2 * eta * Dyy
    Tzz = (zeta - Rational(2, 3) * eta) * trD + 2 * eta * Dzz
    Txy = 2 * eta * Dxy
    Txz = 2 * eta * Dxz
    Tyz = 2 * eta * Dyz

    Txx = Txx.subs({a: _a, b: _b, c: _c})
    Tyy = Tyy.subs({a: _a, b: _b, c: _c})
    Tzz = Tzz.subs({a: _a, b: _b, c: _c})
    Txy = Txy.subs({a: _a, b: _b, c: _c})
    Txz = Txz.subs({a: _a, b: _b, c: _c})
    Tyz = Tyz.subs({a: _a, b: _b, c: _c})

    # Average stress
    Txx_avg = integrate(Txx, (z, 0, h(x,y))) / h(x, y)
    Tyy_avg = integrate(Tyy, (z, 0, h(x,y))) / h(x, y)
    Txy_avg = integrate(Txy, (z, 0, h(x,y))) / h(x, y)



    if not silent:
        # Some replacements for printing
        H, Hx, Hy = symbols('h[0] h[1] h[2]')
        q0, q1, q2 = symbols('q[0] q[1] q[2]')
        dqx0, dqx1, dqx2 = symbols('dqx[0] dqx[1] dqx[2]')
        dqy0, dqy1, dqy2 = symbols('dqy[0] dqy[1] dqy[2]')
        q1q0, q1, q2 = symbols('q[0] q[1] q[2]')
        U, V = symbols('U V')

        replacements = {
            Derivative(h(x, y), x): Hx,
            Derivative(h(x, y), y): Hy,
            h(x, y): H, 
            rho(x, y): q0,
            jx(x, y): q1,
            jy(x, y): q2,
            jz(x, y): 0,
            Derivative(jx(x, y), x): dqx0,
            Derivative(jx(x, y), y): dqy0,
            Derivative(jy(x, y), x): dqx1,
            Derivative(jy(x, y), y): dqy1,
            Derivative(jz(x, y), x): dqx2,
            Derivative(jz(x, y), y): dqy2,
            Ub: U, 
            Vb: V,
            Vt: 0,
            Ut: 0,
            }

        print_all(u, v, Txx_avg, Tyy_avg, Txy_avg, Txx, Tyy, Tzz, Tyz, Txz, Txy, replacements)

# %% [markdown]
# This is another markdown cell

# %%
def print_all(u, v, Txx_avg, Tyy_avg, Txy_avg, Txx, Tyy, Tzz, Tyz, Txz, Txy, replacements):

    z = symbols('z')
    H = symbols('h[0]')

    print('\n---\n')
    print('Wall stress (bottom):')

    # Bottom
    print('tau_xx = ', end='')
    print(simplify(Txx.subs(replacements).subs(z, 0)))
    print('')

    print('tau_yy = ', end='')
    print(simplify(Tyy.subs(replacements).subs(z, 0)))
    print('')

    print('tau_zz = ', end='')
    print(simplify(Tzz.subs(replacements).subs(z, 0)))
    print('')

    print('tau_xy = ', end='')
    print(simplify(Txy.subs(replacements).subs(z, 0)))
    print('')

    print('tau_xz = ', end='')
    print(simplify(Txz.subs(replacements).subs(z, 0)))
    print('')

    print('tau_yz = ', end='')
    print(simplify(Tyz.subs(replacements).subs(z, 0)))
    print('')

    print('\n---\n')
    print('Wall stress (top):')

    # top
    print('tau_xx = ', end='')
    print(simplify(Txx.subs(replacements).subs(z, H)))
    print('')

    print('tau_yy = ', end='')
    print(simplify(Tyy.subs(replacements).subs(z, H)))
    print('')

    print('tau_zz = ', end='')
    print(simplify(Tzz.subs(replacements).subs(z, H)))
    print('')

    print('tau_xy = ', end='')
    print(simplify(Txy.subs(replacements).subs(z, H)))
    print('')

    print('tau_xz = ', end='')
    print(simplify(Txz.subs(replacements).subs(z, H)))
    print('')

    print('tau_yz = ', end='')
    print(simplify(Tyz.subs(replacements).subs(z, H)))
    print('')


    # Stress profiles (z)
    print('\n---\n')
    print('Stress profiles:')

    print('tau_xx = ', end='')
    print(simplify(Txx.subs(replacements)))
    print('')

    print('tau_yy = ', end='')
    print(simplify(Tyy.subs(replacements)))
    print('')

    print('tau_zz = ', end='')
    print(simplify(Tzz.subs(replacements)))
    print('')

    print('tau_xy = ', end='')
    print(simplify(Txy.subs(replacements)))
    print('')

    print('tau_xz = ', end='')
    print(simplify(Txz.subs(replacements)))
    print('')

    print('tau_yz = ', end='')
    print(simplify(Tyz.subs(replacements)))
    print('')

    print('\n---\n')
    print('Average stress:')
    
    print('tau_xx_avg = ', end='')
    print(simplify(Txx_avg.subs(replacements)))
    print('')

    print('tau_yy_avg = ', end='')
    print(simplify(Tyy_avg.subs(replacements)))
    print('')

    print('tau_xy_avg = ', end='')
    print(simplify(Txy_avg.subs(replacements)))

    print('\n---\n')




def stress_profiles(z, h, q, dqx=None, dqy=None, U=1., V=0, eta=1, zeta=1.):

    # From output of print_all

    if dqx is None and dqy is None:
        old = True
    else:
        old = False

    if not old:

        tau_xx = z*(6*eta*(U*h[0]*h[1]*q[0]**2 - 3*h[0]*q[0]*(-2*dqx[0]*(h[0] - z) + h[1]*(U*q[0] - 2*q[1])) + 6*h[1]*q[0]*(h[0] - z)*(U*q[0] - 2*q[1])) - (2*eta - 3*zeta)*(h[0]*q[0]**2*(U*h[1] + V*h[2]) - 3*h[0]*q[0]*(-2*dqx[0]*(h[0] - z) - 2*dqy[1]*(h[0] - z) + h[1]*(U*q[0] - 2*q[1]) + h[2]*(V*q[0] - 2*q[2])) + 6*q[0]*(h[0] - z)*(h[1]*(U*q[0] - 2*q[1]) + h[2]*(V*q[0] - 2*q[2]))))/(3*h[0]**3*q[0]**2)
        
        tau_yy = z*(6*eta*(V*h[0]*h[2]*q[0]**2 - 3*h[0]*q[0]*(-2*dqy[1]*(h[0] - z) + h[2]*(V*q[0] - 2*q[2])) + 6*h[2]*q[0]*(h[0] - z)*(V*q[0] - 2*q[2])) - (2*eta - 3*zeta)*(h[0]*q[0]**2*(U*h[1] + V*h[2]) - 3*h[0]*q[0]*(-2*dqx[0]*(h[0] - z) - 2*dqy[1]*(h[0] - z) + h[1]*(U*q[0] - 2*q[1]) + h[2]*(V*q[0] - 2*q[2])) + 6*q[0]*(h[0] - z)*(h[1]*(U*q[0] - 2*q[1]) + h[2]*(V*q[0] - 2*q[2]))))/(3*h[0]**3*q[0]**2)
        
        tau_zz = -z*(2*eta - 3*zeta)*(h[0]*q[0]**2*(U*h[1] + V*h[2]) - 3*h[0]*q[0]*(-2*dqx[0]*(h[0] - z) - 2*dqy[1]*(h[0] - z) + h[1]*(U*q[0] - 2*q[1]) + h[2]*(V*q[0] - 2*q[2])) + 6*q[0]*(h[0] - z)*(h[1]*(U*q[0] - 2*q[1]) + h[2]*(V*q[0] - 2*q[2])))/(3*h[0]**3*q[0]**2)
        
        tau_xy = eta*z*(h[0]*q[0]**2*(U*h[2] + V*h[1]) - 3*h[0]*q[0]*(-2*dqx[1]*(h[0] - z) - 2*dqy[0]*(h[0] - z) + h[1]*(V*q[0] - 2*q[2]) + h[2]*(U*q[0] - 2*q[1])) + 6*q[0]*(h[0] - z)*(h[1]*(V*q[0] - 2*q[2]) + h[2]*(U*q[0] - 2*q[1])))/(h[0]**3*q[0]**2)
        
        tau_xz = -eta*(U*h[0]**3*q[0] - 30*dqx[2]*z**2*(h[0] - z)**2 + 3*h[0]**2*(h[0] - 2*z)*(U*q[0] - 2*q[1]))/(h[0]**4*q[0])
        
        tau_yz = -eta*(V*h[0]**3*q[0] - 30*dqy[2]*z**2*(h[0] - z)**2 + 3*h[0]**2*(h[0] - 2*z)*(V*q[0] - 2*q[2]))/(h[0]**4*q[0])


        return tau_xx, tau_yy, tau_zz, tau_yz, tau_xz, tau_xy

    else:

        tau_xx = z*(12*eta*h[1]*(-U*q[0] + 3*q[1]) - (2*eta - 3*zeta)*(-3*h[1]*(U*q[0] - 2*q[1]) - 3*h[2]*(V*q[0] - 2*q[2]) + q[0]*(U*h[1] + V*h[2])))/(3*h[0]**2*q[0])
        
        tau_yy = z*(12*eta*h[2]*(-V*q[0] + 3*q[2]) - (2*eta - 3*zeta)*(-3*h[1]*(U*q[0] - 2*q[1]) - 3*h[2]*(V*q[0] - 2*q[2]) + q[0]*(U*h[1] + V*h[2])))/(3*h[0]**2*q[0])
        
        tau_zz = z*(2*eta - 3*zeta)*(3*h[1]*(U*q[0] - 2*q[1]) + 3*h[2]*(V*q[0] - 2*q[2]) - q[0]*(U*h[1] + V*h[2]))/(3*h[0]**2*q[0])
        
        tau_xy = 2*eta*z*(-U*h[2]*q[0] - V*h[1]*q[0] + 3*h[1]*q[2] + 3*h[2]*q[1])/(h[0]**2*q[0])
        
        tau_xz = -eta*(U*h[0]*q[0] + 3*(h[0] - 2*z)*(U*q[0] - 2*q[1]))/(h[0]**2*q[0])
        
        tau_yz = -eta*(V*h[0]*q[0] + 3*(h[0] - 2*z)*(V*q[0] - 2*q[2]))/(h[0]**2*q[0])


        return tau_xx, tau_yy, tau_zz, tau_yz, tau_xz, tau_xy


def velocity_profiles(z, h, q, U=1., V=0):

    u = (U*h[0]**2*q[0] - U*h[0]*q[0]*z - 3*z*(h[0] - z)*(U*q[0] - 2*q[1]))/(h[0]**2*q[0])
    v = (V*h[0]**2*q[0] - V*h[0]*q[0]*z - 3*z*(h[0] - z)*(V*q[0] - 2*q[2]))/(h[0]**2*q[0])

    return u, v

# %%
if __name__ == "__main__":

    stress_sympy_new(silent=True)

    fig, ax = plt.subplots(1, 3, sharey=True, figsize=(9, 3))

    Q = [1., .75, 0.25]
    DQX = [0.0, 0.01, 0.01]
    DQY = [0., 0., 0.]
    H = [1., 0.1, 0.]
    
    z = np.linspace(0., H[0], 100)

    u, v = velocity_profiles(z, H, Q)
    ax[0].plot(u, z, label=r'$u(z)$')
    ax[0].plot(v, z, label=r'$v(z)$')
    
    ax[0].set_xlabel(r'$u, v$')
    ax[0].set_ylabel(r'$z$')

    ax[0].legend()

    tau_xx, tau_yy, tau_zz, tau_yz, tau_xz, tau_xy = stress_profiles(z, H, Q, DQX, DQY)

    ax[1].plot(tau_xx, z, label=r'$\tau_{xx}(z)$')
    ax[1].plot(tau_yy, z, label=r'$\tau_{yy}(z)$')
    ax[1].plot(tau_xy, z, label=r'$\tau_{xy}(z)$')
    # ax[1].plot(tau_zz, z, label=r'$\tau_{zz}(z)$')
    
    ax[2].plot(tau_xz, z, label=r'$\tau_{xz}(z)$')
    ax[2].plot(tau_yz, z, label=r'$\tau_{yz}(z)$')

    tau_xx, tau_yy, tau_zz, tau_yz, tau_xz, tau_xy = stress_profiles(z, H, Q)

    ax[1].plot(tau_xx, z, '--', color='C0')
    ax[1].plot(tau_yy, z, '--', color='C1')
    ax[1].plot(tau_xy, z, '--', color='C2')
    # ax[1].plot(tau_zz, z, '--', color='C3')
    
    ax[2].plot(tau_xz, z, '--', color='C0')
    ax[2].plot(tau_yz, z, '--', color='C1')


    ax[1].legend()
    ax[2].legend()

    ax[1].set_xlabel(r'$\tau$')
    ax[2].set_xlabel(r'$\tau$')

    plt.show()


