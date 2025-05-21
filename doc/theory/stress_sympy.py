# %% [markdown]
# # Viscous stress tensor calculation

# %% [markdown]
# The viscous stress tensor for a Newtonian fluid is usually expressed as a linear function of
# the velocity gradient, i.e., 
# $\tau_{ij} = \eta(u_{i,j} + u_{j,i}) + (\zeta - \frac{2}{3}\eta)u_{i,i}\delta_{ij}$.
# Here, we seek an expression in terms of the gap-averaged densities of conserved 
# variables $\rho(x, y)$, $j_x(x, y)$, and $j_y(x, y)$.
# Here, we use [Sympy](https://sympy.org) to derive the expressions for the stress tensor components.
#
# We start with a common ansatz for the velocity profiles
# 
# $u(z) = a z (h - z) + \frac{U_{top} - U_{bot}}{h} z + U_{bot}$,
# 
# $v(z) = b z (h - z) + \frac{V_{top} - V_{bot}}{h} z + V_{bot}$,
#
# $w(z) = cz^2(h - z)^2$,
#
# for $z \in [0, h]$.
#

# %%
import sympy as sp

# Define Cartesian coordinates
x, y, z = sp.symbols('x y z')

# Define some constants
Ut, Ub, Vt, Vb = sp.symbols('Ut Ub Vt Vb') # Velocities
zeta, eta = sp.symbols('zeta eta') # Viscosities

# The unknown parameters for the nonlinear part, that we need to find
a, b, c = sp.symbols('a b c')

# The gap height is a function of the lateral coordinates
h = sp.Function('h')

# Velocity profiles
u = (Ut - Ub) * z /  h(x, y) + Ub + a * z * (h(x, y) - z)
v = (Vt - Vb) * z /  h(x, y) + Vb + b * z * (h(x, y) - z)
w = c * z**2 * (h(x, y) - z)**2

# %% [markdown]
# The unknown parameters can be found by using the definition of the 
# gap-averaged fluxes, e.g.
# 
# $j_x(x, y) = \frac{\rho(x, y)}{h(x,y)}\int_0^{h(x, y)} u(z) dz$,
#
# where we assume that the density is constant across the gap. 
# Thus, we can solve the latter expression for $a$.

# %%
# Define continuum fields as functions
jx = sp.Function('jx')
jy = sp.Function('jy')
jz = sp.Function('jz')
rho = sp.Function('rho')

# Define equations
eq1 = sp.Eq(jx(x, y), rho(x, y) / h(x, y) * sp.integrate(u, (z, 0, h(x, y))))
eq2 = sp.Eq(jy(x, y), rho(x, y) / h(x, y) * sp.integrate(v, (z, 0, h(x, y))))
eq3 = sp.Eq(jz(x, y), rho(x, y) / h(x, y) * sp.integrate(w, (z, 0, h(x, y))))

# Solve for a, b, c
_a, = sp.solveset(eq1, a)
_b, = sp.solveset(eq2, b)
_c, = sp.solveset(eq3, c)

# %% [markdown]
# It is important to realize that the solutions depend on the lateral coordinates.
# Thus, before we differentiate, we need to substitute the solutions
# into the expressions for the velocity profiles using ```subs```.

# %%
# Compute velocity gradients
du_dx = sp.diff(u.subs(a, _a), x)
du_dy = sp.diff(u.subs(a, _a), y)
du_dz = sp.diff(u.subs(a, _a), z)

dv_dx = sp.diff(v.subs(b, _b), x)
dv_dy = sp.diff(v.subs(b, _b), y)
dv_dz = sp.diff(v.subs(b, _b), z)

dw_dx = sp.diff(w.subs(c, _c), x)
dw_dy = sp.diff(w.subs(c, _c), y)
dw_dz = sp.diff(w.subs(c, _c), z)

# Symmetrize
Dxx = du_dx
Dyy = dv_dy
Dzz = dw_dz
Dxy = (du_dy + dv_dx) / 2
Dxz = (du_dz + dw_dx) / 2 
Dyz = (dv_dz + dw_dy) / 2

# Trace
trD = Dxx + Dyy + Dzz

# Viscous stress
Txx = (zeta - sp.Rational(2, 3) * eta) * trD + 2 * eta * Dxx
Tyy = (zeta - sp.Rational(2, 3) * eta) * trD + 2 * eta * Dyy
Tzz = (zeta - sp.Rational(2, 3) * eta) * trD + 2 * eta * Dzz
Txy = 2 * eta * Dxy
Txz = 2 * eta * Dxz
Tyz = 2 * eta * Dyz

# %% [markdown]
# Finally, we take a gap-average of the viscous stress profiles

# %%
# Average stress
Txx_avg = sp.integrate(Txx, (z, 0, h(x,y))) / h(x, y)
Tyy_avg = sp.integrate(Tyy, (z, 0, h(x,y))) / h(x, y)
Txy_avg = sp.integrate(Txy, (z, 0, h(x,y))) / h(x, y)

# %% [markdown]
# Let's print the functions for the stress profiles. First, we make some 
# simplifying assumptions and useful replacements.

# %%
H, Hx, Hy = sp.symbols('h[0] h[1] h[2]')
q0, q1, q2 = sp.symbols('q[0] q[1] q[2]')
dqx0, dqx1, dqx2 = sp.symbols('dqx[0] dqx[1] dqx[2]')
dqy0, dqy1, dqy2 = sp.symbols('dqy[0] dqy[1] dqy[2]')
U, V = sp.symbols('U V')

replacements = {
    sp.Derivative(h(x, y), x): Hx,
    sp.Derivative(h(x, y), y): Hy,
    h(x, y): H, 
    rho(x, y): q0,
    jx(x, y): q1,
    jy(x, y): q2,
    jz(x, y): 0,
    sp.Derivative(jx(x, y), x): dqx0,
    sp.Derivative(jx(x, y), y): dqy0,
    sp.Derivative(jy(x, y), x): dqx1,
    sp.Derivative(jy(x, y), y): dqy1,
    sp.Derivative(jz(x, y), x): dqx2,
    sp.Derivative(jz(x, y), y): dqy2,
    Ub: U, 
    Vb: V,
    Vt: 0,
    Ut: 0,
    }

# Velocity profiles
print('Velocity profiles:')
print('u = ', sp.simplify(u.subs(a, _a).subs(replacements)))
print('v = ', sp.simplify(v.subs(b, _b).subs(replacements)))

# Stress profiles
print('Stress profiles:')
print('tau_xx = ', sp.simplify(Txx.subs(replacements)))
print('tau_yy = ', sp.simplify(Tyy.subs(replacements)))
print('tau_zz = ', sp.simplify(Tzz.subs(replacements)))
print('tau_xy = ', sp.simplify(Txy.subs(replacements)))
print('tau_xz = ', sp.simplify(Txz.subs(replacements)))
print('tau_yz = ', sp.simplify(Tyz.subs(replacements)))

# %% [markdown]
# We can directly copy these expressions into a Python function...

# %%
def get_velocity_profiles(z, h, q, U=1., V=0.):
    u = (U*h[0]**2*q[0] - U*h[0]*q[0]*z - 3*z*(h[0] - z)*(U*q[0] - 2*q[1]))/(h[0]**2*q[0])
    v = (V*h[0]**2*q[0] - V*h[0]*q[0]*z - 3*z*(h[0] - z)*(V*q[0] - 2*q[2]))/(h[0]**2*q[0])

    return u, v


def get_stress_profiles(z, h, q, dqx, dqy, U=1., V=0., eta=1., zeta=1.):

    tau_xx = z*(6*eta*(U*h[0]*h[1]*q[0]**2 - 3*h[0]*q[0]*(-2*dqx[0]*(h[0] - z) + h[1]*(U*q[0] - 2*q[1])) + 6*h[1]*q[0]*(h[0] - z)*(U*q[0] - 2*q[1])) - (2*eta - 3*zeta)*(h[0]*q[0]**2*(U*h[1] + V*h[2]) - 3*h[0]*q[0]*(-2*dqx[0]*(h[0] - z) - 2*dqy[1]*(h[0] - z) + h[1]*(U*q[0] - 2*q[1]) + h[2]*(V*q[0] - 2*q[2])) + 6*q[0]*(h[0] - z)*(h[1]*(U*q[0] - 2*q[1]) + h[2]*(V*q[0] - 2*q[2]))))/(3*h[0]**3*q[0]**2)
        
    tau_yy = z*(6*eta*(V*h[0]*h[2]*q[0]**2 - 3*h[0]*q[0]*(-2*dqy[1]*(h[0] - z) + h[2]*(V*q[0] - 2*q[2])) + 6*h[2]*q[0]*(h[0] - z)*(V*q[0] - 2*q[2])) - (2*eta - 3*zeta)*(h[0]*q[0]**2*(U*h[1] + V*h[2]) - 3*h[0]*q[0]*(-2*dqx[0]*(h[0] - z) - 2*dqy[1]*(h[0] - z) + h[1]*(U*q[0] - 2*q[1]) + h[2]*(V*q[0] - 2*q[2])) + 6*q[0]*(h[0] - z)*(h[1]*(U*q[0] - 2*q[1]) + h[2]*(V*q[0] - 2*q[2]))))/(3*h[0]**3*q[0]**2)
        
    tau_zz = -z*(2*eta - 3*zeta)*(h[0]*q[0]**2*(U*h[1] + V*h[2]) - 3*h[0]*q[0]*(-2*dqx[0]*(h[0] - z) - 2*dqy[1]*(h[0] - z) + h[1]*(U*q[0] - 2*q[1]) + h[2]*(V*q[0] - 2*q[2])) + 6*q[0]*(h[0] - z)*(h[1]*(U*q[0] - 2*q[1]) + h[2]*(V*q[0] - 2*q[2])))/(3*h[0]**3*q[0]**2)
        
    tau_xy = eta*z*(h[0]*q[0]**2*(U*h[2] + V*h[1]) - 3*h[0]*q[0]*(-2*dqx[1]*(h[0] - z) - 2*dqy[0]*(h[0] - z) + h[1]*(V*q[0] - 2*q[2]) + h[2]*(U*q[0] - 2*q[1])) + 6*q[0]*(h[0] - z)*(h[1]*(V*q[0] - 2*q[2]) + h[2]*(U*q[0] - 2*q[1])))/(h[0]**3*q[0]**2)
        
    tau_xz = -eta*(U*h[0]**3*q[0] - 30*dqx[2]*z**2*(h[0] - z)**2 + 3*h[0]**2*(h[0] - 2*z)*(U*q[0] - 2*q[1]))/(h[0]**4*q[0])
        
    tau_yz = -eta*(V*h[0]**3*q[0] - 30*dqy[2]*z**2*(h[0] - z)**2 + 3*h[0]**2*(h[0] - 2*z)*(V*q[0] - 2*q[2]))/(h[0]**4*q[0])

    return tau_xx, tau_yy, tau_zz, tau_yz, tau_xz, tau_xy

# %% [markdown]
# ... and use it for plotting.

# %%
import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 3, sharey=True, figsize=(9, 3))

q_test = [1., .75, 0.25]
dqx_test = [0.0, 0.01, 0.01]
dqy_test = [0., 0., 0.]
h_test = [1., 0.1, 0.1]

z_test = np.linspace(0., h_test[0], 100)

u, v = get_velocity_profiles(z_test, h_test, q_test)
ax[0].plot(u, z_test, label=r'$u(z)$')
ax[0].plot(v, z_test, label=r'$v(z)$')

ax[0].set_xlabel(r'$u, v$')
ax[0].set_ylabel(r'$z$')

tau_xx, tau_yy, tau_zz, tau_yz, tau_xz, tau_xy = get_stress_profiles(z_test, h_test, q_test, dqx_test, dqy_test)

ax[1].plot(tau_xx, z_test, label=r'$\tau_{xx}(z)$')
ax[1].plot(tau_yy, z_test, label=r'$\tau_{yy}(z)$')
ax[1].plot(tau_xy, z_test, label=r'$\tau_{xy}(z)$')
ax[1].plot(tau_zz, z_test, label=r'$\tau_{zz}(z)$')

ax[2].plot(tau_xz, z_test, label=r'$\tau_{xz}(z)$')
ax[2].plot(tau_yz, z_test, label=r'$\tau_{yz}(z)$')

ax[0].legend()
ax[1].legend()
ax[2].legend()

ax[1].set_xlabel(r'$\tau$')
ax[2].set_xlabel(r'$\tau$')

plt.show()

# %% [markdown]
# Solving the height-averaged balance equations, we need to evaluate the viscous stress at the walls...

# %%
print('Stress bottom:')
print('tau_xx(z=0)= ', sp.simplify(Txx.subs(replacements).subs(z, 0)))
print('tau_yy(z=0)= ', sp.simplify(Tyy.subs(replacements).subs(z, 0)))
print('tau_zz(z=0)= ', sp.simplify(Tzz.subs(replacements).subs(z, 0)))
print('tau_xy(z=0)= ', sp.simplify(Txy.subs(replacements).subs(z, 0)))
print('tau_xz(z=0)= ', sp.simplify(Txz.subs(replacements).subs(z, 0)))
print('tau_yz(z=0)= ', sp.simplify(Tyz.subs(replacements).subs(z, 0)))

# %%
print('Stress top:')
print('tau_xx(z=h)= ', sp.simplify(Txx.subs(replacements).subs(z, H)))
print('tau_yy(z=h)= ', sp.simplify(Tyy.subs(replacements).subs(z, H)))
print('tau_zz(z=h)= ', sp.simplify(Tzz.subs(replacements).subs(z, H)))
print('tau_xy(z=h)= ', sp.simplify(Txy.subs(replacements).subs(z, H)))
print('tau_xz(z=h)= ', sp.simplify(Txz.subs(replacements).subs(z, H)))
print('tau_yz(z=h)= ', sp.simplify(Tyz.subs(replacements).subs(z, H)))

# %% [markdown]
# ...or averaged over the gap height

# %%
print('Stress avg:')
print('tau_xx_avg= ', sp.simplify(Txx_avg.subs(replacements)))
print('tau_yy_avg= ', sp.simplify(Tyy_avg.subs(replacements)))
print('tau_xy_avg= ', sp.simplify(Txy_avg.subs(replacements)))

# %% [markdown]
# All these expressions are hardcoded in {py:mod}`hans.stress`.

# %% [markdown]
# See also: [Holey, H. et al. (2022) Tribology Letters, 70(2), p. 36.](https://doi.org/10.1007/s11249-022-01576-5)