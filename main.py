#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Discretization
Nx = 100
Ny = 100
dt = 1e-5

# Geometry
Lx = 0.01
Ly = 0.01
h1 = 1.e-4
h2 = 5.e-5

# Shear velocities
U = 10.
V = 0.

sx = h2 - h1
sy = 0.


rho_atm = 389.5

# Constants
mu = 2.e-5

dimless = h2**2/(6.*mu*U*Lx)

eos_cubic = {84 : [5.7e-07, -0.0015, 1.3173, -352.6244],
            216 : [5.3e-07, -0.0011, 0.8580, -198.6119],
            292 : [5.5e-07, -0.0011, 0.8643, -194.6684]}

save_ani = 0
maxIt = 1000
plot_dim = 1
writeInterval = 10

# Function definitions
def Height(x):
    X = x/Lx
    alpha = h1/h2
    return alpha + (1 - alpha) * X

def P_analytical(x):
    alpha = h1/h2
    tmp = alpha/(1 - alpha**2)*(1./Height(x)**2 - 1./alpha**2) \
            - 1./(1 - alpha)*(1/Height(x) - 1./alpha)
    return tmp

def eos(rho):
    "Equation of state"
    p = np.poly1d(eos_cubic[292])
    return p(rho)

def heightLinear(x,y):
    "Linear height profile"
    return h1 + sx * x/Lx + sy * y/Ly
    #return h0 + sx * (1. - x/Lx) + sy * (1. - y/Ly)

def stressDivNewtonian(U, h, j, rho):
    "compute flux-dependent term in divergence of stress tensor"
    return 12.*mu/(h**2)*(j/rho-U/2)

# def plot1D(field, comp, time):
#     "plot field component on 2D grid"
#     ax1.plot(field.xx[:,24], field.field[comp][:,24], '-')
#     # ax1.set_ylim(110,170)
#     plt.pause(time)
#     ax1.clear()
#
# def plot2D(field, comp, time):
#     "plot field component on 2D grid"
#     im1 = ax1.imshow(field.field[comp], interpolation='nearest', aspect='1')
#     v1 = np.linspace(np.amin(field.field[comp]), np.amax(field.field[comp]), 11, endpoint=True)
#     cbar = plt.colorbar(im1, ticks = v1, ax = ax1)
#
#     plt.pause(time)
#     cbar.remove()

# Initialization
from field.field import scalarField
from field.field import vectorField
from field.field import tensorField

# Density distribution
rho = scalarField(Nx, Ny, Lx, Ly)
rho.normal(rho_atm, 0.)

# Gap height
height = scalarField(Nx, Ny, Lx, Ly)
height.fromFunctionXY(heightLinear)

# Pressure
press = scalarField(Nx, Ny, Lx, Ly)

# Flux field
flux = vectorField(Nx, Ny, Lx, Ly)

# Divergence of stress tensor
stressDiv = vectorField(Nx, Ny, Lx, Ly)

#press.field[0] = P_analytical(press.xx)*dimless

def solve():

    rho.setDirichlet(rho_atm)
    press.fromFunctionField(eos, rho.field[0])

    pGrad = press.computeGrad()

    stressDiv.fromField(pGrad)
    stressDiv.addFluxContribution(stressDivNewtonian, U, V, height.field[0], \
                                flux.field[0], flux.field[1], rho.field[0])

    flux.updateFlux(stressDiv, dt)

    fluxDiv = flux.computeDiv()
    rho.updateDens(fluxDiv, dt)

    # print(rho.field[0][25,25])

def animate1D(i):
    if i%writeInterval == 0:
        time_text.set_text(' frame number = %.1d' % i)
        line.set_ydata(press.field[0][:,24])  # update the data
    solve()
    #return line , time_text

def animate2D(i):
    if i%writeInterval == 0:
        time_text.set_text(' frame number = %.1d' % i)
        im.set_array(press.field[0])
    solve()

# Initialize figure
if plot_dim == 1:
    fig, ax1 = plt.subplots(1)
    #x = press.xx[:,24]/Lx       #defining 'x'
    x = np.linspace(0,Lx,Nx, endpoint=True)
    ax1.set_ylim(7,100)
    line, = ax1.plot(x, press.field[0][:,24])
    #ax1.plot(x/Lx ,P_analytical(x)*dimless)
    time_text = ax1.text(0.05, 0.95,'',horizontalalignment='left',verticalalignment='top', transform=ax1.transAxes)
    ani = animation.FuncAnimation(fig, animate1D, maxIt, interval=1 ,repeat=True)

elif plot_dim == 2:
    fig, ax1 = plt.subplots(1)
    im = ax1.imshow(press.field[0], interpolation='nearest', vmin=0, vmax=2e-7)
    time_text = ax1.text(0.05, 1.05,'',horizontalalignment='left',verticalalignment='top', transform=ax1.transAxes)
    v1 = np.linspace(0,2e-7, 11, endpoint=True)
    cbar = plt.colorbar(im, ticks=v1, ax = ax1)
    ani = animation.FuncAnimation(fig, animate2D, maxIt, interval=1 ,repeat=False)

if save_ani == 1:
     # ani.save('test.gif',fps=30, writer='imagemagick')
    ani.save('test.mp4',fps=30)
else:
    plt.show()

# fig = plt.figure(figsize=(4,3))
# ax1 = fig.add_subplot(111)
# cs = ax1.contourf(press.xx, press.yy, height.field[0], cmap = 'viridis')
# plt.colorbar(cs)
# plt.show()


# 1) Time Loop --------------------------
# compute pressure
# p       = eos(rho_new) # the eos should be computed from LAMMPS
# compute pressure gradient
# grad_p  = function(p)

# 2) input to MD
# we give the initial condition as input to the sveral MD simulations
# the inputs are rho, grad(p)
# the python script write the input file for LAMMPS (this will also be a function)

# 3) check MD whether the MD simulations converged
# 3.1) read the MD outup and evaluate the averaged quantitites such as sigma

# 4) Continuum part: update the momentum and continuity equation
# momentum equation: we get mom_new out of it
# firstly we evaluate the r.h.s. by computing the gradient of the stress tensor with finite differences.
# sencondly we update the momentum by using the information at the previous step
# continuity equation: we get rho_new out of it

# 5) possible post processing inside the time loop in order to get the stress behaviour in the time

# before closing the loop we update rho_old=rho_new and mom_old=mom_new
# close Time Loop
