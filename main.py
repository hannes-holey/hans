#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# Discretization
Nx = 50
Ny = 50
dt = 0.01

# Geometry
Lx = 1.
Ly = 1.
h0 = 1.
sx = -0.01
sy = -0.0000

# Shear velocities
U = -1.
V = 0.
p_atm = 34.68
rho_atm = 1000.

# Constants
mu = 0.001
eos_cubic = {84 : [5.7e-07, -0.0015, 1.3173, -352.6244],
            216 : [5.3e-07, -0.0011, 0.8580, -198.6119],
            292 : [5.5e-07, -0.0011, 0.8643, -194.6684]}

# Function definitions
def eos(rho):
    "Equation of state"
    p = np.poly1d(eos_cubic[292])
    return p(rho)

def heightLinear(x,y):
    "Linear height profile"
    return h0 + sx * (1. - x/Lx) + sy * (1. - y/Ly)

def stressDivNewtonian(U, h, j, rho):
    "compute flux-dependent term in divergence of stress tensor"
    return 12.*mu/(h**2)*(j/rho-U/2)

def plot1D(field, comp, time):
    "plot field component on 2D grid"
    ax1.plot(field.xx[:,0], field.field[comp][24,:], '-')
    # ax1.set_ylim(np.amin(field.field[comp]), np.amax(field.field[comp]))
    plt.pause(time)
    ax1.clear()

def plot2D(field, comp, time):
    "plot field component on 2D grid"
    im1 = ax1.imshow(field.field[comp], interpolation='nearest', aspect='1')
    v1 = np.linspace(np.amin(field.field[comp]), np.amax(field.field[comp]), 11, endpoint=True)
    cbar = plt.colorbar(im1, ticks = v1, ax = ax1)

    plt.pause(time)
    cbar.remove()

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

# Initialize figure
fig, ax1 = plt.subplots(1)


for i in range(100):

    #press.setDirichlet(p_atm)
    rho.setDirichlet(0.)
    press.fromFunctionField(eos, rho.field[0])

    # Assemble divergence of 'analytic' stress tensor
    pGrad = press.computeGrad()

    #plot2D(press, 0, 0.1)
    plot1D(rho, 0, 0.1)

    stressDiv.fromField(pGrad)
    stressDiv.addFluxContribution(stressDivNewtonian, U, V, height.field[0], \
                                flux.field[0], flux.field[1], rho.field[0])

    flux.updateFlux(stressDiv, dt)
    fluxDiv = flux.computeDiv()
    rho.updateDens(fluxDiv, dt)

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
