#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# Discretization
Nx = 100
Ny = 100

# Geometry
Lx = 1.
Ly = 1.
h0 = 1.
sx = -0.02
sy = -0.03

# Shear velocities
U = 1.
V = 0.

# Constants
mu = 1.
eos_cubic = {84 : [5.7e-07, -0.0015, 1.3173, -352.6244],
            216 : [5.3e-07, -0.0011, 0.8580, -198.6119],
            292 : [5.5e-07, -0.0011, 0.8643, -194.6684]}

# Function definitions
def eos(rho): # Equation of state
    p = np.poly1d(eos_cubic[292])
    return p(rho)

def heightLinear(x,y): # Linear height function
    return h0 + sx * (1. - x/Lx) + sy * (1. - y/Ly)

def stressNewtonian(U,h,j,rho,z): # stress for Newtonian fluid
    return mu*(U/h)+6/h/h*(j/rho-U/2)*(h-2*z)

def avgShearStress(vel,h):
    return mu*vel/h

# Initialization
from field.field import scalarField
rho = scalarField(Nx, Ny, Lx, Ly)
rho.normal(100., 50.)

height = scalarField(Nx, Ny, Lx, Ly)
height.fromFunctionXY(heightLinear)

press = scalarField(Nx, Ny, Lx, Ly)
press.fromFunctionField(eos, rho.field[0])

from field.field import vectorField
flux = vectorField(Nx, Ny, Lx, Ly)

from field.field import tensorField
avg_stress = tensorField(Nx, Ny, Lx, Ly)
avg_stress.getStressNewtonian(press.field[0], avgShearStress, U, V, height.field[0])

pGrad = press.computeGrad()
stressDiv = avg_stress.computeDiv()

fig = plt.figure(figsize=(4,3))
ax1 = fig.add_subplot(111, projection='3d')
ax1.plot_surface(press.xx, press.yy, avg_stress.field[2], cmap = 'viridis')
plt.show()


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
