import numpy as np

def eos(rho): # Equation of state
    alpha = 1 # simple case with alpha = bulk modulus
    p = alpha*rho
    return p

# Discretization
Nx = 10
Ny = 5
rho_new = np.ones((Nx,Ny))
rho_old = rho_new
mom_new = np.zeros((Nx,Ny))
mom_old = mom_new
p       = np.zeros((Nx,Ny))

# 0) Initial conditions
rho_0   = 1.
rho_new = rho_0 * rho_new


# 1) Time Loop --------------------------
# compute pressure 
p       = eos(rho_new) # the eos should be computed from LAMMPS
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






