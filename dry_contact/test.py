import numpy as np
from ContactMechanics import PeriodicFFTElasticHalfSpace
from ContactMechanics.Systems import NonSmoothContactSystem
from SurfaceTopography import Topography

import matplotlib.pyplot as plt

# -----------------
# 1. Grid
# -----------------
nx, ny = 256, 256
sx, sy = 1.0, 1.0  # physical size

# -----------------
# 2. Elastic half-space
# -----------------
E = 210e9
nu = 0.3

substrate = PeriodicFFTElasticHalfSpace(
    (nx, ny),
    young=E,
    physical_sizes=(sx, sy),
    poisson=nu
)

# -----------------
# 3. Rigid indenter
# Example: spherical indenter
# -----------------
x = np.arange(nx) * sx / nx
y = np.arange(ny) * sy / ny

x -= sx / 2
y -= sy / 2

X, Y = np.meshgrid(x, y, indexing='ij')

R = 0.1
h = -(X**2 + Y**2) / (2 * R)

plt.imshow(h, extent=(-sx / 2, sx / 2, -sy / 2, sy / 2))
plt.colorbar(label='Indenter Height')
plt.title('Indenter Topography')

plt.show()

topography = Topography(h, physical_sizes=(sx, sy))

# -----------------
# 4. Contact system
# -----------------
system = NonSmoothContactSystem(substrate, topography)

# -----------------
# 5. Solve dry contact
# -----------------
external_force = 10000000000.0  # total normal force

result = system.minimize_proxy(
    external_force=external_force
)

pressure = result.jac  # contact pressure
displacement = result.x

plt.imshow(displacement)
plt.colorbar(label='Displacement')
plt.title('Displacement Distribution')

plt.show()

print(displacement)
print(np.max(displacement))
print(np.min(displacement))

pressure = result.jac

plt.imshow(pressure)
plt.colorbar(label='Contact Pressure')
plt.title('Contact Pressure Distribution')
plt.show()

print("Total force from pressure:", pressure.sum() * (sx / nx) * (sy / ny))
print("Max pressure:", pressure.max())
print("Contact fraction:", np.mean(pressure > 0))
