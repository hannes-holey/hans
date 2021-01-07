class BoundaryCondition:

    def __init__(self, disc, material):

        self.material = material
        self.disc = disc

    def set_inletDensity(self, q, ax=0):

        rho0 = float(self.material["rho0"])

        if ax == 0:
            q.field[0, 0, :] = rho0
        elif ax == 1:
            q.field[0, :, 0] = rho0

        return q
