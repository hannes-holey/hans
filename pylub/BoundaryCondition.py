import numpy as np


class BoundaryCondition:

    def __init__(self, disc, BC, material):

        self.material = material
        self.BC = BC
        self.disc = disc

    def assert_consistency(self):
        x0 = np.array(list(self.BC["x0"]))
        x1 = np.array(list(self.BC["x1"]))
        y0 = np.array(list(self.BC["y0"]))
        y1 = np.array(list(self.BC["y1"]))

        assert len(x0) == 3
        assert len(x1) == 3
        assert len(y0) == 3
        assert len(y1) == 3

        assert np.all((x0 == "P") == (x1 == "P")), "Inconsistent boundary conditions (x)"
        assert np.all((y0 == "P") == (y1 == "P")), "Inconsistent boundary conditions (y)"

    def fill_ghost_cell(self, q):
        self.assert_consistency()
        q = self.periodic(q)
        q = self.dirichlet(q)
        q = self.neumann(q)
        return q

    def periodic(self, q):
        x0 = np.array(list(self.BC["x0"]))
        y0 = np.array(list(self.BC["y0"]))

        q.field[x0 == "P", 0, :] = q.field[x0 == "P", -2, :]
        q.field[x0 == "P", -1, :] = q.field[x0 == "P", 1, :]
        q.field[y0 == "P", :, 0] = q.field[y0 == "P", :, -2]
        q.field[y0 == "P", :, -1] = q.field[y0 == "P", :, 1]

        return q

    def dirichlet(self, q):

        x0 = np.array(list(self.BC["x0"]))
        x1 = np.array(list(self.BC["x1"]))
        y0 = np.array(list(self.BC["y0"]))
        y1 = np.array(list(self.BC["y1"]))

        rho0 = float(self.material["rho0"])

        q.field[x0 == "D", 0, :] = 2. * rho0 - q.field[x0 == "D", 1, :]
        q.field[x1 == "D", -1, :] = 2. * rho0 - q.field[x0 == "D", -2, :]
        q.field[y0 == "D", :, 0] = 2. * rho0 - q.field[y0 == "D", :, 1]
        q.field[y1 == "D", :, -1] = 2. * rho0 - q.field[y0 == "D", :, -2]

        return q

    def neumann(self, q):
        x0 = np.array(list(self.BC["x0"]))
        x1 = np.array(list(self.BC["x1"]))
        y0 = np.array(list(self.BC["y0"]))
        y1 = np.array(list(self.BC["y1"]))

        q.field[x0 == "N", 0, :] = q.field[x0 == "N", 1, :]
        q.field[x1 == "N", -1, :] = q.field[x0 == "N", -2, :]
        q.field[y0 == "N", :, 0] = q.field[y0 == "N", :, 1]
        q.field[y1 == "N", :, -1] = q.field[y0 == "N", :, -2]

        return q
