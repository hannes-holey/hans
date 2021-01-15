import numpy as np

from pylub.eos import EquationOfState


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

        rhox0 = rhox1 = rhoy0 = rhoy1 = float(self.material["rho0"])

        if "D" in x0 and "px0" in self.BC.keys():
            px0 = float(self.BC["px0"])
            rhox0 = EquationOfState(self.material).isoT_density(px0)

        if "D" in x1 and "px1" in self.BC.keys():
            px1 = float(self.BC["px1"])
            rhox1 = EquationOfState(self.material).isoT_density(px1)

        if "D" in y0 and "py0" in self.BC.keys():
            py0 = float(self.BC["py0"])
            rhoy0 = EquationOfState(self.material).isoT_density(py0)

        if "D" in y1 and "py1" in self.BC.keys():
            py1 = float(self.BC["py1"])
            rhoy1 = EquationOfState(self.material).isoT_density(py1)

        q.field[x0 == "D", 0, :] = 2. * rhox0 - q.field[x0 == "D", 1, :]
        q.field[x1 == "D", -1, :] = 2. * rhox1 - q.field[x0 == "D", -2, :]
        q.field[y0 == "D", :, 0] = 2. * rhoy0 - q.field[y0 == "D", :, 1]
        q.field[y1 == "D", :, -1] = 2. * rhoy1 - q.field[y0 == "D", :, -2]

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
