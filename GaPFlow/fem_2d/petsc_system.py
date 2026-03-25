#
# Copyright 2025 Christoph Huber
#
# ### MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
"""PETSc-based sparse linear solver for Taylor-Hood P2P1 FEM solver."""

import numpy as np
import numpy.typing as npt
from typing import TYPE_CHECKING

from .. import HAS_PETSC

if not HAS_PETSC:
    raise ImportError(
        "petsc4py is required for the Taylor-Hood FEM solver but is not installed.\n"
        "See README.md for installation instructions."
    )

from petsc4py import PETSc

if TYPE_CHECKING:
    from .assembly import P2P1AssemblyInfo

NDArray = npt.NDArray[np.floating]


class PETScSystem:
    """PETSc-based sparse linear solver for the Taylor-Hood P2P1 FEM solver.

    Provides the same .assemble() / .solve() interface as ScipySystem.
    solve() returns a flat 1-D solution vector; the caller unpacks per-variable
    slices of unequal length.

    Parameters
    ----------
    info : P2P1AssemblyInfo
        Precomputed assembly info (sizes and global indices).
    solver_type : str, optional
        "direct" (MUMPS LU) or "iterative" (GMRES). Default: "direct".
    """

    _ILU2_THRESHOLD = 110000

    def __init__(self, info: "P2P1AssemblyInfo", solver_type: str = "direct"):
        self._info = info
        self._solver_type = solver_type
        self.comm = PETSc.COMM_WORLD
        self._create_petsc_objects()

    def _create_petsc_objects(self):
        info = self._info
        local_size = info.local_size
        # In the single-process case global_size == local_size.
        # In MPI, global_size must be summed across ranks; use PETSc's
        # DETERMINE (-1) sentinel so it computes the global automatically.
        global_size = PETSc.DECIDE

        self.mat = PETSc.Mat().create(self.comm)
        self.mat.setSizes([(local_size, global_size), (local_size, global_size)])
        self.mat.setType('aij')
        self.mat.setFromOptions()

        # Pass copies: PETSc's setPreallocationCOO sorts the arrays in-place,
        # which would corrupt the assembly COO index arrays.
        self.mat.setPreallocationCOO(
            info.mat_global_rows.copy(), info.mat_global_cols.copy()
        )
        self.mat.setUp()

        self.vec_rhs = self.mat.createVecLeft()
        self.vec_sol = self.mat.createVecRight()

        self.ksp = PETSc.KSP().create(self.comm)
        self.ksp.setOperators(self.mat)

        if self._solver_type == "iterative":
            self.ksp.setType('gmres')
            self.ksp.setTolerances(rtol=1e-8, atol=1e-12, max_it=1000)
            pc = self.ksp.getPC()
            if self.comm.getSize() > 1:
                pc.setType('bjacobi')
            else:
                fill_level = 2 if local_size > self._ILU2_THRESHOLD else 1
                pc.setType('ilu')
                pc.setFactorLevels(fill_level)
        else:
            self.ksp.setType('preonly')
            pc = self.ksp.getPC()
            pc.setType('lu')
            pc.setFactorSolverType('mumps')

        self.ksp.setFromOptions()

    def assemble(self, coo_values: NDArray, R_local: NDArray) -> None:
        """Assemble sparse matrix and RHS vector.

        Parameters
        ----------
        coo_values : NDArray, shape (nnz,)
            COO Jacobian values.
        R_local : NDArray, shape (local_size,)
            Local residual vector (will be negated for Newton RHS).
        """
        self.mat.setValuesCOO(coo_values, PETSc.InsertMode.INSERT_VALUES)

        self.vec_rhs.zeroEntries()
        self.vec_rhs.setValues(
            self._info.rhs_global_rows, -R_local, PETSc.InsertMode.INSERT_VALUES)

        self.mat.assemblyBegin(PETSc.Mat.AssemblyType.FINAL)
        self.mat.assemblyEnd(PETSc.Mat.AssemblyType.FINAL)
        self.vec_rhs.assemblyBegin()
        self.vec_rhs.assemblyEnd()

    def solve(self) -> NDArray:
        """Solve the assembled linear system.

        Returns
        -------
        NDArray, shape (local_size,)
            Flat solution vector.  The caller unpacks per-variable slices.
        """
        self.ksp.solve(self.vec_rhs, self.vec_sol)
        return self.vec_sol.getArray().copy()[self._info.rhs_global_rows]

    def get_convergence_info(self) -> dict:
        reason = self.ksp.getConvergedReason()
        return {
            'converged': reason > 0,
            'iterations': self.ksp.getIterationNumber(),
            'residual_norm': self.ksp.getResidualNorm(),
            'reason': reason,
        }
