#
# Copyright 2025 Hannes Holey
#           2025 Christoph Huber
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
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

# flake8: noqa: W503

from mpi4py import MPI
import numpy as np
import numpy.typing as npt

from dataclasses import dataclass
from functools import cached_property

from typing import TYPE_CHECKING, Tuple
if TYPE_CHECKING:
    from .problem import Problem

from muGrid import (
    CartesianDecomposition,
    GlobalFieldCollection,
    Communicator,
)
try:
    from muGrid import FFTEngine
    HAS_FFT_ENGINE = True
except ImportError:
    FFTEngine = None
    HAS_FFT_ENGINE = False

NDArray = npt.NDArray[np.floating]


@dataclass
class BCContext:
    """Context object passed to boundary condition callbacks.

    Attributes
    ----------
    problem : Problem
        The GaPFlow Problem instance.
    required_shape : tuple
        Required shape of the returned array.
    ghost_slice : tuple
        Slice for accessing ghost cells.
    interior_slice : tuple
        Slice for accessing adjacent interior cells.
    x_norm : NDArray
        Normalized x-coordinates at the boundary (0 to 1).
    y_norm : NDArray
        Normalized y-coordinates at the boundary (0 to 1).
    """
    problem: 'Problem'
    required_shape: tuple
    ghost_slice: tuple
    interior_slice: tuple
    x_norm: NDArray
    y_norm: NDArray


class DomainDecomposition:
    """
    Manages domain decomposition for MPI-parallel simulations.

    Parameters
    ----------
    grid : dict
        Grid configuration containing numer of grid points and
        boundary conditions.
    numerics : dict, optional
        Numerics configuration. When using FEM solver, the
        Taylor-Hood P2 mass flux grid (ghost depth 2) is also initialised.
    """

    _BND_TO_KEY = {'W': 'xW', 'E': 'xE', 'S': 'yS', 'N': 'yN'}

    def __init__(self, grid: dict, numerics: dict = None):

        self.grid = grid
        self._mpi_comm = MPI.COMM_WORLD
        self._comm = Communicator(self._mpi_comm)

        self._Nx = grid['Nx']
        self._Ny = grid['Ny']

        self._nb_subdivisions = self.split_domain()

        # Density grid (P1)
        self.init_decomposition()

        # Mass flux grid (P2) for Taylor-Hood FEM solver in 2D
        if numerics is not None and numerics['solver'] == 'fem' and grid['dim'] == 2:
            self.init_decomposition_P2()

    def split_domain(self) -> Tuple[int, int]:
        """Determine the number of splits in x and y
        """
        nx_splits = int(np.floor(np.sqrt(self._comm.size)))
        ny_splits = self._comm.size // nx_splits
        return (nx_splits, ny_splits)
    
    def init_decomposition(self) -> None:
        """Standard Decomposition for the main grid (ghost depth 1)
        """
        self._nb_domain_grid_pts = (self._Nx, self._Ny)
        self._decomp = CartesianDecomposition(
            self._comm,
            list(self._nb_domain_grid_pts),
            list(self._nb_subdivisions),
            [1, 1],
            [1, 1],
        )

    def init_decomposition_P2(self) -> None:
        
        nx_splits, ny_splits = self._nb_subdivisions

        if self._Nx % nx_splits != 0 or self._Ny % ny_splits != 0:
            raise ValueError(
                f"Grid dimensions ({self._Nx}, {self._Ny}) must be divisible by "
                f"the number of MPI splits ({nx_splits}, {ny_splits}). "
                f"Choose Nx divisible by {nx_splits} and Ny divisible by {ny_splits}."
            )

        self._nb_domain_grid_pts_v = (2 * self._Nx - 1, 2 * self._Ny - 1)

        self._decomp_v = CartesianDecomposition(
            self._comm,
            list(self._nb_domain_grid_pts_v),
            list(self._nb_subdivisions),
            [2, 2],
            [2, 2],
        )

    @property
    def fc(self) -> GlobalFieldCollection:
        """GlobalFieldCollection on the standard decomposition."""
        return self._decomp.collection

    @property
    def fc_v(self) -> GlobalFieldCollection:
        """GlobalFieldCollection on the P2 decomposition."""
        return self._decomp_v.collection

    # ---------------------------
    # MPI properties
    # ---------------------------

    @property
    def rank(self) -> int:
        """MPI rank of this process."""
        return self._comm.rank

    @property
    def size(self) -> int:
        """Total number of MPI processes."""
        return self._comm.size

    @property
    def nb_domain_grid_pts(self) -> tuple:
        """Global domain size (Nx, Ny)."""
        return self._nb_domain_grid_pts

    @property
    def nb_domain_grid_pts_v(self) -> tuple:
        """Global mass flux grid size (2*Nx-1, 2*Ny-1)."""
        return self._nb_domain_grid_pts_v

    @property
    def nb_subdomain_grid_pts(self) -> tuple:
        """Local subdomain size (inner points)."""
        return tuple(self._decomp.nb_subdomain_grid_pts)

    @property
    def nb_subdomain_grid_pts_v(self) -> tuple:
        """Local mass flux subdomain size (inner points)."""
        return tuple(self._decomp_v.nb_subdomain_grid_pts)

    @property
    def subdomain_locations(self) -> tuple:
        """Start position of this subdomain (without ghost offset)."""
        return tuple(self._decomp.subdomain_locations)

    @property
    def subdomain_locations_v(self) -> tuple:
        """Start position of mass flux subdomain (without ghost offset)."""
        return tuple(self._decomp_v.subdomain_locations)

    @property
    def subdomain_info(self) -> str:
        """Subdomain info string for MPI runs, empty for single rank."""
        if self.size > 1:
            sub_x, sub_y = self.nb_subdomain_grid_pts
            return f" (subdomain: {sub_x}x{sub_y} on {self.size} ranks)"
        return ""

    @property
    def nb_ghost_pts(self) -> int:
        """Total number of ghost points around the subdomain without corners."""
        return 2 * self.nb_subdomain_grid_pts[0] + 2 * self.nb_subdomain_grid_pts[1]

    @property
    def icoordsg(self):
        """Global coordinate indices for each subdomain point (2, Nx_local+2, Ny_local+2)."""
        return self._decomp.icoordsg

    @property
    def icoordsg_v(self):
        """Global coordinate indices for mass flux subdomain (2, Nx_v_local+4, Ny_v_local+4)."""
        return self._decomp_v.icoordsg

    # ---------------------------
    # Local shape utilities
    # ---------------------------

    @property
    def local_shape_padded(self) -> tuple:
        """Local subdomain shape with ghosts: (Nx_local+2, Ny_local+2)."""
        inner = self.nb_subdomain_grid_pts
        return (inner[0] + 2, inner[1] + 2)

    @property
    def local_shape_padded_v(self) -> tuple:
        """Local mass flux subdomain shape with ghosts: (Nx_v_local+4, Ny_v_local+4)."""
        inner = self.nb_subdomain_grid_pts_v
        return (inner[0] + 4, inner[1] + 4)

    # ---------------------------
    # Coordinate arrays
    # ---------------------------

    @cached_property
    def xx(self) -> NDArray:
        """Cell-center x-coordinates in physical units [m] (2D, with ghosts)."""
        grid = self.grid
        dx, Lx = grid['dx'], grid['Lx']
        xx = self.icoordsg[0] * dx + dx / 2.0
        if self.bc_at_W:
            xx[0, :] = -dx / 2.0
        if self.bc_at_E:
            xx[-1, :] = Lx + dx / 2.0
        return xx

    @cached_property
    def yy(self) -> NDArray:
        """Cell-center y-coordinates in physical units [m] (2D, with ghosts)."""
        grid = self.grid
        dy, Ly = grid['dy'], grid['Ly']
        yy = self.icoordsg[1] * dy + dy / 2.0
        if self.bc_at_S:
            yy[:, 0] = -dy / 2.0
        if self.bc_at_N:
            yy[:, -1] = Ly + dy / 2.0
        return yy

    @cached_property
    def xx_norm(self) -> NDArray:
        """Normalized x-coordinates [0, 1] (2D, with ghosts)."""
        return self.xx / self.grid['Lx']

    @cached_property
    def yy_norm(self) -> NDArray:
        """Normalized y-coordinates [0, 1] (2D, with ghosts)."""
        return self.yy / self.grid['Ly']

    # ---------------------------
    # Boundary ownership detection
    # ---------------------------

    @property
    def is_at_xW(self) -> bool:
        """True if this rank owns the West (left, x=0) boundary."""
        return self.subdomain_locations[0] == 0

    @property
    def is_at_xE(self) -> bool:
        """True if this rank owns the East (right, x=Lx) boundary."""
        loc_x = self.subdomain_locations[0]
        local_interior_x = self.nb_subdomain_grid_pts[0]
        return (loc_x + local_interior_x) >= self._Nx

    @property
    def is_at_yS(self) -> bool:
        """True if this rank owns the South (bottom, y=0) boundary."""
        return self.subdomain_locations[1] == 0

    @property
    def is_at_yN(self) -> bool:
        """True if this rank owns the North (top, y=Ly) boundary."""
        loc_y = self.subdomain_locations[1]
        local_interior_y = self.nb_subdomain_grid_pts[1]
        return (loc_y + local_interior_y) >= self._Ny

    @property
    def periodic_x(self) -> bool:
        """True if the x-boundaries are periodic."""
        grid = self.grid
        return all(b == 'P' for b in grid["bc_xW"]) and all(b == 'P' for b in grid["bc_xE"])

    @property
    def periodic_y(self) -> bool:
        """True if the y-boundaries are periodic."""
        grid = self.grid
        return all(b == 'P' for b in grid["bc_yS"]) and all(b == 'P' for b in grid["bc_yN"])

    @property
    def has_full_x(self) -> bool:
        """True if the full x-boundary is owned by this rank."""
        return self.is_at_xW and self.is_at_xE

    @property
    def has_full_y(self) -> bool:
        """True if the full y-boundary is owned by this rank."""
        return self.is_at_yS and self.is_at_yN

    @property
    def bc_at_W(self) -> bool:
        """True if this rank has a physical (non-periodic) West boundary."""
        return self.is_at_xW and not self.periodic_x

    @property
    def bc_at_E(self) -> bool:
        """True if this rank has a physical (non-periodic) East boundary."""
        return self.is_at_xE and not self.periodic_x

    @property
    def bc_at_S(self) -> bool:
        """True if this rank has a physical (non-periodic) South boundary."""
        return self.is_at_yS and not self.periodic_y

    @property
    def bc_at_N(self) -> bool:
        """True if this rank has a physical (non-periodic) North boundary."""
        return self.is_at_yN and not self.periodic_y

    @cached_property
    def index_mask_padded_global(self) -> NDArray:
        """Get global index mask for local padded subdomain shape."""
        return self.icoordsg[0] + self.icoordsg[1] * self._Nx

    @cached_property
    def index_mask_padded_global_v(self) -> NDArray:
        """Global index mask for local mass flux padded subdomain shape."""
        Nx_v = self._nb_domain_grid_pts_v[0]
        return self.icoordsg_v[0] + self.icoordsg_v[1] * Nx_v

    # ---------------------------
    # Global field gathering
    # ---------------------------

    def gather_global(self, local_field: NDArray) -> NDArray:
        """Gather local field to global array on rank 0.

        Parameters
        ----------
        local_field : NDArray
            Local 2D field. If shape matches local_shape_padded, ghost cells
            are excluded. If shape matches nb_subdomain_grid_pts, used directly.

        Returns
        -------
        NDArray or None
            Global field with shape nb_domain_grid_pts on rank 0, None otherwise.
        """
        comm = self._mpi_comm

        # extract inner part if field includes ghosts
        if local_field.shape == self.local_shape_padded:
            local_inner = local_field[1:-1, 1:-1]
        elif local_field.shape == self.nb_subdomain_grid_pts:
            local_inner = local_field
        else:
            raise ValueError(f"Field shape {local_field.shape} doesn't match "
                             f"inner {self.nb_subdomain_grid_pts} or padded {self.local_shape_padded}")

        # gather local fields and their positions
        all_fields = comm.gather(local_inner, root=0)
        all_locs = comm.gather(self.subdomain_locations, root=0)
        all_sizes = comm.gather(self.nb_subdomain_grid_pts, root=0)

        if self.rank == 0:
            global_field = np.zeros(self.nb_domain_grid_pts, dtype=local_field.dtype)
            for field, loc, sz in zip(all_fields, all_locs, all_sizes):
                global_field[loc[0]:loc[0] + sz[0], loc[1]:loc[1] + sz[1]] = field
            return global_field
        return None

    def scatter_global(self, global_field: NDArray) -> NDArray:
        """Scatter global field from rank 0 to local arrays.

        Parameters
        ----------
        global_field : NDArray
            Global field with shape nb_domain_grid_pts. Only needs to be
            valid on rank 0; other ranks can pass None or empty array.

        Returns
        -------
        NDArray
            Local field with shape nb_subdomain_grid_pts (without ghosts).
        """
        comm = self._mpi_comm

        # Gather all locations and sizes
        all_locs = comm.allgather(self.subdomain_locations)
        all_sizes = comm.allgather(self.nb_subdomain_grid_pts)

        # Prepare receive buffer
        local_inner = np.empty(self.nb_subdomain_grid_pts, dtype=np.float64)

        if self.rank == 0:
            # Send each rank's portion
            for dest, (loc, sz) in enumerate(zip(all_locs, all_sizes)):
                chunk = np.ascontiguousarray(
                    global_field[loc[0]:loc[0] + sz[0], loc[1]:loc[1] + sz[1]])
                if dest == 0:
                    local_inner[:] = chunk
                else:
                    comm.Send(chunk, dest=dest, tag=0)
        else:
            comm.Recv(local_inner, source=0, tag=0)

        return local_inner

    # ---------------------------
    # Ghost cell handling
    # ---------------------------

    def update_ghosts(self, exchange_specs, bc_specs, problem: "Problem") -> None:
        """MPI ghost exchange + physical BC application.

        Parameters
        ----------
        exchange_specs : list of (muGrid field, 'P1'|'P2')
            Fields to exchange. 'P1' uses ghost depth 1, 'P2' uses ghost depth 2.
        bc_specs : list of (numpy array, var_name, 'P1_cell'|'P1_nodal'|'P2_nodal')
            Per-variable BC application. var_name is 'rho', 'jx', or 'jy'.
            'P1_cell': depth-1 ghosts, cell-centered Dirichlet (mirror formula).
            'P1_nodal': depth-1 ghosts, nodal Dirichlet (direct assignment).
            'P2_nodal': depth-2 ghosts, nodal Dirichlet (both layers set).
        problem : Problem
            Used for BC callbacks and energy BC application.
        """
        for field, grid_type in exchange_specs:
            if grid_type == 'P1':
                self._exchange_ghosts(field)
            else:
                self._exchange_ghosts_v(field)

        for arr, var_name, disc in bc_specs:
            self._apply_field_bcs(arr, var_name, disc, problem)

        if problem.bEnergy:
            self._exchange_ghosts(problem.fc.get_real_field('total_energy'))
            self._apply_energy_bcs(problem)

    def _exchange_ghosts(self, field) -> None:
        """MPI ghost exchange for a single P1 field."""
        self._decomp.communicate_ghosts(field)

    def _exchange_ghosts_v(self, field) -> None:
        """MPI ghost exchange for a P2 field (ghost depth 2)."""
        self._decomp_v.communicate_ghosts(field)

    def _owns_boundary(self, bnd: str) -> bool:
        """Check if this rank owns the specified boundary."""
        return {'W': self.is_at_xW, 'E': self.is_at_xE,
                'S': self.is_at_yS, 'N': self.is_at_yN}[bnd]

    def _get_bc_slices(self, bnd: str):
        """Return (ghost_slice, interior_slice) for the specified boundary (depth-1 ghosts)."""
        slices = {
            'W': ((slice(0, 1), slice(None)), (slice(1, 2), slice(None))),
            'E': ((slice(-1, None), slice(None)), (slice(-2, -1), slice(None))),
            'S': ((slice(None), slice(0, 1)), (slice(None), slice(1, 2))),
            'N': ((slice(None), slice(-1, None)), (slice(None), slice(-2, -1))),
        }
        return slices[bnd]

    def _get_bc_slices_v(self, bnd: str):
        """Return ((ghost1_slice, ghost2_slice), interior_slice) for depth-2 ghost layers.

        ghost1 is the layer adjacent to the inner domain, ghost2 is the outermost layer.
        interior is the innermost node (used for Neumann forwarding).
        Both ghost layers are set for Dirichlet and Neumann BCs.
        """
        slices = {
            'W': ((slice(1, 2), slice(None)), (slice(0, 1), slice(None)), (slice(2, 3), slice(None))),
            'E': ((slice(-2, -1), slice(None)), (slice(-1, None), slice(None)), (slice(-3, -2), slice(None))),
            'S': ((slice(None), slice(1, 2)), (slice(None), slice(0, 1)), (slice(None), slice(2, 3))),
            'N': ((slice(None), slice(-2, -1)), (slice(None), slice(-1, None)), (slice(None), slice(-3, -2))),
        }
        ghost1, ghost2, interior = slices[bnd]
        return (ghost1, ghost2), interior

    def _apply_field_bcs(self, arr, var_name: str, disc: str, problem: "Problem") -> None:
        """Apply BCs to a single field array.

        Parameters
        ----------
        arr : numpy array
            Field data including ghost layers. Shape (Nx_padded, Ny_padded) for
            P1 fields, or (Nx_v_padded, Ny_v_padded) for P2 fields.
        var_name : str
            Variable name ('rho', 'jx', 'jy') for BC type lookup and callbacks.
        disc : str
            Discretization type:
            'P1_cell'  — depth-1 ghosts, cell-centered Dirichlet (mirror formula).
            'P1_nodal' — depth-1 ghosts, nodal Dirichlet (direct assignment).
            'P2_nodal' — depth-2 ghosts, nodal Dirichlet (both layers set directly).
        problem : Problem
            Used for BC callbacks.
        """
        grid = self.grid
        _VAR_IDX = {'rho': 0, 'jx': 1, 'jy': 2}
        var_idx = _VAR_IDX[var_name]
        bc_callbacks = getattr(problem, '_bc_callbacks', {})
        is_P2 = (disc == 'P2_nodal')

        for bnd in ['W', 'E', 'S', 'N']:
            if not self._owns_boundary(bnd):
                continue

            key = self._BND_TO_KEY[bnd]
            bc_types = grid[f'bc_{key}']

            if all(b == 'P' for b in bc_types):
                continue

            bc_type = bc_types[var_idx]
            if bc_type == 'P':
                continue

            callback = bc_callbacks.get(var_name, {}).get(bnd)

            if is_P2:
                (ghost1, ghost2), interior = self._get_bc_slices_v(bnd)
                if callback is not None:
                    required_shape = arr[ghost1].shape
                    ctx = BCContext(problem, required_shape, ghost1, interior,
                                    self.xx_norm[ghost1], self.yy_norm[ghost1])
                    bc_values = callback(ctx)
                    if bc_values.shape != required_shape:
                        raise ValueError(f"BC callback for {var_name}@{bnd}: "
                                         f"got {bc_values.shape}, expected {required_shape}")
                    arr[ghost1] = bc_values
                    arr[ghost2] = bc_values
                elif bc_type == 'D':
                    bc_vals = grid.get(f'bc_{key}_D_val')
                    target = bc_vals[var_idx] if isinstance(bc_vals, list) else bc_vals
                    arr[ghost1] = target
                    arr[ghost2] = target
                elif bc_type == 'N':
                    arr[ghost1] = arr[interior]
                    arr[ghost2] = arr[interior]
            else:
                ghost, interior = self._get_bc_slices(bnd)
                if callback is not None:
                    required_shape = arr[ghost].shape
                    ctx = BCContext(problem, required_shape, ghost, interior,
                                    self.xx_norm[ghost], self.yy_norm[ghost])
                    bc_values = callback(ctx)
                    if bc_values.shape != required_shape:
                        raise ValueError(f"BC callback for {var_name}@{bnd}: "
                                         f"got {bc_values.shape}, expected {required_shape}")
                    arr[ghost] = bc_values
                elif bc_type == 'D':
                    bc_vals = grid.get(f'bc_{key}_D_val')
                    target = bc_vals[var_idx] if isinstance(bc_vals, list) else bc_vals
                    if disc == 'P1_nodal':
                        arr[ghost] = target
                    else:
                        arr[ghost] = 2.0 * target - arr[interior]
                elif bc_type == 'N':
                    arr[ghost] = arr[interior]

    def _apply_energy_bcs(self, problem: "Problem") -> None:
        """
        Apply boundary conditions to energy field ghost cells.

        Only applies BCs if this rank owns the corresponding boundary.
        If the grid uses periodic BCs (all components), the energy field
        uses the periodic ghost values from _exchange_ghosts() instead.
        """
        grid = self.grid
        energy = problem.energy
        rho = problem.q[0]
        jx = problem.q[1]
        jy = problem.q[2]

        for bnd in ['W', 'E', 'S', 'N']:
            if not self._owns_boundary(bnd):
                continue

            key = self._BND_TO_KEY[bnd]
            bc_types = grid[f'bc_{key}']

            # Skip if all periodic
            if all(b == 'P' for b in bc_types):
                continue

            ghost, interior = self._get_bc_slices(bnd)
            bc_type = getattr(energy, f'bc_{key}')
            T_bc = getattr(energy, f'T_bc_{key}')

            if bc_type == 'D':
                ux = jx[ghost] / rho[ghost]
                uy = jy[ghost] / rho[ghost]
                kinetic = 0.5 * (ux**2 + uy**2)
                energy.energy[ghost] = rho[ghost] * (energy.cv * T_bc + kinetic)
            elif bc_type == 'N':
                energy.energy[ghost] = energy.energy[interior].copy()



class FFTDomainTranslation:
    """
    Manages data transfer between GaPFlow's DomainDecomposition and
    the FFT domain used for elastic deformation computation.

    Handles:
    - FFTEngine instantiation with correct grid size for boundary conditions
    - MPI redistribution between different domain decompositions
    - Zero-padding for non-periodic boundaries

    Parameters
    ----------
    decomp : DomainDecomposition
        GaPFlow's domain decomposition instance.
        Provides: grid, periodic_x, periodic_y, nb_domain_grid_pts,
                  subdomain_locations, nb_subdomain_grid_pts, communicator
    """

    def __init__(self, decomp: DomainDecomposition):
        if not HAS_FFT_ENGINE:
            raise ImportError(
                "FFTDomainTranslation requires muGrid built with FFT support. "
                "Install FFTW (libfftw3-dev) and rebuild muGrid."
            )

        self.decomp = decomp
        self.periodic_x = decomp.periodic_x
        self.periodic_y = decomp.periodic_y
        self.Nx, self.Ny = decomp.nb_domain_grid_pts

        # Compute FFT grid size based on periodicity
        self._compute_fft_grid_size()

        # Create FFTEngine with computed size
        self.fft_engine = FFTEngine(
            [self.Nx_fft, self.Ny_fft],
            decomp._comm  # muGrid Communicator
        )

        # Build exchange plan for MPI redistribution
        self._build_exchange_plan()

    def _compute_fft_grid_size(self):
        """Compute FFT grid size based on boundary conditions.

        Periodic: N
        Semi-periodic (one direction free): 2*N - 1 in free direction
        Non-periodic (both directions free): 2*N in both directions
        """
        if self.periodic_x and self.periodic_y:
            self.Nx_fft = self.Nx
            self.Ny_fft = self.Ny
        elif self.periodic_x and not self.periodic_y:
            # Semi-periodic: x periodic, y free
            self.Nx_fft = self.Nx
            self.Ny_fft = 2 * self.Ny - 1
        elif self.periodic_y and not self.periodic_x:
            # Semi-periodic: x free, y periodic
            self.Nx_fft = 2 * self.Nx - 1
            self.Ny_fft = self.Ny
        else:
            # Fully non-periodic
            self.Nx_fft = 2 * self.Nx
            self.Ny_fft = 2 * self.Ny

    def _build_exchange_plan(self):
        """Precompute send/recv maps for MPI redistribution.

        GaPFlow uses 2D block decomposition; FFTEngine uses Y-stripe decomposition
        (full X dimension on every rank). The exchange plan accounts for both X and Y
        extents of each GaPFlow rank so that block-decomposed data is correctly
        gathered into Y-stripes and scattered back.
        """
        comm = self.decomp._mpi_comm

        # GaPFlow X/Y range for this rank
        src_x_start = self.decomp.subdomain_locations[0]
        src_x_size  = self.decomp.nb_subdomain_grid_pts[0]
        src_x_end   = src_x_start + src_x_size

        src_y_start = self.decomp.subdomain_locations[1]
        src_y_size  = self.decomp.nb_subdomain_grid_pts[1]
        src_y_end   = src_y_start + src_y_size

        # FFT engine Y-range for this rank (X is always full: [0, Nx_fft))
        dst_y_start = self.fft_engine.subdomain_locations[1]
        dst_y_size  = self.fft_engine.nb_subdomain_grid_pts[1]
        dst_y_end   = dst_y_start + dst_y_size

        # Gather all ranks' ranges
        src_ranges = comm.allgather((src_x_start, src_x_end, src_y_start, src_y_end))
        dst_ranges = comm.allgather((dst_y_start, dst_y_end))

        self.send_map = {}
        self.recv_map = {}

        for other_rank in range(comm.size):
            # What I send: my Y-range intersected with their FFT Y-range
            other_dst_y_start, other_dst_y_end = dst_ranges[other_rank]
            send_y_start = max(src_y_start, other_dst_y_start)
            send_y_end   = min(src_y_end,   other_dst_y_end)

            if send_y_start < send_y_end:
                self.send_map[other_rank] = {
                    'y_slice': slice(send_y_start - src_y_start, send_y_end - src_y_start),
                    'size': (src_x_size, send_y_end - send_y_start),
                }

            # What I receive: their GaPFlow Y-range intersected with my FFT Y-range
            other_src_x_start, other_src_x_end, other_src_y_start, other_src_y_end = src_ranges[other_rank]
            recv_y_start = max(dst_y_start, other_src_y_start)
            recv_y_end   = min(dst_y_end,   other_src_y_end)

            if recv_y_start < recv_y_end:
                self.recv_map[other_rank] = {
                    'x_start': other_src_x_start,
                    'x_size':  other_src_x_end - other_src_x_start,
                    'y_slice': slice(recv_y_start - dst_y_start, recv_y_end - dst_y_start),
                    'size': (other_src_x_end - other_src_x_start, recv_y_end - recv_y_start),
                }

    def embed(self, src: np.ndarray, dst: np.ndarray):
        """Transfer data from GaPFlow block domain to FFT Y-stripe domain.

        Parameters
        ----------
        src : np.ndarray
            GaPFlow local block, shape (Nx_local, Ny_local).
        dst : np.ndarray
            FFT engine local buffer, shape (Nx_fft, Ny_fft_local). Will be zeroed
            first; rows beyond Nx remain zero (padding for non-periodic BCs).
        """
        comm = self.decomp._mpi_comm

        dst[:] = 0.0

        send_reqs = []
        for dest_rank, info in self.send_map.items():
            data = np.ascontiguousarray(src[:, info['y_slice']])
            req = comm.Isend(data, dest=dest_rank, tag=100)
            send_reqs.append(req)

        recv_reqs = []
        recv_buffers = []
        for src_rank, info in self.recv_map.items():
            buf = np.empty(info['size'], dtype=src.dtype)
            req = comm.Irecv(buf, source=src_rank, tag=100)
            recv_reqs.append(req)
            recv_buffers.append((buf, info))

        MPI.Request.Waitall(recv_reqs)
        for buf, info in recv_buffers:
            x0, xs = info['x_start'], info['x_size']
            dst[x0:x0 + xs, info['y_slice']] = buf

        MPI.Request.Waitall(send_reqs)

    def extract(self, src: np.ndarray, dst: np.ndarray):
        """Transfer data from FFT Y-stripe domain back to GaPFlow block domain.

        Parameters
        ----------
        src : np.ndarray
            FFT engine local buffer, shape (Nx_fft, Ny_fft_local).
        dst : np.ndarray
            GaPFlow local block, shape (Nx_local, Ny_local).
        """
        comm = self.decomp._mpi_comm

        # Reverse of embed: recv_map entries describe what to send back,
        # send_map entries describe what to receive back.
        send_reqs = []
        for dest_rank, info in self.recv_map.items():
            x0, xs = info['x_start'], info['x_size']
            data = np.ascontiguousarray(src[x0:x0 + xs, info['y_slice']])
            req = comm.Isend(data, dest=dest_rank, tag=200)
            send_reqs.append(req)

        recv_reqs = []
        recv_buffers = []
        for src_rank, info in self.send_map.items():
            buf = np.empty(info['size'], dtype=src.dtype)
            req = comm.Irecv(buf, source=src_rank, tag=200)
            recv_reqs.append(req)
            recv_buffers.append((buf, info['y_slice']))

        MPI.Request.Waitall(recv_reqs)
        for buf, y_slice in recv_buffers:
            dst[:, y_slice] = buf

        MPI.Request.Waitall(send_reqs)
