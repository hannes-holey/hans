"""Global matrix ordering for Taylor-Hood P2P1.

Point-interleaved ordering: residuals are grouped by spatial node.  At every
mass-flux node the block contains jx (0) and jy (1).  At even-even nodes
(which coincide with density nodes) the block additionally contains rho (2)
and, when energy is active, e (3).

Residual index convention:
    0 → jx
    1 → jy
    2 → rho
    3 → e  (only when energy=True)

Parameters used throughout:
    cols_v   -- number of mass-flux columns  (= 2*M - 1  for a global grid)
    M        -- number of density columns    (= (cols_v + 1) // 2)
    energy   -- whether the energy equation is active
"""


def _n_density_before(k: int, cols_v: int, M: int) -> int:
    """Number of density nodes that have been emitted before the block for mass-flux node k.

    A density node lives at every even-even position of the fine grid.
    For mass-flux node k (0-based, F-order on the fine grid with cols_v columns):
        row  r = k // cols_v
        col  c = k % cols_v

    Density nodes before block k:
        - all even rows strictly before row r: ceil(r/2) = (r+1)//2 such rows,
          each contributing M density nodes → ((r+1)//2) * M
        - if r is even: even columns strictly before column c: ceil(c/2) = (c+1)//2
        - if r is odd: no density nodes in this row
    """
    r = k // cols_v
    c = k % cols_v
    # Number of even rows strictly before row r: ceil(r / 2) = (r + 1) // 2
    n = ((r + 1) // 2) * M
    if r % 2 == 0:
        # Number of even columns in {0, 1, ..., c-1} = ceil(c / 2) = (c + 1) // 2
        n += (c + 1) // 2
    return n


def _rows_per_fine_row(r: int, cols_v: int, M: int, n_rho: int) -> int:
    """Total global rows contributed by fine-grid row r."""
    return 2 * cols_v + (M * n_rho if r % 2 == 0 else 0)


def field_to_global(field_idx: int, res_type: int,
                    cols_v: int, M: int, energy: bool = False) -> int:
    """Map a (field_idx, res_type) pair to its global matrix row.

    Parameters
    ----------
    field_idx : int
        0-based F-order index within the field.
        - For res_type in {0, 1}  (jx / jy): index on the mass-flux grid.
        - For res_type in {2, 3}  (rho / e): index on the density grid.
    res_type : int
        Residual type: 0=jx, 1=jy, 2=rho, 3=e.
    cols_v : int
        Number of mass-flux columns.
    M : int
        Number of density columns  (= (cols_v + 1) // 2).
    energy : bool
        Whether the energy equation is active.

    Returns
    -------
    int
        Global row index in the assembled system matrix.
    """
    if res_type not in (0, 1, 2, 3):
        raise ValueError(f"res_type must be 0–3, got {res_type}")
    if energy is False and res_type == 3:
        raise ValueError("res_type=3 (energy) requires energy=True")

    n_rho = 2 if energy else 1  # rho + e  or  rho only

    if res_type in (0, 1):
        # field_idx is a mass-flux index
        k = field_idx
    else:
        # field_idx is a density index; convert to the corresponding even-even
        # mass-flux index.  Density node (i, j) in F-order has
        #   density_idx = i + j * M
        # The corresponding mass-flux node is (2i, 2j) with
        #   mass_flux_idx = 2i + 2j * cols_v
        rows_p = field_idx // M   # density row index
        cols_p = field_idx % M    # density col  index
        k = cols_p * 2 + rows_p * 2 * cols_v

    n_d = _n_density_before(k, cols_v, M)
    global_start = 2 * k + n_d * n_rho

    if res_type in (0, 1):
        return global_start + res_type
    else:
        # rho is at offset 2, e at offset 3 (relative to even-even block start)
        rho_offset = 2 if energy else 2  # always 2: after jx(0) and jy(1)
        if res_type == 2:
            return global_start + rho_offset
        else:  # res_type == 3
            return global_start + rho_offset + 1


def global_to_field(global_idx: int,
                    cols_v: int, M: int, energy: bool = False
                    ) -> tuple:
    """Inverse of field_to_global.

    Returns
    -------
    (field_idx, res_type) : (int, int)
        field_idx is on the mass-flux grid for res_type in {0,1} and on the
        density grid for res_type in {2,3}.
    """
    n_rho = 2 if energy else 1
    rows_v = (cols_v + 1) // 2 * 2 - 1  # not needed directly

    g = global_idx

    # Walk fine-grid rows until we find the row containing g
    r = 0
    while True:
        rpr = _rows_per_fine_row(r, cols_v, M, n_rho)
        if g < rpr:
            break
        g -= rpr
        r += 1

    # Now g is the local offset within fine-grid row r.
    # Row r has cols_v mass-flux nodes.  In this row the ordering is:
    #   for each column c in 0..cols_v-1:
    #       jx(r,c) jy(r,c)  [if r even and c even: rho(r//2, c//2) [e(r//2,c//2)]]
    # So we walk column-by-column.
    c = 0
    while True:
        block_size = 2 + (n_rho if (r % 2 == 0 and c % 2 == 0) else 0)
        if g < block_size:
            break
        g -= block_size
        c += 1

    # g is the offset within the block at (r, c)
    k = c + r * cols_v  # mass-flux linear index (F-order: col-major)

    if g == 0:
        return k, 0   # jx
    elif g == 1:
        return k, 1   # jy
    elif g == 2:
        # rho: density field index
        density_idx = (c // 2) + (r // 2) * M
        return density_idx, 2
    else:
        # e: density field index
        density_idx = (c // 2) + (r // 2) * M
        return density_idx, 3
