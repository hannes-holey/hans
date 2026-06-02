"""Standalone test for _compute_stencils + apply_stencil."""
import numpy as np
from GaPFlow.solver_fem.elements import TaylorHoodP2P1
from GaPFlow.solver_fem.grid_index import GridIndexManager

BLOCK = (('p', 'p'), ('p', 'v'), ('v', 'p'), ('v', 'v'))


class StubDecomp:
    def __init__(self, Nx_p, Ny_p):
        Nx_P2, Ny_P2 = 2*Nx_p-1, 2*Ny_p-1
        self.periodic_x = False; self.periodic_y = False
        self.is_at_xW = True; self.is_at_xE = True
        self.is_at_yS = True; self.is_at_yN = True
        nb_vars = 3
        self.grid = {'bc_xW': ['D']*nb_vars, 'bc_xE': ['D']*nb_vars,
                     'bc_yS': ['D']*nb_vars, 'bc_yN': ['D']*nb_vars}
        self.nb_subdomain_grid_pts = (Nx_p, Ny_p)
        self.index_mask_padded_global = np.full((Nx_p+2, Ny_p+2), -1, dtype=np.int32)
        self.index_mask_padded_global[1:-1, 1:-1] = np.arange(Nx_p*Ny_p).reshape(Nx_p, Ny_p, order='F')
        self.nb_subdomain_grid_pts_P2 = (Nx_P2, Ny_P2)
        self.index_mask_padded_global_P2 = np.full((Nx_P2+4, Ny_P2+4), -1, dtype=np.int32)
        self.index_mask_padded_global_P2[2:-2, 2:-2] = np.arange(Nx_P2*Ny_P2).reshape(Nx_P2, Ny_P2, order='F')

    @property
    def bc_at_W(self): return self.is_at_xW and not self.periodic_x
    @property
    def bc_at_E(self): return self.is_at_xE and not self.periodic_x
    @property
    def bc_at_S(self): return self.is_at_yS and not self.periodic_y
    @property
    def bc_at_N(self): return self.is_at_yN and not self.periodic_y


def _compute_stencils(element):
    stencil = {}
    stencil[3] = {
        (0, 0): np.array(element.stencil_even_even, dtype=np.int32),
        (1, 1): np.array(element.stencil_odd_odd,   dtype=np.int32),
        (0, 1): np.array(element.stencil_even_odd,  dtype=np.int32),
        (1, 0): np.array(element.stencil_odd_even,  dtype=np.int32),
    }
    for idx, combination in enumerate(BLOCK[0:3]):
        stencil[idx] = {}
        res, var = combination[0], combination[1]  # BLOCK = (res, var)
        for origin in stencil[3].keys():
            if res == 'p' and origin != (0, 0):
                continue
            if var == 'v':
                stencil[idx][origin] = stencil[3][origin]
                continue
            points = stencil[3][origin]
            stencil[idx][origin] = np.empty((0, 2), dtype=np.int32)
            for point in points:
                x, y = origin[0] + point[0], origin[1] + point[1]
                if not (x % 2 == 0 and y % 2 == 0):
                    continue
                stencil[idx][origin] = np.vstack([stencil[idx][origin], point])
    return stencil


def apply_stencil(stencil_block, block_type, m_inner_P2, m_padded_P2, m_padded_p):
    res_grid, var_grid = block_type
    inner_pts_2d = np.argwhere(m_inner_P2 >= 0)
    inner_idx = m_inner_P2[inner_pts_2d[:, 0], inner_pts_2d[:, 1]]
    il, cl = [], []
    for origin, offsets in stencil_block.items():
        sel = (inner_pts_2d[:, 0] % 2 == origin[0]) & (inner_pts_2d[:, 1] % 2 == origin[1])
        pts, idx = inner_pts_2d[sel], inner_idx[sel]
        for dx, dy in offsets:
            nx, ny = pts[:, 0] + dx, pts[:, 1] + dy
            if var_grid == 'v':
                contrib = m_padded_P2[nx, ny]
            else:
                contrib = m_padded_p[nx // 2, ny // 2]
            valid = contrib >= 0
            if res_grid == 'v':
                il.append(idx[valid])
            else:
                il.append(m_padded_p[pts[valid, 0] // 2, pts[valid, 1] // 2])
            cl.append(contrib[valid])
    return (np.concatenate(il).astype(np.int32),
            np.concatenate(cl).astype(np.int32))


if __name__ == '__main__':
    Nx_p, Ny_p = 4, 4
    decomp = StubDecomp(Nx_p, Ny_p)
    grid_idx = GridIndexManager(decomp, ['jx', 'jy', 'rho'])
    element = TaylorHoodP2P1(dx=1.0, dy=1.0)

    m_inner_P2  = grid_idx.index_mask_inner_local_P2
    m_padded_P2 = grid_idx.index_mask_padded_local_P2()
    m_padded_p = grid_idx.index_mask_padded_local_p()

    stencil = _compute_stencils(element)

    nb_P2 = (2*Nx_p-1) * (2*Ny_p-1)
    nb_p = Nx_p * Ny_p

    print('block        nnz   inner_range  contrib_range  no_dups')
    all_ok = True
    for bi, bt in enumerate(BLOCK):
        inner, contrib = apply_stencil(stencil[bi], bt, m_inner_P2, m_padded_P2, m_padded_p)
        nb_i = nb_P2 if bt[0] == 'v' else nb_p
        nb_c = nb_P2 if bt[1] == 'v' else nb_p
        ok_i = bool(np.all(inner >= 0) and np.all(inner < nb_i))
        ok_c = bool(np.all(contrib >= 0) and np.all(contrib < nb_c))
        no_dups = len(set(zip(inner.tolist(), contrib.tolist()))) == len(inner)
        ok = ok_i and ok_c and no_dups
        all_ok = all_ok and ok
        print(f'  {bt}  {len(inner):5d}  {ok_i!s:5}        {ok_c!s:5}          {no_dups!s:5}')
        if not no_dups:
            from collections import Counter
            pairs_list = list(zip(inner.tolist(), contrib.tolist()))
            dupes = [(k,v) for k,v in Counter(pairs_list).items() if v>1]
            print(f'    first 3 dupes (inner, contrib): {dupes[:3]}')
            # show fine-grid coords for the first duplicate inner node
            if dupes:
                i0 = dupes[0][0][0]
                c0 = dupes[0][0][1]
                if bt[0] == 'p':
                    pos = np.argwhere(m_padded_p == i0)
                    print(f'    inner p-node {i0} at coarse coords: {pos}')
                else:
                    pos = np.argwhere(m_inner_P2 == i0)
                    print(f'    inner v-node {i0} at fine coords: {pos}')
                if bt[1] == 'p':
                    pos = np.argwhere(m_padded_p == c0)
                    print(f'    contrib p-node {c0} at coarse coords: {pos}')
                else:
                    pos = np.argwhere(m_padded_P2 == c0)
                    print(f'    contrib v-node {c0} at fine coords: {pos}')
    print('\nAll OK:', all_ok)
