#
# Copyright 2026 Christoph Huber
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
"""Per-iteration Newton diagnostics for the Taylor-Hood P2P1 FEM solver.

Activated by setting ``newton_debug: true`` in the ``fem_solver`` config block.
Produces:
  - Console output: per-block residual/update norms and solution ranges.
  - PNG plots: 2D imshow of residual and dq fields per Newton iteration,
    saved to ``<output>/newton_debug/ts{timestep:04d}_it{it:02d}.png``.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ..models.pressure import eos_pressure, eos_rho


# Grid type for each variable / residual name
_VAR_GRID = {'jx': 'v', 'jy': 'v', 'rho': 'p', 'p': 'p', 'E': 'p', 'theta': 'p', 'xi': 'p'}
_RES_GRID = {'momentum_x': 'v', 'momentum_y': 'v', 'mass': 'p', 'energy': 'p', 'fb': 'p', 'R_oss': 'p'}

# Display labels
_VAR_LABEL = {'jx': 'dq jx', 'jy': 'dq jy', 'rho': 'dq rho', 'p': 'dq p', 'E': 'dq E',
              'theta': 'dq theta', 'xi': 'dq xi'}
_RES_LABEL = {'momentum_x': 'R mom_x', 'momentum_y': 'R mom_y',
              'mass': 'R mass', 'energy': 'R energy', 'fb': 'R FB', 'R_oss': 'R OSS'}


def _minmax_title(label, field):
    """Build a title string with min/max (x,y) locations."""
    imin = np.unravel_index(np.argmin(field), field.shape)
    imax = np.unravel_index(np.argmax(field), field.shape)
    return f'{label}\nmin@({imin[0]},{imin[1]})  max@({imax[0]},{imax[1]})'


class NewtonDebugger:
    """Captures and plots Newton iteration diagnostics.

    Parameters
    ----------
    output_dir : str
        Base output directory (same as ``options.output``).
    variables : list of str
        Variable names in solver block order, e.g. ``['jx', 'jy', 'rho']``.
    residuals : list of str
        Residual names in solver block order, e.g. ``['momentum_x', 'momentum_y', 'mass']``.
    res_slices : dict
        Mapping residual name -> slice into the flat R vector.
    sol_slices : dict
        Mapping variable name -> slice into the flat dq vector.
    Nx_p : int
        Number of inner P1 nodes in x.
    Ny_p : int
        Number of inner P1 nodes in y.
    Nx_v : int
        Number of inner P2 nodes in x.
    Ny_v : int
        Number of inner P2 nodes in y.
    """

    def __init__(self, output_dir, variables, residuals,
                 res_slices, sol_slices,
                 Nx_p, Ny_p, Nx_v, Ny_v, terms, problem, quad_mgr=None):
        self.variables = variables
        self.residuals = residuals
        self.res_slices = res_slices
        self.sol_slices = sol_slices
        self.Nx_p = Nx_p
        self.Ny_p = Ny_p
        self.Nx_v = Nx_v
        self.Ny_v = Ny_v
        # Map term name -> residual equation name
        self.term_res = {t.name: t.res for t in terms}
        self.problem = problem
        self.quad_mgr = quad_mgr

        self.plot_dir = os.path.join(output_dir, 'newton_debug')
        if os.path.isdir(self.plot_dir):
            for f in os.listdir(self.plot_dir):
                if f.endswith('.png'):
                    os.remove(os.path.join(self.plot_dir, f))
        os.makedirs(self.plot_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API — called once per Newton iteration
    # ------------------------------------------------------------------

    def step(self, timestep: int, it: int, R: np.ndarray, dq: np.ndarray,
             q: np.ndarray, R_per_term: dict, M_scaled: np.ndarray = None) -> None:
        """Print console summary and save field plots for one Newton step.

        Parameters
        ----------
        timestep : int
            Outer time step index (0-based).
        it : int
            Newton iteration index (0-based).
        R : np.ndarray
            Flat residual vector before the linear solve.
        dq : np.ndarray
            Flat solution increment (unscaled) from the linear solve.
        q : np.ndarray
            Flat solution vector *after* applying dq (i.e. q_new = q_old + alpha*dq).
        R_per_term : dict
            Mapping term name -> flat residual vector for that term only.
        """
        self._print_summary(timestep, it, R, dq, q, R_per_term, M_scaled)
        self._save_plots(timestep, it, R, dq)
        self._save_term_plots(timestep, it, R_per_term)
        self._save_solution_plots(timestep, it, q)
        if 'xi' in self.variables and self.quad_mgr is not None:
            self._save_oss_plots(timestep, it, q, R_per_term)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _block_norm(self, vec, slices, names):
        """Return dict name -> L2 norm of each named slice."""
        return {name: float(np.linalg.norm(vec[slices[name]])) for name in names}

    def _block_range(self, vec, slices, names):
        """Return dict name -> (min, max) of each named slice."""
        return {name: (float(vec[slices[name]].min()),
                       float(vec[slices[name]].max())) for name in names}

    def _print_summary(self, timestep, it, R, dq, q, R_per_term, M_scaled=None):
        r_norms = self._block_norm(R, self.res_slices, self.residuals)
        dq_norms = self._block_norm(dq, self.sol_slices, self.variables)
        q_ranges = self._block_range(q, self.sol_slices, self.variables)

        header = f"  [ts={timestep:04d} it={it:02d}]"
        r_str = "  ||R||: " + "  ".join(
            f"{name}={r_norms[name]:.3e}" for name in self.residuals)
        dq_str = "  ||dq||: " + "  ".join(
            f"{name}={dq_norms[name]:.3e}" for name in self.variables)
        q_str = "  q range: " + "  ".join(
            f"{name}=[{q_ranges[name][0]:.3g}, {q_ranges[name][1]:.3g}]"
            for name in self.variables)

        print(header)
        print(r_str)
        print(dq_str)
        print(q_str)

        if M_scaled is not None:
            abs_vals = np.abs(M_scaled)
            nonzero = abs_vals[abs_vals > 0]
            if len(nonzero):
                cond_est = float(nonzero.max() / nonzero.min())
                print(f"  cond(M_scaled) ~ max/min = {cond_est:.3e}  "
                      f"max={float(nonzero.max()):.3e}  min={float(nonzero.min()):.3e}")

        for term_name, R_term in R_per_term.items():
            res_name = self.term_res[term_name]
            block = R_term[self.res_slices[res_name]]
            print(f"    {term_name:12s}  min={block.min():.3e}  max={block.max():.3e}  ||.||={np.linalg.norm(block):.3e}")
            if term_name == 'R24x':
                grid = _RES_GRID[res_name]
                Nx = self.Nx_v if grid == 'v' else self.Nx_p
                Ny = self.Ny_v if grid == 'v' else self.Ny_p
                field = block.reshape(Nx, Ny, order='F')
                print(f"    R24x field[:8, :] (x=0..7, all y):")
                print(np.array2string(field[:8, :], precision=3, suppress_small=True))

    def _to_2d(self, vec, name, is_residual):
        """Reshape a named block from the flat vector to a 2D (Nx, Ny) array."""
        slices = self.res_slices if is_residual else self.sol_slices
        grid = _RES_GRID[name] if is_residual else _VAR_GRID[name]
        Nx = self.Nx_v if grid == 'v' else self.Nx_p
        Ny = self.Ny_v if grid == 'v' else self.Ny_p
        return vec[slices[name]].reshape(Nx, Ny, order='F')

    def _save_plots(self, timestep, it, R, dq):
        n_vars = len(self.variables)
        n_res = len(self.residuals)
        n_cols = max(n_vars, n_res)

        fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 6),
                                 squeeze=False, facecolor='white')
        for ax in axes.flat:
            ax.set_facecolor('white')
        fig.suptitle(f'Newton diagnostics  ts={timestep:04d}  it={it:02d}',
                     fontsize=11)

        # Top row: residual blocks
        for col, name in enumerate(self.residuals):
            ax = axes[0, col]
            field = self._to_2d(R, name, is_residual=True)
            im = ax.imshow(field.T, origin='lower', aspect='auto',
                           cmap='RdBu_r')
            ax.set_title(_minmax_title(_RES_LABEL[name], field), fontsize=7)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        for col in range(n_res, n_cols):
            axes[0, col].set_visible(False)

        # Bottom row: dq blocks
        for col, name in enumerate(self.variables):
            ax = axes[1, col]
            field = self._to_2d(dq, name, is_residual=False)
            im = ax.imshow(field.T, origin='lower', aspect='auto',
                           cmap='RdBu_r')
            ax.set_title(_minmax_title(_VAR_LABEL[name], field), fontsize=7)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        for col in range(n_vars, n_cols):
            axes[1, col].set_visible(False)

        fname = os.path.join(self.plot_dir,
                             f'ts{timestep:04d}_it{it:02d}.png')
        fig.savefig(fname, dpi=120, bbox_inches='tight', facecolor='white')
        plt.close(fig)

    def _to_2d_res(self, R_term, res_name):
        """Reshape the slice of a per-term residual vector to 2D (Nx, Ny)."""
        grid = _RES_GRID[res_name]
        Nx = self.Nx_v if grid == 'v' else self.Nx_p
        Ny = self.Ny_v if grid == 'v' else self.Ny_p
        return R_term[self.res_slices[res_name]].reshape(Nx, Ny, order='F')

    def _save_term_plots(self, timestep, it, R_per_term):
        # Group terms by residual, preserving order within each group
        groups = {res: [] for res in self.residuals}
        for term_name, res_name in self.term_res.items():
            if term_name in R_per_term:
                groups[res_name].append(term_name)

        n_cols = max((len(names) for names in groups.values()), default=1)
        n_rows = len(self.residuals)

        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(4 * n_cols, 3 * n_rows),
                                 squeeze=False, facecolor='white')
        for ax in axes.flat:
            ax.set_facecolor('white')
        fig.suptitle(f'Per-term residuals  ts={timestep:04d}  it={it:02d}',
                     fontsize=11)

        for row, res_name in enumerate(self.residuals):
            term_names = groups[res_name]
            for col, term_name in enumerate(term_names):
                ax = axes[row, col]
                field = self._to_2d_res(R_per_term[term_name], res_name)
                im = ax.imshow(field.T, origin='lower', aspect='auto',
                               cmap='RdBu_r')
                ax.set_title(_minmax_title(f'{_RES_LABEL[res_name]} {term_name}', field), fontsize=6)
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            for col in range(len(term_names), n_cols):
                axes[row, col].set_visible(False)

        fname = os.path.join(self.plot_dir,
                             f'ts{timestep:04d}_it{it:02d}_terms.png')
        fig.savefig(fname, dpi=120, bbox_inches='tight', facecolor='white')
        plt.close(fig)

    def _save_solution_plots(self, timestep, it, q):
        jx_2d = self._to_2d(q, 'jx', is_residual=False)
        jy_2d = self._to_2d(q, 'jy', is_residual=False)
        prop = self.problem.prop
        if 'p' in self.variables:
            p_2d = self._to_2d(q, 'p', is_residual=False)
            rho_2d = np.array(eos_rho(p_2d, prop))
        else:
            rho_2d = self._to_2d(q, 'rho', is_residual=False)
            p_2d = np.array(eos_pressure(rho_2d, prop))

        fields = [
            (rho_2d, r'$\rho$ [kg/m³]', 'viridis'),
            (p_2d,   r'$p$ [Pa]',        'coolwarm'),
            (jx_2d,  r'$j_x$ [kg/m²s]', 'RdBu_r'),
            (jy_2d,  r'$j_y$ [kg/m²s]', 'RdBu_r'),
        ]

        fig, axes = plt.subplots(1, 4, figsize=(16, 3), facecolor='white')
        for ax in axes.flat:
            ax.set_facecolor('white')
        fig.suptitle(f'Solution fields  ts={timestep:04d}  it={it:02d}', fontsize=11)

        for ax, (field, title, cmap) in zip(axes, fields):
            im = ax.imshow(field.T, origin='lower', aspect='auto', cmap=cmap)
            ax.set_title(_minmax_title(title, field), fontsize=7)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        fname = os.path.join(self.plot_dir,
                             f'ts{timestep:04d}_it{it:02d}_solution.png')
        fig.savefig(fname, dpi=120, bbox_inches='tight', facecolor='white')
        plt.close(fig)

    def _quad_p1(self, name: str) -> np.ndarray:
        """Average quad field over sub-points → inner (Nx_p, Ny_p) array."""
        return self.quad_mgr.quad_fields[name].pg.mean(axis=0)[1:-1, 1:-1]

    def _save_oss_plots(self, timestep: int, it: int, q: np.ndarray,
                        R_per_term: dict) -> None:
        """Save OSS stabilization diagnostic plot.

        All residual contributions are taken directly from R_per_term (the exact
        assembled values) rather than recomputed from quad fields.
        """
        alpha = self.problem.fem_solver.get('oss_theta_alpha', 1.0)

        # Solution fields
        theta = self._to_2d(q, 'theta', is_residual=False)
        xi = self._to_2d(q, 'xi', is_residual=False)

        # Context quad fields (averaged to P1 inner grid for display)
        norm_a = np.sqrt(self._quad_p1('a_vec_x')**2 + self._quad_p1('a_vec_y')**2)
        div_j = self._quad_p1('d_dx_jx') + self._quad_p1('d_dy_jy')

        def _rterm(name):
            """Extract the residual block for a term as a 2D array, or zeros."""
            if name not in R_per_term:
                return None
            res_name = self.term_res[name]
            return self._to_2d_res(R_per_term[name], res_name)

        def _sum_terms(*names):
            """Sum assembled residual blocks for the given term names, or None if all absent."""
            arrays = [_rterm(n) for n in names if _rterm(n) is not None]
            if not arrays:
                return None
            return arrays[0] if len(arrays) == 1 else sum(arrays[1:], arrays[0])

        # R_oss equation contributions (projection equation)
        R_adv  = _sum_terms('R_OSS_advx', 'R_OSS_advy')
        R_div  = _rterm('R_OSS_div')
        R_proj = _rterm('R_OSS_proj')
        R_oss_net = _sum_terms('R_OSS_advx', 'R_OSS_advy', 'R_OSS_div', 'R_OSS_proj')

        # mass equation OSS contributions
        R_lap  = _sum_terms('R_OSS_mass_xx', 'R_OSS_mass_yy',
                            'R_OSS_mass_xy', 'R_OSS_mass_yx')
        R_corr = _sum_terms('R_OSS_corrx', 'R_OSS_corry')
        R_mass_oss_net = _sum_terms('R_OSS_mass_xx', 'R_OSS_mass_yy',
                                    'R_OSS_mass_xy', 'R_OSS_mass_yx',
                                    'R_OSS_corrx', 'R_OSS_corry')

        def _symvlim(*arrays):
            vmax = max((np.abs(a).max() for a in arrays if a is not None), default=1e-30)
            return dict(vmin=-vmax, vmax=vmax)

        panels = [
            # Row 1: solution context
            (theta,       r'$\theta$',                       'viridis', {}),
            (xi,          r'$\xi_h$',                        'RdBu_r',  {}),
            (norm_a,      r'$|\mathbf{a}|$',                 'viridis', {}),
            (div_j,       r'$\nabla\cdot j$ (context)',      'RdBu_r',  {}),
            # Row 2: assembled R_oss contributions
            (R_adv,       r'$R_{oss}$: adv ($a\cdot\nabla\theta$)',  'RdBu_r', _symvlim(R_adv, R_div, R_proj)),
            (R_div,       r'$R_{oss}$: div ($(1-\theta)\nabla\cdot j$)', 'RdBu_r', _symvlim(R_adv, R_div, R_proj)),
            (R_proj,      r'$R_{oss}$: proj ($-\xi$)',       'RdBu_r',  _symvlim(R_adv, R_div, R_proj)),
            (R_oss_net,   r'$R_{oss}$ net',                  'RdBu_r',  _symvlim(R_adv, R_div, R_proj)),
            # Row 3: assembled mass equation OSS contributions
            (R_lap,       r'$R_{mass}$: OSS Laplacian',      'RdBu_r',  _symvlim(R_lap, R_corr, R_mass_oss_net)),
            (R_corr,      r'$R_{mass}$: OSS correction',     'RdBu_r',  _symvlim(R_lap, R_corr, R_mass_oss_net)),
            (R_mass_oss_net, r'$R_{mass}$: OSS net',         'RdBu_r',  _symvlim(R_lap, R_corr, R_mass_oss_net)),
        ]
        panels = [(f, l, c, kw) for f, l, c, kw in panels if f is not None]

        n_cols = 4
        n_rows = (len(panels) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(4.5 * n_cols, 3.5 * n_rows),
                                 squeeze=False, facecolor='white')
        for ax in axes.flat:
            ax.set_facecolor('white')
        fig.suptitle(f'OSS diagnostics  ts={timestep:04d}  it={it:02d}  α={alpha}',
                     fontsize=11)

        for ax, (field, label, cmap, kwargs) in zip(axes.flat, panels):
            im = ax.imshow(field.T, origin='lower', aspect='auto', cmap=cmap, **kwargs)
            ax.set_title(_minmax_title(label, field), fontsize=8)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        for ax in list(axes.flat)[len(panels):]:
            ax.set_visible(False)

        fname = os.path.join(self.plot_dir,
                             f'ts{timestep:04d}_it{it:02d}_oss.png')
        fig.savefig(fname, dpi=120, bbox_inches='tight', facecolor='white')
        plt.close(fig)
