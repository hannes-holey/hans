"""Run Bernoulli venturi case and save results to .npz."""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from GaPFlow.problem import Problem
from GaPFlow.models.pressure import eos_pressure

# Geometry parameters (must match notebook)
H_INLET, H_THROAT = 1.0e-3, 0.5e-3
X_RAMP_START, X_RAMP_END = 0.03, 0.07
THROAT_HALF_WIDTH = 0.005
RHO0, V_INLET = 1000.0, 5.0

CONFIG = Path(__file__).resolve().parents[3] / 'tests' / 'configs' / 'bernoulli_venturi.yaml'
OUTPUT_DIR = Path(__file__).resolve().parent


def venturi_topography(xx):
    """Symmetric venturi with cosine transitions."""
    x_mid = (X_RAMP_START + X_RAMP_END) / 2
    ramp_len = x_mid - THROAT_HALF_WIDTH - X_RAMP_START
    dist = np.clip(np.abs(xx - x_mid) - THROAT_HALF_WIDTH, 0, ramp_len)
    xi = dist / ramp_len
    return np.where(
        (xx >= X_RAMP_START) & (xx <= X_RAMP_END),
        (H_INLET + H_THROAT) / 2 - (H_INLET - H_THROAT) / 2 * np.cos(np.pi * xi),
        H_INLET,
    )


def main():
    problem = Problem.from_yaml(str(CONFIG))

    h_venturi = venturi_topography(problem.topo.xx)
    problem.topo.set_mapped_height(h_venturi)
    problem.q[1][:] = RHO0 * V_INLET

    problem.run()

    # Extract centerline data
    rho = problem.q[0][1:-1, 1:-1]
    jx = problem.q[1][1:-1, 1:-1]
    h = problem.topo.h[1:-1, 1:-1]
    j_center = rho.shape[1] // 2

    dx = problem.grid['dx']
    x = np.arange(rho.shape[0]) * dx + dx / 2

    p_sim = np.asarray(eos_pressure(rho[:, j_center], problem.prop))
    jx_sim = jx[:, j_center]

    # Bernoulli theory
    v_theory = V_INLET * H_INLET / h[:, j_center]
    jx_theory = RHO0 * v_theory
    p_ref = p_sim[-1]
    v_ref = V_INLET * H_INLET / h[-1, j_center]
    p_theory = p_ref + 0.5 * RHO0 * (v_ref**2 - v_theory**2)

    out_path = OUTPUT_DIR / 'bernoulli_venturi.npz'
    np.savez(
        out_path,
        x=x,
        p_sim=p_sim,
        p_theory=p_theory,
        jx_sim=jx_sim,
        jx_theory=jx_theory,
    )
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    main()
