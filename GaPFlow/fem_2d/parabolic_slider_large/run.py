"""
Parabolic slider bearing — large domain memory check (500×500).

Run this script from its directory:
    python run.py
"""

import os
import time
import tracemalloc

import numpy as np
import matplotlib.pyplot as plt
import GaPFlow

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# --- Memory tracking setup ---
tracemalloc.start()
t_wall_start = time.perf_counter()

mem_log = []  # list of (simtime, rss_MB, peak_MB)

def _snapshot_memory(label):
    current, peak = tracemalloc.get_traced_memory()
    mem_log.append({
        'label': label,
        'current_MB': current / 1024**2,
        'peak_MB': peak / 1024**2,
        'wall_s': time.perf_counter() - t_wall_start,
    })

_snapshot_memory('before_load')

problem = GaPFlow.Problem.from_yaml('parabolic_slider_large.yaml')

_snapshot_memory('after_load')

# --- Per-step tracking ---
step_log = []  # list of (step, simtime, current_MB, peak_MB, wall_s)

def track_step():
    current, peak = tracemalloc.get_traced_memory()
    step_log.append({
        'step': problem.step,
        'simtime': problem.simtime,
        'current_MB': current / 1024**2,
        'peak_MB': peak / 1024**2,
        'wall_s': time.perf_counter() - t_wall_start,
    })

problem.add_callback(track_step)
problem._pre_run()  # to initialize the solver and get initial memory usage

_snapshot_memory('after pre_run')
problem.run()
_snapshot_memory('after_run')

t_wall_total = time.perf_counter() - t_wall_start
current_final, peak_final = tracemalloc.get_traced_memory()
tracemalloc.stop()

# --- Print memory summary ---
print()
print('=' * 60)
print('Memory usage summary (tracemalloc)')
print('=' * 60)
print(f"{'Label':<20} {'Current [MB]':>14} {'Peak [MB]':>12} {'Wall [s]':>10}")
print('-' * 60)
for entry in mem_log:
    print(f"{entry['label']:<20} {entry['current_MB']:>14.1f} {entry['peak_MB']:>12.1f} {entry['wall_s']:>10.2f}")
print('-' * 60)
print(f"{'FINAL':<20} {current_final/1024**2:>14.1f} {peak_final/1024**2:>12.1f} {t_wall_total:>10.2f}")
print('=' * 60)
print()

if step_log:
    print(f"{'Step':<6} {'Simtime':>10} {'Current [MB]':>14} {'Peak [MB]':>12} {'Wall [s]':>10}")
    print('-' * 56)
    for s in step_log:
        print(f"{s['step']:<6} {s['simtime']:>10.3e} {s['current_MB']:>14.1f} {s['peak_MB']:>12.1f} {s['wall_s']:>10.2f}")
    print()

# --- Save memory log as numpy ---
if step_log:
    steps = np.array([s['step'] for s in step_log])
    simtimes = np.array([s['simtime'] for s in step_log])
    current_mb = np.array([s['current_MB'] for s in step_log])
    peak_mb = np.array([s['peak_MB'] for s in step_log])
    wall_s = np.array([s['wall_s'] for s in step_log])

    np.savez('memory_log.npz',
             steps=steps, simtimes=simtimes,
             current_mb=current_mb, peak_mb=peak_mb, wall_s=wall_s)
    print('Saved memory_log.npz')

    # --- Plot memory usage ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    ax1.plot(steps, current_mb, 'b-o', ms=4, label='Current')
    ax1.plot(steps, peak_mb, 'r--s', ms=4, label='Peak')
    ax1.set_ylabel('Memory [MB]')
    ax1.set_title('tracemalloc memory usage — parabolic_slider_large (500×500)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(steps, wall_s, 'g-o', ms=4)
    ax2.set_ylabel('Wall time [s]')
    ax2.set_xlabel('Step')
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig('memory_usage.png', dpi=150)
    plt.show()
    print('Saved memory_usage.png')
