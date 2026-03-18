#!/usr/bin/env python
"""Run all FEM 2D test cases and print residual analysis."""

import os
import sys
import time

from GaPFlow import Problem

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIGS_DIR = os.path.join(BASE_DIR, 'configs')

CASES = [
    'inclined_slider',
    'parabolic_slider',
    'journal_bearing',
    'parabolic_slider_2d',
    'parabolic_slider_2d_deform',
    'parabolic_slider_2d_deform_energy',
]


def run_case(name):
    """Run a single test case."""
    config_path = os.path.join(CONFIGS_DIR, f'{name}.yaml')
    print(f'\n{"=" * 70}')
    print(f'  {name}')
    print(f'{"=" * 70}')

    t0 = time.time()
    problem = Problem.from_yaml(config_path)
    problem.run()
    elapsed = time.time() - t0

    print(f'\n  Elapsed: {elapsed:.1f}s')


def main():
    # chdir so relative output paths (data/<name>) land inside fem_2d_tests/
    os.chdir(BASE_DIR)

    cases = CASES
    if len(sys.argv) > 1:
        cases = sys.argv[1:]

    for name in cases:
        try:
            run_case(name)
        except Exception as e:
            print(f'\n  FAILED: {e}')

    print(f'\n{"=" * 70}')
    print('  All cases finished.')
    print(f'{"=" * 70}')


if __name__ == '__main__':
    main()
