# HANS

[![PyPI version](https://badge.fury.io/py/hans.svg)](https://badge.fury.io/py/hans) [![CI](https://github.com/hannes-holey/hans/actions/workflows/ci.yaml/badge.svg?branch=master)](https://github.com/hannes-holey/hans/actions/workflows/ci.yaml) [![Coverage](https://gist.githubusercontent.com/hannes-holey/fac7fa61e1899b1e74b3bab598fe6513/raw/badge.svg)](https://gist.githubusercontent.com/hannes-holey/fac7fa61e1899b1e74b3bab598fe6513/raw/badge.svg)

https://gist.github.com/fac7fa61e1899b1e74b3bab598fe6513.git

This code implements the Height-Averaged Navier-Stokes (HANS) scheme for two-dimensional lubrication problems as described in the following paper:

[Holey, H. et al. (2022) Tribology Letters, 70(2), p. 36.
](https://doi.org/10.1007/s11249-022-01576-5)

## Installation
Packaged versions can be installed via
```
pip install hans
```

## Examples
Run from the command line with
```
mpirun -n <NP> python3 -m hans -i <input_file> [-p] [-r <restart_file>]
```
where `NP` is the number of MPI processes. The plot option (`-p`, `--plot`) is only available for serial execution. Example input files as well as jupyter-notebooks can be found in the [examples](examples/) directory.

The command line interface contains some scripts for plotting and creating animations.
For instance, 1D profiles of converged solutions can be displayed with
```
plot1D_last.py
```

## Tests
Run all tests from the main source directory with
```
pytest
```
or append the path to the test definition file (located in [tests](tests)) to run selected tests only.

## Funding
This work is funded by the German Research Foundation (DFG) through GRK 2450.
