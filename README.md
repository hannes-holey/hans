# HANS

[![PyPI version](https://badge.fury.io/py/hans.svg)](https://badge.fury.io/py/hans) [![CI](https://github.com/hannes-holey/pylub/actions/workflows/ci.yaml/badge.svg?branch=master)](https://github.com/hannes-holey/pylub/actions/workflows/ci.yaml) 

A Height-Averaged Navier-Stokes (HANS) solver for 2D lubrication problems

## Installation
Install from the main source directory via
```
pip install .
```

## Tests
Run serial tests from the main source directory with
```
pytest [name_of_test_file]
```
or without arguments to run all tests.

## Examples
Run from the command line with
```
mpirun -n <NP> python3 -m hans -i <input_file> [-p] [-r <restart_file>]
```
where NP is the number of MPI processes. The plot option (-p, --plot) is only available for serial execution.
Example input files as well as jupyter-notebooks can be found in the *examples* directory.

The command line interface contains some scripts for plotting and creating animations.
For instance, 1D profiles of converged solutions can be displayed with
```
plot1D_last.py
```
