# pylub
A 2D fluid mechanics solver for lubrication

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
mpirun -n <NP> python3 -m pylub -i <input_file> [-p] [-r <restart_file>]
```
where NP is the number of MPI processes. The plot option (-p, --plot) is only available for serial execution.
Example input files as well as jupyter-notebooks can be found in the *examples* direcory.

The command line interface contains some scripts for plotting and creating animations. For instance, 1D profiles of the
converged solution can be displayed with
```
plot1D_last.py
```
