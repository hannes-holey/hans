#!/usr/bin/env python3

import os
import netCDF4
import time
import numpy as np


def getData(path, single=False):

    ncFileList = [os.path.join(path, f) for f in sorted(os.listdir(path)) if f.endswith(".nc")]

    print("Available files:")
    for i, file in enumerate(ncFileList):
        date = time.strftime('%d/%m/%Y %H:%M', time.localtime(os.path.getmtime(file)))
        print(f"{i:3d}: {file} {date:>}")

    if single:
        mask = [int(input("Enter file key: "))]
    else:
        mask = [int(i) for i in input("Enter file keys (space separated): ").split()]

    out = {f: netCDF4.Dataset(f) for i,f in enumerate(ncFileList) if i in mask}

    return out


def getBenchmark():
    availFiles = {}
    i = 0
    for file in sorted(os.listdir("../data/benchmark")):
        availFiles.update({i: os.path.join("../data/benchmark", file)})
        i += 1

    if len(availFiles) == 0:
        print("No *.dat files found.")
        return None
    else:
        print("Available data files for comparison:")

    for key, val in availFiles.items():
        print("{:3d}: {:20s}".format(key, val))

    userInput = input("Enter file key: ")
    if userInput in np.arange(0, len(availFiles)).astype(str):
        ref = np.loadtxt(availFiles[int(userInput)], skiprows=1)
        assert ref.ndim == 2, "Input data file has wrong dimension"
        return ref
    else:
        print("No comparison!")
        return None
