#!/usr/bin/env python3

import os
import netCDF4
import time
import numpy as np


def getData(path, prefix="", suffix="nc", single=False, all=False):

    fileList = sorted(getFromSubDir(path, prefix, suffix))

    print("Available files:")
    for i, file in enumerate(fileList):
        date = time.strftime('%d/%m/%Y %H:%M', time.localtime(os.path.getmtime(file)))
        print(f"{i:3d}: {file} {date:>}")

    loader = {"nc": netCDF4.Dataset,
              "dat": np.loadtxt,
              "out": lambda s: None,
              "txt": lambda s: None,
              "yaml": lambda s: None,
              "yml": lambda s: None}

    if single:
        mask = [int(input("Enter file key: "))]
    else:
        inp = input("Enter file keys (space separated or range [start]-[end] or combination of both): ")

        mask = [int(i) for i in inp.split() if len(i.split("-")) < 2]
        mask_range = [i for i in inp.split() if len(i.split("-")) == 2]

        for j in mask_range:
            mask += list(range(int(j.split("-")[0]), int(j.split("-")[1]) + 1))

    out = {f: loader[suffix](f) for i, f in enumerate(fileList) if i in mask}

    return out


def getFromSubDir(path, prefix, suffix):

    fileList = []

    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            if name.startswith(prefix) and name.endswith(suffix):
                fileList.append(os.path.join(root, name))

    return fileList
