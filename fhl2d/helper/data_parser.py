import os
import netCDF4
import time
import numpy as np


def getData(path, prefix="", suffix="nc", mode="select"):
    """
    Function to interactively select data files for further processing, e.g. for plotting.

    Parameters
    ----------
    path : str
        relative path below which is searched for files
    prefix : str
        filter files that start with prefix, default=""
    suffix : str
        filter files that end with suffix, default="nc"
    mode : str
        can be one of the following, default="select"
        - select: manually select files
        - single: manually select a single file
        - all: select all files found below path with prefix and suffix

    Returns
    ----------
    out : dict
        dictionary where keys are filenames and values are corresponding datasets.
        Datasets currently only implemented for suffices "nc" (netCDF4.Dataset) and "dat" (numpy.ndarray).
        Else, values are None.
    """

    assert mode in ["single", "select", "all"], f"mode must be 'single', select or 'all'"

    fileList = []

    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            if name.startswith(prefix) and name.endswith(suffix):
                fileList.append(os.path.join(root, name))

    fileList = sorted(fileList)

    print("Available files:")
    for i, file in enumerate(fileList):
        date = time.strftime('%d/%m/%Y %H:%M', time.localtime(os.path.getmtime(file)))
        print(f"{i:3d}: {file:<50} {date}")

    loader = {"nc": netCDF4.Dataset,
              "dat": np.loadtxt,
              "out": lambda s: None,
              "txt": lambda s: None,
              "yaml": lambda s: None,
              "yml": lambda s: None}

    if mode == "single":
        mask = [int(input("Enter file key: "))]
    elif mode == "all":
        mask = list(range(len(fileList)))
    elif mode == "select":
        inp = input("Enter file keys (space separated or range [start]-[end] or combination of both): ")

        mask = [int(i) for i in inp.split() if len(i.split("-")) < 2]
        mask_range = [i for i in inp.split() if len(i.split("-")) == 2]

        for j in mask_range:
            mask += list(range(int(j.split("-")[0]), int(j.split("-")[1]) + 1))

    out = {f: loader[suffix](f) for i, f in enumerate(fileList) if i in mask}

    return out


def getSubDirs(path, mode="select"):
    """
    Function to interactively select subdirectories.

    Parameters
    ----------
    path : str
        relative path in which is searched for directories.
    mode : str
        can be one of the following, default="select"
        - select: manually select directories
        - single: manually select a single directory
        - all: select all directories found in path

    Returns
    ----------
    out : list
        list with subdirectory names (relative path)
    """

    assert mode in ["single", "select", "all"], f"mode must be 'single', select or 'all'"

    subdirs = [os.path.join(path, i) for i in sorted(next(os.walk(path))[1])]

    for i, sub in enumerate(subdirs):
        date = time.strftime('%d/%m/%Y %H:%M', time.localtime(os.path.getmtime(sub)))
        print(f"{i:3d}: {sub:<50} {date}")

    if mode == "single":
        mask = [int(input("Enter file key: "))]
    elif mode == "all":
        mask = list(range(len(subdirs)))
    elif mode == "select":
        inp = input("Enter file keys (space separated or range [start]-[end] or combination of both): ")

        mask = [int(i) for i in inp.split() if len(i.split("-")) < 2]
        mask_range = [i for i in inp.split() if len(i.split("-")) == 2]

        for j in mask_range:
            mask += list(range(int(j.split("-")[0]), int(j.split("-")[1]) + 1))

    out = [d for i, d in enumerate(subdirs) if i in mask]

    return out


def get_nc_from_name(fname):
    """
    Get netCDF4 Dataset from filename

    Parameters
    ----------
    fname : str
        filename
    Returns
    ----------
    out : netCDF4.Dataset
        Dataset
    """

    out = netCDF4.Dataset(fname)
    return out
