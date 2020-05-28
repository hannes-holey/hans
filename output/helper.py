#!/usr/bin/env python3

import os
import netCDF4
import time
import numpy as np


def getFiles():
    availFiles = {}
    i = 0

    pathList = os.getcwd().split(sep=os.path.sep)
    assert "MD-FVM" in pathList, "Not in a subdirectory of MD-FVM"

    if pathList[-2:] != ["MD-FVM", "output"]:
        depth = len(pathList) - pathList.index("MD-FVM") - 1
        os.chdir(depth * (".." + os.path.sep) + "output")

    for file in sorted(os.listdir()):
        if file.endswith(".nc"):
            date = time.strftime('%d/%m/%Y %H:%M', time.localtime(os.path.getmtime(file)))
            availFiles.update({i: [file, date]})
            i += 1

    print("Available files:")
    for key, val in availFiles.items():
        print("{:3d}: {:} {:}".format(key, val[1], val[0]))

    files = {}
    ask = True
    j = 0
    while ask is True:
        userInput = input("Enter file key (any other key to exit): ")
        if userInput in np.arange(0, len(availFiles)).astype(str):
            filename = availFiles[int(userInput)][0]
            files.update({j: [filename, netCDF4.Dataset(filename)]})
            j += 1
        else:
            ask = False

    return files


def getFile():
    availFiles = {}
    i = 0

    pathList = os.getcwd().split(sep=os.path.sep)
    assert "MD-FVM" in pathList, "Not in a subdirectory of MD-FVM"

    if pathList[-2:] != ["MD-FVM", "output"]:
        depth = len(pathList) - pathList.index("MD-FVM") - 1
        os.chdir(depth * (".." + os.path.sep) + "output")

    for file in sorted(os.listdir()):
        if file.endswith(".nc"):
            date = time.strftime('%d/%m/%Y %H:%M', time.localtime(os.path.getmtime(file)))
            availFiles.update({i: [file, date]})
            i += 1

    print("Available files:")
    for key, val in availFiles.items():
        print("{:3d}: {:} {:20s}".format(key, val[1], val[0]))

    flag = False
    while flag is False:
        userInput = input("Enter file key: ")
        if userInput in np.arange(0, len(availFiles)).astype(str):
            filename = availFiles[int(userInput)][0]
            file = netCDF4.Dataset(filename)
            flag = True
        else:
            print("File not in list. Try again!")
            flag = False
    return filename, file


def getReference():
    availFiles = {}
    i = 0
    for file in sorted(os.listdir()):
        if file.endswith("dat"):
            availFiles.update({i: file})
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
