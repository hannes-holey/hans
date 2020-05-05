#!/usr/bin/env python3

import os
import h5py
import time
import subprocess
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
        if file.endswith("h5"):
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
        userInput = int(input("Enter file key (any other key to exit): "))
        if userInput in np.arange(0, len(availFiles)):
            # tmp_file = copyTemp(availFiles[int(userInput)][0])
            # files.update({j: [availFiles[int(userInput)][0], h5py.File(tmp_file, 'r')]})
            # subprocess.call("rm " + tmp_file, shell=True)
            files.update({j: [availFiles[userInput][0], h5py.File(availFiles[userInput][0], 'r')]})
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
        if file.endswith("h5"):
            date = time.strftime('%d/%m/%Y %H:%M', time.localtime(os.path.getmtime(file)))
            availFiles.update({i: [file, date]})
            i += 1

    print("Available files:")
    for key, val in availFiles.items():
        print("{:3d}: {:} {:20s}".format(key, val[1], val[0]))

    flag = False
    while flag is False:
        userInput = int(input("Enter file key: "))
        if userInput in np.arange(0, len(availFiles)):
            # tmp_file = copyTemp(availFiles[int(userInput)][0])
            file = h5py.File(availFiles[userInput][0], 'r')
            # file = h5py.File(tmp_file, 'r')
            # subprocess.call("rm " + tmp_file, shell=True)
            flag = True
        else:
            print("File not in list. Try again!")
            flag = False
    return file


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

    userInput = int(input("Enter file key: "))
    if userInput in np.arange(0, len(availFiles)):
        ref = np.loadtxt(availFiles[userInput], skiprows=1)
        assert ref.ndim == 2, "Input data file has wrong dimension"
        return ref
    else:
        print("No comparison!")
        return None


def copyTemp(filename):

    tmp_filename = os.path.splitext(filename)[0] + "_tmp.h5"

    subprocess.call("cp " + filename + " " + tmp_filename, shell=True)

    return tmp_filename
