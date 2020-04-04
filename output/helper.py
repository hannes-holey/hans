#!/usr/bin/env python3

import os
import h5py
import subprocess
import numpy as np

def getFiles():
    availFiles = {}
    i = 0
    for file in sorted(os.listdir()):
        if file.endswith('h5'):
            availFiles.update({i: file})
            i +=1

    print("Available files:")
    for key, val in availFiles.items():
        print("{:3d}: {:20s}".format(key, val))

    files = {}
    ask = True
    j = 0
    while ask == True:
        userInput = input("Enter file key (any other key to exit): ")
        if userInput in np.arange(0, len(availFiles)).astype(str):
            tmp_file = copyTemp(availFiles[int(userInput)])
            files.update({j: [availFiles[int(userInput)], h5py.File(tmp_file, 'r')]})
            subprocess.call("rm " + tmp_file, shell = True)
            j += 1
        else:
            ask = False

    return files

def getFile():
    availFiles = {}
    i = 0
    for file in sorted(os.listdir()):
        if file.endswith('h5'):
            availFiles.update({i: file})
            i +=1

    print("Available files:")
    for key, val in availFiles.items():
        print("{:3d}: {:20s}".format(key, val))

    flag = False
    while flag == False:
        userInput = input("Enter file key: ")
        if userInput in np.arange(0, len(availFiles)).astype(str):
            tmp_file = copyTemp(availFiles[int(userInput)])
            file = h5py.File(tmp_file, 'r')
            subprocess.call("rm " + tmp_file, shell = True)
            flag = True
        else:
            print("File not in list. Try again!")
            flag = False
    return file

def copyTemp(filename):

    tmp_filename = os.path.splitext(filename)[0] + "_tmp.h5"

    subprocess.call("cp " + filename + " " + tmp_filename, shell = True)

    return tmp_filename
