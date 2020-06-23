#!/usr/bin/env python3

from helper import getFiles

files = getFiles()

for file in files.values():

    print(file[0] + ": \n" + 40 * "-")
    for name in file[1].ncattrs():
        print("{:20s}: {:>}".format(name, getattr(file[1], name)))
    print(40 * "-")
