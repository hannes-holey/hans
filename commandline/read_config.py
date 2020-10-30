#!/usr/bin/env python3

from fhl2d.helper.data_parser import getData

files = getData(".")

for filename, data in files.items():

    print(filename + ": \n" + 40 * "-")
    for name in data.ncattrs():
        print("{:20s}: {:>}".format(name, getattr(data, name)))
    print(40 * "-")
