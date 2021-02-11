#!/usr/bin/env python3

from pylub.plottools import Plot

if __name__ == "__main__":

    files = Plot(".")

    print(files)

    for filename, data in files.ds.items():

        print(filename + ": \n" + 40 * "-")
        for name in data.ncattrs():
            print("{:20s}: {:>}".format(name, getattr(data, name)))
        print(40 * "-")
