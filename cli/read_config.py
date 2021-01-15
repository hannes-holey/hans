#!/usr/bin/env python3

from pylub.tools import getData

if __name__ == "__main__":

    files = getData(".")

    for filename, data in files.items():

        print(filename + ": \n" + 40 * "-")
        for name in data.ncattrs():
            print("{:20s}: {:>}".format(name, getattr(data, name)))
        print(40 * "-")
