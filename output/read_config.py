#!/usr/bin/env python3

import h5py
import sys
import os

for file in sys.argv[1:]:
    if file in os.listdir() and file.endswith(".h5"):

        f1 = h5py.File(file, 'r')
        group = f1.get('/config')
        start = group.attrs['Start time:']
        print(50 * "-" + "\n" + "{:31s}".format(file)+ start + "\n" + 50 * "-")

        for sub_key, sub_val in group.items():
            print(sub_key + ":")

            for item in sub_val.attrs.keys():
                print("\t{:20s}: {:>}".format(item, sub_val.attrs[item]))

    else:
        print(file + ": no hdf5 file")
