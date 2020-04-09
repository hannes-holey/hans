#!/usr/bin/env python3

from helper import getFiles

files = getFiles()

for file in files.values():

        group = file[1].get('/config')

        start = group.attrs['tStart']
        commit = group.attrs['commit']
        print(50 * "-" )
        print("{:30s} {:s}\n{:s}".format(file[0], start, commit))
        print(50 * "-" )

        for sub_key, sub_val in group.items():
            print(sub_key + ":")

            for item in sub_val.attrs.keys():
                print("\t{:20s}: {:>}".format(item, sub_val.attrs[item]))
