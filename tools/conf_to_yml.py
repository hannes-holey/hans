#!/usr/bin/env python3

import os
import sys
from ruamel.yaml import YAML
from snippets.loadNetCDF import getData

config_file = sys.argv[1]
readme_file = "../README.yml"

files = getData("../data", single=True)

for filename, data in files.items():
    commit = getattr(data, "commit")

yaml = YAML()
yaml.explicit_start = True      # ensures the file begins with "---"
yaml.width = 80
yaml.indent(mapping=4, sequence=4, offset=2)

with open(config_file) as cf:
    append = yaml.load(cf)

if os.path.exists(readme_file):
    with open(readme_file) as rf:
        yml = yaml.load(rf)
    yml["software_packages"][0]["version"] = commit
else:
    yml = {}

yml.update(append)

with open(readme_file, "w+") as f:
    yaml.dump(yml, f)
