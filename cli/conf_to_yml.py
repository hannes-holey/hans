#!/usr/bin/env python3

import sys
from ruamel.yaml import YAML
from pylub.tools import getData

config_file = sys.argv[1]
readme_template = "/home/hannes/.dtool_readme_FVM.yml"
readme_out = "README.yml"

yaml = YAML()
yaml.explicit_start = True      # ensures the file begins with "---"
yaml.width = 80
yaml.indent(mapping=4, sequence=4, offset=2)

with open(config_file) as cf:
    append = yaml.load(cf)
    name = append["options"]["name"]

files = getData(".", prefix=name, mode="single")

for filename, data in files.items():
    commit = getattr(data, "commit")

with open(readme_template) as rf:
    yml = yaml.load(rf)
yml["software_packages"][0]["version"] = commit

yml.update(append)

with open(readme_out, "w+") as f:
    yaml.dump(yml, f)
