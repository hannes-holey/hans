#
#


import sys

file_lines = open(sys.argv[1], 'r').readlines()
header_lines = sys.stdin.readlines()

while file_lines[0].startswith('#'):
    file_lines = file_lines[1:]

file_lines.insert(0, '#\n')
for header_line in header_lines[::-1]:
    file_lines.insert(0, '# {}'.format(header_line).strip() + '\n')
file_lines.insert(0, '#\n')

open(sys.argv[1], 'w').writelines(file_lines)
