#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import sys

plt.style.use('presentation')
plt.figure(figsize=(12,7))

filename = sys.argv[1]
A = np.loadtxt(filename, skiprows=1)

plt.plot(A[:,0], A[:,1], '-s')
plt.show()
