#!/usr/bin/env python3

import matplotlib.pyplot as plt
from pylub.helper.plot import Plot

if __name__ == "__main__":

    files = Plot(".", mode="single")
    fig, ax, ani = files.animate2D()
    plt.show()
    # ani.save(f"animation.mp4", fps=30)
