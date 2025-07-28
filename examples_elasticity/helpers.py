#%%
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import scienceplots
import warnings
import time

def plot3D(array, savename="", bSave=False, bShow=False, title="", dx=1., dy=1.):
    """3D plot of 2D numpy array with color coding
    """
    # Create meshgrid for X and Y coordinates
    x = np.arange(array.shape[0]) * dx
    y = np.arange(array.shape[1]) * dy
    xx, yy = np.meshgrid(x, y, indexing='ij') # indexing method ij

    # Figure and Plot
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(xx, yy, array, cmap=cm.cividis)

    # Labels and title
    ax.set_xlabel('$x$ [m]')
    ax.set_ylabel('$y$ [m]')
    ax.set_zlabel('$z$')
    plt.draw() # to get zaxis offset
    fig.suptitle(title + " - z: " + str(ax.zaxis.offsetText.get_text()), y=1., x=0.4, 
                 horizontalalignment='center', fontsize=14, color='darkblue', weight='bold')

    # Ticks and Adjust
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.zaxis.set_major_locator(MaxNLocator(nbins=4))
    
    ax.tick_params(axis='x', pad=2) 
    ax.tick_params(axis='y', pad=2)
    plt.subplots_adjust(left=0, right=0.8, bottom=0, top=1)

    # Save/Show
    if bSave:
        plt.savefig("{}.png".format(savename), dpi=300, 
                    bbox_inches='tight', pad_inches=0.2)
    if bShow:
        ax.set_box_aspect(None, zoom=0.85)
        plt.show()

    plt.close()


def plot_1D_series(series_list, labels=None, title="Plot", xlabel="Index", ylabel="Value",
                   bSave=False, bShow=False, savename="", bMiddleShift=True):
    """Flexible plotting for Green's or deformation slices. Can be automatically centered
    """
    plt.style.use('science')
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.figure(figsize=(5, 4), constrained_layout=True)

    # if np.array is given
    if isinstance(series_list, np.ndarray):
        if series_list.ndim == 1:
            series_list = [series_list]
        elif series_list.ndim == 2:
            series_list = [row for row in series_list]

    # shift to middle and plot
    if bMiddleShift:
        for i, series in enumerate(series_list):
            arr = np.squeeze(series) 
            mid_index = len(arr) // 2
            x_shifted = np.arange(len(arr)) - mid_index
            label = labels[i] if labels and i < len(labels) else f"Series {i+1}"
            plt.plot(x_shifted, arr, label=label)
    else:
        for i, series in enumerate(series_list):
            arr = np.squeeze(series)
            x = np.arange(len(arr))
            label = labels[i] if labels and i < len(labels) else f"Series {i+1}"
            plt.plot(x, arr, label=label)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if labels:
        plt.legend()
    plt.grid(True)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The figure layout has changed to tight")
        plt.tight_layout()

    # Save/Show
    if bSave:
        plt.savefig("{}.png".format(savename), dpi=300, 
                    bbox_inches='tight', pad_inches=0.2)
    if bShow:
        plt.show()

    plt.close()


class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self
    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.interval = self.end - self.start
        print(f"Runtime: {self.interval:.6f} seconds")


if __name__ == "__main__":
    X = np.arange(-5, 5, 0.25)
    Y = np.arange(-3, 3, 0.2)
    xx, yy = np.meshgrid(X, Y)
    R = np.sqrt(xx**2 + yy**2)
    Z = np.sin(R)

    #plot3D(Z, bShow=True, savename="test", bSave=True, title="Test",
    #       dx=1e-03, dy=1)

    plot_1D_series((X, Y), labels=["A", "B"], title="Test", bShow=True, bSave=True, savename="Test")


# %%
