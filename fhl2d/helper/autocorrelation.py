import numpy as np


def getTimeACF(time, field, dim, length=1):
    """
    Compute time autocorrelation function from grid data.
    Choose only k vectors perpendicular to the box edges.

    Parameters
    ----------
    time : numpy.ndarray
        1D-array containing the time
    field : numpy.ndarray
        field to compute the ACF, shape=(len(time),Nx,Ny,Nz)
    dim : numpy.ndarray
        boolean array, which axes are resolved by grid
    length : int
        scale factor for wave vector (default = 1)

    Returns
    ----------
    out_ac : numpy.ndarray
        output containig time ACF, shape=(len(time), ndim + 1)
    wave_vec[dim] : numpy.ndarray
        array containing the wave_vectors (rows) for which ACF is computed
    """

    # dimensions
    ndim = np.sum(dim)

    # array of the three shortest wavevectors in 3D
    wave_vec = np.eye(3, dtype=int) * length

    # all possible axes over which FFT is computed
    fft_axes = np.arange(1,4)

    # init output array, 1st column: time
    out_ac = np.empty([len(time), ndim + 1])
    out_ac[:,0] = time

    # compute FFT
    field_fft = np.real(np.fft.fftn(field, axes=fft_axes[dim]))

    # Compute time autocorrelation for given k vectors
    for i in range(ndim):
        ikx, iky, ikz = wave_vec[dim][i]
        C = compute_ACF(field_fft[:, ikx, iky, ikz])

        out_ac[:, i + 1] = C

    return out_ac, wave_vec[dim]


def compute_ACF(time_series):
    """
    Compute normalized time autocorrelation function from the
    time series of a given Fourier coefficient

    Parameters
    ----------
    time_series : numpy.ndarray
        2D-array containing the time series data

    Returns
    ----------
    C : numpy.ndarray
        ACF function C(k,t), shape=(len(time))
    """

    var = np.var(time_series)
    mean = np.mean(time_series)
    C = np.correlate(time_series - mean, time_series - mean, mode="full")
    C = C[C.size // 2:]
    C /= len(C)
    C /= var

    return C
