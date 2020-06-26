"""
Module for discrete Hilbert transform that uses the discrete Fourier transform 
at its core.
"""

import numpy as np
from scipy import fftpack

from hilbert.preprocess import pad_edge_mean, mirror

def hilbert_fft(x, axis=-1, bad_value='eps', min_value=None):
    """Compute the Hilbert Transform using the FFT.

    Parameters
    ----------
    x : array-like
        Input signal such that y[n] = H{x[n]} will ultimately be returned
    axis : int, optional
        For nd-arrays, axis to perform over, by default -1
    bad_value : {None, 'eps', float}, optional
        Inf's and NaN's set to bad_value. If 'eps', sets to machine epsilon of 
        x dtype. If None, skips check. Default is 'eps'.
    min_value : {None, 'eps', float}, optional
        Values below min_value set to min_value. If 'eps', sets to machine 
        epsilon of x dtype. By default None (skip).
    
    Returns
    -------
    ndarray
        Hilbert transformed data

    References
    -----------
    -   P. Henrici, Applied and Computational Complex Analysis Vol III 
        (Wiley-Interscience, 1986).
    """

    if bad_value == 'eps':
        bad_value = np.finfo(x.dtype).eps

    if min_value == 'eps':
        min_value = np.finfo(x.dtype).eps

    len_axis = x.shape[axis]
    time_vec = fftpack.fftfreq(len_axis)
    
    # Trick for having an arbitrary axis
    slice_add_dims = x.ndim*[None]
    slice_add_dims[axis] = slice(None)
    slice_add_dims = tuple(slice_add_dims)
    time_vec = time_vec[slice_add_dims]

    # Perform Hilbert
    x = fftpack.ifft(x, axis=axis, overwrite_x=True)   
    x *= 1j*np.sign(time_vec)
    x = fftpack.fft(x, axis=axis, overwrite_x=True)
    
    if bad_value:
        x[np.isnan(x)] = bad_value
        x[np.isinf(x)] = bad_value
        
    if min_value:
        x[x < min_value] = min_value

    return x.real
    

if __name__ == '__main__':  # pragma: no cover
    # import timeit as _timeit

    # x = np.abs(10e3*np.random.rand(33, 33, 900))+1.0

    # start = _timeit.default_timer()
    # out = mirror_hilbertfft(x)
        
    # start -= _timeit.default_timer()
    # print('Scipy Time (Trial 1): {:.3g} sec'.format(-start))
    pass
