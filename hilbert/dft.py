"""
Module for discrete Hilbert transform that uses the discrete Fourier transform 
at its core.
"""

import numpy as np
from scipy import fftpack

from scipy.signal import hilbert as _hilbert_analytic_marple

def hilbert_fft_henrici(x, axis=-1, bad_value='eps', min_value=None):
    """Compute the Hilbert Transform using the FFT (Henrici).

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
    fx = fftpack.ifft(x, axis=axis, overwrite_x=False)   
    fx *= 1j*np.sign(time_vec)
    hx = fftpack.fft(fx, axis=axis, overwrite_x=True).real
    
    if bad_value:
        hx[np.isnan(hx)] = bad_value
        hx[np.isinf(hx)] = bad_value
        
    if min_value:
        hx[hx < min_value] = min_value

    return hx
    

def hilbert_fft_marple(x, axis=-1, bad_value='eps', min_value=None):
    """Compute the Hilbert Transform using the Marple implemetation FFT.

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
    -   L. Marple, "Computing the discrete-time “analytic” signal via FFT," 
        IEEE Trans. Signal Process. 47(9), 2600–2603 (1999).
    """

    if bad_value == 'eps':
        bad_value = np.finfo(x.dtype).eps

    if min_value == 'eps':
        min_value = np.finfo(x.dtype).eps

    # SciPy implementation returns the `analytic` signal; thus, taking imag
    # x = _hilbert_analytic_marple(x, axis=axis).imag
    
    # Transcribed from https://www.gaussianwaves.com/2017/04/analytic-signal-hilbert-transform-and-fft/
    N = x.shape[axis]
    fx = fftpack.fft(x,N, axis=axis, overwrite_x=False)
    h = np.zeros(N)
    
    if N % 2 == 0:  # Even-length
        h[0] = 1
        h[1:N//2] = 2
        h[N//2] = 1
    else:  # Odd length
        h[0] = 1
        h[1:(N+1)//2] = 2

    hx = fx*h
    hx = fftpack.ifft(hx, N, axis=axis, overwrite_x=True).imag
    
    
    if bad_value:
        hx[np.isnan(hx)] = bad_value
        hx[np.isinf(hx)] = bad_value
        
    if min_value:
        hx[hx < min_value] = min_value

    return hx

def hilbert_scipy(x, *args, **kwargs):
    """Compute the Hilbert Transform using SciPy's hilbert function

    Parameters
    ----------
    x : array-like
        Input signal such that y[n] = H{x[n]} will ultimately be returned
    
    Returns
    -------
    ndarray
        Hilbert transformed data

    """
    return _hilbert_analytic_marple(x, *args, **kwargs).imag

if __name__ == '__main__':  # pragma: no cover
    # import timeit as _timeit

    # x = np.abs(10e3*np.random.rand(33, 33, 900))+1.0

    # start = _timeit.default_timer()
    # out = mirror_hilbertfft(x)
        
    # start -= _timeit.default_timer()
    # print('Scipy Time (Trial 1): {:.3g} sec'.format(-start))
    pass
