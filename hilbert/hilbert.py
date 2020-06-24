"""
Module for discrete Hilbert transform
"""

import numpy as np
from scipy import fftpack

def hilbertfft(y, pad_factor=1, n_edge=1, axis=-1, copy=True, bad_value=1e-8, min_value=None, return_pad=False, **kwargs):
    """Compute the Hilbert Transform using the FFT.

    Parameters
    ----------
    y : {array-like}, shape (n_samples, n_features)
        Input signal such that H{y[n]} will ultimately be returned
    pad_factor : int, optional
        The multiple number of spectra-length edge-value pads that will be
        applied before and after the original spectra, by default 1
    n_edge : int, optional
        For edge values, take a mean of n_edge neighbors, by default 1
    axis : int, optional
        For nd-arrays, axis to perform over, by default -1
    copy : bool, optional
        Copy or over-write input data, by default True
    bad_value : [type], optional
        Inf's and NaN's set to bad_value
    min_value : [type], optional
        Values below min_value set to min_value, by default None
    return_pad : bool, optional
        Return the full padded signal, by default False

    Returns
    -------
    ndarray
        Hilbert transformed data

    References
    -----------
    -    C H Camp Jr, Y J Lee, and M T Cicerone, "Quantitative, Comparable
          Coherent Anti-Stokes Raman Scattering (CARS) Spectroscopy: Correcting
          Errors in Phase Retrieval," Journal of Raman Spectroscopy (2016).
          arXiv:1507.06543.
    -    A D Poularikas, "The Hilbert Transform," in The Handbook of
          Formulas and Tables for Signal Processing (ed., A. D. Poularikas),
          Boca Raton, CRC Press LLC (1999).
    """    
    
    # Pad
    y_pad, window = pad_edge_mean(y, pad_factor*y.shape[axis], n_edge=n_edge, axis=axis)
    len_axis = y_pad.shape[axis]
    time_vec = fftpack.fftfreq(len_axis)
    
    slice_add_dims = y.ndim*[None]
    slice_add_dims[axis] = slice(None)
    slice_add_dims = tuple(slice_add_dims)

    # Perform Hilbert
    y_pad = fftpack.ifft(y_pad, axis=axis, overwrite_x=True)   
    y_pad *= 1j*np.sign(time_vec[slice_add_dims])
    y_pad = fftpack.fft(y_pad, axis=axis, overwrite_x=True)
    
    if bad_value:
        y_pad[np.isnan(y_pad)] = bad_value
        y_pad[np.isinf(y_pad)] = bad_value
        
    if min_value:
        y_pad[y_pad < min_value] = min_value

    slice_vec_get_y_from_pad = y.ndim*[slice(None)]
    slice_vec_get_y_from_pad[axis] = np.where(window==1)[0]
    slice_vec_get_y_from_pad = tuple(slice_vec_get_y_from_pad)
    
    if return_pad:
        return y_pad.real
    
    if copy:
        return y_pad[slice_vec_get_y_from_pad].real
    else:
        y *= 0
        y += y_pad[slice_vec_get_y_from_pad].real


def pad_edge_mean(y, pad_width, n_edge=1, axis=-1):
    """
    Pad data y with edge-values or near-edge mean values along axis
    
    Parameters
    ----------
    
    y : ndarray
        Input array
        
    pad_width : int
        Size of padding on each side of y
        
    n_edge : int, optional
        Number of edge points to average for the pad value, by default 1
        
    axis : int, optional
        Axis to pad, by default -1
        
    Returns
    -------
    (y_pad, window)
    
    y_pad : ndarray
        Padded y
        
    window : ndarray (1D)
        Mask with 0's for pad regions, 1's for original size
        
    """
    if pad_width == 0:  # No padding
        window = np.ones((y.shape[axis]), dtype=np.integer)
        y_pad = y
    elif pad_width > 0:
        orig_shape = y.shape
        pad_shape = list(orig_shape)
        pad_shape[axis] += pad_width*2
        
        window = np.zeros((pad_shape[axis]), dtype=np.integer)
        window[pad_width:-pad_width] = 1
        
        y_pad = np.zeros(pad_shape, dtype=y.dtype)
        slice_vec = y.ndim*[slice(None)]
        slice_vec[axis] = slice(pad_width,-pad_width)
        y_pad[tuple(slice_vec)] = y
        
        y_slice_vec_low = y.ndim*[slice(None)]
        y_slice_vec_low[axis] = slice(0,n_edge)
        y_slice_vec_high = y.ndim*[slice(None)]
        y_slice_vec_high[axis] = slice(-n_edge,None)
        
        y_pad_slice_vec_low = y.ndim*[slice(None)]
        y_pad_slice_vec_low[axis] = slice(0,pad_width)
        y_pad_slice_vec_high = y.ndim*[slice(None)]
        y_pad_slice_vec_high[axis] = slice(-pad_width,None)
        
        y_pad[tuple(y_pad_slice_vec_low)] += y[tuple(y_slice_vec_low)].mean(axis=axis, keepdims=True)
        y_pad[tuple(y_pad_slice_vec_high)] += y[tuple(y_slice_vec_high)].mean(axis=axis, keepdims=True)
    else:
        raise ValueError('pad_width must be >= 0')
        
    return y_pad, window


if __name__ == '__main__':  # pragma: no cover
    import timeit as _timeit

    x = np.abs(10e3*np.random.rand(330, 330, 900))+1.0

    start = _timeit.default_timer()
    out = hilbertfft(x)
    
    start -= _timeit.default_timer()
    print('Scipy Time (Trial 1): {:.3g} sec'.format(-start))

