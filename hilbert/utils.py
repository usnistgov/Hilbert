"""
Module for preprocessing of signals
"""

import numpy as np

def _make_pad_window(signal_len, pad_len):
    """Returns a window (vector) that identifies (True) the original portion
    of the signal prior to padding.

    Parameters
    ----------
    signal_len : int
        Length of original signal along axis to-be padded
    pad_len : int
        Length of padding applied to each side

    Returns
    -------
    ndarray (1D)
        Window array where True's are location of original signal and False's
        are padding locations.
    """

    signal_pad_len = signal_len + 2*pad_len
    window = np.zeros(signal_pad_len, dtype=np.bool)
    window[pad_len:-pad_len] = True
    return window

def _make_pad_list(ndim, pad_width, axis=-1):
    """Create the pad_sequency for numpy.pad

    Parameters
    ----------
    ndim : int
        Number of dimensions of expected signal
    pad_width : int
        Number of values padded to each edge along specified axis
    axis : int, optional
        Axis to pad, by default -1

    Returns
    -------
    list
        list of lists

    See Also
    ---------
    numpy.pad : Numpy padding routine
    """    

    out = ndim*[[0,0]]
    out[axis] = [pad_width, pad_width]
    return out
    
def pad(x, pad_width, mode='mean', stat_length=1, constant_values=0,
        reflect_type='even', axis=-1, **kwargs):
    """Aims to simplify padding over numpy.pad

    Parameters
    ----------
    x : ndarray
        Input vector or array (n-D)
    pad_width : int
        Number of values to pad to EACH end of specified axis
    mode : str, optional
        Method of padding, by default 'mean'. Selected allowed values are 
        {'constant', 'edge', 'maximum', 'mean', 'median', 'minimum', 'reflect',
        'symmetric', 'wrap'}. See numpy.pad for all options.
    stat_length : int, optional
        Used in ‘maximum’, ‘mean’, ‘median’, and ‘minimum’. Number of values 
        at edge of each axis used to calculate the statistic value, 
        by default 1
    constant_values : int, optional
        Used in ‘constant’. The values to set the padded values for each axis,
        by default 0
    reflect_type{'even', 'odd'}, optional
        Used in 'reflect', and 'symmetric'. The 'even' style is the default 
        with an unaltered reflection around the edge value. For the 'odd' 
        style, the extended part of the array is created by subtracting the 
        reflected values from two times the edge value.
    axis : int, optional
        Axis to apply padding, by default -1
    kwargs : dict, optional
        Sent to numpy.pad

    Returns
    -------
    tuple
        (Padded x, window)
    """

    if pad_width == 0:
        return x

    pad_list = _make_pad_list(x.ndim, pad_width, axis)
    window = _make_pad_window(x.shape[axis], pad_width)

    # * All these checks are dumb, but np.pad has enforced requirements
    # * on keywords and modes
    if mode == 'constant':
        x_pad = np.pad(array=x, pad_width=pad_list, mode=mode, 
                       constant_values=constant_values, 
                       **kwargs)
    elif mode in ['mean', 'median', 'maximum', 'minimum']:
        x_pad = np.pad(array=x, pad_width=pad_list, mode=mode, 
                       stat_length=stat_length, **kwargs)
    elif mode in ['reflect', 'symmetric','mirror']:
        if mode == 'mirror':
            mode = 'symmetric'
        x_pad = np.pad(array=x, pad_width=pad_list, mode=mode, 
                       reflect_type=reflect_type, **kwargs)
    else:
        x_pad = np.pad(array=x, pad_width=pad_list, mode=mode, **kwargs)

    # return (x_pad, window)
    return x_pad

def depad(x_pad, pad_width, axis=-1):
    """Remove padding from x along the speficied axis

    Parameters
    ----------
    x_pad : ndarray
        Padded input vector or array (n-D)
    window : ndarray (1D)
        Boolean vector with True indicated original signal, False pad locations
    axis : int, optional
        Axis along which padding was applied, by default -1
    """

    if pad_width == 0:
        return x_pad

    ndim = x_pad.ndim
    if axis < 0:
        axis = ndim + axis

    depadding_slicer = tuple([slice(pad_width,-pad_width,None) if num == axis 
                              else slice(None,None,None) 
                              for num in range(ndim)])

    return x_pad[depadding_slicer]


def hilbert_pad_wrap(hilbert_lambda, pad_lambda, depad_lambda):
    """Wraps a hilbert function with padding and depadding

    Parameters
    ----------
    hilbert_lambda : function
        Single-parameter hilbert function that takes just an array
    pad_lambda : function
        Single-parameter padding function that takes just an array
    depad_lambda : function
        Single-parameter de-padding function that takes just an array

    Returns
    -------
    function
        Returns a mono-parameter function that just takes in a single array
    """    

    return lambda x: depad_lambda(hilbert_lambda(pad_lambda(x)))
    
def hilbert_pad_simple(x, hilbert_lambda, m_multi_size=1, stat_len=1, axis=-1):
    """Simplified function for performing the Hilbert transform with padding.

    Parameters
    ----------
    x : array-like
        Signal to perform Hilbert transform on
    hilbert_lambda : function
        Single-parameter function for performing the Hilbert transform
    m_multi_size : int, optional
        Multiples of the signal size (along axis) to pad, by default 1
    stat_len : int, optional
        Padding value is the end-point mean across stat_len pixels, by 
        default 1
    axis : int, optional
        Axis to perform Hilbert along, by default -1

    Returns
    -------
    array-like
        Hilbert transform of x along axis.
    """
    if m_multi_size == 0:  # No padding, just Hilbert transform
        hx = hilbert_lambda(x)
    else:
        x_size = x.shape[axis]
        x_pad = pad(x, pad_width=m_multi_size*x_size, stat_length=stat_len, axis=axis)
        hx_pad = hilbert_lambda(x_pad)
        hx = depad(hx_pad, pad_width=m_multi_size*x_size, axis=axis)
    return hx


if __name__ == '__main__':
    from hilbert.dft import hilbert_fft_henrici
    fcn = hilbert_pad_wrap(lambda x: hilbert_fft_henrici(x), lambda x: pad(x,10), lambda x: depad(x,10))
    X = np.random.randn(2,1001)
    print(fcn(X))