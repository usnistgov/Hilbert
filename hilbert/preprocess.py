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
    elif mode in ['reflect', 'symmetric']:
        x_pad = np.pad(array=x, pad_width=pad_list, mode=mode, 
                       reflect_type=reflect_type, **kwargs)
    else:
        x_pad = np.pad(array=x, pad_width=pad_list, mode=mode, **kwargs)

    return (x_pad, window)

    