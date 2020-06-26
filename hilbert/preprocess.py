"""
Module for preprocessing of signals
"""

import numpy as np

def pad_edge_mean(x, pad_width, n_edge=1, axis=-1):
    """
    Pad data x with edge-values or near-edge mean values along axis

    Parameters
    ----------

    x : ndarray
        Input array

    pad_width : int
        Size of padding on each side of x

    n_edge : int, optional
        Number of edge points to average for the pad value, by default 1

    axis : int, optional
        Axis to pad, by default -1

    Returns
    -------
    (x_pad, window)

    x_pad : ndarray
        Padded x

    window : ndarray (1D)
        Mask with 0's for pad regions, 1's for original size

    """
    if pad_width == 0:  # No padding
        window = np.ones((x.shape[axis]), dtype=np.integer)
        x_pad = x
    elif pad_width > 0:
        orig_shape = x.shape
        pad_shape = list(orig_shape)
        pad_shape[axis] += pad_width*2

        window = np.zeros((pad_shape[axis]), dtype=np.integer)
        window[pad_width:-pad_width] = 1
        window = window.astype(bool)

        x_pad = np.zeros(pad_shape, dtype=x.dtype)
        slice_vec = x.ndim*[slice(None)]
        slice_vec[axis] = slice(pad_width, -pad_width)
        x_pad[tuple(slice_vec)] = x

        y_slice_vec_low = x.ndim*[slice(None)]
        y_slice_vec_low[axis] = slice(0, n_edge)
        y_slice_vec_high = x.ndim*[slice(None)]
        y_slice_vec_high[axis] = slice(-n_edge, None)

        y_pad_slice_vec_low = x.ndim*[slice(None)]
        y_pad_slice_vec_low[axis] = slice(0, pad_width)
        y_pad_slice_vec_high = x.ndim*[slice(None)]
        y_pad_slice_vec_high[axis] = slice(-pad_width, None)

        x_pad[tuple(y_pad_slice_vec_low)] += x[tuple(y_slice_vec_low)
                                                         ].mean(axis=axis, keepdims=True)
        x_pad[tuple(y_pad_slice_vec_high)
                   ] += x[tuple(y_slice_vec_high)].mean(axis=axis, keepdims=True)
    else:
        raise ValueError('pad_width must be >= 0')

    return x_pad, window


def mirror(x, axis=-1, prepend=False):
    """Mirror a signal x along an axis

    Parameters
    ----------
    x : array-like
        Input signal/array
    axis : int, optional
        axis to mirror, by default -1
    prepend : bool, optional
        Typically, a signal is appended to the rear (right) of the signal. This option prepends the signal. By default False
    """

    if not prepend:
        return np.append(x, np.flip(x, axis=axis), axis=axis)
    else:
        return np.append(np.flip(x, axis=axis), x, axis=axis)
