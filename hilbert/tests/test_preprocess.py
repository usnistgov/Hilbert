"""
Testing for preprocessing methods (signal manipulation prior to DHT)

"""

import numpy as np
from numpy.testing import assert_array_almost_equal

from hilbert.preprocess import pad

def test_pad():
    """ Test out padding procedure """
    
    #1D case
    x = np.linspace(-100, 100, 1000)
    x_pad, window = pad(x, pad_width=x.shape[-1], stat_length=1, axis=-1)

    assert x_pad.size == x.size*3
    assert x_pad[window].size == x.size
    assert np.allclose(x_pad[:x.size], x[0])
    assert np.allclose(x_pad[-x.size:], x[-1])

    #2D case, axis=-1
    x = np.linspace(-100, 100, 1000)
    x = np.repeat(x[None,:], repeats=2, axis=0)
    x_pad, window = pad(x, pad_width=x.shape[-1], stat_length=1, axis=-1)

    assert x_pad.size == x.size*3
    assert np.allclose(True, x_pad[...,window] == x)
    assert np.allclose(x_pad[..., :x.shape[-1]], x[0,0])
    assert np.allclose(x_pad[..., -x.shape[-1]:], x[0,-1])

    #2D case, axis=0
    x = np.linspace(-100, 100, 1000)
    x = np.repeat(x[:,None], repeats=2, axis=1)
    x_pad, window = pad(x, pad_width=x.shape[0], stat_length=1, axis=0)

    assert x_pad.size == x.size*3
    assert np.allclose(True, x_pad[window, ...] == x)
    assert np.allclose(x_pad[:x.shape[-1],...], x[0,0])
    assert np.allclose(x_pad[-x.shape[-1]:, ...], x[-1,0])


def test_mirror():
    """ Test out padding procedure """
    
    #1D case
    x = np.linspace(-100, 100, 1000)
    x_mirror, window = pad(x, pad_width=x.size, mode='symmetric', axis=-1)

    assert x_mirror.size == x.size*3
    assert x_mirror[-1] == x[0]
    assert x_mirror[0] == x[-1]
    assert x_mirror[x.size-1] == x_mirror[-1]
    assert x_mirror[x.size-1] == x[0]

    #2D case, axis=-1
    x = np.linspace(-100, 100, 1000)
    x = np.repeat(x[None,:], repeats=2, axis=0)
    x_mirror, window = pad(x, pad_width=x.shape[-1], mode='symmetric', axis=-1)

    assert window.sum() == x.shape[-1]
    assert x_mirror.shape[0] == x.shape[0]
    assert x_mirror.shape[1] == x.shape[1]*3
    assert x_mirror.size == x.size*3
    assert np.allclose(x_mirror[...,-1], x[0,0])
    assert np.allclose(x_mirror[...,0], x[-1,-1])
    assert np.allclose(x_mirror[...,x.shape[-1]-1], x_mirror[...,x.shape[-1]])
    assert np.allclose(x_mirror[...,x.shape[-1]-1], x[...,0])

    #2D case, axis=0
    x = np.linspace(-100, 100, 1000)
    x = np.repeat(x[:,None], repeats=2, axis=1)
    x_mirror, window = pad(x, pad_width=x.shape[0], mode='symmetric', axis=0)

    assert window.sum() == x.shape[0]
    assert x_mirror.shape[-1] == x.shape[-1]
    assert x_mirror.shape[0] == x.shape[0]*3
    assert x_mirror.size == x.size*3
    assert np.allclose(x_mirror[0,...], x[-1,-1])
    assert np.allclose(x_mirror[-1,...], x[0,0])
    assert np.allclose(x_mirror[x.shape[0]-1,...], x_mirror[x.shape[0],...])
    assert np.allclose(x_mirror[x.shape[0]-1,...], x[0,...])
    

