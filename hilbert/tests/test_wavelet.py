"""
Testing for Hilbert transform methods that use wavelets at their core

Using the math relation a^2 / (a^2 + x^2) (Lorentz/Cauchy) has an 
analytical Hilbert transform: x^2 / (a^2 + x^2)
"""

import numpy as np
from numpy.testing import assert_array_almost_equal

from hilbert.wavelet import hilbert_haar, _haar_matrix

import pytest


def test_haar():

    # Not power-of-2
    n = np.linspace(-100, 100, 1000)
    x = 2/(2**2 + n**2)
    hilb_x = hilbert_haar(x)
    hilb_x_analytical = n/(2**2 + n**2)
    assert_array_almost_equal(hilb_x_analytical, hilb_x, decimal=1)

    # Power-of-2
    n = np.linspace(-100, 100, 1024)
    x = 2/(2**2 + n**2)
    hilb_x = hilbert_haar(x)
    hilb_x_analytical = n/(2**2 + n**2)
    assert_array_almost_equal(hilb_x_analytical, hilb_x, decimal=1)

    # 2D version
    x2 = np.vstack((x, x/2))
    hilb_x = hilbert_haar(x2)
    hilb_x_analytical = np.vstack((n/(2**2 + n**2), 0.5*n/(2**2 + n**2)))
    assert_array_almost_equal(hilb_x_analytical, hilb_x, decimal=1)

def test_haar_errors():
    n = np.linspace(-100, 100, 1000)
    x = 2/(2**2 + n**2)

    # Wrong axis
    with pytest.raises(NotImplementedError):
        _ = hilbert_haar(x, axis=0)

    # > 2 dimensions
    with pytest.raises(ValueError):
        _ = hilbert_haar(np.random.randn(3,3,3))


def test_haar_matrix():
    hm, hilb_hm = _haar_matrix(4)
    assert hm.shape == (4,4)
    assert hilb_hm.shape == (4,4)

    # Wrong dimensionsal size
    with pytest.raises(ValueError):
        _haar_matrix(3)