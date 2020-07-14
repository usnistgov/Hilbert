"""
Testing for Hilbert transform methods that use the DFT at their core

Using the math relation a^2 / (a^2 + x^2) (Lorentz/Cauchy) has an 
analytical Hilbert transform: x^2 / (a^2 + x^2)
"""

import numpy as np
from numpy.testing import assert_array_almost_equal

from hilbert.dft import hilbert_fft_henrici, hilbert_fft_marple


def test_hilbert_fft_henrici():
    n = np.linspace(-100, 100, 1000)
    x = 2/(2**2 + n**2)
    hilb_x = hilbert_fft_henrici(x)
    hilb_x_analytical = n/(2**2 + n**2)
    assert_array_almost_equal(hilb_x_analytical, hilb_x, decimal=2)
    assert np.allclose(0, hilb_x.sum()) # DHT enforced -- not real

    # 2D version
    x2 = np.vstack((x, x/2))
    hilb_x = hilbert_fft_henrici(x2)
    hilb_x_analytical = np.vstack((n/(2**2 + n**2), 0.5*n/(2**2 + n**2)))
    assert_array_almost_equal(hilb_x_analytical, hilb_x, decimal=2)

def test_hilbert_henrici_replace_min_value():
    """Replace min_value"""
    n = np.linspace(-100, 100, 1000)
    x = 2/(2**2 + n**2)
    hilb_x_analytical = n/(2**2 + n**2)

    # Replace all values (min_value LARGE)
    hilb_x = hilbert_fft_henrici(x, min_value=100)
    assert_array_almost_equal(100, hilb_x, decimal=4)

    # Replace all values (min_value small)
    hilb_x = hilbert_fft_henrici(x)
    assert_array_almost_equal(hilb_x_analytical, hilb_x, decimal=2)


def test_hilbert_marple():
    n = np.linspace(-100, 100, 1000)
    x = 2/(2**2 + n**2)
    hilb_x = hilbert_fft_marple(x)
    hilb_x_analytical = n/(2**2 + n**2)
    assert_array_almost_equal(hilb_x_analytical, hilb_x, decimal=2)
    assert np.allclose(0, hilb_x.sum()) # DHT enforced -- not real

    # 2D version
    x2 = np.vstack((x, x/2))
    hilb_x = hilbert_fft_marple(x2)
    hilb_x_analytical = np.vstack((n/(2**2 + n**2), 0.5*n/(2**2 + n**2)))
    assert_array_almost_equal(hilb_x_analytical, hilb_x, decimal=2)

def test_hilbert_marple_replace_min_value():
    """Replace min_value"""
    n = np.linspace(-100, 100, 1000)
    x = 2/(2**2 + n**2)
    hilb_x_analytical = n/(2**2 + n**2)

    # Replace all values (min_value LARGE)
    hilb_x = hilbert_fft_marple(x, min_value=100)
    assert_array_almost_equal(100, hilb_x, decimal=4)

    # Replace all values (min_value small)
    hilb_x = hilbert_fft_henrici(x)
    assert_array_almost_equal(hilb_x_analytical, hilb_x, decimal=2)
