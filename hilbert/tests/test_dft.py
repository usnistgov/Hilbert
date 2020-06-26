"""
Testing for Hilbert transform methods that use the DFT at their core

Using the math relation a^2 / (a^2 + x^2) (Lorentz/Cauchy) has an 
analytical Hilbert transform: x^2 / (a^2 + x^2)
"""

import numpy as np
from numpy.testing import assert_array_almost_equal

from hilbert.dft import hilbert_fft


def test_hilbert_no_pad():
    n = np.linspace(-100, 100, 1000)
    x = 2/(2**2 + n**2)
    hilb_x = hilbert_fft(x)
    hilb_x_analytical = n/(2**2 + n**2)
    assert_array_almost_equal(hilb_x_analytical, hilb_x, decimal=2)
    assert np.allclose(0, hilb_x.sum()) # DHT enforced -- not real

def test_hilbert_replace_min_value():
    """Replace min_value"""
    n = np.linspace(-100, 100, 1000)
    x = 2/(2**2 + n**2)
    hilb_x_analytical = n/(2**2 + n**2)

    # Replace all values (min_value LARGE)
    hilb_x = hilbert_fft(x, min_value=100)
    assert_array_almost_equal(100, hilb_x, decimal=4)

    # Replace all values (min_value small)
    hilb_x = hilbert_fft(x)
    assert_array_almost_equal(hilb_x_analytical, hilb_x, decimal=2)
