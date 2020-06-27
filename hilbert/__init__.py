from .dft import hilbert_fft_henrici, hilbert_fft_marple
from .preprocess import pad_edge_mean, mirror

from ._version import __version__

# Can be set to whatever flavor of Hilbert-fft a user wants to use
hilbert_fft = hilbert_fft_henrici 

__all__ = ['hilbert_fft', 'hilbert_fft_marple', 'hilbert_fft_henrici', '__version__', 'pad_edge_mean', 'mirror']
