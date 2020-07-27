from .dft import hilbert_fft_henrici, hilbert_fft_marple
from .wavelet import hilbert_haar

from .mlhilbert import MLHilb
from .synthdata import GaussianTrainingData, LorentzianTrainingData, SincTrainingData
from .preprocess import pad, depad, hilbert_pad_wrap, hilbert_pad_simple
from .metrics import rss, mse

from ._version import __version__

# Can be set to whatever flavor of Hilbert-fft a user wants to use
hilbert_fft = hilbert_fft_henrici 

__all__ = ['hilbert_fft', 'hilbert_fft_marple', 'hilbert_fft_henrici', 
           '__version__', 'MLHilb', 'GaussianTrainingData',
           'LorentzianTrainingData', 'SincTrainingData', 'VoigtTrainingData',
           '__version__', 'pad', 'depad', 'hilbert_pad_wrap',
            'hilbert_pad_simple','rss','mse']
