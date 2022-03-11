from .dft import hilbert_fft_henrici, hilbert_fft_marple, hilbert_scipy
from .wavelet import hilbert_haar
from .estimators import DHT, DHT_Pad, LeDHT

from .synthdata import GaussianTrainingData, LorentzianTrainingData, SincTrainingData, SyntheticSpectra
from .utils import pad, depad, hilbert_pad_wrap, hilbert_pad_simple
from .metrics import rss, mse, mlhilb_scorer

from ._version import __version__

# Can be set to whatever flavor of Hilbert-fft a user wants to use
hilbert_fft = hilbert_fft_henrici

__all__ = ['DHT', 'DHT_Pad', 'LeDHT', 'hilbert_fft', 'hilbert_fft_marple',
           'hilbert_fft_henrici', 'hilbert_scipy', '__version__', 'MLHilb', 
           'LorentzianTrainingData', 'SincTrainingData',
           '__version__', 'pad', 'depad', 'hilbert_pad_wrap',
           'hilbert_pad_simple','rss','mse', 'mlhilb_scorer',
           'mlhilb_train_dev_cv']
