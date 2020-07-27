"""
Testing for synthetic data generation methods
"""

import numpy as np
from numpy.testing import assert_array_almost_equal

from hilbert.synthdata import SincTrainingData, GaussianTrainingData, LorentzianTrainingData

def test_sinc_max_width():
    """ Testing my defined max width that is the +/- 1-order peak of the sinc
    is equal to the full window width Dn"""

    n = np.linspace(-100,100,100001)
    dn = n[1] - n[0]
    sincds = SincTrainingData(n, n_samples=1)
    # Check gradient ~= 0
    assert np.abs(np.gradient(np.squeeze(sincds.fcn(n, np.array([1, 0, sincds.max_width])).real))[0]) < 1e-8
    assert np.abs(np.gradient(np.squeeze(sincds.fcn(n, np.array([1, 0, sincds.max_width])).real))[-1]) < 1e-8

    n = np.linspace(-400,400,100001)
    dn = n[1] - n[0]
    sincds = SincTrainingData(n, n_samples=1)
    # Check gradient ~= 0
    assert np.abs(np.gradient(np.squeeze(sincds.fcn(n, np.array([1, 0, sincds.max_width])).real))[0]) < 1e-6
    assert np.abs(np.gradient(np.squeeze(sincds.fcn(n, np.array([1, 0, sincds.max_width])).real))[-1]) < 1e-6


def test_Gauss_max_width():
    """ Testing my defined max width such that the window width == `* FWHM"""

    n = np.linspace(-100,100,100001)
    dn = n[1] - n[0]
    gaussds = GaussianTrainingData(n, n_samples=1)
    
    # Check end-points are around 0.5 (FWHM amp)
    assert np.allclose(np.squeeze(gaussds.fcn(n, np.array([1, 0, gaussds.max_width])).real)[0], 0.5)
    assert np.allclose(np.squeeze(gaussds.fcn(n, np.array([1, 0, gaussds.max_width])).real)[-1], 0.5)
    
    n = np.linspace(-400,400,100001)
    dn = n[1] - n[0]
    gaussds = GaussianTrainingData(n, n_samples=1)
    
    # Check end-points are around 0.5 (FWHM amp)
    assert np.allclose(np.squeeze(gaussds.fcn(n, np.array([1, 0, gaussds.max_width])).real)[0], 0.5)
    assert np.allclose(np.squeeze(gaussds.fcn(n, np.array([1, 0, gaussds.max_width])).real)[-1], 0.5)
    

def test_Lorentz_max_width():
    """ Testing my defined max width such that the window width == `* FWHM"""

    n = np.linspace(-100,100,100001)
    dn = n[1] - n[0]
    lords = LorentzianTrainingData(n, n_samples=1)
    
    # Check end-points are around 0.5 (FWHM amp)
    assert np.allclose(np.squeeze(lords.fcn(n, np.array([1, 0, lords.max_width])).real)[0], 0.5)
    assert np.allclose(np.squeeze(lords.fcn(n, np.array([1, 0, lords.max_width])).real)[-1], 0.5)
    
    n = np.linspace(-400,400,100001)
    dn = n[1] - n[0]
    lords = LorentzianTrainingData(n, n_samples=1)
    
    # Check end-points are around 0.5 (FWHM amp)
    assert np.allclose(np.squeeze(lords.fcn(n, np.array([1, 0, lords.max_width])).real)[0], 0.5)
    assert np.allclose(np.squeeze(lords.fcn(n, np.array([1, 0, lords.max_width])).real)[-1], 0.5)

def test_Lorentz_amp():
    n = np.linspace(-100,100,1001)
    out = LorentzianTrainingData(n, stack_Hf_f=False, n_samples=1001, 
                                 random_state=0)
    isinstance(out.conditions_[0,0], float)
    isinstance(out.conditions_[0,1], float)
    isinstance(out.conditions_[0,2], float)

    f_max_to_Hf_max_factor = 0.5
    wiggle_room = 0.9  # ie within wiggle room factor

    assert np.alltrue(out.f_.max(axis=-1)>=0.9), \
                      '{}'.format(out.f_.max(axis=-1)[out.f_.max(axis=-1)<0.9])
    assert np.alltrue(out.Hf_.max(axis=-1) > 0.9*f_max_to_Hf_max_factor)

    out = LorentzianTrainingData(n, stack_Hf_f=False, n_samples=1001, 
                                 random_state=0, amp=2)
    assert np.alltrue(out.Hf_.max(axis=-1) > wiggle_room*2*f_max_to_Hf_max_factor)
    
    assert np.alltrue(out.f_.max(axis=-1)>=1.8), \
                      '{}'.format(out.f_.max(axis=-1)[out.f_.max(axis=-1)<1.8])
    out = LorentzianTrainingData(n, stack_Hf_f=False, n_samples=101, 
                                 random_state=0, amp=[2.,4.])
    assert np.alltrue((out.f_.max(axis=-1)>=2) &
                      (out.f_.max(axis=-1)<4))
    assert np.alltrue((out.Hf_.max(axis=-1)>=2*f_max_to_Hf_max_factor) &
                      (out.Hf_.max(axis=-1)<4*f_max_to_Hf_max_factor))

    out = LorentzianTrainingData(n, stack_Hf_f=False, n_samples=101, 
                                 random_state=0, amp=lambda: 10*np.random.rand(1)[0]+3)
    assert np.alltrue((out.f_.max(axis=-1)>=3) &
                      (out.f_.max(axis=-1)<13))
    assert np.alltrue((out.Hf_.max(axis=-1)>=3*f_max_to_Hf_max_factor) &
                      (out.Hf_.max(axis=-1)<13*f_max_to_Hf_max_factor))

def test_Gaussian_amp():
    n = np.linspace(-100,100,1001)
    out = GaussianTrainingData(n, stack_Hf_f=False, n_samples=1001, 
                                 random_state=0)
    isinstance(out.conditions_[0,0], float)
    isinstance(out.conditions_[0,1], float)
    isinstance(out.conditions_[0,2], float)

    f_max_to_Hf_max_factor = 0.61

    assert np.alltrue(out.f_.max(axis=-1)>=0.9), \
                      '{}'.format(out.f_.max(axis=-1)[out.f_.max(axis=-1)<0.9])
    assert np.alltrue(out.Hf_.max(axis=-1) > 0.9*f_max_to_Hf_max_factor)

    out = GaussianTrainingData(n, stack_Hf_f=False, n_samples=1001, 
                                 random_state=0, amp=2)
    isinstance(out.conditions_[0,0], float)
    isinstance(out.conditions_[0,1], float)
    isinstance(out.conditions_[0,2], float)
    
    assert np.alltrue(out.f_.max(axis=-1)>=1.8), \
                      '{}'.format(out.f_.max(axis=-1)[out.f_.max(axis=-1)<1.8])
    out = GaussianTrainingData(n, stack_Hf_f=False, n_samples=101, 
                                 random_state=0, amp=[2.,4.])
    isinstance(out.conditions_[0,0], float)
    isinstance(out.conditions_[0,1], float)
    isinstance(out.conditions_[0,2], float)

    assert np.alltrue((out.f_.max(axis=-1)>=2) &
                      (out.f_.max(axis=-1)<4))
    assert np.alltrue((out.Hf_.max(axis=-1)>=2*f_max_to_Hf_max_factor) &
                      (out.Hf_.max(axis=-1)<4*f_max_to_Hf_max_factor))

    out = GaussianTrainingData(n, stack_Hf_f=False, n_samples=101, 
                                 random_state=0, amp=lambda: 10*np.random.rand(1)[0]+3)
    isinstance(out.conditions_[0,0], float)
    isinstance(out.conditions_[0,1], float)
    isinstance(out.conditions_[0,2], float)
    assert np.alltrue((out.f_.max(axis=-1)>=3) &
                      (out.f_.max(axis=-1)<13))
    assert np.alltrue((out.Hf_.max(axis=-1)>=3*f_max_to_Hf_max_factor) &
                      (out.Hf_.max(axis=-1)<13*f_max_to_Hf_max_factor))

def test_Sinc_amp():
    n = np.linspace(-100,100,1001)
    out = SincTrainingData(n, stack_Hf_f=False, n_samples=1001, 
                                 random_state=0)
    isinstance(out.conditions_[0,0], float)
    isinstance(out.conditions_[0,1], float)
    isinstance(out.conditions_[0,2], float)
    assert np.alltrue(out.f_.max(axis=-1)>=0.9), \
                      '{}'.format(out.f_.max(axis=-1)[out.f_.max(axis=-1)<0.9])
    assert np.alltrue(out.Hf_.max(axis=-1)>=0.65), \
                      '{}'.format(out.Hf_.max(axis=-1)[out.Hf_.max(axis=-1)<0.65])

    out = SincTrainingData(n, stack_Hf_f=False, n_samples=1001, 
                                 random_state=0, amp=2)
    isinstance(out.conditions_[0,0], float)
    isinstance(out.conditions_[0,1], float)
    isinstance(out.conditions_[0,2], float)
    
    assert np.alltrue(out.f_.max(axis=-1)>=1.8), \
                      '{}'.format(out.f_.max(axis=-1)[out.f_.max(axis=-1)<1.8])

    assert np.alltrue(out.Hf_.max(axis=-1)>=1.35), \
                      '{}'.format(out.Hf_.max(axis=-1)[out.Hf_.max(axis=-1)<1.35])

    out = SincTrainingData(n, stack_Hf_f=False, n_samples=101, 
                                 random_state=0, amp=[2.,4.])
    isinstance(out.conditions_[0,0], float)
    isinstance(out.conditions_[0,1], float)
    isinstance(out.conditions_[0,2], float)
    assert np.alltrue((out.f_.max(axis=-1)>=2) &
                      (out.f_.max(axis=-1)<4))
    assert np.alltrue((out.Hf_.max(axis=-1)>=1.3) &
                      (out.Hf_.max(axis=-1)<=3))

    out = SincTrainingData(n, stack_Hf_f=False, n_samples=101, 
                                 random_state=0, amp=lambda: 10*np.random.rand(1)[0]+3)
    assert np.alltrue((out.f_.max(axis=-1)>=3) &
                      (out.f_.max(axis=-1)<13))