"""
Testing for synthetic data generation methods
"""

import numpy as np
from numpy.testing import assert_array_almost_equal

from hilbert.synthdata import (SincTrainingData, GaussianTrainingData,
                               LorentzianTrainingData, SyntheticSpectra)

def test_sinc_max_width():
    """ Testing my defined max width that is the +/- 1-order peak of the sinc
    is equal to the full window width Dn"""

    n = np.linspace(-100,100,1001)
    
    sincds = SincTrainingData(n, n_samples=1)
    # Check gradient ~= 0
    assert np.abs(np.gradient(np.squeeze(sincds.fcn(n, np.array([1, 0, sincds.max_width])).real))[0]) < 1e-4
    assert np.abs(np.gradient(np.squeeze(sincds.fcn(n, np.array([1, 0, sincds.max_width])).real))[-1]) < 1e-4

    n = np.linspace(-400,400,100001)
    
    sincds = SincTrainingData(n, n_samples=1)
    # Check gradient ~= 0
    assert np.abs(np.gradient(np.squeeze(sincds.fcn(n, np.array([1, 0, sincds.max_width])).real))[0]) < 1e-6
    assert np.abs(np.gradient(np.squeeze(sincds.fcn(n, np.array([1, 0, sincds.max_width])).real))[-1]) < 1e-6


def test_Gauss_max_width():
    """ Testing my defined max width such that the window width == `* FWHM"""

    n = np.linspace(-100,100,1001)
    
    gaussds = GaussianTrainingData(n, n_samples=1)

    # Check end-points are around 0.5 (FWHM amp)
    assert np.allclose(np.squeeze(gaussds.fcn(n, np.array([1, 0, gaussds.max_width])).real)[0], 0.5)
    assert np.allclose(np.squeeze(gaussds.fcn(n, np.array([1, 0, gaussds.max_width])).real)[-1], 0.5)

    n = np.linspace(-400,400,100001)
    
    gaussds = GaussianTrainingData(n, n_samples=1)

    # Check end-points are around 0.5 (FWHM amp)
    assert np.allclose(np.squeeze(gaussds.fcn(n, np.array([1, 0, gaussds.max_width])).real)[0], 0.5)
    assert np.allclose(np.squeeze(gaussds.fcn(n, np.array([1, 0, gaussds.max_width])).real)[-1], 0.5)


def test_Lorentz_max_width():
    """ Testing my defined max width such that the window width == `* FWHM"""

    n = np.linspace(-100,100,1001)
    
    lords = LorentzianTrainingData(n, n_samples=1)

    # Check end-points are around 0.5 (FWHM amp)
    assert np.allclose(np.squeeze(lords.fcn(n, np.array([1, 0, lords.max_width])).real)[0], 0.5)
    assert np.allclose(np.squeeze(lords.fcn(n, np.array([1, 0, lords.max_width])).real)[-1], 0.5)

    n = np.linspace(-400,400,100001)
    
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
                      (out.Hf_.max(axis=-1)<4.1*f_max_to_Hf_max_factor))

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
    assert np.alltrue(out.Hf_.max(axis=-1)>=0.64), \
                      '{}'.format(out.Hf_.max(axis=-1)[out.Hf_.max(axis=-1)<0.65])

    out = SincTrainingData(n, stack_Hf_f=False, n_samples=1001,
                                 random_state=0, amp=2)
    isinstance(out.conditions_[0,0], float)
    isinstance(out.conditions_[0,1], float)
    isinstance(out.conditions_[0,2], float)

    assert np.alltrue(out.f_.max(axis=-1)>=1.8), \
                      '{}'.format(out.f_.max(axis=-1)[out.f_.max(axis=-1)<1.8])

    assert np.alltrue(out.Hf_.max(axis=-1)>=1.28), \
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

def test_dynamic_update_from_ABC_class():
    """ Make sure when certain properties are changed, the whole synthetic
    dataset is regenerated """

    n = np.linspace(-100,100,1001)
    

    # Changing n_samples
    n_samples_1 = 11
    n_samples_2 = 21
    lords = LorentzianTrainingData(n, n_samples=n_samples_1)
    assert lords.f_.shape[0] == n_samples_1
    assert lords.Hf_.shape[0] == n_samples_1

    lords.n_samples = n_samples_2
    assert lords.f_.shape[0] == n_samples_2
    assert lords.Hf_.shape[0] == n_samples_2

    del n_samples_1, n_samples_2, lords

    # stack_Hf_f=False, n_samples=1000, random_state=None,
    #              amp=1., center=None, width=None

    # Changing stack_Hf_f
    n_samples_1 = 11
    lords = LorentzianTrainingData(n, n_samples=n_samples_1)
    assert lords.stack_Hf_f == False
    assert lords.f_.shape[0] == n_samples_1
    assert lords.Hf_.shape[0] == n_samples_1

    lords.stack_Hf_f = True
    assert lords.f_.shape[0] == n_samples_1*2
    assert lords.Hf_.shape[0] == n_samples_1*2

    del n_samples_1, lords

    # Changing amp
    n_samples_1 = 11
    lords = LorentzianTrainingData(n, n_samples=n_samples_1, amp=1.0,
                                   center=[-10.,10.])
    assert lords.f_.shape[0] == n_samples_1
    assert lords.Hf_.shape[0] == n_samples_1
    assert np.alltrue(lords.f_.max(axis=-1) >= 0.99)

    lords.amp = 2.0
    assert lords.f_.shape[0] == n_samples_1
    assert lords.Hf_.shape[0] == n_samples_1
    assert np.alltrue(lords.f_.max(axis=-1) > 1.95)

    del n_samples_1, lords

    # Changing center
    n_samples_1 = 11
    lords = LorentzianTrainingData(n, n_samples=n_samples_1, center=0.0)
    temp = 1.0*lords.f_
    assert np.allclose(temp, lords.f_)

    lords.center = 10.0
    assert not np.allclose(temp, lords.f_)

    del n_samples_1, lords

    # Changing width
    n_samples_1 = 11
    lords = LorentzianTrainingData(n, n_samples=n_samples_1, width=10.0)
    temp = 1.0*lords.f_
    assert np.allclose(temp, lords.f_)

    lords.width = 20.0
    assert not np.allclose(temp, lords.f_)

    del n_samples_1, lords

    # Regenerate
    n_samples_1 = 11
    lords = LorentzianTrainingData(n, n_samples=n_samples_1, width=10.0)
    temp = 1.0*lords.f_
    assert np.allclose(temp, lords.f_)

    lords.regenerate()
    assert not np.allclose(temp, lords.f_)

    del n_samples_1, lords

    # Re-writing n_samples with same value -- to see if it changed
    n_samples_1 = 11
    lords = LorentzianTrainingData(n, n_samples=n_samples_1)
    temp = 1.0*lords.f_
    assert np.allclose(temp, lords.f_)

    lords.n_samples = n_samples_1
    assert not np.allclose(temp, lords.f_)

    del n_samples_1, lords

def test_synthspectra():
    n_spectra = 10
    n = np.linspace(-100,100,1001)

    lineshape_inst = LorentzianTrainingData(n, n_samples=1, amp=1.0,
                                            center=[-10.0,10.0],
                                            width=[10.0,20.])

    synthspect = SyntheticSpectra(lineshape_inst, n_spectra=n_spectra,
                                  n_peak_lims=[1,30])

    assert synthspect.f_.shape[0] == n_spectra
    assert synthspect.f_.shape[1] == n.size

    temp = 1.0*synthspect.f_

    synthspect.regenerate()

    assert synthspect.f_.shape[0] == n_spectra
    assert synthspect.f_.shape[1] == n.size
    assert not np.allclose(temp, synthspect.f_)


def test_amp_lims():
    """ Testing to make sure that the amp setting does what it's supposed to do.
    Using Lorentzian as a proxy for the Abtsract class"""

    n_spectra = 10
    n = np.linspace(-100,100,1001)

    # Constant
    lineshape_inst = LorentzianTrainingData(n, n_samples=1, amp=1.0)
    assert np.allclose(lineshape_inst.conditions_[:,0], 1.0)

    # Range [min, max)
    lineshape_inst = LorentzianTrainingData(n, n_samples=1, amp=[0.0, 1.0])
    assert np.alltrue(lineshape_inst.conditions_[:,0] < 1.0)
    assert np.alltrue(lineshape_inst.conditions_[:,0] >= 0.0)

def test_width_lims():
    """ Testing to make sure that the width setting does what it's supposed to
     do. Using Lorentzian as a proxy for the Abtsract class"""

    n_spectra = 10
    n = np.linspace(-100,100,1001)

    # Constant
    lineshape_inst = LorentzianTrainingData(n, n_samples=1, width=10.0)
    assert np.allclose(lineshape_inst.conditions_[:,2], 10.0)

    # Range [min, max)
    lineshape_inst = LorentzianTrainingData(n, n_samples=1, width=[1.0, 10.0])
    assert np.alltrue(lineshape_inst.conditions_[:,2] < 10.0)
    assert np.alltrue(lineshape_inst.conditions_[:,2] >= 1.0)

def test_center_lims():
    """ Testing to make sure that the center setting does what it's supposed to
     do. Using Lorentzian as a proxy for the Abtsract class"""

    n_spectra = 10
    n = np.linspace(-100,100,1001)

    # Constant
    lineshape_inst = LorentzianTrainingData(n, n_samples=1, center=10.0)
    assert np.allclose(lineshape_inst.conditions_[:,1], 10.0)

    # Range [min, max)
    lineshape_inst = LorentzianTrainingData(n, n_samples=1, center=[1.0, 10.0])
    assert np.alltrue(lineshape_inst.conditions_[:,1] < 10.0)
    assert np.alltrue(lineshape_inst.conditions_[:,1] >= 1.0)

def test_amp_reps():
    """ Testing to make sure that the amp_reps does what it's supposed to
     do. Using Lorentzian as a proxy for the Abtsract class"""

    n_samples = 10
    n = np.linspace(-100,100,1001)

    amps = [1.0, 2.0, 3.1]
    lineshape_inst = LorentzianTrainingData(n, n_samples=n_samples, 
                                            amp_reps=amps)

    assert np.unique(lineshape_inst.conditions_[:,0]).size == len(amps)
    assert np.unique(lineshape_inst.conditions_[:,1]).size == n_samples
    assert np.unique(lineshape_inst.conditions_[:,2]).size == n_samples
    assert lineshape_inst.f_.shape[0] == n_samples * len(amps)