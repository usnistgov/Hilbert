"""
Module for generating synthetic data.
"""

from abc import (ABC, abstractclassmethod, abstractproperty, 
                 abstractstaticmethod)
import itertools

import numpy as np

from scipy.special import dawsn, wofz, expit

class AbstractRandomTrainingDataGenerator(ABC):    
    """ Abstract class for generating synthetic data. Assumes a lineshape has
    exactly 1 amplitude, width, and center parameter
    
    Parameters
    ----------
    n : array-like, shape (n_features,)
        The independent variable
    stack_Hf_f : bool, optional
        Stack Hf -f in the results, by default True
    n_samples : int, optional
        Number of samples to generate, by default 1000
    random_state : int, optional
        Random number generator seed, by default None
    amp : float, list, or function-like
        Amplitude constant, list of [min,max) for uniform distribution, 
        or function to generate an amplitude.
    
    Attributes
    ----------
    config : dict
        Configuration settings
        
    """
    
    # Configuration dictionary (example from Gaussian)
    m_width_is_fwhm = 2*np.sqrt(2*np.log(2))
    config = {'m_widths_from_edge' : 0.5 * m_width_is_fwhm,  # 1/2 FWHMs from edge
              'width_over_Dn_is_max_width' : 1 / m_width_is_fwhm,  # 1 FWHM is max width
              'width_over_dn_is_min_width' : 3 / m_width_is_fwhm  # min FWHM = 3 dn
             }
    
    def __init__(self, n, stack_Hf_f=False, n_samples=1000, random_state=None, 
                 amp=1., center=None, width=None):     
        self.n = n
        self._amp_fcn = self._ret_val_fcn(amp)
        if width is None:
            self._width_fcn = self._ret_val_fcn([self.min_width, self.max_width])
        else:
            self._width_fcn = self._ret_val_fcn(width)
        if center is None:
            self._center_fcn = None
        else:
            self._center_fcn = self._ret_val_fcn(center)

        self.n_samples = n_samples
        self.stack_Hf_f = stack_Hf_f
        
        np.random.seed(random_state)
        
        self.f_ = None
        self.Hf_ = None
        self.conditions_ = None
        
        self.generate()
    
    def _ret_val_fcn(self, value):
        if isinstance(value, float):
            return lambda value=value: value
        elif isinstance(value, int):
            return lambda value=value: float(value)
        elif isinstance(value, (list, tuple)):
            assert len(value) == 2, 'value must be a float, 2-entry list/tuple, or a function'
            assert value[0] < value[1], 'value list/tuple must be [min, max)'
            minner = value[0]
            maxer = value[1]

            return lambda minner=minner, maxer=maxer: (maxer - minner)*np.random.rand(1)[0] + minner
        else:  # Assumes it's an executable with no inputs
            return value

    @property
    def dn(self):
        return self.n[1] - self.n[0]
    
    @property
    def Dn(self):
        return self.n[-1] - self.n[0]
    
    @property
    def min_width(self):
        return self.config['width_over_dn_is_min_width'] * self.dn

    @property
    def max_width(self):
        return self.config['width_over_Dn_is_max_width'] * self.Dn

    def min_ctr(self, width):
        return self.n.min() + self.config['m_widths_from_edge']*width

    def max_ctr(self, width):
        return self.n.max() - self.config['m_widths_from_edge']*width

    def generate_conditions(self):
        conditions_vec = []

        # TODO: width and ctr as fcn's that can be user-set. If None, revert to the given limits

        for _ in range(self.n_samples):
            a = self._amp_fcn()
            # width = (self.max_width - self.min_width) * np.random.rand(1)[0] + \
            #          self.min_width
            width = self._width_fcn()
            max_ctr = self.max_ctr(width)
            min_ctr = self.min_ctr(width)
            if max_ctr < min_ctr:
                raise ValueError('Max-Min Center is not possible: {},{}. Check config[m_widths_from_edge]'.format(min_ctr, max_ctr))

            if self._center_fcn is None:
                ctr = (max_ctr - min_ctr) * np.random.rand(1)[0] + min_ctr
            else:
                ctr = self._center_fcn()
                if ctr < min_ctr:
                    ctr = min_ctr
                    print('Warning: ctr < min_ctr. Setting to min_ctr...')
                if ctr > max_ctr:
                    ctr = max_ctr
                    print('Warning: ctr > max_ctr. Setting to max_ctr...')

            conditions_vec.append([a, ctr, width])
        self.conditions_ = np.array(conditions_vec)           

    def generate(self):      
        f = []
        Hf = []
        
        self.generate_conditions()
        temp = self.fcn(self.n, self.conditions_)
        
        f = temp.real
        Hf = temp.imag
        del temp
        
        if self.stack_Hf_f:
            self.f_ = np.vstack((f, Hf))
            self.Hf_ = np.vstack((Hf, -f))
            self.conditions_ = np.vstack((self.conditions_, self.conditions_))
        else:
            self.f_ = f
            self.Hf_ = Hf
        
    # @abstractstaticmethod
    # def fwhm(width):
    #     raise NotImplementedError

    @abstractclassmethod
    def fcn(cls, n, cmat):
        raise NotImplementedError
        
    @classmethod
    def _check_cmat_shape(cls, cmat):
        if cmat.ndim == 1:
            assert cmat.size == 2 + 1
            return cmat[None,:]
        elif cmat.ndim == 2:
            assert cmat.shape[1] == 2 + 1
            return cmat
        else:
            raise ValueError('cmat need be a 2D array')
        
    def __repr__(self):
        if self.f_ is None:
            return 'Empty'
        else:
            return '{} training samples generated ({} features)'.format(self.f_.shape[0], self.f_.shape[1])
        
class LorentzianTrainingData(AbstractRandomTrainingDataGenerator):
    """ Generate Lorentzian-Dispersive synthetic data (see notes). Unless 
    stack_Hf_f = True, f_ will be the even function (Lorentzian), and Hf_ will
    be the odd function (Dispersive).
    
    Parameters
    ----------
    n : array-like, shape (n_features,)
        The independent variable
    stack_Hf_f : bool, optional
        Stack Hf -f in the results, by default True
    n_samples : int, optional
        Number of samples to generate, by default 1000
    random_state : int, optional
        Random number generator seed, by default None
    amp : float, list, or function-like
        Amplitude constant, list of [min,max) for uniform distribution, 
        or function to generate an amplitude.
    
    Attributes
    ----------
    config : dict
        Configuration settings
        
    Notes
    -----
    This class generates the Lorentzian - Dispersive lineshape pair, sometimes
    casually referred to as a complex Lorentzian in the literature [1]_.

    .. math:: f(x) + iH\{f\}(x) = \frac{A\Gamma}{\Omega - n - i\Gamma}

    where :math:`\Omega` is the center position, :math:`\Gamma` is the 
    half-width, and :math:`A` is the amplitude. 
    
    -   **Note** this is not a standard definition (i.e., an extra 
        :math:`\Gamma` in the numerator), but we have used it so that 
        the user can select the amplitude directly.

    .. [1] A. D. Poularikas, "Hilbert Transform," in The Handbook of Forumulas
        and Tables for Signal Processing, A. D. Poularikas, ed. (CRC, 1999).

    """

    m_width_is_fwhm = 2
    config = {'m_widths_from_edge' : 0.5 * m_width_is_fwhm,  # 1/2 FWHMs from edge
              'width_over_Dn_is_max_width' : 1 / m_width_is_fwhm,  # 1 FWHM is max width
              'width_over_dn_is_min_width' : 3 / m_width_is_fwhm}  # min FWHM = 3 dn

    @classmethod
    def fcn(cls, n, cmat):
        """ Returns a Lorentzian (real part) and its Hilbert transform (imag part) """
        
        cmat = cls._check_cmat_shape(cmat)
        amp = cmat[...,0][:,None]
        ctr = cmat[...,1][:,None]
        width = cmat[...,2][:,None]
        
        return amp*width*width/((n-ctr)**2 + width**2) + 1j*amp*width*(n-ctr) / ((n-ctr)**2 + width**2)

    @staticmethod
    def fwhm(width):
        """Full-width at half-max, where width is Gamma"""
        return 2*width
    
class GaussianTrainingData(AbstractRandomTrainingDataGenerator):
    """ Generate Gaussian-Dawson synthetic data (see notes). Unless 
    stack_Hf_f = True, f_ will be the even function (Gaussian), and Hf_ will
    be the odd function (Dawson).
    
    Parameters
    ----------
    n : array-like, shape (n_features,)
        The independent variable
    stack_Hf_f : bool, optional
        Stack Hf -f in the results, by default True
    n_samples : int, optional
        Number of samples to generate, by default 1000
    random_state : int, optional
        Random number generator seed, by default None
    amp : float, list, or function-like
        Amplitude constant, list of [min,max) for uniform distribution, 
        or function to generate an amplitude.
    
    Attributes
    ----------
    config : dict
        Configuration settings
        
    Notes
    -----
    This class generates the Gaussian - Dawson lineshape pair [1]_.

    .. math:: 
    
        f(x) + iH\{f\}(x) = A \exp \left [-\frac{(n - n_0)^2}{2\sigma^2} \right] + \\
        i\frac{2A}{\sqrt{\pi}} \mathcal{D}\{\sqrt{\frac{1}{2\sigma^2}}(n-n_0)\}

    where :math:`n_0` is the center position, :math:`\sigma` is a width 
    parameter, and :math:`A` is the amplitude. 
    
    .. [1] https://en.wikipedia.org/wiki/Dawson_function#Relation_to_Hilbert_transform_of_Gaussian


    """
    m_width_is_fwhm = 2*np.sqrt(2*np.log(2))
    config = {'m_widths_from_edge' : 0.5 * m_width_is_fwhm,  # 1/2 FWHMs from edge
              'width_over_Dn_is_max_width' : 1 / m_width_is_fwhm,  # 1 FWHM is max width
              'width_over_dn_is_min_width' : 3 / m_width_is_fwhm}  # min FWHM = 3 dn

    @classmethod
    def fcn(cls, n, cmat):
        """ Returns a Gaussian (real part) and its Hilbert transform (imag part) """
        
        cmat = cls._check_cmat_shape(cmat)
        amp = cmat[:,0][:,None]
        ctr = cmat[:,1][:,None]
        width = cmat[:,2][:,None]
        
        return amp * np.exp(-(n-ctr)**2/(2*width**2)) + 1j*(amp * 2 / np.sqrt(np.pi) * dawsn(np.sqrt(1/(2*width**2))*(n-ctr)))

    @staticmethod
    def fwhm(width):
        """Full-width at half-max, where width is sigma"""
        return 2*width*np.sqrt(2*np.log(2))
    
class SincTrainingData(AbstractRandomTrainingDataGenerator):
    """ Generate Sinc and Hilbert of Sinc synthetic data pair (see notes). Unless 
    stack_Hf_f = True, f_ will be the even function (Sinc), and Hf_ will
    be the odd function (Hilbert of Sinc).
    
    Parameters
    ----------
    n : array-like, shape (n_features,)
        The independent variable
    stack_Hf_f : bool, optional
        Stack Hf -f in the results, by default True
    n_samples : int, optional
        Number of samples to generate, by default 1000
    random_state : int, optional
        Random number generator seed, by default None
    
    Attributes
    ----------
    config : dict
        Configuration settings
        
    Notes
    -----
    This class generates the Sinc - Hilbert of Sinc lineshape pair [1]_.

    .. math:: 
    
        f(x) + iH\{f\}(x) = A \text{sinc}[2f(n-n_0)] + iA\frac{1 - \cos[2f(n-n_0)]}{2f\pi(n-n_0)}

    where :math:`n_0` is the center position, :math:`f` is a frequency 
    parameter inversely proportional to width, and :math:`A` is the amplitude. 
    
    .. [1] A. D. Poularikas, "Hilbert Transform," in The Handbook of Forumulas 
        and Tables for Signal Processing, A. D. Poularikas, ed. (CRC, 1999).
    

    """
    m_width_max_to_max = 2.459018059619502  # Empirical
    config = {'m_widths_from_edge' : 0.5 * m_width_max_to_max,  # 1/2 from edge
              'width_over_Dn_is_max_width' : 1 / m_width_max_to_max,  # 1 FWHM is max width
              'width_over_dn_is_min_width':9 / m_width_max_to_max}  # min FWHM = 3 dn

            
    @classmethod
    def fcn(cls, n, cmat):
        """ Returns a Sinc (real part) and its Hilbert transform (imag part) """
        
        cmat = cls._check_cmat_shape(cmat)
        amp = cmat[:,0][:,None]
        ctr = cmat[:,1][:,None]
        width = cmat[:,2][:,None]
        
        f = 1/width
        denom = 2*f*np.pi*(n-ctr)
        denom[denom == 0] = 1

        return amp*np.sinc(2*f*(n-ctr)) + 1j*amp*((1-np.cos(2*f*np.pi*(n-ctr)))/denom)
        
   
# class VoigtTrainingData(AbstractRandomTrainingDataGenerator):
#     m_widths = 2
#     # AbstractTrainingDataGenerator.config['n_fwhm_is_max_width'] = 4
            
#     @classmethod
#     def fcn(cls, n, cmat):
#         """ Returns a Voigt (real part) and its Hilbert transform (imag part) """
        
#         cmat = cls._check_cmat_shape(cmat)
#         amp = cmat[:,0][:,None]
#         ctr = cmat[:,1][:,None]
#         width_lorentz = cmat[:,2][:,None]
#         width_gaussian = cmat[:,3][:,None]
        
#         z = ((n-ctr)+1j*width_lorentz)/(width_gaussian*np.sqrt(2))
#         z0 = (0+1j*width_lorentz)/(width_gaussian*np.sqrt(2))
#         return amp*wofz(z)/wofz(z0)

#     @classmethod
#     def fwhm(cls, width_g, width_l):
#         """ Approximate FWHM """
#         fwhm_l = LorentzianTrainingData.fwhm(width_l)
#         fwhm_g = GaussianTrainingData.fwhm(width_g)

#         return fwhm_l/2 + np.sqrt(fwhm_l**2/4 + fwhm_g**2)

#     @property
#     def min_width(self):
#         """ Minimum resolvable width. We'll just say it's one period."""
#         return 2 / (3*self.dn)
    

if __name__ == '__main__':
    n = np.linspace(-100,100,1001)
    out = LorentzianTrainingData(n, stack_Hf_f=False, n_samples=1000, 
                                 random_state=0, amp=lambda: 1, width=[1,20.])

    # print(out.Hf_.max(axis=-1).min())
    # assert np.alltrue(out.Hf_.max(axis=-1) > 0.45)

    # assert np.alltrue(out.f_.max(axis=-1)>=0.9), \
    #                   '{}'.format(out.f_.max(axis=-1)[out.f_.max(axis=-1)<0.9])

    # out = LorentzianTrainingData(n, stack_Hf_f=False, n_samples=1001, 
    #                              random_state=0, amp=2)
    
    # assert np.alltrue(out.f_.max(axis=-1)>=1.8), \
    #                   '{}'.format(out.f_.max(axis=-1)[out.f_.max(axis=-1)<1.8])
    # out = LorentzianTrainingData(n, stack_Hf_f=False, n_samples=101, 
    #                              random_state=0, amp=[2.,4.])
    # assert np.alltrue((out.f_.max(axis=-1)>=2) &
    #                   (out.f_.max(axis=-1)<4))

    # out = LorentzianTrainingData(n, stack_Hf_f=False, n_samples=101, 
    #                              random_state=0, amp=lambda: 10*np.random.rand(1)[0]+3)
    # assert np.alltrue((out.f_.max(axis=-1)>=3) &
    #                   (out.f_.max(axis=-1)<13))

    
    # print(out.f_.max(axis=-1))
    

    # out = LorentzianTrainingData(n, stack_Hf_f=False, n_samples=11, amp=2)
    # assert np.allclose(out.f_.max(axis=-1),1)
    
    # out = LorentzianTrainingData(n, stack_Hf_f=False, n_samples=1001, amp=[0,1])
    # assert print(out.f_)
    




    