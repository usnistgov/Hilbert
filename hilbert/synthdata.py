"""
Module for generating synthetic data.
"""

from abc import (ABC, abstractclassmethod)

import numpy as np

from scipy.special import dawsn

class AbstractRandomTrainingDataGenerator(ABC):
    """ Abstract class for generating synthetic data. Assumes a lineshape has
    exactly 1 amplitude, width, and center parameter

    Parameters
    ----------
    n : array-like, shape (n_features,)
        The independent variable
    n_samples : int, optional
        Number of samples to generate, by default 1000
    amp : float, list, or function-like
        Amplitude constant, list of [min,max) for uniform distribution,
        or function to generate an amplitude.
    amp_reps : list or array-like (of float)
        Supercedes amp. Generate replicate sets of lineshape only varying
        in amplitude. E.g., [1.0 ,2.0] would be identical sets of lineshapes but one
        with amp 1 and one with amp 2. n_samples will be re-written with new
        larger value. Only float & int constants currently supported.
    center : float, list, or function-like
        Center-n constant, list of [min,max) for uniform distribution,
        or function to generate an amplitude.
    width : float, list, or function-like
        Width constant, list of [min,max) for uniform distribution,
        or function to generate an amplitude.
    random_state : int, optional
        Random number generator seed, by default None
    f_is_even : bool, optional
        Whether f is an even function (ergo Hf is odd) or vice versa, by 
        default True
    stack_Hf_f : bool, optional
        Stack Hf -f in the results, by default False

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

    def __init__(self, n, n_samples=1000, amp=1., amp_reps=None, center=None, 
                 width=None, random_state=None, f_is_even=True, 
                 stack_Hf_f=False, **kwargs):

        # Whether to re-update after changing params
        # Before all params are initially set, we don't want this
        self._can_update = False
        self.n = n

        self._amp = None
        self._amp_reps = None
        self._width = None
        self._center = None
        self._n_samples = None
        self._stack_Hf_f = None
        self._f_is_even = None

        self._amp_fcn = None
        self._width_fcn = None
        self._center_fcn = None

        self.amp = amp
        self.amp_reps = amp_reps
        self.width = width
        self.center = center
        self.n_samples = n_samples
        self.stack_Hf_f = stack_Hf_f
        self.f_is_even = f_is_even

        np.random.seed(random_state)

        self.f_ = None
        self.Hf_ = None
        self.conditions_ = None

        if kwargs:
            for k in kwargs:
                self.config.update({k:kwargs[k]})

        self.generate_conditions()
        self.generate()

        self._can_update = True

    def regenerate(self):
        """Create new conditions and regenerate f_ and Hf"""
        if self._can_update:
            self.generate_conditions()
            self.generate()

    @property
    def f_is_even(self):
        return self._f_is_even

    @f_is_even.setter
    def f_is_even(self, value):
        self._f_is_even = value
        self.regenerate()

    @property
    def stack_Hf_f(self):
        return self._stack_Hf_f

    @stack_Hf_f.setter
    def stack_Hf_f(self, value):
        self._stack_Hf_f = value
        self.regenerate()

    @property
    def n_samples(self):
        return self._n_samples

    @n_samples.setter
    def n_samples(self, value):
        self._n_samples = value
        self.regenerate()

    @property
    def n_samples_total(self):
        if self.f_ is not None:
            return self.f_.shape[0]
            
    @property
    def amp(self):
        return self._amp

    @amp.setter
    def amp(self, value):
        self._amp = value
        self._amp_fcn = self._ret_val_fcn(self._amp)
        self.regenerate()

    @property
    def amp_reps(self):
        return self._amp_reps

    @amp_reps.setter
    def amp_reps(self, value):
        if value is not None:
            assert np.alltrue([isinstance(v, (int, float)) for v in value])
        self._amp_reps = value
        self.regenerate()

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, value):
        self._width = value

        if self._width is None:
            self._width_fcn = self._ret_val_fcn([self.min_width, self.max_width])
        else:
            self._width_fcn = self._ret_val_fcn(self._width)

        self.regenerate()

    @property
    def center(self):
        return self._center

    @center.setter
    def center(self, value):
        self._center = value

        if self._center is None:
            self._center_fcn = None
        else:
            self._center_fcn = self._ret_val_fcn(self._center)

        self.regenerate()

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
        """Generate amplitude, width, center conditions for simulation"""

        # TODO: Refactor for improved memory usage

        ctr_list = []
        amp_list = []
        width_list = []

        remaining_samples = self.n_samples

        while remaining_samples > 0:
            temp_widths = np.array([self._width_fcn() for _ in range(remaining_samples)])

            if self.amp_reps is None:
                temp_amps = np.array([self._amp_fcn() for _ in range(remaining_samples)])
            else:
                temp_amps = np.array([self.amp_reps[0] for _ in range(remaining_samples)])

            max_ctr = self.max_ctr(self.min_width)
            min_ctr = self.min_ctr(self.min_width)

            if self._center_fcn is None:
                temp_ctrs = (max_ctr - min_ctr) * np.random.rand(remaining_samples) + min_ctr
            else:
                temp_ctrs = np.array([self._center_fcn() for _ in range(remaining_samples)])

            idx_list= []
            for num, (w,c) in enumerate(zip(temp_widths, temp_ctrs)):
                if ((c >= self.min_ctr(w)) & (c <= self.max_ctr(w))):
                    idx_list.append(num)
            temp_ctrs = temp_ctrs[idx_list]
            temp_amps = temp_amps[idx_list]
            temp_widths = temp_widths[idx_list]

            ctr_list.extend(temp_ctrs.tolist())
            width_list.extend(temp_widths.tolist())
            amp_list.extend(temp_amps.tolist())

            remaining_samples -= temp_ctrs.size
        if self.amp_reps is not None:
            for num, a in enumerate(self.amp_reps):
                if num == 0:
                    continue
                else:
                    amp_list.extend(self.n_samples*[a])
                    width_list.extend(width_list[:self.n_samples])
                    ctr_list.extend(ctr_list[:self.n_samples])

        self.conditions_ = np.vstack((amp_list, ctr_list, width_list)).T

    def generate(self):
        """Generate f_ and Hf_ based on self.conditions"""
        f = []
        Hf = []

        temp = self.fcn(self.n, self.conditions_)

        if self.f_is_even:
            f = temp.real
            Hf = temp.imag
        else:
            f = -temp.imag
            Hf = temp.real

        del temp

        if self.stack_Hf_f:
            self.f_ = np.vstack((f, Hf))
            self.Hf_ = np.vstack((Hf, -1*f))
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
    """Generate Lorentzian-Dispersive synthetic data (see notes). Unless
    stack_Hf_f = True, f_ will be the even function (Lorentzian), and Hf_ will
    be the odd function (Dispersive).

    Parameters
    ----------
    n : array-like, shape (n_features,)
        The independent variable
    n_samples : int, optional
        Number of samples to generate, by default 1000
    amp : float, list, or function-like
        Amplitude constant, list of [min,max) for uniform distribution,
        or function to generate an amplitude.
    center : float, list, or function-like
        Center-n constant, list of [min,max) for uniform distribution,
        or function to generate an amplitude.
    width : float, list, or function-like
        Width constant, list of [min,max) for uniform distribution,
        or function to generate an amplitude.
    random_state : int, optional
        Random number generator seed, by default None
    f_is_even : bool, optional
        Whether f is an even function (ergo Hf is odd) or vice versa, by 
        default True
    stack_Hf_f : bool, optional
        Stack Hf -f in the results, by default False

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
    n_samples : int, optional
        Number of samples to generate, by default 1000
    amp : float, list, or function-like
        Amplitude constant, list of [min,max) for uniform distribution,
        or function to generate an amplitude.
    center : float, list, or function-like
        Center-n constant, list of [min,max) for uniform distribution,
        or function to generate an amplitude.
    width : float, list, or function-like
        Width constant, list of [min,max) for uniform distribution,
        or function to generate an amplitude.
    random_state : int, optional
        Random number generator seed, by default None
    f_is_even : bool, optional
        Whether f is an even function (ergo Hf is odd) or vice versa, by 
        default True
    stack_Hf_f : bool, optional
        Stack Hf -f in the results, by default False

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
    n_samples : int, optional
        Number of samples to generate, by default 1000
    amp : float, list, or function-like
        Amplitude constant, list of [min,max) for uniform distribution,
        or function to generate an amplitude.
    center : float, list, or function-like
        Center-n constant, list of [min,max) for uniform distribution,
        or function to generate an amplitude.
    width : float, list, or function-like
        Width constant, list of [min,max) for uniform distribution,
        or function to generate an amplitude.
    random_state : int, optional
        Random number generator seed, by default None
    f_is_even : bool, optional
        Whether f is an even function (ergo Hf is odd) or vice versa, by 
        default True
    stack_Hf_f : bool, optional
        Stack Hf -f in the results, by default False

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

class SyntheticSpectra:
    """Generate synthetic spectra/signals

        Parameters
        ----------
        n : array-like, shape (n_features,)
            The independent variable
        lineshape_inst : instance subclassed from AbstractRandomTrainingDataGenerator
            Instance of a class that generates lineshapes and their Hilbert
            transform, example:
            LorentzianTrainingData(n, n_samples=10, amp=[0.1, 3.0],
                                   center=None, width=[1.2, 15])
        n_spectra : int, optional
            Number of spectra/signal to generate, by default 100
        n_peak_lims : list of 2 ints, optional
            [Min, Max) number of peaks that can be in a single spectrum,
            by default [1,30]
        """
    def __init__(self, lineshape_inst, n_spectra=100, n_peak_lims=[1,30]):

        self._can_update = False

        self._lineshape_inst = lineshape_inst
        self._n_spectra = None
        self._n_peak_lims = None

        self.n_spectra = n_spectra
        self.n_peak_lims = n_peak_lims
        self.lineshape_inst = lineshape_inst
        self._can_update = True

        self.regenerate()


    def regenerate(self):
        """ Regenerate new set of spectra/signals"""
        if not self._can_update:
            return None
        self.f_ = []
        self.Hf_ = []
        self.n_peaks_ = []
        self.conditions_ = []

        for _ in range(self.n_spectra):
            n_peaks = np.random.randint(self.n_peak_lims[0],
                                        self.n_peak_lims[1])
            self.n_peaks_.append(n_peaks)
            self.lineshape_inst.n_samples = n_peaks
            self.conditions_.append(self.lineshape_inst.conditions_)

            
            self.f_.append(self.lineshape_inst.f_.sum(axis=0))
            self.Hf_.append(self.lineshape_inst.Hf_.sum(axis=0))
            
        self.f_ = np.array(self.f_)
        self.Hf_ = np.array(self.Hf_)

    @property
    def lineshape_inst(self):
        return self._lineshape_inst

    @lineshape_inst.setter
    def lineshape_inst(self, value):
        self._lineshape_inst = value
        self.regenerate()

    @property
    def n_spectra(self):
        return self._n_spectra

    @n_spectra.setter
    def n_spectra(self, value):
        self._n_spectra = value
        self.regenerate()

    @property
    def n_peak_lims(self):
        return self._n_peak_lims

    @n_peak_lims.setter
    def n_peak_lims(self, value):
        self._n_peak_lims = value
        self.regenerate()







