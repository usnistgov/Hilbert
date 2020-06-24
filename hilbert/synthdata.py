from abc import ABC, abstractclassmethod
import itertools

import numpy as np

from scipy.special import dawsn, wofz, expit

class AbstractTrainingDataGenerator(ABC):
    """ Abstract class for generating synthetic data.
    
    Parameters
    ----------
    n : array-like, shape (n_features,)
        The independent variable

    Attributes
    ----------
    n_widths : int
        Number of width parameter associated with a given lineshape
        
    max_width_factor : int
        Lineshape center-to-edge-to-width maximum ratio
        
    
    """
    n_widths = 1
    max_width_factor = 2
    
    def __init__(self, n, stack_results=True, randomizer=False, n_rand=1000, subsample_center=2, 
                 subsample_width=2, random_state=None, rand_over=15):
        self.n = n
        
        self.subsample_center = subsample_center
        self.subsample_width = subsample_width
        self.randomizer = randomizer
        self.n_rand = n_rand
        self.rand_over = rand_over
        self.stack_results = stack_results
        
        np.random.seed(random_state)
        
        self.X_train_ = None
        self.Y_train_ = None
        self.conditions_ = None
        
        self.generate()
               

    @property
    def min_center(self):
        return self.n.min() + 10*np.abs(self.dn)
    
    @property
    def max_center(self):
        return self.n.max() - 10*np.abs(self.dn)
    
    @property
    def dn(self):
        return self.n[1] - self.n[0]
    
    @property
    def Dn(self):
        return self.n[-1] - self.n[0]
    
    @property
    def center_steps(self):
        return int(np.floor((self.max_center-self.min_center)/self.dn))//self.subsample_center
    
    @property
    def center_vec(self):
        return np.linspace(self.min_center, self.max_center, self.center_steps)
    
    def generate_conditions(self):
        conditions_vec = []
        a = 1
        
        if not self.randomizer:
            for ctr in self.center_vec:
                max_width = np.min(((self.n[-1] - ctr)/self.max_width_factor, (ctr - self.n[0])/self.max_width_factor))
                min_width = 3*self.dn
                width_steps = int(np.floor((max_width-3*self.dn)/self.dn))//self.subsample_width
                width_vec = np.linspace(min_width, max_width, width_steps)

                for ws in itertools.product(width_vec, repeat=self.n_widths):
                    q = [a, ctr]
                    q.extend(ws)
                    conditions_vec.append(q)
            self.conditions_ = np.array(conditions_vec)
                    
        else:
            r_a_vec = a*np.ones(self.n_rand * self.rand_over)
            r_ctr_vec = (self.max_center - self.min_center)*np.random.rand(self.n_rand * self.rand_over) + self.min_center
            
            max_width = self.Dn/self.max_width_factor
            min_width = 3*self.dn
            
            r_width_vec = (max_width - min_width)*np.random.rand(self.n_rand * self.rand_over, self.n_widths) + min_width
            
            self.conditions_ = np.hstack((r_a_vec[:,None], r_ctr_vec[:,None], r_width_vec))
            
            max_width_vec = np.vstack(((self.n[-1] - self.conditions_[:,1])/2, 
                                       (self.conditions_[:,1] - self.n[0])/2)).min(axis=0)
            
            self.conditions_ = self.conditions_[self.conditions_[:,2] - max_width_vec < 0]
            
            if self.conditions_.shape[0] > self.n_rand:
                self.conditions_ = self.conditions_[:self.n_rand,...]
            
                        
    def generate(self):      
        f = []
        Hf = []
        
        self.generate_conditions()
        temp = self.fcn(self.n, self.conditions_)
        
        f = temp.real
        Hf = temp.imag
        del temp
        
        if self.stack_results:
            self.X_train_ = np.vstack((f, Hf))
            self.Y_train_ = np.vstack((Hf, -f))
            self.conditions_ = np.vstack((self.conditions_, self.conditions_))
        else:
            self.X_train_ = f
            self.Y_train_ = Hf
        
    @abstractclassmethod
    def fcn(cls, n, cmat):
        raise NotImplementedError
        
    @classmethod
    def _check_cmat_shape(cls, cmat):
        if cmat.ndim == 1:
            assert cmat.size == 2 + cls.n_widths
            return cmat[None,:]
        elif cmat.ndim == 2:
            assert cmat.shape[1] == 2 + cls.n_widths
            return cmat
        else:
            raise ValueError('cmat need be a 2D array')
        
    def __repr__(self):
        if self.X_train_ is None:
            return 'Empty'
        else:
            return '{} training samples generated ({} features)'.format(self.X_train_.shape[0], self.X_train_.shape[1])
        
class LorentzianTrainingData(AbstractTrainingDataGenerator):
    n_widths = 1
            
    @classmethod
    def fcn(cls, n, cmat):
        """ Returns a Lorentzian (real part) and its Hilbert transform (imag part) """
        
        cmat = cls._check_cmat_shape(cmat)
        amp = cmat[...,0][:,None]
        ctr = cmat[...,1][:,None]
        width = cmat[...,2][:,None]
        
        return amp*width*width/((n-ctr)**2 + width**2) + 1j*amp*width*(n-ctr) / ((n-ctr)**2 + width**2)
    
    
class GaussianTrainingData(AbstractTrainingDataGenerator):
    n_widths = 1
            
    @classmethod
    def fcn(cls, n, cmat):
        """ Returns a Gaussian (real part) and its Hilbert transform (imag part) """
        
        assert cmat.shape[1] == 2 + cls.n_widths
        amp = cmat[:,0][:,None]
        ctr = cmat[:,1][:,None]
        width = cmat[:,2][:,None]
        
        return amp * np.exp(-(n-ctr)**2/(2*width**2)) + 1j*(amp * 2 / np.sqrt(np.pi) * dawsn(np.sqrt(1/(2*width**2))*(n-ctr)))
    
class SincTrainingData(AbstractTrainingDataGenerator):
    n_widths = 1
            
    @classmethod
    def fcn(cls, n, cmat):
        """ Returns a Sinc (real part) and its Hilbert transform (imag part) """
        
        assert cmat.shape[1] == 2 + cls.n_widths
        amp = cmat[:,0][:,None]
        ctr = cmat[:,1][:,None]
        width = cmat[:,2][:,None]
        
        f = 1/width
        a = 2*np.pi*f
        denom = a*(n-ctr)
        denom[denom == 0] = 1

        return amp*np.sinc(a/np.pi*(n-ctr)) + 1j* ((1-np.cos(a*(n-ctr)))/denom)
        
        
class VoigtTrainingData(AbstractTrainingDataGenerator):
    n_widths = 2
    max_width_factor = 4
            
    @classmethod
    def fcn(cls, n, cmat):
        """ Returns a Voigt (real part) and its Hilbert transform (imag part) """
        
        assert cmat.shape[1] == 2 + cls.n_widths
        amp = cmat[:,0][:,None]
        ctr = cmat[:,1][:,None]
        width_lorentz = cmat[:,2][:,None]
        width_gaussian = cmat[:,3][:,None]
        
        z = ((n-ctr)+1j*width_lorentz)/(width_gaussian*np.sqrt(2))
        z0 = (0+1j*width_lorentz)/(width_gaussian*np.sqrt(2))
        return amp*wofz(z)/wofz(z0)
    
    