"""
Scikit-Learn compatible estimators for Hilbert transforms

"""
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.base import RegressorMixin, BaseEstimator
from scipy.signal import hilbert as _scipy_hilbert
from hilbert.utils import hilbert_pad_simple
from hilbert.dft import hilbert_scipy

class DHT(BaseEstimator, RegressorMixin):
    """ Class that performs the DHT. Default is SciPy's DHT """
    def __init__(self, step_size=30000, 
                 hilbert_fcn=hilbert_scipy):
        super().__init__()
        self.step_size = step_size
        self.hilbert_fcn = hilbert_fcn
        
    def fit(self, X, y):
        pass
        
    def predict(self, X):
        assert (X.ndim <= 2), '3D not setup for this file'
        if X.shape[0] > self.step_size:
            out = 0*X
            left_to_do = X.shape[0]
            for num in range(X.shape[0] // self.step_size + X.shape[0] % self.step_size):
                if left_to_do >= self.step_size:
                    out[num*self.step_size:(num+1)*self.step_size,:] = self.hilbert_fcn(X[num*self.step_size:(num+1)*self.step_size,:])
                    left_to_do -= self.step_size
                else:
                    out[num*self.step_size:,:] = self.hilbert_fcn(X[num*self.step_size:,:])
                    left_to_do = 0
            return out                    
        else:
            return self.hilbert_fcn(X)

class DHT_Pad(BaseEstimator, RegressorMixin):
    """ Class that pads the input data and performs the DHT from SciPy """
    def __init__(self, hilbert_fcn=hilbert_scipy, 
                 pad_factor=1, step_size=30000):
        super().__init__()
        self.pad_factor = pad_factor
        self.step_size = step_size
        self.hilbert_fcn = hilbert_fcn
        
    def fit(self, X, y):
        pass
        
    def predict(self, X):
        assert (X.ndim <= 2), '3D not setup for this file'

        if X.shape[0] > self.step_size:
            out = 0*X
            left_to_do = X.shape[0]
            for num in range(X.shape[0] // self.step_size + X.shape[0] % self.step_size):
                if left_to_do >= self.step_size:
                    out[num*self.step_size:(num+1)*self.step_size,:] = hilbert_pad_simple(X[num*self.step_size:(num+1)*self.step_size,:], self.hilbert_fcn, m_multi_size=self.pad_factor)
                    left_to_do -= self.step_size
                else:
                    out[num*self.step_size:,:] = hilbert_pad_simple(X[num*self.step_size:,:], self.hilbert_fcn, m_multi_size=self.pad_factor)
                    left_to_do = 0
            return out                    
        else:
            return hilbert_pad_simple(X, self.hilbert_fcn, m_multi_size=self.pad_factor)


class LeDHT(RegressorMixin, BaseEstimator):
    def __init__(self, regressor_inst):
        self.regressor_inst = regressor_inst
    
    def fit(self, X, y):
        self.regressor_inst.fit(X, y)
    
    def predict(self, X):
        return np.dot(X, self.regressor_inst.coef_.T) + self.regressor_inst.intercept_