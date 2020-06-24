"""
Module for discrete Hilbert transform that's enhanced with machine learning aspects
"""
import numpy as np
import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from crikit.cri.algorithms.kk import hilbertfft


class MLHilb(TransformerMixin, BaseEstimator):
    """ Machine-learning based Hilbert transform designed for improved edge-effect

    Parameters
    ----------
    regressor: sklearn-like instantiated regressor object
        Regression object with `fit` method
    pad_first : int, optional
        Padding factor for `first' FFT-based Hilbert transform, by default 0
    pad_second : int, optional
        Padding factor for `second' FFT-based Hilbert transform, by default 1

    Attributes
    ----------
    n_features_ : int
        Number of features
    """        
    
    def __init__(self, regressor, pad_first=0, pad_second=1):
        self.regressor = regressor
        self.pad_first = pad_first
        self.pad_second = pad_second        

    def fit(self, X, y):
        """Fit the ML-Hilbert transform on a training set (X) with known Hilbert transforms in y (should really be Y)

        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            Training data following Y[n] = H{X[n]}
        y : {array-like}, shape (n_samples, n_features)
            Target data following Y[n] = H{X[n]}
        
        Returns
        -------
        self : object
            Returns self.

        Raises
        ------
        ValueError
            X and y must have the same shape
        """        
        
        # Currently Numpy does not support complex values
        X = check_array(X, accept_sparse=False, dtype='float')

        if X.shape != y.shape:
            raise ValueError('X and y must have the same shape')
            
        y = check_array(y, accept_sparse=False, dtype='float')
        
        self.n_features_ = X.shape[1]

        # `first` Hilbert transform
        HA_first_ = hilbertfft(X, pad_factor=self.pad_first)

        # `second` Hilbert transform
        HA_second_ = hilbertfft(X, pad_factor=self.pad_second)
        
        # Different of first and second Hilbert
        dHA_ = HA_first_ - HA_second_

        # Error b/w known Hilbert and the `first` Hilbert
        E_ = y - HA_first_
        
        # Find relation b/w error and difference b/w Hilbert transforms
        self.regressor.fit(dHA_, E_)
        self.transfer_matrix_ = 1*self.regressor.coef_
        self.transfer_matrix_intercept_ = 1*self.regressor.intercept_
        
        # Return the transformer (per sklearn requirements for `fit` methods)
        return self

    def transform(self, X):
        """ Applies ML-Hilbert transform to data

        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        X_transformed : array, shape (n_samples, n_features)
            The array containing the hilbert transform of the values
            in ``X``.
        """
        # Check is fit had been called
        check_is_fitted(self, 'n_features_')

        # Input validation
        X = check_array(X, accept_sparse=False, dtype='float', ensure_2d=False)

        # Check that the input is of the same shape as the one passed
        # during fit.
        if X.ndim == 2:
            if X.shape[1] != self.n_features_:
                raise ValueError('Shape of input is different from what was seen'
                                 'in `fit`')
        elif X.ndim == 1:
            if X.size != self.n_features_:
                raise ValueError('Shape of input is different from what was seen'
                                 'in `fit`')
            
        HB_first_ = hilbertfft(X, pad_factor=self.pad_first)
        HB_second_ = hilbertfft(X, pad_factor=self.pad_second)
        dHB_ = HB_first_ - HB_second_
        self.baseline_ = -1 * np.dot(dHB_, self.transfer_matrix_.T)-self.transfer_matrix_intercept_
                
        return HB_first_ - self.baseline_
    