"""
Module for calculating various loss metrics common for regression methods.
"""

import numpy as np

def mse(y_true, y_pred, mean_centered=False, axis=-1):
    """Calculate the mean-squared error (MSE) with or without mean-centering
    the data first.

    Parameters
    ----------
    y_true : array-like
        The ground truth
    y_pred : array-like
        Estimated/predicted/measured values
    mean_centered : bool, optional
        Mean-center y_true and y_pred before calculation, by default False
    axis : int, optional
        Which axis contains the features, by default -1

    Returns
    -------
    array-like
        A non-negative floating point value (the best value is 0.0), or an 
        array of floating point values, one for each individual sample.

    """    
    
    return rss(y_true, y_pred, mean_centered=mean_centered)/y_true.shape[axis]
    

def rss(y_true, y_pred, mean_centered=False, axis=-1):
    """Calculate the residual sum-of-squares (RSS) with or without 
    mean-centering the data first.

    Parameters
    ----------
    y_true : array-like
        The ground truth
    y_pred : array-like
        Estimated/predicted/measured values
    mean_centered : bool, optional
        Mean-center y_true and y_pred before calculation, by default False
    axis : int, optional
        Which axis contains the features, by default -1

    Returns
    -------
    array-like
        A non-negative floating point value (the best value is 0.0), or an 
        array of floating point values, one for each individual sample.

    """    
    if not mean_centered:
        return np.sum((y_true - y_pred)**2, axis=axis)
    if mean_centered:
        return np.sum(((y_true - y_true.mean(axis=axis, keepdims=True)) - 
                       (y_pred - y_pred.mean(axis=axis, keepdims=True)))**2, 
                      axis=axis)

def mlhilb_scorer(estimator, X, y_true, *args, **kwargs):
    """A customized scorer for mlhilb"""
    
    # TODO: turn this into a class so weights can be altered
    weight = {'mean':0.0, 'median':1.0, 'min':1.0, 'max':1.0, 'std':1.0}
    
    y_pred = estimator.predict(X)
    mse_out = mse(y_true, y_pred, mean_centered=True)
    score = 0.0
    score += weight['mean']*np.log10(np.mean(mse_out))
    score += weight['median']*np.log10(np.median(mse_out))
    score += weight['max']*np.log10(np.max(mse_out) - np.median(mse_out))
    score += weight['std']*np.log10(np.std(mse_out))
    score += weight['min']*np.log10(np.median(mse_out) - np.min(mse_out))
    
    # For scorers, bigger is BETTER; thus, we have to negate our score
    # b/c for it, smaller is better
    return -score