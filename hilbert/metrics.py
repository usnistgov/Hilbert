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