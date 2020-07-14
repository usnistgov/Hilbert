"""
Module for discrete Hilbert transform that uses wavelets at its core.
"""

import numpy as np

def _haar_matrix(N):
    """Generate a Haar matrix and it's analytic Hilbert transformed version

    Parameters
    ----------
    N : int
        Size of axis=-1 dimension of matrix (will be number of sample point in signal)

    Returns
    -------
    tuple (ndarray, ndarray)
        Square Haar matrix, square Hilbert of Haar matrix

    References
    -----------
    -   http://fourier.eng.hmc.edu/e161/lectures/Haar/index.html

    """

    if np.log2(N) % 1 != 0:
        raise ValueError('N must be a power of 2')
    haar_mat = np.zeros((N,N))
    hilb_haar_mat = np.zeros((N,N))
    p_vec = np.zeros((N))
    q_vec = np.zeros((N))
    t = np.linspace(0,1,N)
    for k in range(N):
        
        if k == 0:
            p = 0
            q = 0
            haar_mat[k,:] = 1
            hilb_haar_mat[k,:] = 0
        else:
            p = int(np.floor(np.log2(k)))
            q = k - 2**p + 1
            wavelet_ctr = (q-0.5)/2**p  # Center of wavelet
            pos_lim = (t >= (q-1)/2**p) & (t < wavelet_ctr)
            neg_lim = (t >= wavelet_ctr) & (t < q/2**p)
           
            haar_mat[k,pos_lim] = 2**(p/2)
            haar_mat[k,neg_lim] = -2**(p/2)
            haar_mat[k,:] /= np.sqrt(N)
            
            wavelet_width = 0.5/2**p
            pos_ctr = wavelet_ctr - wavelet_width/2
            neg_ctr = wavelet_ctr + wavelet_width/2           
            
                       
            n0 = (t - pos_ctr + wavelet_width/2)
            
            # At discontinuities, use mean
            if np.sum(n0==0)>0:
                locs = np.where(n0==0)[0]
                for l in locs:
                    if l == 0:
                        n0[l] = n0[l+1]
                    elif l == N-1:
                        n0[l] = n0[l-1]
                    else:
                        n0[l] = 0.5*(n0[l+1] + n0[l-1])
                
            d0 = (t - pos_ctr - wavelet_width/2)
            
            n1 = (t - neg_ctr - wavelet_width/2)
            
            # At discontinuities, use mean
            if np.sum(n1==0)>0:
                locs = np.where(n1==0)[0]
                for l in locs:
                    if l == 0:
                        n1[l] = n1[l+1]
                    elif l == N-1:
                        n1[l] = n1[l-1]
                    else:
                        n1[l] = 0.5*(n1[l+1] + n1[l-1])
                        
            d1 = (t - neg_ctr + wavelet_width/2)
            
            hilb_haar_mat[k,:] = (1/np.pi)*np.log(np.abs(n0/d0)*np.abs(n1/d1))
            hilb_haar_mat[k,:] *= 2**(p/2)
            hilb_haar_mat[k,:] /= np.sqrt(N)
        
        p_vec[k] = p
        q_vec[k] = q
        
    return haar_mat, hilb_haar_mat

def hilbert_haar(x, pad_type='reflect', axis=-1):
    """Use the Haar wavelet decomposition to approximate the Hilbert transform

    Parameters
    ----------
    x : array-like
        Input signal such that y[n] = H{x[n]} will ultimately be returned
    pad_type : str
        Type of signal padding to perform, by default 'reflect'. See numpy.pad
    axis : int, optional
        For nd-arrays, axis to perform over, by default -1

    Notes
    -----
    -   This implementation is based on the concepts in Ref 1, but does not 
        implement that paper's algorithm exactly.
    -   This method requires the sample be a power-of-2; thus, some padding
        is performed, if necessary.

    Refernces
    ----------
    1.   C. Zhou, L. Yang, Y. Liu, and Z. Yang, "A novel method for 
         computing the Hilbert transform with Haar multiresolution 
         approximation," J. Comput. Appl. Math. 223(2), 585–597 (2009).

    """    

    if axis != -1:
        raise NotImplementedError('Only axis=-1 is currently supported')

    pad_total = (int(2**np.ceil(np.log2(x.shape[axis]))) - 
                 x.shape[axis])
    if pad_total == 0:
        pad_for_2 = False
        x_pad = x
    else:
        pad_for_2 = True
        pad_first = int(np.ceil(pad_total/2))
        pad_last = pad_total - pad_first
        if x.ndim == 1:
            x_pad = np.pad(x, [pad_first, pad_last], pad_type)
        else:
            x_pad = np.pad(x, [[0,0],[pad_first, pad_last]], pad_type)

    hm, hilb_hm = _haar_matrix(x_pad.shape[axis])
    if x.ndim == 1:
        coeffs = np.dot(hm, x_pad)
        out = np.dot(hilb_hm.T,coeffs)
    elif x.ndim == 2:
        coeffs = np.dot(hm, x_pad.T)
        out = np.dot(hilb_hm.T,coeffs).T
    else:
        raise ValueError('Only 1- and 2D x is currently supported')

    # Remove padding used to get it to a power of 2 in len
    if pad_for_2:
        if out.ndim == 2:
            out = out[:,pad_first:-pad_last]
        else:  # Must be 1D
            out = out[pad_first:-pad_last]

    return out
