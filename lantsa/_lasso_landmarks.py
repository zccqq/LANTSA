# -*- coding: utf-8 -*-

import numpy as np
from numba import jit


@jit(nopython=True)
def cyclic_descent_loop(X, g, w, rand_idx, shrinkFactor, threshold):
    
    for j in rand_idx:
        wjold = w[j]
        
        gj = g + w[j] * X[:, j]
        wj = np.sum(X[:, j] * gj)
        
        w[j] = np.sign(wj) * max(np.abs(wj) - threshold, 0) / shrinkFactor[j]
        
        g = g - X[:, j] * (w[j] - wjold)
    
    return w, g


def cyclic_descent(X, r, w, rand_idx, active, norms, threshold, rng):
    
    num_rand = rand_idx.shape[0]
    
    if num_rand == 0:
        return w, r
    else:
        rand_idx = rand_idx[rng.permutation(num_rand)]
    
    w, r = cyclic_descent_loop(X, r, w, rand_idx, norms, threshold)
    
    return w, r


def lasso_fit(diag_col, X, w, r, threshold, reltol, norms, rng):
    
    active = w != 0
    
    wold = w
    
    while True:
        
        rand_idx = np.setdiff1d(np.flatnonzero(active), diag_col)
        w, r = cyclic_descent(X, r, w, rand_idx, active, norms, threshold, rng)
        active = w != 0
        
        if np.linalg.norm((w - wold) / (1.0 + np.abs(wold)), np.inf) < reltol:
            
            wold = w
            
            potentially_active = np.abs(np.matmul(r, X)) > threshold
            
            if np.any(potentially_active):
                
                new_active = active | potentially_active
                
                rand_idx = np.setdiff1d(np.flatnonzero(new_active), diag_col)
                w, r = cyclic_descent(X, r, w, rand_idx, new_active, norms, threshold, rng)
                new_active = w != 0
            else:
                new_active = active
            
            if np.all(new_active == active):
                break
            else:
                active = new_active
            
            if np.linalg.norm((w - wold) / (1.0 + np.abs(wold)), np.inf) < reltol:
                break
        
        wold = w
        
    return w, r


def lasso_landmarks(X, S, stats, idx, rng):
    
    norms = stats['normsSt']
    XS = stats['XSt']
    
    w_size = S.shape[0]
    w_vec = stats['W'][:w_size, idx]
    
    diag_col = np.flatnonzero(S==idx)
    
    w_vec, r = lasso_fit(diag_col, XS, w_vec, stats['R'][:, idx],
                         stats['Lambda'], stats['reltol'], norms, rng)
    
    stats['R'][:, idx] = r
    stats['W'][:w_size, idx] = w_vec
    
    return stats



















