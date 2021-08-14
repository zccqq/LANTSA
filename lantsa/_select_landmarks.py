# -*- coding: utf-8 -*-

from typing import Optional
from anndata import AnnData

import numpy as np
import pandas as pd

from scipy.sparse import issparse
from sklearn.preprocessing import normalize
from tqdm import trange

from ._lasso_landmarks import lasso_landmarks


def select_landmarks(
    adata: AnnData,
    n_landmarks: int,
    Lambda: float = 0.5,
    reltol: float = 1e-3,
    use_highly_variable: Optional[bool] = None,
    random_state: int = 0,
    copy: bool = False,
) -> Optional[AnnData]:
    '''
    Select landmark samples [Matsushima19]_.
    
    Parameters
    ----------
    adata
        Annotated data matrix.
    n_landmarks
        Number of landmarks to be selected.
    Lambda
        Hyperparameter for sparsity regularization.
    reltol
        Relative tolerance in optimization.
    use_highly_variable
        Whether to use highly variable genes only, stored in `adata.var['highly_variable']`.
        By default uses them if they have been determined beforehand.
    random_state
        Change to use different initial states for the optimization.
    copy
        Return a copy instead of writing to ``adata``.
    
    Returns
    -------
    Depending on ``copy``, returns or updates ``adata`` with the following fields.
    
    .obs['is_landmark']
        Boolean indicator of landmark samples.
    '''
    
    adata = adata.copy() if copy else adata
    
    if adata.n_obs < n_landmarks:
        n_landmarks = adata.n_obs
    
    if use_highly_variable is True and 'highly_variable' not in adata.var.keys():
        raise ValueError(
            'Did not find adata.var[\'highly_variable\']. '
            'Either your data already only consists of highly-variable genes '
            'or consider running `pp.highly_variable_genes` first.'
        )
    
    if use_highly_variable is None:
        use_highly_variable = True if 'highly_variable' in adata.var.keys() else False
    
    adata_use = (
        adata[:, adata.var['highly_variable']] if use_highly_variable else adata
    )
    
    
    X = adata_use.X.toarray().T if issparse(adata_use.X) else adata_use.X.T
    X = normalize(X.astype(np.float64), axis=0)
    
    rng = np.random.RandomState(seed=random_state)
    
    n = adata.n_obs
    
    max_t = n_landmarks * 2
    num_I_t = 1
    num_rand_idx = n_landmarks * 2 * num_I_t
    if n >= num_rand_idx:
        rand_idx = rng.choice(np.arange(n), num_rand_idx)
    else:
        rand_idx = []
        for it in range(num_rand_idx // n + 1):
            rand_idx.append(rng.permutation(n))
        rand_idx = np.concatenate(rand_idx)
    
    S_t = np.array([], dtype=int)
    
    norms = np.sum(X**2, axis=0)
    
    stats = {
        'reltol': reltol,
        'Lambda': Lambda,
        'normsSt': np.array([]),
        'XSt': np.ndarray((X.shape[0], 0)),
        'W': np.zeros((n_landmarks, n)),
        'R': X,
    }
    
    # select landmark samples
    pbar = trange(max_t)
    
    for t in pbar:
        
        I_t = rand_idx[(t*num_I_t):((t+1)*num_I_t)]
        
        if S_t.shape[0] != 0:
            for i in I_t:
                stats = lasso_landmarks(X, S_t, stats, i, rng)
        
        candidate = np.setdiff1d(np.arange(n), S_t)
        
        grad_L = np.matmul(stats['R'][:, I_t].T, X[:, candidate])
        grad = np.zeros((num_I_t, n))
        grad[:, candidate] = np.minimum(grad_L+Lambda, np.maximum(0, grad_L-Lambda))
        
        for j in range(num_I_t):
            grad[j, I_t[j]] = 0
        
        grad2sum = np.sum(grad**2, axis=0)
        idx = np.argsort(grad2sum)[::-1]
        max_vals = grad2sum[idx]
        
        if max_vals[0] != 0:
            dSt = idx[0]
            S_t = np.append(S_t, dSt)
            pbar.set_postfix_str(f'seleted landmarks: {S_t.shape[0]}')
            stats['normsSt'] = np.append(stats['normsSt'], norms[dSt])
            stats['XSt'] = np.concatenate((stats['XSt'], X[:, dSt][:, None]), axis=1)
            if S_t.shape[0] == n_landmarks:
                break
    
    landmarks = pd.Series(False, index=adata.obs_names)
    landmarks.iloc[S_t] = True
    
    
    adata.uns['select_landmarks'] = {}
    
    landmarks_dict = adata.uns['select_landmarks']
    
    landmarks_dict['params'] = {}
    landmarks_dict['params']['n_landmarks'] = np.count_nonzero(landmarks)
    landmarks_dict['params']['Lambda'] = Lambda
    landmarks_dict['params']['reltol'] = reltol
    landmarks_dict['params']['use_highly_variable'] = use_highly_variable
    landmarks_dict['params']['random_state'] = random_state
    
    adata.obs['is_landmark'] = landmarks
    
    return adata if copy else None



















