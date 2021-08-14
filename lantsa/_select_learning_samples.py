# -*- coding: utf-8 -*-

from typing import Literal, Optional
from anndata import AnnData

import numpy as np
import pandas as pd

from scipy.sparse import issparse
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import euclidean_distances

_Method = Literal['MiniBatchKMeans', 'Random']


def select_learning_samples(
    adata: AnnData,
    n_learning: int,
    method: _Method = 'MiniBatchKMeans',
    n_components: Optional[int] = None,
    use_highly_variable: Optional[bool] = None,
    random_state: int = 0,
    copy: bool = False,
) -> Optional[AnnData]:
    '''
    Select learning samples.
    
    Parameters
    ----------
    adata
        Annotated data matrix.
    n_learning
        Number of learning samples to be selected.
    method
        Method to use for learning sample selection.
        
        * ``'MiniBatchKMeans'``
            Use `scikit-learn` :class:`~sklearn.cluster.MiniBatchKMeans`
            to select learning samples.
        * ``'Random'``
            Randomly choose learning samples.
        
    n_components
        Number of principal components to use as the input of
        :class:`~sklearn.cluster.MiniBatchKMeans`.
        Defaults to the number of features devided by 50 with a minimum number of 20.
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
    
    .obs['is_learning']
        Boolean indicator of learning samples.
    '''
    
    adata = adata.copy() if copy else adata
    
    if method not in ['MiniBatchKMeans', 'Random']:
        raise ValueError('method needs to be \'MiniBatchKMeans\' or \'Random\'')
    
    if n_learning > adata.n_obs:
        raise ValueError(f"n_learning {n_learning} needs to be less than n_obs {adata.shape[0]}")
    
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
    
    
    if method == 'MiniBatchKMeans':
        
        X = adata_use.X.toarray() if issparse(adata_use.X) else adata_use.X
        
        if n_components is None:
            n_components = min(max(adata_use.n_vars // 50, 20), adata.n_obs-1)
        if n_components < adata_use.n_vars:
            X = PCA(n_components=n_components, svd_solver='arpack', random_state=random_state).fit_transform(X)
        else:
            n_components = None
        
        kmeans = MiniBatchKMeans(n_clusters=n_learning, random_state=random_state).fit(X)
        learning_samples = pd.Series(False, index=adata_use.obs_names)
        learning_samples.iloc[np.unique(np.argmin(euclidean_distances(X, kmeans.cluster_centers_), axis=0))] = True
    
    elif method == 'Random':
        
        n_components = None
        
        rng = np.random.RandomState(seed=random_state)
        learning_samples = pd.Series(False, index=adata_use.obs_names)
        learning_samples.iloc[rng.choice(np.arange(adata_use.shape[0]), size=n_learning, replace=False)] = True
    
    
    adata.uns['select_training_samples'] = {}
    
    learn_dict = adata.uns['select_training_samples']
    
    learn_dict['params'] = {}
    learn_dict['params']['n_learning'] = np.count_nonzero(learning_samples)
    learn_dict['params']['method'] = method
    learn_dict['params']['n_components'] = n_components
    learn_dict['params']['use_highly_variable'] = use_highly_variable
    learn_dict['params']['random_state'] = random_state
    
    adata.obs['is_learning'] = learning_samples
    
    return adata if copy else None



















