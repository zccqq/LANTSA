# -*- coding: utf-8 -*-

from typing import Optional
from anndata import AnnData

import numpy as np
import pandas as pd

from scipy.sparse import issparse
from sklearn.neighbors import KNeighborsClassifier
from natsort import natsorted


def transfer_labels(
    adata: AnnData,
    adata_learning: AnnData,
    groups: str,
    n_neighbors: int = 10,
    discriminant_dim: Optional[int] = None,
    subspace_key: Optional[str] = None,
    copy: bool = False,
) -> Optional[AnnData]:
    '''
    Transfer labels from learning samples to other samples [Shi21]_.
    
    Parameters
    ----------
    adata
        Annotated data matrix.
    adata_learning
        Annotated data matrix of learning samples with given labels.
    groups
        The key of the observations grouping to transfer.
    n_neighbors
        Number of neighbors for :class:`~sklearn.neighbors.KNeighborsClassifier`.
    discriminant_dim
        Dimension of discriminant matrix to use.
        Defaults to the stored dimension of discriminant matrix.
    subspace_key
        If not specified, it looks `adata_learning.uns['subspace_analysis']` for subspace settings
        and `adata_learning.varm['discriminant']` for discriminant matrix
        (default storage places for :func:`~lantsa.subspace_analysis`).
        If specified, it looks `adata_learning.uns[subspace_key]` for subspace settings
        and `adata_learning.varm[adata_learning.uns[subspace_key]['discriminant_key']]` for discriminant matrix.
    copy
        Return a copy instead of writing to ``adata``.
    
    Returns
    -------
    Depending on ``copy``, returns or updates ``adata`` with the following fields.
    
    .obs[``groups``]
        Array of dim (number of samples) that stores the transferred labels for each cell.
    .uns['transfer_labels']['probability'] : :class:`~numpy.recarray`
        Structured array to be indexed by group id storing the transferred probability
        for each cell for each group.
    '''
    
    adata = adata.copy() if copy else adata
    
    if 'is_learning' not in adata.obs.keys():
        raise ValueError(
            'Could not find adata.obs[\'is_learning\'].'
            'Please run select_learning_samples first.'
        )
    
    if subspace_key is None:
        subspace_key = 'subspace_analysis'
    
    if subspace_key not in adata_learning.uns.keys():
        raise ValueError(
            f'Could not find adata_learning.uns[\'{subspace_key}\'].'
            'Please run subspace_analysis first.'
        )
    
    discriminant_key = adata_learning.uns[subspace_key]['discriminant_key']
    
    if discriminant_dim is None:
        discriminant_dim = adata_learning.varm[discriminant_key].shape[1]
    
    if groups not in adata_learning.obs.keys():
        raise KeyError(
            f'Could not find key {groups} in adata_learning.obs.columns.'
        )
    
    use_highly_variable = adata_learning.uns[subspace_key]['params']['use_highly_variable']
    
    if use_highly_variable is True and 'highly_variable' not in adata.var.keys():
        raise ValueError(
            'Did not find adata.var[\'highly_variable\']. '
            'Either your data already only consists of highly-variable genes '
            'or consider running `pp.highly_variable_genes` first.'
        )
    
    if use_highly_variable is True and 'highly_variable' not in adata_learning.var.keys():
        raise ValueError(
            'Did not find adata_learning.var[\'highly_variable\']. '
            'Either your data already only consists of highly-variable genes '
            'or consider running `pp.highly_variable_genes` first.'
        )
    
    adata_use = (
        adata[:, adata.var['highly_variable']] if use_highly_variable else adata
    )
    
    adata_learning_use = (
        adata_learning[:, adata_learning.var['highly_variable']] if use_highly_variable else adata_learning
    )
    
    
    clusters_learning = adata_learning_use.obs[groups].to_numpy()
    P = adata_learning_use.varm[discriminant_key][:, :discriminant_dim]
    
    X_learning = adata_learning_use.X.toarray() if issparse(adata_learning_use.X) else adata_learning_use.X
    X_prediction = adata_use[~adata_use.obs['is_learning'], :].X.toarray() if issparse(adata_use.X) else adata_use[~adata_use.obs['is_learning'], :].X
    
    neigh = KNeighborsClassifier(n_neighbors=n_neighbors, metric='euclidean')
    neigh.fit(np.matmul(X_learning, P), clusters_learning)
    clusters_prediction = neigh.predict(np.matmul(X_prediction, P))
    
    clusters = pd.Series(index=adata_use.obs_names, dtype=int)
    clusters.loc[adata_use.obs['is_learning']] = clusters_learning
    clusters.loc[~adata_use.obs['is_learning']] = clusters_prediction
    
    clusters_proba_learning = pd.DataFrame(neigh.predict_proba(np.matmul(X_learning, P)),
                                           index=adata.obs_names[adata.obs['is_learning']],
                                           columns=np.unique(adata_learning.obs[groups]))
    clusters_proba_prediction = pd.DataFrame(neigh.predict_proba(np.matmul(X_prediction, P)),
                                             index=adata.obs_names[~adata.obs['is_learning']],
                                             columns=np.unique(adata_learning.obs[groups]))
    clusters_proba = pd.DataFrame(0., index=adata.obs_names, columns=np.unique(adata_learning.obs[groups]))
    clusters_proba.loc[adata.obs['is_learning'], :] = clusters_proba_learning
    clusters_proba.loc[~adata.obs['is_learning'], :] = clusters_proba_prediction
    
    
    adata.uns['transfer_labels'] = {}
    
    transfer_dict = adata.uns['transfer_labels']
    
    transfer_dict['params'] = {}
    transfer_dict['params']['groups'] = groups
    transfer_dict['params']['n_neighbors'] = n_neighbors
    transfer_dict['params']['discriminant_dim'] = discriminant_dim
    transfer_dict['params']['subspace_key'] = subspace_key
    
    transfer_dict['probability'] = clusters_proba.to_records(index=None)
    
    adata.obs[groups] = pd.Categorical(
        values=clusters.astype('U'),
        categories=natsorted(np.unique(clusters)),
    )
    
    return adata if copy else None



















