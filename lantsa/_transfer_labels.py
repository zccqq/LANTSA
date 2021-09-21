# -*- coding: utf-8 -*-

from typing import Optional, Union
from anndata import AnnData

import numpy as np
import pandas as pd

from scipy.sparse import issparse
from sklearn.neighbors import KNeighborsClassifier
from natsort import natsorted

from ._subspace_analysis import _Metric, _Metric_fn


def transfer_labels(
    adata: AnnData,
    adata_learning: AnnData,
    groups: str,
    n_neighbors: int = 10,
    metric: Union[_Metric, _Metric_fn] = 'cosine',
    n_discriminant: Optional[int] = None,
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
    metric
        Distance metric for :class:`~sklearn.neighbors.KNeighborsClassifier`.
    n_discriminant
        Number of discriminant vectors to use.
        By default uses all stored discriminant vectors.
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
    .obsm['transfer_proba'] : :class:`~pandas.DataFrame`
        DataFrame storing the transferred probability for each cell for each group.
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
    
    if n_discriminant is None:
        n_discriminant = adata_learning.uns[subspace_key]['discriminant'].shape[1]
    
    if groups not in adata_learning.obs.keys():
        raise KeyError(
            f'Could not find key {groups} in adata_learning.obs.columns.'
        )
    
    adata_use = adata[:, adata_learning.uns[subspace_key]['var_names_use']]
    
    adata_learning_use = adata_learning[:, adata_learning.uns[subspace_key]['var_names_use']]
    
    
    clusters_learning = adata_learning_use.obs[groups]
    P = adata_learning.uns[subspace_key]['discriminant'][:, :n_discriminant]
    
    X_learning = adata_learning_use.X.toarray() if issparse(adata_learning_use.X) else adata_learning_use.X
    X_prediction = adata_use[~adata_use.obs['is_learning'], :].X.toarray() if issparse(adata_use.X) else adata_use[~adata_use.obs['is_learning'], :].X
    
    neigh = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)
    neigh.fit(np.matmul(X_learning, P), clusters_learning)
    PX = np.matmul(X_prediction, P)
    clusters_prediction = neigh.predict(PX)
    
    clusters = pd.Series(index=adata_use.obs_names, dtype=int)
    clusters.loc[adata_use.obs['is_learning']] = clusters_learning
    clusters.loc[~adata_use.obs['is_learning']] = clusters_prediction
    
    clusters_proba_columns = np.sort(adata_learning.obs[groups].unique())
    clusters_proba_prediction = pd.DataFrame(neigh.predict_proba(PX),
                                             index=adata.obs_names[~adata.obs['is_learning']],
                                             columns=clusters_proba_columns)
    clusters_proba = pd.DataFrame(0., index=adata.obs_names, columns=clusters_proba_columns)
    clusters_proba.loc[adata.obs['is_learning'], :] = pd.get_dummies(clusters_learning, dtype=float)
    clusters_proba.loc[~adata.obs['is_learning'], :] = clusters_proba_prediction
    
    
    adata.uns['transfer_labels'] = {}
    
    transfer_dict = adata.uns['transfer_labels']
    
    transfer_dict['params'] = {}
    transfer_dict['params']['groups'] = groups
    transfer_dict['params']['n_neighbors'] = n_neighbors
    transfer_dict['params']['metric'] = str(metric)
    transfer_dict['params']['n_discriminant'] = n_discriminant
    transfer_dict['params']['subspace_key'] = subspace_key
    
    adata.obsm['transfer_proba'] = clusters_proba
    
    adata.obs[groups] = pd.Categorical(
        values=clusters.astype('U'),
        categories=natsorted(map(str, np.unique(clusters))),
    )
    
    return adata if copy else None



















