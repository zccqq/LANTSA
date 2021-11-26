# -*- coding: utf-8 -*-

from typing import Callable, Optional, Union
from anndata import AnnData
from ._compat import Literal

import torch
import numpy as np

from scipy.sparse import issparse, spmatrix, csr_matrix

from ._initialize import initialize_Z
from ._solve import solve_P, solve_Z


_Metric = Literal['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan',
                  'braycurtis', 'canberra', 'chebyshev', 'correlation',
                  'dice', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis',
                  'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean',
                  'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']
_Metric_fn = Callable[[np.ndarray, np.ndarray], float]


def subspace_analysis(
    adata: AnnData,
    n_neighbors: Optional[int] = None,
    n_pcs: Optional[int] = None,
    metric: Union[_Metric, _Metric_fn] = 'euclidean',
    Lambda: float = 0.1,
    n_iterations: int = 500,
    n_discriminant: Optional[int] = None,
    Z_mask: Optional[Union[np.ndarray, spmatrix]] = None,
    use_landmarks: Optional[bool] = None,
    use_highly_variable: Optional[bool] = None,
    device: Optional[str] = None,
    random_state: int = 0,
    key_added: Optional[str] = None,
    copy: bool = False,
) -> Optional[AnnData]:
    '''
    Subspace analysis for representation learning [Wong17]_ [Shi21]_.
    
    Parameters
    ----------
    adata
        Annotated data matrix.
    n_neighbors
        Number of neighbors for constructing a :class:`~sklearn.neighbors.NearestNeighbors`
        graph as hard constraints in subspace analysis.
        Defaults to the number of samples devided by 10 with a minimum number of 20.
    n_pcs
        Number of principal components to use for constructing a
        :class:`~sklearn.neighbors.NearestNeighbors` graph as hard
        constraints in subspace analysis.
        Defaults to the number of features devided by 50 with a minimum number of 20.
    metric
        Distance metric to use for constructing a
        :class:`~sklearn.neighbors.NearestNeighbors` graph as hard
        constraints in subspace analysis.
    Lambda
        Hyperparameter for sparsity regularization.
    n_iterations
        Number of iterations for the optimization.
    n_discriminant
        Number of discriminant vectors to store for label transfer.
        By default stores all discriminant vectors.
    Z_mask
        Customized sample-by-sample graph of shape (`adata.n_obs`, `adata.n_obs`)
        as hard constraints in subspace analysis.
        By default computes a :class:`~sklearn.neighbors.NearestNeighbors` graph.
    use_landmarks
        Whether to use landmarks as hard constraints in subspace analysis,
        stored in `adata.obs['is_landmarks']`.
        By default uses them if they have been selected beforehand.
    use_highly_variable
        Whether to use highly variable genes only, stored in `adata.var['highly_variable']`.
        By default uses them if they have been determined beforehand.
    device
        The desired device for `PyTorch` computation. By default uses cuda if cuda is avaliable
        cpu otherwise.
    random_state
        Change to use different initial states for the optimization.
    key_added
        If not specified, the subspace data is stored in `adata.uns['subspace_analysis']`,
        representation is stored in `adata.obsp['representation']` and
        discriminant matrix is stored in `adata.varm['discriminant']`.
        If specified, the subspace data is added to `adata.uns[key_added]`,
        representation is stored in `adata.obsp[key_added+'_representation']` and
        discriminant matrix is stored in `adata.varm[key_added+'_discriminant']`.
    copy
        Return a copy instead of writing to ``adata``.
    
    Returns
    -------
    Depending on ``copy``, returns or updates ``adata`` with the following fields.
    
    See ``key_added`` parameter description for the storage path of
    representation and discriminant.
    
    representation : :class:`~scipy.sparse.csr_matrix` (.obsp)
        The subspace representation of samples.
    discriminant : :class:`~numpy.ndarray` (.uns[``key_added``])
        The discriminant vectors for label transfer.
    '''
    
    adata = adata.copy() if copy else adata
    
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    
    device = torch.device(device)
    
    if use_landmarks is None:
        use_landmarks = True if 'is_landmark' in adata.obs else False
    
    landmarks = np.flatnonzero(adata.obs['is_landmark']) if use_landmarks else np.arange(adata.n_obs)
    
    if n_neighbors is None:
        n_neighbors = min(max(adata.n_obs // 10, 20), adata.n_obs)
    
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
    
    if n_discriminant is None:
        n_discriminant = adata_use.n_vars
    
    
    X = adata_use.X.toarray().T if issparse(adata_use.X) else adata_use.X.T
    m, n = X.shape
    if Z_mask is None:
        Z_mask, n_pcs = initialize_Z(X, landmarks, n_neighbors, n_pcs, random_state)
    else:
        if len(Z_mask.shape) != 2 or Z_mask.shape[0] != X.shape[1] or Z_mask.shape[1] != landmarks.shape[0]:
            raise ValueError(
                'The shape of Z_mask needs to be (adata.n_obs, adata.n_obs) '
                f'({adata.n_obs}, {adata.n_obs}), but given {Z_mask.shape}.'
            )
        Z_mask = Z_mask.toarray() if issparse(Z_mask) else Z_mask
        Z_mask[Z_mask != 0] = 1
        n_pcs = None
    
    X = torch.Tensor(X).type(torch.float32).to(device)
    Z_mask = torch.Tensor(Z_mask).type(torch.float32).to(device)
    
    Z, E = solve_Z(X, landmarks, Lambda, Z_mask, n_iterations, device)  
    
    P = solve_P(X, landmarks, Z)
    
    Z = Z.cpu().numpy()
    P = P.cpu().numpy()
    
    eps = 1e-14
    Z[Z < eps] = 0
    
    
    if key_added is None:
        key_added = 'subspace_analysis'
        conns_key = 'representation'
        dists_key = 'representation'
    else:
        conns_key = key_added + '_representation'
        dists_key = key_added + '_representation'
    
    adata.uns[key_added] = {}
    
    subspace_dict = adata.uns[key_added]
    
    subspace_dict['connectivities_key'] = conns_key
    subspace_dict['distances_key'] = dists_key
    subspace_dict['discriminant'] = P[:, :n_discriminant]
    subspace_dict['var_names_use'] = adata_use.var_names.to_numpy()
    
    subspace_dict['params'] = {}
    subspace_dict['params']['n_neighbors'] = np.count_nonzero(Z) // Z.shape[0]
    subspace_dict['params']['n_pcs'] = n_pcs
    subspace_dict['params']['metric'] = str(metric)
    subspace_dict['params']['Lambda'] = Lambda
    subspace_dict['params']['n_iterations'] = n_iterations
    subspace_dict['params']['n_discriminant'] = n_discriminant
    subspace_dict['params']['use_landmarks'] = use_landmarks
    subspace_dict['params']['use_highly_variable'] = use_highly_variable
    subspace_dict['params']['random_state'] = random_state
    subspace_dict['params']['method'] = 'umap'
    
    row_idx = np.repeat(np.arange(n), landmarks.shape[0])
    col_idx = np.repeat(landmarks[None,:], n, axis=0).reshape(-1)
    adata.obsp[conns_key] = csr_matrix((Z.reshape(-1), (row_idx, col_idx)), shape=(n, n))
    
    return adata if copy else None



















