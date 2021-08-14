# -*- coding: utf-8 -*-

from typing import Optional
from anndata import AnnData

import torch
import numpy as np

from scipy.sparse import issparse, csr_matrix

from ._initialize import initialize_Z
from ._solve import solve_P, solve_Z


def subspace_analysis(
    adata: AnnData,
    Lambda: float = 0.1,
    n_neighbors: Optional[int] = None,
    n_iterations: int = 500,
    depth: int = 1,
    discriminant_dim: Optional[int] = None,
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
    Lambda
        Hyperparameter for sparsity regularization.
    n_neighbors
        Number of neighbors for constructing a :class:`~sklearn.neighbors.NearestNeighbors`
        graph to hard constrain subspace analysis.
        Defaults to the number of samples devided by 10 with a minimum number of 20.
    n_iterations
        Number of iterations for the optimization.
    depth
        How many times to loop subspace analysis.
    discriminant_dim
        Dimension of discriminant matrix to store for label transfer.
        Defaults to the number of features devided by 10 with a minimum number of 50.
    use_landmarks
        Whether to use landmarks to constrain subspace analysis, stored in `adata.obs['is_landmarks']`.
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
    
    representation : :class:`~scipy.sparse.csr_matrix` (`.obsp`)
        The subspace representation of samples.
    discriminant : :class:`~numpy.ndarray` (`.varm`)
        The discriminant matrix for label transfer.
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
    
    if discriminant_dim is None:
        discriminant_dim = min(max(adata_use.n_vars // 10, 50), adata_use.n_vars)
    
    
    X = adata_use.X.toarray().T if issparse(adata_use.X) else adata_use.X.T
    Zi = initialize_Z(X, n_neighbors, landmarks, random_state)
    
    X = torch.Tensor(X).type(torch.float32).to(device)
    Zi = torch.Tensor(Zi).type(torch.float32).to(device)
    
    obj_final = float('inf')
    P = torch.eye(adata_use.n_vars).to(device)
    
    for dep in range(depth):
        
        # fixing P, solve Z
        X_input = torch.matmul(P.T, X)
        Z, E = solve_Z(X_input, Lambda, Zi, n_iterations, device)  
        
        # fixing Z, solve P
        P = solve_P(X, Z, P)
        
        # check convergence
        M = torch.matmul(P.T, X - torch.matmul(X, Z))
        obj = torch.sum(torch.norm(M, 2, dim=0))
        
        if torch.abs(obj - obj_final) / obj < 0.001:
            break
        else:
            obj_final = obj
    
    Z = Z.cpu().numpy()
    P = P.cpu().numpy()
    
    eps = 1e-14
    Z[Z < eps] = 0
    
    
    if key_added is None:
        key_added = 'subspace_analysis'
        conns_key = 'representation'
        dists_key = 'representation'
        discriminant_key = 'discriminant'
    else:
        conns_key = key_added + '_representation'
        dists_key = key_added + '_representation'
        discriminant_key = key_added + '_discriminant'
    
    adata.uns[key_added] = {}
    
    subspace_dict = adata.uns[key_added]
    
    subspace_dict['connectivities_key'] = conns_key
    subspace_dict['distances_key'] = dists_key
    subspace_dict['discriminant_key'] = discriminant_key
    
    subspace_dict['params'] = {}
    subspace_dict['params']['n_neighbors'] = np.count_nonzero(Z) // Z.shape[0]
    subspace_dict['params']['Lambda'] = Lambda
    subspace_dict['params']['n_iterations'] = n_iterations
    subspace_dict['params']['depth'] = depth
    subspace_dict['params']['discriminant_dim'] = discriminant_dim
    subspace_dict['params']['use_landmarks'] = use_landmarks
    subspace_dict['params']['use_highly_variable'] = use_highly_variable
    subspace_dict['params']['random_state'] = random_state
    subspace_dict['params']['method'] = 'umap'
    
    adata.obsp[conns_key] = csr_matrix(Z)
    
    if use_highly_variable:
        adata.varm[discriminant_key] = np.zeros(shape=(adata.n_vars, discriminant_dim))
        adata.varm[discriminant_key][adata.var['highly_variable']] = P[:, :discriminant_dim]
    else:
        adata.varm[discriminant_key] = P[:, :discriminant_dim]
    
    return adata if copy else None



















