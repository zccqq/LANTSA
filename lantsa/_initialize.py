# -*- coding: utf-8 -*-

import numpy as np

from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


def initialize_Z(X, n_neighbors, landmarks, random_state):
    
    m, n = X.shape
    n_components = min(max(m // 50, 20), n-1)
    if n_components < m:
        X_new = PCA(n_components=n_components, svd_solver='arpack', random_state=random_state).fit_transform(X.T)
    else:
        X_new = X.T
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean').fit(X_new[landmarks, :])
    Zi = np.zeros((n, n))
    Zi[:, landmarks] = nbrs.kneighbors_graph(X_new).toarray()
    np.fill_diagonal(Zi, 0)
    
    return Zi



















