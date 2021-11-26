# -*- coding: utf-8 -*-

import torch

from tqdm import trange


def solve_l1l2(W, Lambda):
    
    NW = torch.norm(W, 2, dim=0, keepdim=True)
    E = (NW > Lambda) * (NW - Lambda) / NW
    
    return E * W


def solve_Z(X, landmarks, Lambda, Z_mask, n_iterations, device):
    
    mu = 1e-6
    max_mu = 1e30
    rho = 1.1
    tol1 = 1e-4
    tol2 = 1e-5
    
    m, n = X.shape
    l = landmarks.shape[0]
    Xl = X[:,landmarks]
    n = torch.tensor(n).to(device)
    normfX = torch.norm(X, 'fro')
    norm2X = torch.norm(X, 2)
    eta = norm2X ** 2 + n + 1
    tensor0 = torch.tensor(0).to(device)
    
    # intialize
    J = torch.zeros(n, l).to(device)
    Z = torch.zeros(n, l).to(device)
    E = torch.zeros(m, l).to(device)
    
    Y1 = torch.zeros(m, l).to(device)
    Y2 = torch.zeros(1, l).to(device)
    Y3 = torch.zeros(n, l).to(device)
    
    pbar = trange(n_iterations)
    
    for Iter in pbar:
        Em = E
        Zm = Z
        
        temp = Z + Y3 / mu
        
        J = torch.maximum(temp - 1 / mu, tensor0)
        
        temp = Xl - torch.matmul(X, Z) + Y1 / mu
        E = solve_l1l2(temp, Lambda / mu)
        
        H = - torch.matmul(X.T, (Xl - torch.matmul(X, Z) - E + Y1 / mu)) - (1 - torch.sum(Z, axis=0, keepdims=True) + Y2 / mu) + (Z - J + Y3 / mu)
        M = Z - H / eta
        Z = Z_mask * M
        
        xmaz = Xl - torch.matmul(X, Z)
        leq1 = xmaz - E
        leq2 = 1 - torch.sum(Z, axis=0, keepdims=True)
        leq3 = Z - J
        relChgZ = torch.norm(Z - Zm, 'fro') / normfX
        relChgE = torch.norm(E - Em, 'fro') / normfX
        relChg = torch.max(relChgE, relChgZ)
        recErr = torch.norm(leq1, 'fro') / normfX
        
        pbar.set_postfix_str(f'relChg: {relChg.item():.3e}, recErr: {recErr.item():.3e}')
        
        if relChg < tol1 and recErr < tol2:
            pbar.set_postfix_str(f'relChg: {relChg.item():.3e}, recErr: {recErr.item():.3e}, converged!')
            break
        else:
            Y1 = Y1 + mu * leq1
            Y2 = Y2 + mu * leq2
            Y3 = Y3 + mu * leq3
            mu = min(max_mu, mu * rho)
    
    return Z, E


def solve_P(X, landmarks, Z):
    
    eps = 1e-14
    Xl = X[:,landmarks]
    temp = X - torch.matmul(Xl, Z.T)
    D = torch.diag(0.5 / torch.norm(temp, 2, dim=0) + eps)
    S = torch.matmul(torch.matmul(X - torch.matmul(Xl, Z.T), D), (X - torch.matmul(Xl, Z.T)).T)
    S = (S + S.T) / 2
    DS, Pall = torch.eig(S, eigenvectors=True)
    Pu = Pall[:, DS[:, 0] > 10^-3]
    
    return Pu



















