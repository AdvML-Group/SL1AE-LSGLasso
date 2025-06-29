#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 12:14:01 2021

@author: Midas
"""

import numpy as np

class L2PCA:
    '''
        L2-norm Principal Component Analysis
    '''

    def __init__(self, k, eps=1e-9):
        self.k = k
        self.eps = eps
        self.X = None
        self.U = None
        self.V = None

    def fit(self, X):
        self.X = X
        self.U, self.V = self.opt_alg(self.X, self.k)
        return self.U, self.V

    def predict(self, ):
        return self.U @ self.V.T

    def opt_alg(self, X, k):
        # Optimization: SVD Algorithm
        #  min_{U,V} ||X - UV'||_F^2 (X: p-by-n, U: p-by-k, V: n-by-k)
        U, S, V = self.svd(X, k)
        S2 = S ** 0.5
        z = np.sqrt(self.L1Norm(V @ S2) / self.L1Norm(U @ S2))# @是一个操作符，表示矩阵-向量乘法
        return U @ S2 * z, V @ S2 / z

    def L1Norm(self, X):
        return np.sum(np.abs(X))
        
    def svd(self, X, k):
        # X => Uk * Sk * Vk'
        if k > min(X.shape): k = min(X.shape)
        U, S, V = np.linalg.svd(X, full_matrices=False)
        return U[:, 0:k], np.diag(S[0:k]), V[0:k, :].T
