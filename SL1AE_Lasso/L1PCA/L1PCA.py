#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 00:00:31 2021

@author: Midas
"""

import numpy as np
from utils import *
class L1PCA:
    '''
        L1-norm Principal Component Analysis
    '''

    def __init__(self, k, maxiter=10000, thresh=1e-7, eps=1e-9, verbose=False):
        self.k = k
        self.maxiter = maxiter
        self.thresh = thresh
        self.eps = eps
        self.verbose = verbose
        self.X = None # p-by-n
        self.U = None # p-by-k
        self.V = None # n-by-k

    def fit(self, X):
        self.X = X
        self.U, self.V = self.opt_alg(X, self.k)
        return self.U, self.V

    def predict(self, ):
        return self.U @ self.V.T

    def L1Norm(self, X):
        return np.sum(np.abs(X))

    def LfNorm(self, X):
        return np.sqrt(np.sum(X ** 2))

    def cost_fn(self, X, U, V):
        return self.L1Norm(X - U @ V.T)

    def svd(self, X, k):
        # X => Uk * Sk * Vk'
        if k > min(X.shape): k = min(X.shape)
        U, S, V = np.linalg.svd(X, full_matrices=False)
        return U[:, 0:k], np.diag(S[0:k]), V[0:k, :].T

    def opt_alg(self, X, k):
        # Optimization: ALM Algorithm
        #  min_{U,V} ||X - UV'||_1 (X: p-by-n, U: p-by-k, V: n-by-k)
        #  => min_{E,U,V} ||E||_1 + mu/2 * ||E - (X - UV' + A / mu)||_F^2
        p, n = X.shape
        mu = 1.0 / self.LfNorm(X)
        rho = 1.1
        U, V = self.init_UV(X, k)
        A = np.zeros([p, n])
        loss = [self.cost_fn(X, U, V)]
        iter = 1
        if self.verbose:
            print('%d-th iteration: loss=%.10f' % (0, loss[-1]))
        while iter <= self.maxiter:
            # Update E
            E = self.update_E(X, U, V, A, mu)
            # Update U, V
            U, V = self.update_UV(X, E, A, mu, k)
            # Update Parameters
            A = A + mu * (X - U @ V.T - E)
            mu = rho * mu
            # Check Convergence
            loss.append(self.cost_fn(X, U, V))
            #if self.verbose:
            #    print('%d-th iteration: loss=%.10f' % (iter, loss[-1]))
            if iter > 1 and abs(loss[-1] - loss[-2]) / abs(loss[-1]) < self.thresh:
                break
            iter += 1
        return U, V

    def init_UV(self, X, k):
        p, n = X.shape
        U = np.random.random([p, k]) - 0.5
        V = np.random.random([n, k]) - 0.5
        z = np.sqrt(self.L1Norm(V) / self.L1Norm(U))
        return U * z, V / z

    def update_E(self, X, U, V, A, mu):
        # min_E ||E||_1 + mu/2 * ||E - (X - UV' + A/mu)||_F^2
        P = X - U @ V.T + A / mu
        return np.sign(P) * np.maximum(np.abs(P) - 1.0 / mu, 0)

    def update_UV(self, X, E, A, mu, k):
        # min_{U,V} mu/2 * ||Q - UV'||_F^2, s.t. Q = X - E + A/mu
        Q = X - E + A / mu # p-by-n
        F, S, G = self.svd(Q, k) # Q = FSG'
        return F, (S @ G.T).T

def disp_img(X, filename):
    from utils import normalize, imgrid
    import matplotlib.pyplot as plt
    im = imgrid(normalize(X), 10, 10, 60, 40, 2)
    plt.figure()
    plt.axis('off')
    plt.imshow(im, cmap='gray', vmin=0, vmax=1)
    plt.savefig(filename, bbox_inches='tight')

def test_PCA():
    from scipy.io import loadmat
    from L2PCA import L2PCA
    # from utils import compute_reconstruction_performance
    from datetime import datetime

    filename = './data/Face_GT_10x10_60x40_diming.mat'
    # data = loadmat(filename)['A_hat']
    # X = data[:, 0:165]
    # filename = 'Face_GT_10x10_60x40_diming.mat'
    # data = loadmat(filename)['Dn']

    # # ATnT
    # clean_data = loadmat(filename)['D']
    # x0 = clean_data[:, 0:100]
    # print(x0)
    # disp_img(x0, './PCA_results/x0.png')
    #
    # data = loadmat(filename)['Dn']    # ATnT
    # # data = loadmat(filename)['fea']
    # # data = data.T
    # X = data[:, 0:100]

    # YanleB
    # clean_data = loadmat(filename)['D']
    # x0 = clean_data[:, 0:165]
    # data = loadmat(filename)['Dn']
    # X = data[:, 0:165]

    clean_data = loadmat(filename)['D']
    x0 = clean_data[:, 0:165]
    data = loadmat(filename)['Dn']
    X = data[:, 0:165]
    clfs = {'L2PCA': L2PCA(k=6),
            'L1PCA': L1PCA(k=6, maxiter=5000, thresh=1e-6, verbose=True)} # k = 5~15

    disp_img(X, './PCA_results/X.png')



    for clfname, clf in clfs.items():
        file = './PCA_results/' + clfname
        U, V = clf.fit(X)
        X_ = clf.predict()
        filename = './PCA_results/X_' + clfname + '.png'
        disp_img(X_, filename)



def main():
    test_PCA()

if __name__ == '__main__':
    main()
