import numpy as np
import math
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
import pandas as pd
from scipy.optimize import linear_sum_assignment as linear_assignment


def normalize(W):
    # assume: each column represent a data
    p, n = W.shape
    Wn = np.zeros([p, n])
    for i in range(n):
        Wi = W[:, i]
        Wn[:, i] = (Wi - np.min(Wi)) / (np.max(Wi) - np.min(Wi)) # scale to [0, 1]
    return Wn


def imgrid(X, nh, nw, h, w, d):
    # X: (#features)-by-(#images)
    # nh: #images per column
    # nw: #images per row
    # (h, w): (height, width) of one image
    # d: interval between two adjacent images
    p, n = X.shape
    imH = nh * h + (nh - 1) * d
    imW = nw * w + (nw - 1) * d
    im = np.zeros([imH, imW])
    imR, imC, cnt = 0, 0, 0
    for i in range(nh):
        for j in range(nw):
            im[imR:imR+h, imC:imC+w] = np.reshape(X[:,cnt], [h, w], order='F')
            imC = imC + w + d
            cnt += 1
            if cnt >= n: break
        imR = imR + h + d
        imC = 0
    return im

def compute_reconstruction_performance(X0, X, X_, is_print=False):
    # compute reconstruction errors of l1-ae (X0 -> X -> X_)
    errs = []

    for Y in [X, X0]:  # train & test errors
        errs.append(
            (
                np.linalg.norm(Y - X_, ord='fro'),
                np.linalg.norm(Y - X_, ord='fro') / np.linalg.norm(Y, ord='fro')
            )
        )

    if is_print:
        for index, prefix in enumerate(['Training', 'Testing']):
            print('%s Performance:' % prefix, end=' ')
            # print("index", index)
            print('RE=%.6f, RRE=%.6f' % errs[index])

    return errs


