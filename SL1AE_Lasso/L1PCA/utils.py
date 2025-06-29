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


def compute_AR_reconstruction_performance(X, X_, is_print=False):
    # print("X.dtype", X.dtype)
    # print("X_.dtype", X_.dtype)
    # print("X.shape", X.shape)
    # print("X_.shape", X_.shape)
    # AR第一组
    a = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14,
         16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
         30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 42, 43, 44,
         45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 56, 57, 58, 59,
         60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 71, 72, 73, 74,
         75, 76, 77, 78, 79, 81, 82, 83, 84, 85, 86, 87, 88, 89,
         90, 91, 92, 93, 94, 95, 96, 98, 99, 100, 101, 102, 103, 104,
         105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118,
         120, 121, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134,
         135, 136, 137, 138, 139, 140, 141, 143, 144, 145, 146, 147, 148, 149]
    # AR第二组
    # a = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14,
    #  15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 29,
    #  30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 42, 43, 44,
    #  46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
    #  60, 61, 62, 63, 64, 65, 66, 67, 69, 70, 71, 72, 73, 74,
    #  75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89,
    #  90, 91, 92, 93, 94, 95, 97, 98, 99, 100, 101, 102, 103, 104,
    #  105, 106, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
    #  120, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134,
    #  135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148]

    # AR第三组
    # a = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
    #      15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26, 27, 28, 29,
    #      30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 41, 42, 43, 44,
    #      46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
    #      60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 73, 74,
    #      75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 89,
    #      90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 101, 102, 103, 104,
    #      105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 118, 119,
    #      120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 131, 132, 133, 134,
    #      135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 147, 148, 149]

    # AR第四组
    # a = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14,
    #      15, 16, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
    #      30, 31, 32, 33, 34, 35, 36, 38, 39, 40, 41, 42, 43, 44,
    #      45, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
    #      60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 72, 73, 74,
    #      75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89,
    #      91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104,
    #      105, 106, 107, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
    #      120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 132, 133, 134,
    #      135, 136, 137, 138, 139, 141, 142, 143, 144, 145, 146, 147, 148, 149]
    # a = [0,1,2,4,5,7,8,9,10,
    #      11,12,13,15,16,18,19,20,21,
    #      22,23,24,26,27,29,30,31,32,
    #      33,34,35,37,38,40,41,42,43,
    #      44,45,46,48,49,51,52,53,54,
    #      55,56,57,59,60,62,63,64,65,
    #      66,67,68,70,71,73,74,75,76,
    #      77,78,79,81,82,84,85,86,87,
    #      88,89,90,92,93,95,96,97,98,
    #      99,100,101,103,104,106,107,108,109,
    #      110,111,112,114,115,117,118,119,120,
    #      121,122,123,125,126,128,129,130,131,
    #      132,133,134,136,137,139,140,141,142,
    #      143,144,145,147,148,150,151,152,153,
    #      154,155,156,158,159,161,162,163,164]
    X_0 = []
    X__ = []
    for i in a:
        X_0.append(X[:, i])
    X_0 = np.array(X_0)
    X_0 = X_0.T
    # print("X_0.shape", X_0.shape)
    # print("X_0.dtype", X_0.dtype)
    for i in a:
        X__.append(X_[:,i])
    X__ = np.array(X__)
    X__ = X__.T

    errs = []

    for Y in [X_0]:  # train & test errors
        errs.append(
            (
                np.linalg.norm(Y - X__, ord='fro'),
                np.linalg.norm(Y - X__, ord='fro') / np.linalg.norm(Y, ord='fro')
            )
        )

    if is_print:
        for index, prefix in enumerate(['Testing']):
            print('%s Performance:' % prefix, end=' ')
            # print("index", index)
            print('RE=%.6f, RRE=%.6f' % errs[index])

    return errs
    print("c", c)
    c = np.array(c)
    print("c", c)
    print("c.shape", c.shape)
    c = c.T
    print("c", c)
    print("c.shape", c.shape)


