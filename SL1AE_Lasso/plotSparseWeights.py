import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('TkAgg')

import numpy as np


def plot_one_weight_rgb(X, filename):
    plt.figure()
    plt.axis('off')
    plt.imshow(X, cmap=plt.cm.bwr)
    plt.colorbar()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


def plot_one_weight_gray(X, filename):
    X = np.abs(X)
    X = (X - np.min(X[:])) / (np.max(X[:]) - np.min(X[:]) + 1e-11)
    plt.figure()
    plt.axis('off')
    plt.imshow(X, cmap='gray', vmin=0, vmax=1)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


def plot_whole_weight_col_partition(X, width, path, prefix='', mode='gray'):
    m, n = X.shape
    i, w = 0, width
    img_idx = 0
    while i < n:
        j = n if i + w > n else i + w
        A = X[:, i:j]
        filename = path + prefix + str(img_idx) + '.png'
        if mode == 'gray':
            plot_one_weight_gray(A, filename)
        else:
            plot_one_weight_rgb(A, filename)
        i += w
        img_idx += 1


def plot_whole_weight_row_partition(X, width, path, prefix='', mode='gray'):
    m, n = X.shape
    i, w = 0, width
    img_idx = 0
    while i < m:
        j = m if i + w > m else i + w
        A = X[i:j, ]
        filename = path + prefix + str(img_idx) + '.png'
        if mode == 'gray':
            plot_one_weight_gray(A, filename)
        else:
            plot_one_weight_rgb(A, filename)
        i += w
        img_idx += 1


def plot_ae_solution():
    path = './results/2022-12-05 19:54:13/'
    filename = 'model_weights_encoder_layer_1.npy'
    X = np.load(path + filename)
    plot_whole_weight_col_partition(X, 100, path, '', 'gray')
    # plot_whole_weight_row_partition(X, 100, path, '', 'gray')


if __name__ == '__main__':

    plot_ae_solution()







