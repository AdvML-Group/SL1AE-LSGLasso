import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors
mpl.use('TkAgg')

import numpy as np

def plot_one_weight_rgb(X, filename):

    cmap = mcolors.ListedColormap(
        ['black', 'palegreen', 'cyan', 'skyblue', 'orange', 'wheat','beige','azure', 'lightskyblue','thistle','bisque','peachpuff','ivory', 'red'])
    bounds = [0, 0.00000000000001, 0.001, 0.01, 0.1, 0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99999999999, 1] # 6
    norm = mcolors.BoundaryNorm(bounds, cmap.N)



    plt.rcParams['figure.figsize'] = (100, 100)

    plt.axis('off')

    a = np.arange(X.shape[1] + 1)
    b = np.arange(X.shape[0] + 1)
    A, B = np.meshgrid(a, b)

    plt.pcolormesh(A, B, X, cmap=cmap, norm=norm, edgecolors='white', linewidth=1)
    plt.gca().set_aspect(1)
    plt.colorbar(ticks=bounds)
    plt.savefig(filename, bbox_inches='tight', format='png')
    plt.close()


def plot_one_weight_gray(X, filename):
    X = np.abs(X)
    X = (X - np.min(X[:])) / (np.max(X[:]) - np.min(X[:]) + 1e-11)
    plt.figure()
    plt.axis('off')
    plt.imshow(X, cmap='gray', vmin=0, vmax=1)
    plt.savefig(filename, bbox_inches='tight', format='png')
    plt.close()


def plot_whole_weight_col_partition(X, width, path, prefix='', mode='gray'):
    m, n = X.shape
    i, w = 0, width
    img_idx = 0
    X = np.abs(X)
    X = normalize_matrix(X)
    print(X)
    print('Normalization',np.sum(X == 0))

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


def normalize_matrix(matrix):
    """
    """
    matrix = np.array(matrix)
    min_val = np.min(matrix)
    max_val = np.max(matrix)
    print("max", max_val)
    normalized_matrix = (matrix - min_val) / (max_val - min_val)
    return normalized_matrix



def plot_ae_solution():
    # path = './results/2022-12-05 19:54:13/'
    path = './results/fwq/Face_GT\L21/2025-03-04 19_14_17 l21_l1_parser()_0.5/'
    # path = './results/wg_L21_L1/AR/2025-03-04 16_21_18 l21_l1_parser()_0.7/'
    filename = 'model_weights_encoder_layer_3.npy'
    X = np.load(path + filename)
    # normalize_matrix(X)
    print("x",X)
    print(np.sum(X == 0))

    print("X.shape",X.shape)
    plot_whole_weight_col_partition(X, 50, path, '', 'gray1 ')
    # plot_whole_weight_row_partition(X, 100, path, '', 'gray')


if __name__ == '__main__':

    plot_ae_solution()







