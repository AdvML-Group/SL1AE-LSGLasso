import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.cluster import KMeans
from sklearn import metrics
import numpy as np
import os
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib as mpl
# mpl.use('TkAgg')
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
# from sklearn.utils.linear_assignment_ import linear_assignment
from scipy.optimize import linear_sum_assignment as linear_assignment
from L1PCA.utils import *


def disp_img(X, filename, args):
    # im = imgrid(normalize(X), *[10, 14, 55, 40, 2])
    im = imgrid(normalize(X), *args.image_grid)
    plt.figure()
    plt.axis('off')
    plt.imshow(im, cmap='gray', vmin=0, vmax=1)
    # plt.show()
    plt.savefig(filename, bbox_inches='tight',format='png')
    plt.close()



def save_arg_into_txt(args):
    path = args.save_dir
    f = open(path + 'args.txt', 'w')
    for key, val in args.__dict__.items():
        f.write(str(key) + ":\t" + str(val) + "\n")
    f.close()


# def compute_weight_sparsity(model):
#     # compute the sparsity of weights in each layer
#     paras = [p for p in model.parameters()]
#
#     enc_wsp = []
#     n_paras = len(paras) // 2
#     for idx_paras in range(1, n_paras + 1, 2):
#         enc_W = paras[idx_paras - 1].data
#         enc_wsp.append(1.0 * torch.sum(enc_W == 0) / (enc_W.shape[0] * enc_W.shape[1]))
#
#     dec_wsp = []
#     n_paras = len(paras)
#     for idx_paras in range(n_paras // 2, n_paras, 2):
#         dec_W = paras[idx_paras].data
#         dec_wsp.append(1.0 * torch.sum(dec_W == 0) / (dec_W.shape[0] * dec_W.shape[1]))
#
#     return enc_wsp, dec_wsp
#
#
# def compute_weight_pseudo_sparsity(model, thresh):
#     # compute the pseudo sparsity (l2) of weights in each layer
#     paras = [p for p in model.parameters()]
#
#     enc_wsp = []
#     n_paras = len(paras) // 2
#     for idx_paras in range(1, n_paras + 1, 2):
#         enc_W = paras[idx_paras - 1].data
#         enc_wsp.append(1.0 * torch.sum(torch.abs(enc_W) <= thresh) / (enc_W.shape[0] * enc_W.shape[1]))
#
#     dec_wsp = []
#     n_paras = len(paras)
#     for idx_paras in range(n_paras // 2, n_paras, 2):
#         dec_W = paras[idx_paras].data
#         dec_wsp.append(1.0 * torch.sum(torch.abs(dec_W) <= thresh) / (dec_W.shape[0] * dec_W.shape[1]))
#
#     return enc_wsp, dec_wsp
def compute_weight_sparsity(model):
    # compute the sparsity of weights in each layer
    paras = [p for p in model.parameters()]

    enc_wsp = []
    sum_enc_wsp = []
    enc_num_0 = 0
    enc_num = 0
    n_paras = len(paras) // 2
    for idx_paras in range(1, n_paras + 1, 2):
        enc_W = paras[idx_paras - 1].data
        enc_num_0 = 1.0 * torch.sum(enc_W == 0) + enc_num_0
        enc_num = (enc_W.shape[0] * enc_W.shape[1]) + enc_num
        enc_wsp.append(1.0 * torch.sum(enc_W == 0) / (enc_W.shape[0] * enc_W.shape[1]))
    sum_enc_wsp.append(enc_num_0 / enc_num)
    # enc_wsp = num_0 / num

    dec_wsp = []
    sum_dec_wsp = []
    dec_num_0 = 0
    dec_num = 0
    n_paras = len(paras)
    for idx_paras in range(n_paras // 2, n_paras, 2):
        dec_W = paras[idx_paras].data
        dec_num_0 = 1.0 * torch.sum(dec_W == 0) + dec_num_0
        dec_num = (dec_W.shape[0] * dec_W.shape[1]) + dec_num
        dec_wsp.append(1.0 * torch.sum(dec_W == 0) / (dec_W.shape[0] * dec_W.shape[1]))
    sum_dec_wsp.append(dec_num_0 / dec_num)
    sum_wsp = []
    sum_wsp.append((enc_num_0 + dec_num_0) / (enc_num + dec_num))




    return enc_wsp, dec_wsp, sum_enc_wsp, sum_dec_wsp, sum_wsp


def compute_weight_pseudo_sparsity(model, thresh):
    # compute the pseudo sparsity (l2) of weights in each layer
    paras = [p for p in model.parameters()]

    enc_wsp = []
    sum_enc_wsp = []
    enc_num_0 = 0
    enc_num = 0
    n_paras = len(paras) // 2
    for idx_paras in range(1, n_paras + 1, 2):
        enc_W = paras[idx_paras - 1].data
        enc_num_0 = 1.0 * torch.sum(torch.abs(enc_W) <= thresh) + enc_num_0
        enc_num = enc_W.shape[0] * enc_W.shape[1] + enc_num
        enc_wsp.append(1.0 * torch.sum(torch.abs(enc_W) <= thresh) / (enc_W.shape[0] * enc_W.shape[1]))
    sum_enc_wsp.append(enc_num_0 / enc_num)

    dec_wsp = []
    sum_dec_wsp = []
    dec_num_0 = 0
    dec_num = 0
    n_paras = len(paras)
    for idx_paras in range(n_paras // 2, n_paras, 2):
        dec_W = paras[idx_paras].data
        dec_num_0 = 1.0 * torch.sum(torch.abs(dec_W) <= thresh) + dec_num_0
        dec_num = dec_W.shape[0] * dec_W.shape[1] + dec_num
        dec_wsp.append(1.0 * torch.sum(torch.abs(dec_W) <= thresh) / (dec_W.shape[0] * dec_W.shape[1]))
    sum_dec_wsp.append(dec_num_0 / dec_num)
    sum_wsp = []
    sum_wsp.append((enc_num_0 + dec_num_0) / (enc_num + dec_num))

    return enc_wsp, dec_wsp, sum_enc_wsp, sum_dec_wsp, sum_wsp


def save_model_weights(model, paras_save_path):
    # save model weights as numpy array
    paras = [p for p in model.parameters()]

    if not os.path.exists(paras_save_path):
        os.mkdir(paras_save_path)

    for paras_idx in range(0, len(paras)//2, 2):
        paras_name = paras_save_path + 'model_weights_encoder_layer_' + str(paras_idx // 2 + 1) + '.npy'
        paras_var = paras[paras_idx].data.cpu().numpy()
        np.save(paras_name, paras_var)

    for paras_idx in range(len(paras)//2, len(paras), 2):
        paras_name = paras_save_path + 'model_weights_decoder_layer_' + str((paras_idx - len(paras) // 2) // 2 + 1) + '.npy'
        paras_var = paras[paras_idx].data.cpu().numpy()
        np.save(paras_name, paras_var)


def compute_relative_reconstruction_error(X, model, X0):
    # compute the relative reconstruction error (RRE) of l1-ae (X0 -> X -> X_)
    model.eval()

    X_ = model(X)
    # rre = torch.sqrt(torch.sum(torch.pow(X - X_, 2.0))) / torch.sqrt(torch.sum(torch.pow(X, 2.0)))
    rre = torch.norm(X0 - X_, p='fro') / torch.norm(X0, p='fro')

    model.train()

    return rre.item()

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
            print('%s Performance123:' % prefix, end=' ')
            print("index:", index)
            print("prefix:", prefix)
            print('RE=%.6f, RRE=%.6f' % errs[index])

    return errs


def save_reconstruction_performance(X0, X, X_, args, is_print=False):
    # save reconstruction errors of l1-ae (X0 -> X -> X_)
    errs = []
    file_writer = open(args.save_dir + 'model_performance.txt', 'w')

    for Y in [X, X0]:  # train & test errors
        N = Y.shape[0] * Y.shape[1]
        errs.append(
            (
                1.0 / N * np.linalg.norm(Y - X_, ord='fro'),
                np.linalg.norm(Y - X_, ord='fro'),
                np.linalg.norm(Y - X_, ord='fro') / np.linalg.norm(Y, ord='fro'),
                1.0 / N * np.sum(np.abs(Y[:] - X_[:])),
                np.sum(np.abs(Y[:] - X_[:])),
                np.sum(np.abs(Y[:] - X_[:])) / np.sum(np.abs(Y[:])),
                1.0 / N * np.max(np.abs(Y[:] - X_[:])),
                np.max(np.abs(Y[:] - X_[:])),
                np.max(np.abs(Y[:] - X_[:])) / np.max(np.abs(Y[:]))
            )
        )

    file_writer_content = ""
    for index, prefix in enumerate(['Training', 'Testing']):
        file_writer_content += ('%s Performance: ' % prefix)
        file_writer_content += ('L2: ARE=%.6f, RE=%.6f, RRE=%.6f || L1: ARE=%.6f, RE=%.6f, RRE=%.6f || Linf: ARE=%.6f, RE=%.6f, RRE=%.6f' % errs[index])
        file_writer_content += "\n"

    file_writer.write(file_writer_content)

    if is_print:
        print('-' * 10)
        print(file_writer_content)

    file_writer.close()

def save_model(model, args):
    path = args.save_dir + 'pretrained_model'
    torch.save(model.state_dict(), path)
