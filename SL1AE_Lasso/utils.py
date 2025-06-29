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



def AR_disp_img(X, filename, args):
    im = imgrid(normalize(X), *args.AR_image_grid)
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

def compute_AR_relative_reconstruction_error(X, model, X0, args):

    XX = X.cpu().data.numpy()

    XX = XX.T
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

    # yalb
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
    # a=[0, 1, 2, 3, 4, 6, 7, 9, 10]
    X_0 = []
    for i in a:

        X_0.append(XX[:, i])

    X_0 = np.array(X_0)

    X_0 = torch.tensor(X_0)

    X_0 = X_0.to(args.device)


    model.eval()

    X_ = model(X_0)
    # rre = torch.sqrt(torch.sum(torch.pow(X - X_, 2.0))) / torch.sqrt(torch.sum(torch.pow(X, 2.0)))
    rre = torch.norm(X0 - X_, p='fro') / torch.norm(X0, p='fro')

    model.train()

    return rre.item()

def Xx0(X):
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

    # yalb
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
    # a=[0, 1, 2, 3, 4, 6, 7, 9, 10]
    X_0 = []
    X__ = []
    for i in a:
        # print("X_的列", X[:, i])
        # print("X_的列", X[:, i].shape)
        X_0.append(X[:, i])
    X_0 = np.array(X_0)
    X_0 = X_0.T # 输入图片删除墨镜后的数据
    return X_0


def Clustering(output, args):
    file_writer = open(args.save_dir + 'cul.txt', 'w')
    y_pred = KMeans(n_clusters=10, random_state=9).fit_predict(output)
    a = metrics.calinski_harabasz_score(output, y_pred)
    file_writer_content = ""
    file_writer_content += ('%.2f' % (a))
    file_writer_content += '\n'
    file_writer.write(file_writer_content)
    file_writer.close()

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




def compute_AR_reconstruction_performance(X, X_, is_print=False):
    print("X.shape = ", X.shape)
    print("X_.shape = ", X_.shape)
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

    # yalb
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
    # a=[0, 1, 2, 3, 4, 6, 7, 9, 10]
    X__ = []
    for i in a:
        X__.append(X_[:,i])
    X__ = np.array(X__)
    X__ = X__.T # 重构数据删除对应墨镜索引后的数据
    print("X__.shape", X__.shape)
    # 计算重构损失
    errs = []

    for Y in [X]:  # train & test errors
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

# X0, X, X_, args, is_print=False
def save_AR_reconstruction_performance(X, X_, args,is_print=False):
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

    # yalb
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
    X__ = []
    for i in a:
        X__.append(X_[:, i])
    X__ = np.array(X__)
    X__ = X__.T  # Input the data after removing the sunglasses from the image.
    print("X_.shape", X__.shape)
    errs = []
    file_writer = open(args.save_dir + 'model_performance.txt', 'w')

    for Y in [X]:  # train & test errors
        N = Y.shape[0] * Y.shape[1]
        errs.append(
            (
                1.0 / N * np.linalg.norm(Y - X__, ord='fro'),
                np.linalg.norm(Y - X__, ord='fro'),
                np.linalg.norm(Y - X__, ord='fro') / np.linalg.norm(Y, ord='fro'),
                1.0 / N * np.sum(np.abs(Y[:] - X__[:])),
                np.sum(np.abs(Y[:] - X__[:])),
                np.sum(np.abs(Y[:] - X__[:])) / np.sum(np.abs(Y[:])),
                1.0 / N * np.max(np.abs(Y[:] - X__[:])),
                np.max(np.abs(Y[:] - X__[:])),
                np.max(np.abs(Y[:] - X__[:])) / np.max(np.abs(Y[:]))
            )
        )

    file_writer_content = ""
    for index, prefix in enumerate(['Testing']):
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



def read_data(args, output):
    df = pd.read_csv(args.dbtxt, sep=',', header=None)
    # print("df", df)
    df1 = df.drop(labels=range(150, 400), axis=1)
    # x = df1.drop([0], axis=0).to_numpy(dtype=np.float64)
    x = output
    # y_true = df1.iloc[0].to_numpy(dtype=np.int32)
    # YaleB
    # y_true = [1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,
    #           4,4,4,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5,5,5,5,6,6,6,6,6,6,6,6,6,6,6,
    #           7,7,7,7,7,7,7,7,7,7,7,8,8,8,8,8,8,8,8,8,8,8,9,9,9,9,9,9,9,9,9,9,9,
    #           10,10,10,10,10,10,10,10,10,10,10,11,11,11,11,11,11,11,11,11,11,11,
    #           12,12,12,12,12,12,12,12,12,12,12,13,13,13,13,13,13,13,13,13,13,13,
    #           14,14,14,14,14,14,14,14,14,14,14,15,15,15,15,15,15,15,15,15,15,15]
    # AR
    y_true = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
              3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,
              5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,
              7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,
              9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10]
    print('y_tru', y_true)
    # print('y_tru', y_true)
    num_class = len(np.unique(y_true))
    return x, y_true, num_class

def run_kmeans(x, y_true, num_class):
    # kmeans = KMeans(n_clusters=num_class, init='random', n_init=10).fit(x)
    kmeans = KMeans(n_clusters=num_class, init='k-means++', n_init=10 ).fit(x)
    loss = kmeans.inertia_
    y_pred = kmeans.labels_ + 1
    C = confusion_matrix(y_true, y_pred)
    accuracy = np.sum(np.diag(C)) / np.sum(C)
    return loss, y_pred, C, accuracy

def reorder_confusion_matrix(C):
    indexes = linear_assignment(-C) # -C: minimize => maximize
    indexes = np.array(indexes).T
    # print("c",C)
    C_opt = C[:, indexes[:,1]]
    accuracy_opt = np.sum(np.diag(C_opt)) / np.sum(C_opt)
    return C_opt, accuracy_opt
