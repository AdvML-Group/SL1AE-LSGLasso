from autoencoder import *
from L1PCA.utils import *

from utils import *
from train_l1_ae_ss import *
from logger import *
from sklearn.metrics import confusion_matrix
from datetime import datetime
import os
import pandas as pd
import argparse
from scipy.io import loadmat

import torch
import torch.nn as nn
import torch.optim as optim
import shutil
import time
import numpy as np
import random
# for sd in np.arange(6,11,1):
#     for lamd in np.arange(0.031, 0.08, 0.0005):
def l2_parser():
    """ predefined arguments for l1-autoencoder with l2 regularization """
    parser = argparse.ArgumentParser(description="Robust Data Reconstruction Using L1-Autoencoder")
    #-------------------------------------------------------------------------------------------------------------#
    parser.add_argument("--db", type=str, default="./data/Face_GT_10x10_60x40_diming.mat")
    parser.add_argument("--dbtxt", type=str, default="./data/ATNTFaceImages.txt")
    parser.add_argument("--image_grid", nargs='*', type=int, default=[10, 10, 60, 40, 2])
    parser.add_argument("--save_dir", type=str, default='./results/')
    parser.add_argument("--log_dir", type=str, default='./logs')
    parser.add_argument("--nlayers", nargs='*', type=int, default=[60 * 40,30, 20, 8])  # l1-ae: single hidden layer
    parser.add_argument("--batch_size", type=int, default=150)
    parser.add_argument("--epoch", type=int, default=25000)
    parser.add_argument("--pretrain_epoch", type=int, default=10000)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    #-------------------------------------------------------------------------------------------------------------#
    parser.add_argument("--reg_enc", type=str, choices=['l2', 'l1', 'l21', 'l21_l1'], default='l2')
    parser.add_argument("--reg_enc_lambda", type=float, default=0.01)
    """
            Comparison with sparse regularization:
                (1) pseudo sparsity: compute ||abs(W) <= e||_0, i.e., e=1e-3
                (2) magnitude: compute ||W||_1
                (3) just simply search some lambdas to get a over-smooth results, which is used as a baseline.
            Two Version:
                (1) pseudo sparsity:
                (2) real sparsity: W[abs(W) <= e]=0, this one will have a bad reconstruction (even small numbers carry information)
        """
    parser.add_argument("--reg_pseudo_thresh", type=float, default=1e-3)
    #-------------------------------------------------------------------------------------------------------------#
    parser.add_argument("--reg_dec", type=str, choices=['l2', 'l1', 'l21', 'l21_l1'], default='l2')
    parser.add_argument("--reg_dec_lambda", type=float, default=0.01)
    #-------------------------------------------------------------------------------------------------------------#
    parser.add_argument("--is_closure_used", type=bool, default=False)
    parser.add_argument("--verbose", type=bool, default=True)
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument("--activation_s", type=float, default=10.0)
    parser.add_argument("--lr_step_size", type=int, default=5000)
    parser.add_argument("--lr_gamma", type=float, default=0.1)
    #-------------------------------------------------------------------------------------------------------------#
    return parser


def l1_parser():
    """ predefined arguments for l1-autoencoder with l1 regularization """
    parser = argparse.ArgumentParser(description="Robust Data Reconstruction Using L1-Autoencoder")
    #-------------------------------------------------------------------------------------------------------------#
    parser.add_argument("--db", type=str, default="./data/Face_YaleB_15x11_64x64_diming.mat")
    parser.add_argument("--dbtxt", type=str, default="./data/ATNTFaceImages.txt")
    parser.add_argument("--image_grid", nargs='*', type=int, default=[15, 11, 64, 64, 2])
    parser.add_argument("--save_dir", type=str, default='./results/')
    parser.add_argument("--log_dir", type=str, default='./logs')
    parser.add_argument("--nlayers", nargs='*', type=int, default=[64 * 64, 500, 8])  # l1-ae: single hidden layer
    parser.add_argument("--batch_size", type=int, default=165)
    parser.add_argument("--epoch", type=int, default=29000)
    parser.add_argument("--pretrain_epoch", type=int, default=10000)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    #-------------------------------------------------------------------------------------------------------------#
    parser.add_argument("--reg_enc", type=str, choices=['l2', 'l1', 'l21', 'l21_l1'], default='l1')
    parser.add_argument("--reg_enc_lambda", type=float, default=0.006)  # recon_loss=18873/encoder_weight_sparsity=20.13%
    """
        Early-stopping for fixed around 20% sparsity    
    """
    #-------------------------------------------------------------------------------------------------------------#
    parser.add_argument("--reg_dec", type=str, choices=['l2', 'l1', 'l21', 'l21_l1'], default='l1')
    parser.add_argument("--reg_dec_lambda", type=float, default=0.006)
    #-------------------------------------------------------------------------------------------------------------#
    parser.add_argument("--is_closure_used", type=bool, default=False)
    parser.add_argument("--verbose", type=bool, default=True)
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument("--activation_s", type=float, default=10.0)
    parser.add_argument("--lr_step_size", type=int, default=5000)
    parser.add_argument("--lr_gamma", type=float, default=0.1)
    #-------------------------------------------------------------------------------------------------------------#
    return parser


def l21_parser():
    """ predefined arguments for l1-autoencoder with l21 regularization """
    parser = argparse.ArgumentParser(description="Robust Data Reconstruction Using L1-Autoencoder")
    #-------------------------------------------------------------------------------------------------------------#
    parser.add_argument("--db", type=str, default="./data/Face_GT_10x10_60x40_diming.mat")
    parser.add_argument("--dbtxt", type=str, default="./data/ATNTFaceImages.txt")
    parser.add_argument("--image_grid", nargs='*', type=int, default=[10, 10, 60, 40, 2])
    parser.add_argument("--save_dir", type=str, default='./results/')
    parser.add_argument("--log_dir", type=str, default='./logs')
    parser.add_argument("--nlayers", nargs='*', type=int, default=[60 * 40, 30, 20, 8])  # l1-ae: single hidden layer
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--epoch", type=int, default=30000)
    parser.add_argument("--pretrain_epoch", type=int, default=10000)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    #-------------------------------------------------------------------------------------------------------------#
    parser.add_argument("--reg_enc", type=str, choices=['l2', 'l1', 'l21', 'l21_l1'], default='l21')
    parser.add_argument("--reg_enc_lambda", type=float, default=0.8)  # recon_loss=20062/encoder_weight_sparsity=19.02%/epochs=25000
    """ 
        BUGS: for large lambdas (e.g. 1.5), after 25,000 epochs, l2/enc_w_sp=40%->0.00% 
        Reasons:
            (1) loss v.s. reg
            (2) learning rate is decreasing from 1e-3 to 1e-6
        Solutions:
            (1) fixed learning rate
            (2) after 25,000 epochs, if enc_w_sp lies in 15%~20%, then early-stop the training procedure => only training 25000 epochs
    """
    #-------------------------------------------------------------------------------------------------------------#
    parser.add_argument("--reg_dec", type=str, choices=['l2', 'l1', 'l21', 'l21_l1'], default='l21')
    parser.add_argument("--reg_dec_lambda", type=float, default=0.8)
    #-------------------------------------------------------------------------------------------------------------#
    parser.add_argument("--is_closure_used", type=bool, default=False)
    parser.add_argument("--verbose", type=bool, default=True)
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument("--activation_s", type=float, default=10.0)
    parser.add_argument("--lr_step_size", type=int, default=5000)
    parser.add_argument("--lr_gamma", type=float, default=0.1)
    #-------------------------------------------------------------------------------------------------------------#
    return parser


def l21_l1_parser():
    """ predefined arguments for l1-autoencoder with l21_l1 regularization """
    parser = argparse.ArgumentParser(description="Robust Data Reconstruction Using L1-Autoencoder")
    #-------------------------------------------------------------------------------------------------------------#
    parser.add_argument("--db", type=str, default="./data/Face_AR_10x15_55x40_one_glass.mat")
    parser.add_argument("--dbtxt", type=str, default="./data/ATNTFaceImages.txt")
    parser.add_argument("--image_grid", nargs='*', type=int, default=[10, 15, 55, 40, 2])
    parser.add_argument("--save_dir", type=str, default='./results/')
    parser.add_argument("--log_dir", type=str, default='./logs/')
    parser.add_argument("--nlayers", nargs='*', type=int, default=[55 * 40,30,20, 8])
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--epoch", type=int, default=30000)
    parser.add_argument("--pretrain_epoch", type=int, default=10000)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    #-------------------------------------------------------------------------------------------------------------#
    parser.add_argument("--reg_enc", type=str, choices=['l2', 'l1', 'l21', 'l21_l1'], default='l21_l1')
    parser.add_argument("--reg_enc_lambda", type=float, default=0.4)   # recon_loss=18776/encoder_weight_sparsity=20.69%
    parser.add_argument("--reg_enc_c", type=float, default=3)
    parser.add_argument("--reg_enc_m", type=float, default=0.5)
    parser.add_argument("--reg_enc_solver", type=str, choices=['exact', 'approx'], default='approx')  # exact: l1->l21, approx: l21->l1
    #-------------------------------------------------------------------------------------------------------------#
    parser.add_argument("--reg_dec", type=str, choices=['l2', 'l1', 'l21', 'l21_l1'], default='l21_l1')
    parser.add_argument("--reg_dec_lambda", type=float, default=0.4)
    parser.add_argument("--reg_dec_c", type=float, default=3)
    parser.add_argument("--reg_dec_m", type=float, default=0.5)
    parser.add_argument("--reg_dec_solver", type=str, choices=['exact', 'approx'], default='approx')  # exact: l1->l21, approx: l21->l1
    #-------------------------------------------------------------------------------------------------------------#
    parser.add_argument("--is_closure_used", type=bool, default=False)   # this version only supports first-order optimizer !!!
    parser.add_argument("--verbose", type=bool, default=True)
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument("--activation_s", type=float, default=10.0)
    # parser.add_argument("--lr_step_size", type=int, default=5000)
    parser.add_argument("--lr_step_size", type=int, default=5000)
    parser.add_argument("--lr_gamma", type=float, default=0.1)
    #-------------------------------------------------------------------------------------------------------------#
    return parser


def l21_l1_exact_parser():
    """ predefined arguments for l1-autoencoder with l21_l1 regularization """
    parser = argparse.ArgumentParser(description="Robust Data Reconstruction Using L1-Autoencoder")
    #-------------------------------------------------------------------------------------------------------------#
    parser.add_argument("--db", type=str, default="./data/ATNTFace400Image56x46_all_noise_new_8.mat")
    # parser.add_argument("--image_grid", nargs='*', type=int, default=[10, 10, 56, 46, 2])
    parser.add_argument("--save_dir", type=str, default='./results/')
    parser.add_argument("--log_dir", type=str, default='./logs')
    # parser.add_argument("--nlayers", nargs='*', type=int, default=[56 * 46, 8])  # l1-ae: single hidden layer
    parser.add_argument("--nlayers", nargs='*', type=int, default=[55 * 40, 8])  # l1-ae: single hidden layer
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--epoch", type=int, default=29000)
    parser.add_argument("--pretrain_epoch", type=int, default=10000)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    #-------------------------------------------------------------------------------------------------------------#
    parser.add_argument("--reg_enc", type=str, choices=['l2', 'l1', 'l21', 'l21_l1'], default='l21_l1')
    parser.add_argument("--reg_enc_lambda", type=float, default=0.2)   # recon_loss=?/encoder_weight_sparsity=?
    parser.add_argument("--reg_enc_c", type=float, default=2.0)
    parser.add_argument("--reg_enc_m", type=float, default=0.5)
    parser.add_argument("--reg_enc_solver", type=str, choices=['exact', 'approx'], default='exact')  # exact: l1->l21, approx: l21->l1
    #-------------------------------------------------------------------------------------------------------------#
    parser.add_argument("--reg_dec", type=str, choices=['l2', 'l1', 'l21', 'l21_l1'], default='l21_l1')
    parser.add_argument("--reg_dec_lambda", type=float, default=0.2)
    parser.add_argument("--reg_dec_c", type=float, default=2.0)
    parser.add_argument("--reg_dec_m", type=float, default=0.5)
    parser.add_argument("--reg_dec_solver", type=str, choices=['exact', 'approx'], default='exact')  # exact: l1->l21, approx: l21->l1
    #-------------------------------------------------------------------------------------------------------------#
    parser.add_argument("--is_closure_used", type=bool, default=False)   # this version only supports first-order optimizer !!!
    parser.add_argument("--verbose", type=bool, default=True)
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument("--activation_s", type=float, default=10.0)
    parser.add_argument("--lr_step_size", type=int, default=5000)
    parser.add_argument("--lr_gamma", type=float, default=0.1)
    #-------------------------------------------------------------------------------------------------------------#
    return parser


def main():
    start = time.perf_counter()
    parser = l21_l1_parser()
    args = parser.parse_args()
    args.reg_dec_lambda = args.reg_enc_lambda
    print(args)
    print(args.reg_dec_lambda)

    args.save_dir += '3_layers_1/' + datetime.now().strftime("%Y-%m-%d %H_%M_%S") + ' l21_l1_parser()'+ '_' +str(args.reg_enc_lambda)  +'/'
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    save_arg_into_txt(args)

    # dataset
    clean_data = loadmat(args.db)['DIDX']
    x0 = clean_data[:, 0:100]
    disp_img(x0, args.save_dir + 'X0.png', args)
    
    data = loadmat(args.db)['DnIDX']
    x = data[:, 0:100]
    disp_img(x, args.save_dir + 'X.png', args)


    gt = torch.from_numpy(x0.T).float()
    input = torch.from_numpy(x.T).float()
    activation = sReLU(args.activation_s)
    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
    args.log_dir = args.log_dir + TIMESTAMP
    logger = Logger(args.log_dir)
    # print("args.nlayers", args.nlayers)
    model = AutoEncoder(args.nlayers, activation)
    criterion = nn.L1Loss(reduction='sum')

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    train(input, model, logger, criterion, optimizer, scheduler, args, gt)  # add ground truth


    output = torch.transpose(model(input.to(args.device)).detach().cpu(), 1, 0).numpy()
    print("output",output.shape)
    filename = 'X_recon.png'
    disp_img(output, args.save_dir + filename, args)
    # dataset
    compute_reconstruction_performance(x0, x, output, True)
    save_reconstruction_performance(x0, x, output, args, True)
    save_model(model, args)



if __name__ == '__main__':
    seed = 3
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    main()
