import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from autoencoder import *
from logger import *
from L1PCA.utils import *
from utils import *
import numpy
import numpy as np
import os

import matplotlib.pyplot as plt
import matplotlib as mpl
# mpl.use('TkAgg')
import time


def train(x, model: AutoEncoder, logger: Logger, criterion, optimizer, scheduler, args, gt):
    """
        Training L1-AutoEncoder Using Mini-Batch SGD
    """

    n = x.shape[0]
    print("n", n)
    num_of_batch = int(math.ceil(n / args.batch_size))
    print("num_of_batch", num_of_batch)
    loss_history = []

    gt = gt.to(args.device)
    x = x.to(args.device)
    model.to(args.device)

    file_writer = open(args.save_dir + 'train_loss.txt', 'w')
    for i in range(args.epoch):
        total_loss = 0.0
        total_reg = 0.0
        total_reg_enc = 0.0
        total_reg_dec = 0.0





        for j in range(num_of_batch):
            start, end = j * args.batch_size, min((j + 1) * args.batch_size, n)
            xb = x[start:end, :]

            model.zero_grad()
            xb_ = model(xb)
            # reconstruction loss
            loss = criterion(xb_, xb)
            total_loss += loss.item()

            # regularization loss
            if args.reg_enc == 'l2-' and args.reg_dec == 'l2-':
                """ stochastic gradient descent based standard weight-decay """

                paras = [p for p in model.parameters()]

                for idx_paras in range(0, len(paras) // 2, 2):
                    reg_enc = torch.sum(torch.pow(paras[idx_paras], 2.0))
                    loss += args.reg_enc_lambda * reg_enc
                    total_reg_enc += reg_enc.item()
                    total_reg += args.reg_enc_lambda * reg_enc.item()

                for idx_paras in range(len(paras) // 2, len(paras), 2):
                    reg_dec = torch.sum(torch.pow(paras[idx_paras], 2.0))
                    loss += args.reg_dec_lambda * reg_dec
                    total_reg_dec += reg_dec.item()
                    total_reg += args.reg_dec_lambda * reg_dec.item()

                loss.backward()
                optimizer.step()

            else:
                """ proximal gradient descent based sparse weight-decay """

                # gradient step: W_{t+0.5}, b_{t+1} <-  W_{t}, b_{t}
                loss.backward()
                optimizer.step()

                # project step: W_{t+1} <- W_{t+0.5}
                lr = optimizer.state_dict()['param_groups'][0]['lr']
                paras = [p for p in model.parameters()]

                # encoder
                n_paras = len(paras) // 2
                enc_layer = n_paras // 2
                enc_layer_idx = 0  # only used in l21_l1
                enc_lambda = args.reg_enc_lambda
                enc_c = args.reg_enc_c if args.reg_enc == 'l21_l1' else 0  # only used in l21_l1
                enc_m = args.reg_enc_m if args.reg_enc == 'l21_l1' else 0  # only used in l21_l1

                for idx_paras in range(1, n_paras + 1, 2):
                    # weights in encoder layers
                    enc_W = paras[idx_paras - 1].data  # W_{t+0.5}

                    if args.reg_enc == 'l2':
                        # l2-norm based regularization
                        enc_reg = torch.sum(torch.pow(enc_W, 2.0))
                        total_reg_enc += enc_lambda * enc_reg.item()

                        # min_w ||w - w_{t+0.5}||_2^2 + lr * lambda * ||w||_2^2
                        paras[idx_paras - 1].data = 1.0 / (1.0 + enc_lambda * lr) * enc_W

                    elif args.reg_enc == 'l1':
                        # l1-norm based regularization
                        enc_reg = torch.sum(torch.abs(enc_W))
                        total_reg_enc += enc_lambda * enc_reg.item()

                        # min_w ||w - w_{t+0.5}||_2^2 + lr * lambda * ||w||_1
                        enc_lambda1 = lr * enc_lambda
                        enc_O = torch.zeros(1).to(args.device)
                        paras[idx_paras - 1].data = torch.sign(enc_W) * torch.maximum(torch.abs(enc_W) - enc_lambda1, enc_O)

                    elif args.reg_enc == 'l21':
                        # l21-norm based regularization
                        enc_reg = torch.sum(torch.norm(enc_W, p=2, dim=0))
                        total_reg_enc += enc_lambda * enc_reg.item()

                        # min_w ||w - w_{t+0.5}||_2^2 + lr * lambda * ||w||_2
                        enc_lambda1 = lr * enc_lambda
                        enc_O = torch.zeros(1).to(args.device)
                        paras[idx_paras - 1].data = torch.reshape(torch.maximum(1 - enc_lambda1 / (torch.norm(enc_W, p=2, dim=0) + 1e-11), enc_O), (1, -1)) * enc_W

                    elif args.reg_enc == 'l21_l1':
                        # l21_l1-norm based regularization
                        enc_beta_l = enc_m + (1 - enc_m) * enc_layer_idx / (enc_layer - 1 + 1e-11)  # add 1e-11 for single hidden layer

                        enc_reg_ell1 = torch.sum(torch.abs(enc_W))

                        enc_reg_ell21 = torch.sum(torch.norm(enc_W, p=2, dim=0))
                        enc_reg = enc_beta_l * enc_reg_ell1 + (1 - enc_beta_l) * enc_reg_ell21
                        total_reg_enc += enc_lambda * enc_reg.item()

                        # min_w ||w - w_{t+0.5}||_2^2 + lr * lambda_1 * ||w||_1 + lr * lambda_2 * ||w||_2
                        enc_lambda1 = lr * enc_lambda * enc_beta_l
                        enc_lambda2 = lr * enc_lambda * (1 - enc_beta_l)
                        enc_O = torch.zeros(1).to(args.device)


                        if args.reg_enc_solver == 'exact':
                            # exact solution
                            enc_U = torch.sign(enc_W) * torch.maximum(torch.abs(enc_W) - enc_lambda1, enc_O)
                            paras[idx_paras - 1].data = torch.reshape(torch.maximum(1 - enc_lambda2 / (torch.norm(enc_U, p=2, dim=0) + 1e-11), enc_O), (1, -1)) * enc_U

                        elif args.reg_enc_solver == 'approx':

                            enc_U = torch.reshape(torch.maximum(1 - enc_lambda2 / (torch.norm(enc_W, p=2, dim=0) + 1e-11), enc_O), (1, -1)) * enc_W
                            paras[idx_paras - 1].data = torch.sign(enc_U) * torch.maximum(torch.abs(enc_U) - enc_lambda1, enc_O)

                        else:
                            # errors
                            print('Error! This %s solver is not implemented!' % args.reg_enc_solver)
                            exit()

                        enc_layer_idx += 1  # enc_1 -> enc_n
                        enc_lambda /= enc_c  # decreasing lambda

                    else:
                        # errors
                        print('Error! This %s regularization is not implemented!' % args.reg_enc)
                        exit()


                # decoder
                n_paras = len(paras)
                dec_layer = enc_layer
                dec_layer_idx = enc_layer - 1  # only used in l21_l1
                dec_lambda = args.reg_dec_lambda
                dec_c = args.reg_dec_c if args.reg_dec == 'l21_l1' else 0  # only used in l21_l1
                dec_m = args.reg_dec_m if args.reg_dec == 'l21_l1' else 0  # only used in l21_l1
                if args.reg_dec == 'l21_l1':
                    dec_lambda /= (dec_c ** (n_paras // 4 - 1 + 1e-11))  # only used in l21_l1 & add 1e-11 for single hidden layer
                # print('dec_lambda', dec_lambda)
                for idx_paras in range(n_paras // 2, n_paras, 2):
                    # weights in decoder layers
                    dec_W = paras[idx_paras].data

                    if args.reg_dec == 'l2':
                        # l2-norm based regularization
                        dec_reg = torch.sum(torch.pow(dec_W, 2.0))
                        total_reg_dec += dec_lambda * dec_reg.item()

                        # min_w ||w - w_{t+0.5}||_2^2 + lr * lambda * ||w||_2^2
                        paras[idx_paras].data = 1.0 / (1.0 + dec_lambda * lr) * dec_W

                    elif args.reg_dec == 'l1':
                        # l1-norm based regularization
                        dec_reg = torch.sum(torch.abs(dec_W))
                        total_reg_dec += dec_lambda * dec_reg.item()

                        # min_w ||w - w_{t+0.5}||_2^2 + lr * lambda * ||w||_1
                        dec_lambda1 = lr * dec_lambda
                        dec_O = torch.zeros(1).to(args.device)
                        paras[idx_paras].data = torch.sign(dec_W) * torch.maximum(torch.abs(dec_W) - dec_lambda1, dec_O)

                    elif args.reg_dec == 'l21':
                        # l21-norm based regularization
                        dec_reg = torch.sum(torch.norm(dec_W, p=2, dim=0))
                        total_reg_dec += dec_lambda * dec_reg.item()

                        # min_w ||w - w_{t+0.5}||_2^2 + lr * lambda * ||w||_2
                        dec_lambda1 = lr * dec_lambda
                        dec_O = torch.zeros(1).to(args.device)
                        paras[idx_paras].data = torch.reshape(torch.maximum(1 - dec_lambda1 / (torch.norm(dec_W, p=2, dim=0) + 1e-11), dec_O), (1, -1)) * dec_W

                    elif args.reg_dec == 'l21_l1':
                        # l21_l1-norm based regularization
                        # print('dec_lambda', dec_lambda)
                        dec_beta_l = dec_m + (1 - dec_m) * dec_layer_idx / (dec_layer - 1 + 1e-11)  # add 1e-11 for single hidden layer
                        dec_reg_ell1 = torch.sum(torch.abs(dec_W))
                        dec_reg_ell21 = torch.sum(torch.norm(dec_W, p=2, dim=0))
                        dec_reg = dec_beta_l * dec_reg_ell1 + (1 - dec_beta_l) * dec_reg_ell21
                        total_reg_dec += dec_lambda * dec_reg.item()

                        # min_w ||w - w_{t+0.5}||_2^2 + lr * lambda_1 * ||w||_1 + lr * lambda_2 * ||w||_2
                        dec_lambda1 = lr * dec_lambda * dec_beta_l
                        dec_lambda2 = lr * dec_lambda * (1 - dec_beta_l)
                        dec_O = torch.zeros(1).to(args.device)


                        if args.reg_dec_solver == 'exact':
                            # exact solution
                            dec_U = torch.sign(dec_W) * torch.maximum(torch.abs(dec_W) - dec_lambda1, dec_O)
                            paras[idx_paras].data = torch.reshape(torch.maximum(1 - dec_lambda2 / (torch.norm(dec_U, p=2, dim=0) + 1e-11), dec_O), (1, -1)) * dec_U

                        elif args.reg_dec_solver == 'approx':
                            # approximate solution
                            dec_U = torch.reshape(torch.maximum(1 - dec_lambda2 / (torch.norm(dec_W, p=2, dim=0) + 1e-11), dec_O), (1, -1)) * dec_W
                            paras[idx_paras].data = torch.sign(dec_U) * torch.maximum(torch.abs(dec_U) - dec_lambda1, dec_O)

                        else:
                            # errors
                            print('Error! This %s solver is not implemented!' % args.reg_dec_solver)
                            exit()

                        dec_layer_idx -= 1  # dec_n -> dec_1
                        dec_lambda *= dec_c  # increasing lambda


                    else:
                        # errors
                        print('Error! This %s regularization is not implemented!' % args.reg_dec)
                        exit()


                total_reg += total_reg_enc + total_reg_dec





        scheduler.step()
        total_err = compute_relative_reconstruction_error(x, model, gt)


        loss_history.append((i + 1, total_loss + total_reg, total_loss, total_reg, total_reg_enc, total_reg_dec, total_err))

        if args.reg_enc == 'l2' and args.reg_dec == 'l2':
            enc_wsp, dec_wsp, sum_enc_wsp, sum_dec_wsp, sum_wsp = compute_weight_pseudo_sparsity(model, args.reg_pseudo_thresh)
        else:
            enc_wsp, dec_wsp, sum_enc_wsp, sum_dec_wsp, sum_wsp = compute_weight_sparsity(model)




        if i == 0 or (i + 1) % args.print_every == 0:

            if args.verbose:

                file_writer_content = ""

                # 1. loss
                file_writer_content += ('%d-th epoch: J=%.6f, loss=%.6f, reg=%.6f, enc_reg=%.6f, dec_reg=%.6f. || err=%.6f. ' % loss_history[-1]) + ' || '
                # 2. weight sparsity

                file_writer_content += args.reg_enc + '/enc_w_sp: '
                for wsp_idx, wsp_val in enumerate(enc_wsp):
                    file_writer_content += ('%.2f%%' % (wsp_val * 100)) + ' '
                file_writer_content += ' || ' + args.reg_dec + '/dec_w_sp: '
                for wsp_idx, wsp_val in enumerate(dec_wsp):
                    file_writer_content += ('%.2f%%' % (wsp_val * 100)) + ' '
                file_writer_content += ' || ' + args.reg_dec + '/sum_enc_wsp: '
                for wsp_idx, wsp_val in enumerate(sum_enc_wsp):
                    file_writer_content += ('%.2f%%' % (wsp_val * 100)) + ' '
                file_writer_content += ' || ' + args.reg_dec + '/sum_dec_wsp: '
                for wsp_idx, wsp_val in enumerate(sum_dec_wsp):
                    file_writer_content += ('%.2f%%' % (wsp_val * 100)) + ' '
                file_writer_content += ' || ' + args.reg_dec + '/sum_wsp: '
                for wsp_idx, wsp_val in enumerate(sum_wsp):
                    # if 9 < wsp_val * 100 < 11:
                    file_writer_content += ('%.2f%%' % (wsp_val * 100)) + ' '

                print(file_writer_content)

                file_writer_content += '\n'
                file_writer.write(file_writer_content)


            if logger:
                # 1. Log scalar values (scalar summary)
                info = {'J': loss_history[-1][1], 'loss': loss_history[-1][2], 'reg': loss_history[-1][3], 'ERR': loss_history[-1][6]}

                for tag, value in info.items():
                    logger.scalar_summary(tag, value, i + 1)

                # 2. Log values and gradients of the parameters (histogram summary)
                for tag, value in model.named_parameters():
                    tag = tag.replace('.', '/')
                    logger.histo_summary(tag, value.data.cpu().numpy(), i + 1)
                    logger.histo_summary(tag + '/grad', value.grad.data.cpu().numpy(), i + 1)

                # 3. Log images (image summary)
                with torch.no_grad():
                    x_ = torch.transpose(model(x).detach().cpu(), 1, 0).numpy()
                    
                    x_ = imgrid(normalize(x_), 10, 10, 56, 46, 2)

                info = {'reconstruction_images': [x_]}

                for tag, images in info.items():
                    logger.image_summary(tag, images, i + 1)


        if i == args.epoch - 1:  # last epoch
            save_model_weights(model, args.save_dir)

    for wsp_idx, wsp_val in enumerate(sum_wsp):
        print("wsp_val",wsp_val)

    file_writer.close()
    return enumerate(sum_wsp)
