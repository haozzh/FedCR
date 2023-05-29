#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import ssl

import copy
import itertools
import random
import torch
import numpy as np
from utils.options import args_parser
from utils.seed import setup_seed
from utils.logg import get_logger
from models.Nets import client_model
from models.Nets_VIB import client_model_VIB
from utils.utils_dataset import DatasetObject
from models.distributed_training_utils import Client, Server
torch.set_printoptions(
    precision=8,
    threshold=1000,
    edgeitems=3,
    linewidth=150, 
    profile=None,
    sci_mode=False  
)
if __name__ == '__main__':

    ssl._create_default_https_context = ssl._create_unverified_context
    # parse args
    args = args_parser()

    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    setup_seed(args.seed)


    data_path = 'Folder/'
    data_obj = DatasetObject(dataset=args.dataset, n_client=args.num_users, seed=args.seed, rule=args.rule, class_main=args.class_main, data_path=data_path, frac_data=args.frac_data, dir_alpha=args.dir_a)

    clnt_x = data_obj.clnt_x;
    clnt_y = data_obj.clnt_y;
    tst_x = data_obj.tst_x;
    tst_y = data_obj.tst_y

    # build model
    if args.method == 'FedCR':
            net_glob = client_model_VIB(args, args.dimZ, args.alpha, args.dataset).to(args.device)
    else:
        if args.dataset == 'CIFAR100':
            net_glob = client_model('cifar100_LeNet').to(args.device)
        elif args.dataset == 'CIFAR10':
            net_glob = client_model('cifar10_LeNet').to(args.device)
        elif args.dataset == 'EMNIST':
            net_glob = client_model('emnist_NN', [1 * 28 * 28, 10]).to(args.device)
        elif args.dataset == 'FMNIST':
            net_glob = client_model('FMNIST_CNN', [1 * 28 * 28, 10]).to(args.device)
        else:
            exit('Error: unrecognized model')

    total_num_layers = len(net_glob.state_dict().keys())
    net_keys = [*net_glob.state_dict().keys()]

    if args.method == 'fedrep' or args.method == 'fedper' or args.method == 'fedbabu':
        if 'CIFAR100' in  args.dataset:
            w_glob_keys = [net_glob.weight_keys[i] for i in [0, 1, 2, 3]]
        elif 'CIFAR10' in args.dataset:
            w_glob_keys = [net_glob.weight_keys[i] for i in [0, 1, 2, 3]]
        elif 'EMNIST' in args.dataset:
            w_glob_keys = [net_glob.weight_keys[i] for i in [0, 1]]
        elif 'FMNIST' in args.dataset:
            w_glob_keys = [net_glob.weight_keys[i] for i in [0, 1, 2, 3]]
        else:
            exit('Error: unrecognized data1')
    elif args.method == 'lg':
        if 'CIFAR100' in  args.dataset:
            w_glob_keys = [net_glob.weight_keys[i] for i in [3, 4]]
        elif 'CIFAR10' in args.dataset:
            w_glob_keys = [net_glob.weight_keys[i] for i in [3, 4]]
        elif 'EMNIST' in args.dataset:
            w_glob_keys = [net_glob.weight_keys[i] for i in [1, 2]]
        elif 'FMNIST' in args.dataset:
            w_glob_keys = [net_glob.weight_keys[i] for i in [3, 4]]
        else:
            exit('Error: unrecognized data2')
    elif args.method == 'FedCR':
        if 'CIFAR100' in args.dataset:
            w_glob_keys = [net_glob.weight_keys[i] for i in [0, 1, 2, 3, 4]]
        elif 'CIFAR10' in args.dataset:
            w_glob_keys = [net_glob.weight_keys[i] for i in [0, 1, 2, 3, 4]]
        elif 'EMNIST' in args.dataset:
            w_glob_keys = [net_glob.weight_keys[i] for i in [0, 1, 2]]
        elif 'FMNIST' in args.dataset:
            w_glob_keys = [net_glob.weight_keys[i] for i in [0, 1, 2, 3, 4]]
        else:
            exit('Error: unrecognized data3')
    elif args.method == 'fedavg' or args.method == 'ditto' or args.method == 'maml':
        w_glob_keys = []
    else:
        exit('Error: unrecognized data4')

    w_glob_keys = list(itertools.chain.from_iterable(w_glob_keys))

    clients = [Client(model=copy.deepcopy(net_glob).to(args.device), args=args, trn_x=data_obj.clnt_x[i],
                      trn_y=data_obj.clnt_y[i], tst_x=data_obj.tst_x[i], tst_y=data_obj.tst_y[i], n_cls = data_obj.n_cls, dataset_name=data_obj.dataset, id_num=i) for i in range(args.num_users)]

    server = Server(model = (net_glob).to(args.device), args = args, n_cls = data_obj.n_cls)

    logger = get_logger(args.filepath)
    logger.info('--------args----------')
    for k in list(vars(args).keys()):
        logger.info('%s: %s' % (k, vars(args)[k]))
    logger.info('--------args----------\n')
    logger.info('total_num_layers')
    logger.info(total_num_layers)
    logger.info('net_keys')
    logger.info(net_keys)
    logger.info('w_glob_keys')
    logger.info(w_glob_keys)

    logger.info('start training!')

    results_loss = [];
    results_acc = []

    for client in clients:

        client.compute_weight_update(w_glob_keys, server, last=False)

        results_test, loss_test1 = client.evaluate(data_x=client.tst_x, data_y=client.tst_y,
                                                   dataset_name=data_obj.dataset)


    results_loss.append(loss_test1)
    results_acc.append(results_test)

    results_loss = np.mean(results_loss)
    results_acc = np.mean(results_acc)

    logger.info('Final Epoch:[{}]\tlr =\t{:.5f}\tloss=\t{:.5f}\tacc_test=\t{:.5f}'.format(iter, args.lr, results_loss, results_acc))

    args.lr = args.lr * (args.lr_decay)

    logger.info('finish training!')






