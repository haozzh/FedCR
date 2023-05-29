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
from models.NetsSR import Model_CMI
from models.Nets_VIB import client_model_VIB
from utils.utils_dataset import DatasetObject
from models.distributed_training_utilsSR import Client, Server
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


    net_glob = Model_CMI(args, args.dimZ, args.alpha, args.dataset).to(args.device)


    total_num_layers = len(net_glob.state_dict().keys())
    net_keys = [*net_glob.state_dict().keys()]

    if args.method == 'fedSR':
        w_glob_keys = []

    w_glob_keys = list(itertools.chain.from_iterable(w_glob_keys))

    clients = [Client(model=Model_CMI(args, args.dimZ, args.alpha, args.dataset).to(args.device), args=args, trn_x=data_obj.clnt_x[i],
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

    for iter in range(args.epochs + 1):
        net_glob.train()

        m = max(int(args.frac * args.num_users), 1)
        if iter == args.epochs:
            m = args.num_users
        participating_clients = random.sample(clients, m)

        last = iter == args.epochs

        for client in participating_clients:


            client.synchronize_with_server(server, w_glob_keys)

            client.compute_weight_update(w_glob_keys, server, last)



        server.aggregate_weight_updates(clients=participating_clients, iter=iter)


#-----------------------------------------------test--------------------------------------------------------------------

#-----------------------------------------------test--------------------------------------------------------------------

        if iter % args.test_freq==args.test_freq-1 or iter>=args.epochs-10:
            results_loss =[]; results_acc = []
            results_loss_last = []; results_acc_last = []
            for client in clients:

                results_test, loss_test1 = client.evaluate(data_x=client.tst_x, data_y=client.tst_y,
                                                                         dataset_name=data_obj.dataset)
                if last:
                    results_test_last, loss_test1_last = client.evaluate(data_x=client.tst_x, data_y=client.tst_y,
                                                                         dataset_name=data_obj.dataset)

                results_loss.append(loss_test1)
                results_acc.append(results_test)

                if last and args.method == 'fedSR':
                    results_loss_last.append(loss_test1_last)
                    results_acc_last.append(results_test_last)

            results_loss = np.mean(results_loss)
            results_acc = np.mean(results_acc)
            if last:
                logger.info('Final Epoch:[{}]\tlr =\t{:.5f}\tloss=\t{:.5f}\tacc_test=\t{:.5f}'.
                            format(iter, args.lr, results_loss, results_acc))
            else:
                logger.info('Epoch:[{}]\tlr =\t{:.5f}\tloss=\t{:.5f}\tacc_test=\t{:.5f}'.
                            format(iter, args.lr, results_loss, results_acc))

            if last and args.method == 'fedSR':
                results_loss_last = np.mean(results_loss_last)
                results_acc_last = np.mean(results_acc_last)
                logger.info('Final FT Epoch:[{}]\tlr =\t{:.5f}\tloss=\t{:.5f}\tacc_test=\t{:.5f}'.
                            format(iter, args.lr, results_loss_last, results_acc_last))

        args.lr = args.lr * (args.lr_decay)

    logger.info('finish training!')





