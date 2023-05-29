#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=500, help="rounds of training")
    parser.add_argument('--test_freq', type=int, default=1, help="frequency of test")
    parser.add_argument('--num_users', type=int, default=100, help="number of users")
    parser.add_argument('--frac', type=float, default=0.1, help="the fraction of clients")
    parser.add_argument('--local_ep', type=int, default=2, help="the number of local epochs")
    parser.add_argument('--last_local_ep', type=int, default=10, help="the number of local epochs of last")
    parser.add_argument('--local_rep_ep', type=int, default=1, help="the number of local epochs of Fed_Rep's feature") #ten local epochs to train the local head, followed by one or five epochs for the representation
    parser.add_argument('--local_bs', type=int, default=48, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=10, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--globallr', type=float, default=1, help="Global learning rate")
    parser.add_argument('--momentum', type=float, default=0, help="local SGD momentum (default: 0.0)")
    parser.add_argument('--weigh_delay', type=float, default=0, help="local SGD weigh_delay")
    parser.add_argument('--mu', type=float, default=0, help='the value of Ditto')
    parser.add_argument('--mu1', type=float, default=0, help='the value_L1 of Factorization')
    parser.add_argument('--tau1', type=float, default=0, help='the value_cosine similarity of Factorization')
    parser.add_argument('--lr_decay', type=float, default=0.997, help='the value of lr_decay')
    parser.add_argument('--alpha', default=0, type=float, help='the value of alpha for Fed_VIB')
    parser.add_argument('--beta', default=0.001, type=float, help='the value of beta for Fed_VIB')
    parser.add_argument('--sync', type=str, default='True', help='If the model is synchronized for FedCR')
    parser.add_argument('--beta_PAC', default=1, type=float, help='the value of beta for FedPAC')
    parser.add_argument('--beta2', default=0, type=float, help='the value of beta2 for Z')
    parser.add_argument('--dimZ', default = 256, type=int, help='dimension of encoding Z in Fed_VIB')
    parser.add_argument('--dimZ_PAC', default=1024, type=int, help='dimension of Z in Factorized')
    parser.add_argument('--CMI', default=0.001, type=float, help='the value of CMI in FedSR')
    parser.add_argument('--L2R', default=0.001, type=float, help='the value of L2R in FedSR')
    parser.add_argument('--num_avg_train', default = 15, type=int, help='the number of samples when\
            perform multi-shot train')
    parser.add_argument('--num_avg', default = 30, type=int, help='the number of samples when\
            perform multi-shot prediction')

    parser.add_argument('--filepath', type=str, default='filepath', help='whether error accumulation or not')

    # model arguments
    parser.add_argument('--method', type=str, default='fedCR', help='method name')

    # other arguments
    parser.add_argument('--dataset', type=str, default='CIFAR10', help="name of dataset")
    parser.add_argument('--frac_data', type=float, default=0.7, help="fraction of frac_data")
    parser.add_argument('--rule', type=str, default='noniid', help='whether noniid or Dirichet')
    parser.add_argument('--class_main', type=int, default=5, help='the value of class_main for noniid')
    parser.add_argument('--dir_a', default=0.5, type=float, help='the value of dir_a for dirichlet')
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--seed', type=int, default=23, help='random seed (default: 23)')
    args = parser.parse_args()
    return args


