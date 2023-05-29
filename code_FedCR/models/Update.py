#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
import copy
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

    def train(self, net):
        net.train()
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum, weight_decay=self.args.weigh_delay)
        net_pre = copy.deepcopy(net)
        state_pre = [parameter for parameter in net_pre.parameters()]
        # train and update
        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)

                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        state_now = [parameter for parameter in net.parameters()]
        grads = [torch.zeros_like(param) for param in state_now]
        for state_now_, state_pre_, grad in zip(state_now, state_pre, grads):
            grad.data[:] = state_now_ - state_pre_
        return grads, sum(epoch_loss) / len(epoch_loss)

