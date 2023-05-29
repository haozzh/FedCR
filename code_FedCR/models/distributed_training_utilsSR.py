import torch
import copy
import math
from torch import nn, autograd
import numpy as np
from torch.utils.data import DataLoader
from utils.utils_dataset import Dataset
from models.Nets_VIB import KL_between_normals
import itertools
import torch.nn.functional as F

max_norm = 10

def add(target, source):
    for name in target:
        target[name].data += source[name].data.clone()


def add_mome(target, source, beta_):
    for name in target:
        target[name].data = (beta_ * target[name].data + source[name].data.clone())


def add_mome2(target, source1, source2, beta_1, beta_2):
    for name in target:
        target[name].data = beta_1 * source1[name].data.clone() + beta_2 * source2[name].data.clone()


def add_mome3(target, source1, source2, source3, beta_1, beta_2, beta_3):
    for name in target:
        target[name].data = beta_1 * source1[name].data.clone() + beta_2 * source2[name].data.clone() + beta_3 * source3[name].data.clone()

def add_2(target, source1, source2, beta_1, beta_2):
    for name in target:
        target[name].data += beta_1 * source1[name].data.clone() + beta_2 * source2[name].data.clone()

def scale(target, scaling):
    for name in target:
        target[name].data = scaling * target[name].data.clone()


def scale_ts(target, source, scaling):
    for name in target:
        target[name].data = scaling * source[name].data.clone()


def subtract(target, source):
    for name in target:
        target[name].data -= source[name].data.clone()


def subtract_(target, minuend, subtrahend):
    for name in target:
        target[name].data = minuend[name].data.clone() - subtrahend[name].data.clone()


def average(target, sources):
    for name in target:
        target[name].data = torch.mean(torch.stack([source[name].data for source in sources]), dim=0).clone()


def weighted_average(target, sources, weights):
    for name in target:
        summ = torch.sum(weights)
        n = len(sources)
        modify = [weight / summ * n for weight in weights]
        target[name].data = torch.mean(torch.stack([m * source[name].data for source, m in zip(sources, modify)]),
                                       dim=0).clone()


def computer_norm(source1, source2):
    diff_norm = 0

    for name in source1:
        diff_source = source1[name].data.clone() - source2[name].data.clone()
        diff_norm += torch.pow(torch.norm(diff_source),2)

    return (torch.pow(diff_norm, 0.5))

def majority_vote(target, sources, lr):
    for name in target:
        threshs = torch.stack([torch.max(source[name].data) for source in sources])
        mask = torch.stack([source[name].data.sign() for source in sources]).sum(dim=0).sign()
        target[name].data = (lr * mask).clone()


def get_mdl_params(model_list, n_par=None):
    if n_par == None:
        exp_mdl = model_list[0]
        n_par = 0
        for name, param in exp_mdl.named_parameters():
            n_par += len(param.data.reshape(-1))

    param_mat = np.zeros((len(model_list), n_par)).astype('float32')
    for i, mdl in enumerate(model_list):
        idx = 0
        for name, param in mdl.named_parameters():
            temp = param.data.cpu().numpy().reshape(-1)
            param_mat[i, idx:idx + len(temp)] = temp
            idx += len(temp)
    return np.copy(param_mat)


def get_other_params(model_list, n_par=None):
    if n_par == None:
        exp_mdl = model_list[0]
        n_par = 0
        for name in exp_mdl:
            n_par += len(exp_mdl[name].data.reshape(-1))

    param_mat = np.zeros((len(model_list), n_par)).astype('float32')
    for i, mdl in enumerate(model_list):
        idx = 0
        for name in mdl:
            temp = mdl[name].data.cpu().numpy().reshape(-1)
            param_mat[i, idx:idx + len(temp)] = temp
            idx += len(temp)
    return np.copy(param_mat)






class DistributedTrainingDevice(object):
    '''
  A distributed training device (Client or Server)
  data : a pytorch dataset consisting datapoints (x,y)
  model : a pytorch neural net f mapping x -> f(x)=y_
  hyperparameters : a python dict containing all hyperparameters
  '''

    def __init__(self, model, args):
        self.model = model
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()

class Client(DistributedTrainingDevice):

    def __init__(self, model, args, trn_x, trn_y, tst_x, tst_y, n_cls, dataset_name, id_num=0):
        super().__init__(model, args)

        self.trn_gen = DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name),
                                  batch_size=self.args.local_bs, shuffle=True)

        self.tst_x = tst_x
        self.tst_y = tst_y
        self.n_cls = n_cls
        self.id = id_num
        self.local_epoch = int(np.ceil(trn_x.shape[0] / self.args.local_bs))
        # Parameters
        self.W = {name: value for name, value in self.model.named_parameters()}

        self.state_params_diff = 0.0
        self.train_loss = 0.0
        self.n_par = get_mdl_params([self.model]).shape[1]


    def synchronize_with_server(self, server, w_glob_keys):
        # W_client = W_server

        if self.args.method != 'fedSR':
            for name in self.W:
                if name in w_glob_keys:
                    self.W[name].data = server.W[name].data.clone()

        else:

            self.W = {name: value for name, value in server.model.named_parameters()}


    def train_cnn(self, w_glob_keys, server, last):

        self.model.train()


        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=self.args.momentum,
                                        weight_decay=self.args.weigh_delay)

        #.add_param_group({'params':[self.model.r_mu,self.model.r_sigma,self.model.C]})

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1)

        local_eps = self.args.local_ep
        if last:
            if self.args.method =='fedSR':
                local_eps= self.args.last_local_ep
                if 'CIFAR100' in self.args.dataset:
                    w_glob_keys = [self.model.weight_keys[i] for i in [0, 1, 2, 3]]
                elif 'CIFAR10' in self.args.dataset:
                    w_glob_keys = [self.model.weight_keys[i] for i in [0, 1, 2, 3]]
                elif 'EMNIST' in self.args.dataset:
                    w_glob_keys = [self.model.weight_keys[i] for i in [0, 1]]
                elif 'FMNIST' in self.args.dataset:
                    w_glob_keys = [self.model.weight_keys[i] for i in [0, 1, 2, 3]]
                w_glob_keys = list(itertools.chain.from_iterable(w_glob_keys))

            elif 'maml' in self.args.method:
                local_eps = self.args.last_local_ep / 2
                w_glob_keys = []
            else:
                local_eps = max(self.args.last_local_ep, self.args.local_ep - self.args.local_rep_ep)

        # all other methods update all parameters simultaneously
        else:
            for name, param in self.model.named_parameters():
                param.requires_grad = True

        # train and update
        epoch_loss = []


        self.dir_Z_u = torch.zeros(self.n_cls, 1, self.args.dimZ, dtype=torch.float32, device=self.args.device)
        self.dir_Z_sigma = torch.ones(self.n_cls, 1, self.args.dimZ, dtype = torch.float32, device = self.args.device)

        for iter in range(local_eps):

            if last:
                for name, param in self.model.named_parameters():
                    if name in w_glob_keys:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True

            loss_by_epoch = []
            accuracy_by_epoch = []

            trn_gen_iter = self.trn_gen.__iter__()
            batch_loss = []

            for i in range(self.local_epoch):

                images, labels = trn_gen_iter.__next__()
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                labels = labels.reshape(-1).long()

                batch_size = images.size()[0]

                encoder_Z_distr, decoder_logits, regL2R= self.model(images, self.args.num_avg_train)

                decoder_logits_mean = torch.mean(decoder_logits, dim=0)

                loss = nn.CrossEntropyLoss(reduction='none')
                decoder_logits = decoder_logits.permute(1, 2, 0)
                cross_entropy_loss = loss(decoder_logits, labels[:, None].expand(-1, self.args.num_avg_train))
                # estimate E_{eps in N(0, 1)} [log q(y | z)]
                cross_entropy_loss_montecarlo = torch.mean(cross_entropy_loss, dim=-1)
                minusI_ZY_bound = torch.mean(cross_entropy_loss_montecarlo, dim=0)


                r_sigma_softplus = F.softplus(self.model.r_sigma)


                r_mu = self.model.r_mu[labels]
                r_sigma = r_sigma_softplus[labels]
                z_mu_scaled = encoder_Z_distr[0]*self.model.C
                z_sigma_scaled = encoder_Z_distr[1]*self.model.C
                regCMI = torch.log(r_sigma) - torch.log(z_sigma_scaled) + \
                        (z_sigma_scaled**2+(z_mu_scaled-r_mu)**2)/(2*r_sigma**2) - 0.5

                regCMI = regCMI.sum(1).mean()
                regL2R = regL2R / len(labels)

                total_loss = torch.mean(minusI_ZY_bound)  + self.args.CMI * regCMI + self.args.L2R * regL2R

                prediction = torch.max(decoder_logits_mean, dim=1)[1]
                accuracy = torch.mean((prediction == labels).float())

                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=max_norm)
                optimizer.step()

                loss_by_epoch.append(total_loss.item())
                accuracy_by_epoch.append(accuracy.item())

            scheduler.step()
            epoch_loss.append(sum(loss_by_epoch) / len(loss_by_epoch))



        return sum(epoch_loss) / len(epoch_loss)

    def compute_weight_update(self, w_glob_keys, server, last=False):

        # Training mode
        self.model.train()

        # W = SGD(W, D)
        self.train_loss = self.train_cnn(w_glob_keys, server, last)



    @torch.no_grad()
    def evaluate(self, data_x, data_y, dataset_name):
        self.model.eval()
        # testing
        I_ZX_bound_by_epoch_test = []
        I_ZY_bound_by_epoch_test = []
        loss_by_epoch_test = []
        accuracy_by_epoch_test = []

        n_tst = data_x.shape[0]
        tst_gen = DataLoader(Dataset(data_x, data_y, dataset_name=dataset_name), batch_size=self.args.bs, shuffle=False)
        tst_gen_iter = tst_gen.__iter__()
        for i in range(int(np.ceil(n_tst / self.args.bs))):
            data, target = tst_gen_iter.__next__()
            data, target = data.to(self.args.device), target.to(self.args.device)
            target = target.reshape(-1).long()
            batch_size = data.size()[0]
            prior_Z_distr = torch.zeros(batch_size, self.args.dimZ).to(self.args.device), torch.ones(batch_size,self.args.dimZ).to(self.args.device)
            encoder_Z_distr, decoder_logits, regL2R =  self.model(data, self.args.num_avg)

            decoder_logits_mean = torch.mean(decoder_logits, dim=0)
            loss = nn.CrossEntropyLoss(reduction='none')
            decoder_logits = decoder_logits.permute(1, 2, 0)
            cross_entropy_loss = loss(decoder_logits, target[:, None].expand(-1, self.args.num_avg))

            cross_entropy_loss_montecarlo = torch.mean(cross_entropy_loss, dim=-1)

            I_ZX_bound_test = torch.mean(KL_between_normals(encoder_Z_distr, prior_Z_distr))
            minusI_ZY_bound_test = torch.mean(cross_entropy_loss_montecarlo, dim=0)
            total_loss_test = torch.mean(minusI_ZY_bound_test + self.args.beta * I_ZX_bound_test)

            prediction = torch.max(decoder_logits_mean, dim=1)[1]
            accuracy_test = torch.mean((prediction == target).float())

            I_ZX_bound_by_epoch_test.append(I_ZX_bound_test.item())
            I_ZY_bound_by_epoch_test.append(minusI_ZY_bound_test.item())

            loss_by_epoch_test.append(total_loss_test.item())
            accuracy_by_epoch_test.append(accuracy_test.item())

        I_ZX = np.mean(I_ZX_bound_by_epoch_test)
        I_ZY = np.mean(I_ZY_bound_by_epoch_test)
        loss_test = np.mean(loss_by_epoch_test)
        accuracy_test = np.mean(accuracy_by_epoch_test)
        accuracy_test = 100.00 * accuracy_test
        return accuracy_test, loss_test

'''
------------------------------------------------------------------------------------------------------------------------

------------------------------------------------------------------------------------------------------------------------
'''


class Server(DistributedTrainingDevice):

    def __init__(self, model, args, n_cls):
        super().__init__(model, args)

        # Parameters
        self.W = {name: value for name, value in self.model.named_parameters()}
        self.local_epoch = 0
        self.n_cls = n_cls
        if self.args.method == 'FedCR':
            self.dir_global_Z_u = torch.zeros(self.n_cls, 1, self.args.dimZ, dtype=torch.float32, device=self.args.device)
            self.dir_global_Z_sigma = torch.ones(self.n_cls, 1, self.args.dimZ, dtype = torch.float32, device = self.args.device)


    def aggregate_weight_updates(self, clients, iter, aggregation="mean"):

        # Warning: Note that K is different for unbalanced dataset
        self.local_epoch = clients[0].local_epoch
        # dW = aggregate(dW_i, i=1,..,n)
        if aggregation == "mean":
            average(target=self.W, sources=[client.W for client in clients])





    @torch.no_grad()
    def evaluate(self, data_x, data_y, dataset_name):
        self.model.eval()
        # testing
        I_ZX_bound_by_epoch_test = []
        I_ZY_bound_by_epoch_test = []
        loss_by_epoch_test = []
        accuracy_by_epoch_test = []

        n_tst = data_x.shape[0]
        tst_gen = DataLoader(Dataset(data_x, data_y, dataset_name=dataset_name), batch_size=self.args.bs, shuffle=False)
        tst_gen_iter = tst_gen.__iter__()
        for i in range(int(np.ceil(n_tst / self.args.bs))):
            data, target = tst_gen_iter.__next__()
            data, target = data.to(self.args.device), target.to(self.args.device)
            target = target.reshape(-1).long()
            batch_size = data.size()[0]
            prior_Z_distr = torch.zeros(batch_size, self.args.dimZ).to(self.args.device), torch.ones(batch_size,self.args.dimZ).to(self.args.device)
            encoder_Z_distr, decoder_logit, regL2R =  self.model(data, self.args.num_avg)

            decoder_logits_mean = torch.mean(decoder_logits, dim=0)
            loss = nn.CrossEntropyLoss(reduction='none')
            decoder_logits = decoder_logits.permute(1, 2, 0)
            cross_entropy_loss = loss(decoder_logits, target[:, None].expand(-1, self.args.num_avg))

            cross_entropy_loss_montecarlo = torch.mean(cross_entropy_loss, dim=-1)

            I_ZX_bound_test = torch.mean(KL_between_normals(encoder_Z_distr, prior_Z_distr))
            minusI_ZY_bound_test = torch.mean(cross_entropy_loss_montecarlo, dim=0)
            total_loss_test = torch.mean(minusI_ZY_bound_test + self.args.beta * I_ZX_bound_test)

            prediction = torch.max(decoder_logits_mean, dim=1)[1]
            accuracy_test = torch.mean((prediction == target).float())

            I_ZX_bound_by_epoch_test.append(I_ZX_bound_test.item())
            I_ZY_bound_by_epoch_test.append(minusI_ZY_bound_test.item())

            loss_by_epoch_test.append(total_loss_test.item())
            accuracy_by_epoch_test.append(accuracy_test.item())

        I_ZX = np.mean(I_ZX_bound_by_epoch_test)
        I_ZY = np.mean(I_ZY_bound_by_epoch_test)
        loss_test = np.mean(loss_by_epoch_test)
        accuracy_test = np.mean(accuracy_by_epoch_test)
        accuracy_test = 100.00 * accuracy_test
        return accuracy_test, loss_test



