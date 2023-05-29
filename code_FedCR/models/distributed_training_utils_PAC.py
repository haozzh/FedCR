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



def personalized_classifier(args, client, clients):

    #penalty method
    class Model_quadratic_programming():
        def __init__(self):
            self.coe_client = torch.rand(int(len(clients)) -1, dtype=torch.float32, device=args.device, requires_grad=True)
            self.coe_client_last = 1 - torch.sum(self.coe_client)

        def final_solve(self):

            A3 = 0; A3_ = 0; A4 = 0;
            for i in range(int(len(clients))):
                for j in range(int(len(clients))):

                    C1 = client.P_y * torch.squeeze(client.F_x) - clients[i].P_y * torch.squeeze(clients[i].F_x)
                    A2 = ( C1 @ C1.t()).trace()

                    if j < int(len(clients)) -1:
                        A3 = A3 + self.coe_client[j] * A2
                    else:
                        A3 = A3 + self.coe_client_last * A2

                if i < int(len(clients)) - 1:
                    A3_ = A3_ + self.coe_client[i] * A3
                else:
                    A3_ = A3_ + self.coe_client_last * A3

                if i < int(len(clients)) - 1:
                    A4 = A4 + self.coe_client[i] * clients[i].Var_
                else:
                    A4 = A4 + self.coe_client_last * clients[i].Var_

            # penalty method
            final_solve = torch.sum(A4 + A3_ + F.relu(-self.coe_client)) + F.relu(-self.coe_client_last) # coe_client > 0 penalty method
            #final_solve = torch.sum(A5)

            return final_solve

    Model_quadratic = Model_quadratic_programming()

    lr = 0.1
    for i in range(50):
        loss = Model_quadratic.final_solve()
        loss.backward()

        Model_quadratic.coe_client.data.sub_(lr * Model_quadratic.coe_client.grad)

        Model_quadratic.coe_client.grad.zero_()
        Model_quadratic.coe_client_last = 1 - torch.sum(Model_quadratic.coe_client)

        coe_client_all = torch.rand(int(len(clients)), dtype=torch.float32, device=args.device, requires_grad=False)
        coe_client_all[:-1] = Model_quadratic.coe_client.clone().detach()
        coe_client_all[-1:] = Model_quadratic.coe_client_last.clone().detach()

    return coe_client_all




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

        self.lo_fecture_output = torch.zeros(self.n_cls, 1, self.args.dimZ_PAC, dtype=torch.float32, device=self.args.device)

        self.P_y = torch.zeros(self.n_cls, 1, dtype=torch.float32, device=self.args.device)
        self.F_x = torch.zeros(self.n_cls, 1, self.args.dimZ_PAC, dtype=torch.float32, device=self.args.device)
        self.Var = torch.zeros(self.n_cls, 1, dtype=torch.float32, device=self.args.device)
        self.Var_ = torch.zeros(self.n_cls, 1, dtype=torch.float32, device=self.args.device)

        self.state_params_diff = 0.0
        self.train_loss = 0.0
        self.n_par = get_mdl_params([self.model]).shape[1]


    def synchronize_with_server(self, server, w_glob_keys):
        # W_client = W_server

        if self.args.method != 'fedavg':
            for name in self.W:
                if name in w_glob_keys:
                    self.W[name].data = server.W[name].data.clone()
        else:
            self.model = copy.deepcopy(server.model)
            self.W = {name: value for name, value in self.model.named_parameters()}


    def train_cnn(self, w_glob_keys, server, last):

        self.model.train()

        local_eps = self.args.local_ep
        if last:
            if self.args.method =='fedavg':
                local_eps= self.args.last_local_ep
                if 'CIFAR100' in self.args.dataset:
                    w_glob_keys = [self.model.weight_keys[i] for i in [0, 1, 2, 3]]
                elif 'CIFAR10' in self.args.dataset:
                    w_glob_keys = [self.model.weight_keys[i] for i in [0, 1, 2, 3]]
                elif 'MNIST' in self.args.dataset:
                    w_glob_keys = [self.model.weight_keys[i] for i in [0, 1]]
                elif 'FMNIST' in self.args.dataset:
                    w_glob_keys = [self.model.weight_keys[i] for i in [0, 1, 2, 3]]
                w_glob_keys = list(itertools.chain.from_iterable(w_glob_keys))
            else:
                local_eps = max(self.args.last_local_ep, self.args.local_ep - self.args.local_rep_ep)

        # all other methods update all parameters simultaneously
        else:
            for name, param in self.model.named_parameters():
                param.requires_grad = True


        # train and update
        epoch_loss = []


        for iter in range(local_eps):
            flag=0
            head_eps = local_eps - self.args.local_rep_ep
                # for FedRep, first do local epochs for the head
            if (iter < head_eps and self.args.method == 'fedPAC') or last:
                for name, param in self.model.named_parameters():
                    if name in w_glob_keys:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True

            # then do local epochs for the representation
            elif (iter >= head_eps and self.args.method == 'fedPAC') and not last:
                flag =1
                for name, param in self.model.named_parameters():
                    if name in w_glob_keys:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False

            trn_gen_iter = self.trn_gen.__iter__()
            batch_loss = []

            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=self.args.momentum,
                                        weight_decay=self.args.weigh_delay)

            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1)


            for i in range(self.local_epoch):

                images, labels = trn_gen_iter.__next__()
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                labels = labels.reshape(-1).long()

                optimizer.zero_grad()
                log_probs, fecture_output = self.model(images)


                loss_f_i = self.loss_func(log_probs, labels)

                if flag==1:
                    for cls in range(len(labels)):
                        if cls == 0:
                            dir_g_fecture_output = server.fecture_output[labels[cls]].clone().detach()
                        else:
                            dir_g_fecture_output = torch.cat((dir_g_fecture_output, server.fecture_output[labels[cls]].clone().detach()), 0).clone().detach()

                    R_i = torch.norm(fecture_output - dir_g_fecture_output) / len(labels)
                else:
                    R_i =0

                loss = loss_f_i + self.args.beta_PAC * R_i

                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=max_norm)
                optimizer.step()
                batch_loss.append(loss.item())

            scheduler.step()
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return sum(epoch_loss) / len(epoch_loss)


    def local_feature(self):

        trn_gen_iter = self.trn_gen.__iter__()
        batch_loss = []

        for i in range(self.local_epoch):
            images, labels = trn_gen_iter.__next__()
            images, labels = images.to(self.args.device), labels.to(self.args.device)
            labels = labels.reshape(-1).long()

            log_probs, fecture_output = self.model(images)

            for cls in range(len(labels)):
                if self.lo_fecture_output[labels[cls]].equal(torch.zeros(1, self.args.dimZ_PAC, dtype=torch.float32, device=self.args.device)):
                    self.lo_fecture_output[labels[cls]] = fecture_output[cls].clone().detach()

                    self.Var[labels[cls]] = (fecture_output[cls].t().clone().detach()) @(fecture_output[cls].clone().detach())
                else:
                    self.lo_fecture_output[labels[cls]] = (fecture_output[cls].clone().detach() + self.lo_fecture_output[labels[cls]]) / 2
                    self.Var[labels[cls]] = (self.Var[labels[cls]] + (fecture_output[cls].clone().detach()) @(fecture_output[cls].clone().detach()) ) / 2

                self.P_y[labels[cls]] = self.P_y[labels[cls]] + 1

        self.P_y = self.P_y / torch.sum(self.P_y)
        self.F_x = self.lo_fecture_output.clone().detach()
        self.Var_ = torch.sum(self.P_y * self.Var.trace() - (self.P_y * torch.squeeze(self.F_x)) ** 2)

    def compute_weight_update(self, w_glob_keys, server, last=False):

        # Training mode
        self.model.train()

        # W = SGD(W, D)
        self.train_loss = self.train_cnn(w_glob_keys, server, last)


    @torch.no_grad()
    def evaluate(self, data_x, data_y, dataset_name):
        self.model.eval()
        # testing
        test_loss = 0
        acc_overall = 0
        n_tst = data_x.shape[0]
        tst_gen = DataLoader(Dataset(data_x, data_y, dataset_name=dataset_name), batch_size=self.args.bs, shuffle=False)
        tst_gen_iter = tst_gen.__iter__()
        for i in range(int(np.ceil(n_tst / self.args.bs))):
            data, target = tst_gen_iter.__next__()
            data, target = data.to(self.args.device), target.to(self.args.device)
            log_probs, fecture_output = self.model(data)
            # sum up batch loss
            test_loss += nn.CrossEntropyLoss(reduction='sum')(log_probs, target.reshape(-1).long()).item()
            # get the index of the max log-probability
            log_probs = log_probs.cpu().detach().numpy()
            log_probs = np.argmax(log_probs, axis=1).reshape(-1)
            target = target.cpu().numpy().reshape(-1).astype(np.int32)
            batch_correct = np.sum(log_probs == target)
            acc_overall += batch_correct
            '''
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
            '''

        test_loss /= n_tst
        accuracy = 100.00 * acc_overall / n_tst
        return accuracy, test_loss


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

        self.fecture_output = torch.zeros(self.n_cls, 1, self.args.dimZ_PAC, dtype=torch.float32, device=self.args.device)


    def aggregate_weight_updates(self, clients, iter, aggregation="mean"):

        # Warning: Note that K is different for unbalanced dataset
        self.local_epoch = clients[0].local_epoch
        # dW = aggregate(dW_i, i=1,..,n)
        if aggregation == "mean":
            average(target=self.W, sources=[client.W for client in clients])


    def global_feature_centroids(self, clients):


        for cls in range(self.n_cls):

            clients_all_fecture_output = True

            for i in range(len(clients)):

                if clients[i].lo_fecture_output[cls].equal(torch.zeros(1, self.args.dimZ_PAC, dtype=torch.float32, device=self.args.device)):
                    pass
                elif isinstance(clients_all_fecture_output, bool):
                    clients_all_fecture_output = clients[i].lo_fecture_output[cls].clone().detach()

                else:
                    clients_all_fecture_output = torch.cat((clients_all_fecture_output, clients[i].lo_fecture_output[cls].clone().detach()), 0).clone().detach()

            if not isinstance(clients_all_fecture_output, bool):

                self.fecture_output[cls] = torch.mean(clients_all_fecture_output)


    def Get_classifier(self, clients, w_glob_keys):
        for client in clients:
            modify = personalized_classifier(args = self.args, client= client, clients=clients)
            for name in self.W:
                if name not in w_glob_keys:
                    client.W[name].data = torch.mean(
                        torch.stack([m * source.W[name].data for source, m in zip(clients, modify)]),
                        dim=0).clone()

    @torch.no_grad()
    def evaluate(self, data_x, data_y, dataset_name):
        self.model.eval()
        # testing
        test_loss = 0
        acc_overall = 0
        n_tst = data_x.shape[0]
        tst_gen = DataLoader(Dataset(data_x, data_y, dataset_name=dataset_name), batch_size=self.args.bs, shuffle=False)
        tst_gen_iter = tst_gen.__iter__()
        for i in range(int(np.ceil(n_tst / self.args.bs))):
            data, target = tst_gen_iter.__next__()
            data, target = data.to(self.args.device), target.to(self.args.device)
            log_probs = self.model(data)
            # sum up batch loss
            test_loss += nn.CrossEntropyLoss(reduction='sum')(log_probs, target.reshape(-1).long()).item()
            # get the index of the max log-probability
            log_probs = log_probs.cpu().detach().numpy()
            log_probs = np.argmax(log_probs, axis=1).reshape(-1)
            target = target.cpu().numpy().reshape(-1).astype(np.int32)
            batch_correct = np.sum(log_probs == target)
            acc_overall += batch_correct
            '''
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
            '''

        test_loss /= n_tst
        accuracy = 100.00 * acc_overall / n_tst
        return accuracy, test_loss



