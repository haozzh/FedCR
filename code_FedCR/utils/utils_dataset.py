from scipy import io
import numpy as np
import torchvision
from torchvision import transforms
import torch
from torch.utils import data
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import ssl

ssl._create_default_https_context = ssl._create_unverified_context


class DatasetObject:
    def __init__(self, dataset, n_client, seed, rule, class_main, data_path, frac_data=0.7, dir_alpha=0):
        self.dataset = dataset
        self.n_client = n_client
        self.seed = seed
        self.rule = rule
        self.frac_data = frac_data
        self.dir_alpha = dir_alpha
        self.class_main = class_main

        self.name = "Data%s_nclient%d_seed%d_rule%s_alpha%s_class_main%d_frac_data%s" % (
        self.dataset, self.n_client, self.seed, self.rule, self.dir_alpha, self.class_main, self.frac_data)
        self.data_path = data_path
        self.set_data()

    def set_data(self):
        # Prepare data if not ready
        if not os.path.exists('Data/%s' % (self.name)):
            # Get Raw data
            if self.dataset == 'MNIST':
                transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
                trnset = torchvision.datasets.MNIST(root='Data/', train=True, download=True, transform=transform)
                tstset = torchvision.datasets.MNIST(root='Data/', train=False, download=True, transform=transform)

                trn_load = torch.utils.data.DataLoader(trnset, batch_size=60000, shuffle=False, num_workers=1)
                tst_load = torch.utils.data.DataLoader(tstset, batch_size=10000, shuffle=False, num_workers=1)
                self.channels = 1;
                self.width = 28;
                self.height = 28;
                self.n_cls = 10;

            if self.dataset == 'FMNIST':
                transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
                trnset = torchvision.datasets.FashionMNIST(root='Data/', train=True, download=True, transform=transform)
                tstset = torchvision.datasets.FashionMNIST(root='Data/', train=False, download=True,
                                                           transform=transform)

                trn_load = torch.utils.data.DataLoader(trnset, batch_size=60000, shuffle=False, num_workers=1)
                tst_load = torch.utils.data.DataLoader(tstset, batch_size=10000, shuffle=False, num_workers=1)
                self.channels = 1;
                self.width = 28;
                self.height = 28;
                self.n_cls = 10;

            if self.dataset == 'CIFAR10':
                transform = transforms.Compose([transforms.ToTensor()])

                trnset = torchvision.datasets.CIFAR10(root='Data/', train=True, download=True, transform=transform)
                tstset = torchvision.datasets.CIFAR10(root='Data/', train=False, download=True, transform=transform)

                trn_load = torch.utils.data.DataLoader(trnset, batch_size=50000, shuffle=False, num_workers=1)
                tst_load = torch.utils.data.DataLoader(tstset, batch_size=10000, shuffle=False, num_workers=1)
                self.channels = 3;
                self.width = 32;
                self.height = 32;
                self.n_cls = 10;

            if self.dataset == 'CIFAR100':
                transform = transforms.Compose([transforms.ToTensor()])

                trnset = torchvision.datasets.CIFAR100(root='Data/', train=True, download=True, transform=transform)
                tstset = torchvision.datasets.CIFAR100(root='Data/', train=False, download=True, transform=transform)

                trn_load = torch.utils.data.DataLoader(trnset, batch_size=50000, shuffle=False, num_workers=1)
                tst_load = torch.utils.data.DataLoader(tstset, batch_size=10000, shuffle=False, num_workers=1)
                self.channels = 3;
                self.width = 32;
                self.height = 32;
                self.n_cls = 100;

            if self.dataset == 'MNIST' or self.dataset == 'FMNIST' or self.dataset == 'CIFAR10' or self.dataset == 'CIFAR100':
                trn_itr = trn_load.__iter__()
                tst_itr = tst_load.__iter__()
                # labels are of shape (n_data,)
                trn_x, trn_y = trn_itr.__next__()
                tst_x, tst_y = tst_itr.__next__()

                trn_x = trn_x.numpy();
                trn_y = trn_y.numpy().reshape(-1, 1)
                tst_x = tst_x.numpy();
                tst_y = tst_y.numpy().reshape(-1, 1)

                concat_datasets_x = np.concatenate((trn_x, tst_x), axis=0)
                concat_datasets_y = np.concatenate((trn_y, tst_y), axis=0)

                self.trn_x = trn_x
                self.trn_y = trn_y
                self.tst_x = tst_x
                self.tst_y = tst_y

            if self.dataset == 'EMNIST':
                emnist = io.loadmat("Data/Raw/matlab/emnist-letters.mat")
                # load training dataset
                x_train = emnist["dataset"][0][0][0][0][0][0]
                x_train = x_train.astype(np.float32)

                # load training labels
                y_train = emnist["dataset"][0][0][0][0][0][1] - 1 # make first class 0

                # take first 10 classes of letters
                trn_idx = np.where(y_train < 10)[0]

                y_train = y_train[trn_idx]
                x_train = x_train[trn_idx]

                mean_x = np.mean(x_train)
                std_x = np.std(x_train)

                # load test dataset
                x_test = emnist["dataset"][0][0][1][0][0][0]
                x_test = x_test.astype(np.float32)

                # load test labels
                y_test = emnist["dataset"][0][0][1][0][0][1] - 1 # make first class 0

                tst_idx = np.where(y_test < 10)[0]

                y_test = y_test[tst_idx]
                x_test = x_test[tst_idx]

                x_train = x_train.reshape((-1, 1, 28, 28))
                x_test  = x_test.reshape((-1, 1, 28, 28))

                # normalise train and test features

                trn_x = (x_train - mean_x) / std_x
                trn_y = y_train

                tst_x = (x_test  - mean_x) / std_x
                tst_y = y_test

                self.channels = 1; self.width = 28; self.height = 28; self.n_cls = 10;
                concat_datasets_x = np.concatenate((trn_x, tst_x), axis=0)
                concat_datasets_y = np.concatenate((trn_y, tst_y), axis=0)

                self.trn_x = trn_x
                self.trn_y = trn_y
                self.tst_x = tst_x
                self.tst_y = tst_y


            # Shuffle Data
            np.random.seed(self.seed)
            rand_perm = np.random.permutation(len(concat_datasets_y))
            concat_datasets_x = concat_datasets_x[rand_perm]
            concat_datasets_y = concat_datasets_y[rand_perm]

            assert len(concat_datasets_y) % self.n_client == 0
            n_data_per_clnt = int((len(concat_datasets_y)) / self.n_client)
            clnt_data_list = np.ones(self.n_client).astype(int) * n_data_per_clnt

            n_data_per_clnt_train = int(n_data_per_clnt * self.frac_data)
            n_data_per_clnt_tst = n_data_per_clnt - n_data_per_clnt_train
            clnt_data_list_train = np.ones(self.n_client).astype(int) * n_data_per_clnt_train
            clnt_data_list_tst = np.ones(self.n_client).astype(int) * n_data_per_clnt_tst
            ###

            cls_per_client = self.class_main
            n_cls = self.n_cls
            n_client = self.n_client

            # Distribute training datapoints
            idx_list = [np.where(concat_datasets_y == i)[0] for i in range(self.n_cls)]
            idx_count_list = [0 for i in range(self.n_cls)]
            cls_amount = np.asarray([len(idx_list[i]) for i in range(self.n_cls)])
            n_data = np.sum(cls_amount)
            total_clnt_data_list = np.asarray([0 for i in range(n_client)])
            clnt_cls_idx = [[[] for kk in range(n_cls)] for jj in range(n_client)]  # Store the indeces of data points


            if self.rule == 'Dirichlet':
                cls_priors = np.random.dirichlet(alpha=[self.dir_alpha] * self.n_cls, size=self.n_client)
                prior_cumsum = np.cumsum(cls_priors, axis=1)

                concat_clnt_x = np.asarray(
                    [np.zeros((clnt_data_list[clnt__], self.channels, self.height, self.width)).astype(np.float32) for
                     clnt__ in range(self.n_client)])
                concat_clnt_y = np.asarray(
                    [np.zeros((clnt_data_list[clnt__], 1)).astype(np.int64) for clnt__ in range(self.n_client)])

                while (np.sum(clnt_data_list) != 0):
                    curr_clnt = np.random.randint(self.n_client)
                    # If current node is full resample a client
                    if clnt_data_list[curr_clnt] <= 0:
                        continue
                    clnt_data_list[curr_clnt] -= 1
                    curr_prior = prior_cumsum[curr_clnt]
                    while True:
                        cls_label = np.argmax(np.random.uniform() <= curr_prior)
                        # Redraw class label if trn_y is out of that class
                        if cls_amount[cls_label] <= 0:
                            continue
                        cls_amount[cls_label] -= 1

                        concat_clnt_x[curr_clnt][clnt_data_list[curr_clnt]] = concat_datasets_x[idx_list[cls_label][cls_amount[cls_label]]]
                        concat_clnt_y[curr_clnt][clnt_data_list[curr_clnt]] = concat_datasets_y[idx_list[cls_label][cls_amount[cls_label]]]

                        break

                concat_clnt_x = np.asarray(concat_clnt_x)
                concat_clnt_y = np.asarray(concat_clnt_y)

                clnt_x = np.asarray(
                    [np.zeros((clnt_data_list_train[clnt__], self.channels, self.height, self.width)).astype(np.float32)
                     for
                     clnt__ in range(self.n_client)])
                clnt_y = np.asarray(
                    [np.zeros((clnt_data_list_train[clnt__], 1)).astype(np.int64) for clnt__ in range(self.n_client)])
                tst_x = np.asarray(
                    [np.zeros((clnt_data_list_tst[clnt__], self.channels, self.height, self.width)).astype(np.float32)
                     for
                     clnt__ in range(self.n_client)])
                tst_y = np.asarray(
                    [np.zeros((clnt_data_list_tst[clnt__], 1)).astype(np.int64) for clnt__ in range(self.n_client)])

                for jj in range(n_client):
                    rand_perm = np.random.permutation(len(concat_clnt_y[jj]))
                    concat_clnt_x[jj] = concat_clnt_x[jj][rand_perm]
                    concat_clnt_y[jj] = concat_clnt_y[jj][rand_perm]

                    clnt_x[jj] = concat_clnt_x[jj][:n_data_per_clnt_train, :, :, :]
                    tst_x[jj] = concat_clnt_x[jj][n_data_per_clnt_train:, :, :, :]

                    clnt_y[jj] = concat_clnt_y[jj][:n_data_per_clnt_train, :]
                    tst_y[jj] = concat_clnt_y[jj][n_data_per_clnt_train:, :]


                cls_means = np.zeros((self.n_client, self.n_cls))
                for clnt in range(self.n_client):
                    for cls in range(self.n_cls):
                        cls_means[clnt,cls] = np.mean(clnt_y[clnt]==cls)
                prior_real_diff = np.abs(cls_means-cls_priors)
                print('--- Max deviation from prior: %.4f' %np.max(prior_real_diff))
                print('--- Min deviation from prior: %.4f' %np.min(prior_real_diff))

            if self.rule == 'noniid':
                while np.sum(total_clnt_data_list) != n_data:
                    # Still there are data to distibute
                    # Get a random client that among the ones that has the least # of data with respect to totat data it is supposed to have
                    min_amount = np.min(total_clnt_data_list - clnt_data_list)
                    min_idx_list = np.where(total_clnt_data_list - clnt_data_list == min_amount)[0]
                    np.random.shuffle(min_idx_list)
                    cur_clnt = min_idx_list[0]
                    print(
                        'Current client %d, total remaining amount %d' % (cur_clnt, n_data - np.sum(total_clnt_data_list)))

                    # Get its class list
                    cur_cls_list = np.asarray([(cur_clnt + jj) % n_cls for jj in range(cls_per_client)])
                    # Get the class that has minumum amount of data on the client
                    cls_amounts = np.asarray([len(clnt_cls_idx[cur_clnt][jj]) for jj in range(n_cls)])
                    min_to_max = cur_cls_list[np.argsort(cls_amounts[cur_cls_list])]
                    cur_idx = 0
                    while cur_idx != len(min_to_max) and cls_amount[min_to_max[cur_idx]] == 0:
                        cur_idx += 1
                    if cur_idx == len(min_to_max):
                        # This client is not full, it needs data but there is no class data left
                        # Pick a random client and assign its data to this client
                        while True:
                            rand_clnt = np.random.randint(n_client)
                            print('Random client %d' % rand_clnt)
                            if rand_clnt == cur_clnt:  # Pick a different client
                                continue
                            rand_clnt_cls = np.asarray([(rand_clnt + jj) % n_cls for jj in range(cls_per_client)])
                            # See if random client has an intersection class with the current client
                            cur_list = np.asarray([(cur_clnt + jj) % n_cls for jj in range(cls_per_client)])
                            np.random.shuffle(cur_list)
                            cls_idx = 0
                            is_found = False
                            while cls_idx != cls_per_client:
                                if cur_list[cls_idx] in rand_clnt_cls and len(
                                        clnt_cls_idx[rand_clnt][cur_list[cls_idx]]) > 1:
                                    is_found = True
                                    break
                                cls_idx += 1
                            if not is_found:  # No class intersection, choose another client
                                continue
                            found_cls = cur_list[cls_idx]
                            # Assign this class instance to curr client
                            total_clnt_data_list[cur_clnt] += 1
                            total_clnt_data_list[rand_clnt] -= 1
                            transfer_idx = clnt_cls_idx[rand_clnt][found_cls][-1]
                            del clnt_cls_idx[rand_clnt][found_cls][-1]
                            clnt_cls_idx[cur_clnt][found_cls].append(transfer_idx)
                            # print('Class %d is transferred from %d to %d' %(found_cls, rand_clnt, cur_clnt))
                            break
                    else:
                        cur_cls = min_to_max[cur_idx]
                        # Assign one data point from this class to the task
                        total_clnt_data_list[cur_clnt] += 1
                        cls_amount[cur_cls] -= 1
                        clnt_cls_idx[cur_clnt][cur_cls].append(idx_list[cur_cls][cls_amount[cur_cls]])
                        # print('Chosen client: %d, chosen class: %d' %(cur_clnt, cur_cls))

                for i in range(n_cls):
                    assert 0 == cls_amount[i], 'Missing datapoints'
                assert n_data == np.sum(total_clnt_data_list), 'Missing datapoints'

                concat_clnt_x = np.asarray(
                    [np.zeros((clnt_data_list[clnt__], self.channels, self.height, self.width)).astype(np.float32) for
                     clnt__ in range(self.n_client)])
                concat_clnt_y = np.asarray(
                    [np.zeros((clnt_data_list[clnt__], 1)).astype(np.int64) for clnt__ in range(self.n_client)])

                clnt_x = np.asarray(
                    [np.zeros((clnt_data_list_train[clnt__], self.channels, self.height, self.width)).astype(np.float32) for
                     clnt__ in range(self.n_client)])
                clnt_y = np.asarray(
                    [np.zeros((clnt_data_list_train[clnt__], 1)).astype(np.int64) for clnt__ in range(self.n_client)])
                tst_x = np.asarray(
                    [np.zeros((clnt_data_list_tst[clnt__], self.channels, self.height, self.width)).astype(np.float32) for
                     clnt__ in range(self.n_client)])
                tst_y = np.asarray(
                    [np.zeros((clnt_data_list_tst[clnt__], 1)).astype(np.int64) for clnt__ in range(self.n_client)])

                for jj in range(n_client):
                    concat_clnt_x[jj] = concat_datasets_x[np.concatenate(clnt_cls_idx[jj]).astype(np.int32)]
                    concat_clnt_y[jj] = concat_datasets_y[np.concatenate(clnt_cls_idx[jj]).astype(np.int32)]

                for jj in range(n_client):
                    rand_perm = np.random.permutation(len(concat_clnt_y[jj]))
                    concat_clnt_x[jj] = concat_clnt_x[jj][rand_perm]
                    concat_clnt_y[jj] = concat_clnt_y[jj][rand_perm]

                    clnt_x[jj] = concat_clnt_x[jj][:n_data_per_clnt_train, :, :, :]
                    tst_x[jj] = concat_clnt_x[jj][n_data_per_clnt_train:, :, :, :]

                    clnt_y[jj] = concat_clnt_y[jj][:n_data_per_clnt_train, :]
                    tst_y[jj] = concat_clnt_y[jj][n_data_per_clnt_train:, :]

            self.clnt_x = clnt_x;
            self.clnt_y = clnt_y
            self.tst_x = tst_x;
            self.tst_y = tst_y

            # Save data
            os.mkdir('Data/%s' % (self.name))

            np.save('Data/%s/clnt_x.npy' % (self.name), clnt_x)
            np.save('Data/%s/clnt_y.npy' % (self.name), clnt_y)

            np.save('Data/%s/tst_x.npy' % (self.name), tst_x)
            np.save('Data/%s/tst_y.npy' % (self.name), tst_y)

            if not os.path.exists('Model'):
                os.mkdir('Model')

        else:
            print("Data is already downloaded")

            self.clnt_x = np.load('Data/%s/clnt_x.npy' % (self.name))
            self.clnt_y = np.load('Data/%s/clnt_y.npy' % (self.name))
            self.n_client = len(self.clnt_x)

            self.tst_x = np.load('Data/%s/tst_x.npy' % (self.name))
            self.tst_y = np.load('Data/%s/tst_y.npy' % (self.name))

            if self.dataset == 'MNIST':
                self.channels = 1;self.width = 28;self.height = 28;self.n_cls = 10;
            if self.dataset == 'FMNIST':
                self.channels = 1; self.width = 28; self.height = 28;self.n_cls = 10;
            if self.dataset == 'CIFAR10':
                self.channels = 3; self.width = 32;self.height = 32;self.n_cls = 10;
            if self.dataset == 'CIFAR100':
                self.channels = 3;self.width = 32;self.height = 32;self.n_cls = 100;
            if self.dataset == 'EMNIST':
                self.channels = 1; self.width = 28; self.height = 28; self.n_cls = 10;

        print('Class frequencies:')

        # train
        count = 0
        for clnt in range(self.n_client):
            print("Client %3d: " % clnt +
                  ', '.join(["%.3f" % np.mean(self.clnt_y[clnt] == cls) for cls in range(self.n_cls)]) +
                  ', Amount:%d' % self.clnt_y[clnt].shape[0])
            count += self.clnt_y[clnt].shape[0]

        print('Total Amount:%d' % count)
        print('-----------------------------------------------------------')

        # test
        count = 0
        for clnt in range(self.n_client):
            print("Client %3d: " % clnt +
                  ', '.join(["%.3f" % np.mean(self.tst_y[clnt] == cls) for cls in range(self.n_cls)]) +
                  ', Amount:%d' % self.tst_y[clnt].shape[0])
            count += self.tst_y[clnt].shape[0]

        print('Total Amount:%d' % count)
        print('-----------------------------------------------------------')


class Dataset(torch.utils.data.Dataset):

    def __init__(self, data_x, data_y=True, train=False, dataset_name=''):
        self.name = dataset_name

        if self.name == 'MNIST' or self.name == 'EMNIST' or self.name == 'FMNIST':
            self.X_data = torch.tensor(data_x).float()
            self.y_data = data_y
            if not isinstance(data_y, bool):
                self.y_data = torch.tensor(data_y).float()

        elif self.name == 'CIFAR10':
            self.train = train
            self.transform = transforms.Compose([transforms.ToTensor()])
            self.X_data = torch.tensor(data_x).float()
            self.y_data = data_y
            if not isinstance(data_y, bool):
                self.y_data = data_y.astype('float32')
            self.augment_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])])
            self.noaugmt_transform = transforms.Compose(
                [transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])])

        elif self.name == 'CIFAR100':
            self.train = train
            self.transform = transforms.Compose([transforms.ToTensor()])
            self.X_data = torch.tensor(data_x).float()
            self.y_data = data_y
            if not isinstance(data_y, bool):
                self.y_data = data_y.astype('float32')
            self.augment_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])])
            self.noaugmt_transform = transforms.Compose(
                [transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])])

    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, idx):

        if self.name == 'MNIST' or self.name == 'EMNIST' or self.name == 'FMNIST':
            X = self.X_data[idx]
            if isinstance(self.y_data, bool):
                return X
            else:
                y = self.y_data[idx]
                return X, y

        elif self.name == 'CIFAR10':
            img = self.X_data[idx]
            if self.train:
                img = self.augment_transform(img)
            else:
                img = self.noaugmt_transform(img)
            if isinstance(self.y_data, bool):
                return img
            else:
                y = self.y_data[idx]
                return img, y

        elif self.name == 'CIFAR100':
            img = self.X_data[idx]
            if self.train:
                img = self.augment_transform(img)
            else:
                img = self.noaugmt_transform(img)
            if isinstance(self.y_data, bool):
                return img
            else:
                y = self.y_data[idx]

                return img, y

if __name__ == '__main__':
    data_path = 'Folder/'  # The folder to save Data & Model
    n_client = 100
    data_obj = DatasetObject(dataset='EMNIST', n_client=n_client, seed=23, rule = 'noniid', class_main=5, data_path=data_path,
                             frac_data=0.7, dir_alpha=1)
    tst_x = data_obj.clnt_x
    tst_y = data_obj.clnt_y

    '''
    trn_gen = data.DataLoader(Dataset(tst_x[0], tst_y[0], train=True, dataset_name='CIFAR10'), batch_size=32, shuffle=True)

    tst_gen_iter = trn_gen.__iter__()
    for i in range(int(np.ceil(300 / 32))):
        data, target = tst_gen_iter.__next__()
        print(target.shape)
        targets = target.reshape(-1)
        print(targets.shape)
        print(target[2])
        print(target[2,0])
        print(target[2].type(torch.long))
        print(target[target[1].type(torch.long)])
        print(target[:, None].shape)
        print(target[:, None].expand(-1,-1, 5))
        
    '''