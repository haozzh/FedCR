import torch
import torch.nn as nn
import torch.nn.functional as F


class Model_CMI(nn.Module):

    def __init__(self, args, dimZ=256, alpha=0, dataset = 'EMNIST'):
        # the dimension of Z
        super().__init__()

        self.alpha = alpha
        self.dimZ = dimZ
        self.device = args.device
        self.dataset = dataset

        if self.dataset == 'EMNIST':
            self.n_cls = 10
        if self.dataset == 'FMNIST':
            self.n_cls = 10
        if self.dataset == 'CIFAR10':
            self.n_cls = 10
        if self.dataset == 'CIFAR100':
            self.n_cls = 100

        self.r_mu = nn.Parameter(torch.zeros(self.n_cls, self.dimZ)).to(self.device)
        self.r_sigma = nn.Parameter(torch.ones(self.n_cls,  self.dimZ)).to(self.device)
        self.C = nn.Parameter(torch.ones([])).to(self.device)

        if self.dataset == 'EMNIST':
            self.n_cls = 10
            self.fc1 = nn.Linear(1 * 28 * 28, 1024)
            self.fc2 = nn.Linear(1024, 1024)
            self.fc3 = nn.Linear(1024, 2 * self.dimZ)
            self.fc4 = nn.Linear(self.dimZ, self.n_cls)
            self.weight_keys = [['fc1.weight', 'fc1.bias'],
                                ['fc2.weight', 'fc2.bias'],
                                ['fc3.weight', 'fc3.bias'],
                                ['fc4.weight', 'fc4.bias']]

        if self.dataset == 'FMNIST':
            self.n_cls = 10
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=5)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv2 = nn.Conv2d(in_channels=4, out_channels=12, kernel_size=5)
            self.fc1 = nn.Linear(12 * 4 * 4, 1024)
            self.fc2 = nn.Linear(1024, 1024)
            self.fc3 = nn.Linear(1024, 2 * self.dimZ)
            self.fc4 = nn.Linear(self.dimZ, self.n_cls)

            self.weight_keys = [['conv1.weight', 'conv1.bias'],
                                ['conv2.weight', 'conv2.bias'],
                                ['fc1.weight', 'fc1.bias'],
                                ['fc2.weight', 'fc2.bias'],
                                ['fc3.weight', 'fc3.bias'],
                                ['fc4.weight', 'fc4.bias']]

        if self.dataset == 'CIFAR10':
            self.n_cls = 10
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5)
            self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(64 * 5 * 5, 1024)
            self.fc2 = nn.Linear(1024, 1024)
            self.fc3 = nn.Linear(1024, 2 * self.dimZ)
            self.fc4 = nn.Linear(self.dimZ, self.n_cls)
            self.weight_keys = [['conv1.weight', 'conv1.bias'],
                                ['conv2.weight', 'conv2.bias'],
                                ['fc1.weight', 'fc1.bias'],
                                ['fc2.weight', 'fc2.bias'],
                                ['fc3.weight', 'fc3.bias'],
                                ['fc4.weight', 'fc4.bias']]

        if self.dataset == 'CIFAR100':
            self.n_cls = 100
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5)
            self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(64 * 5 * 5, 1024)
            self.fc2 = nn.Linear(1024, 1024)
            self.fc3 = nn.Linear(1024, 2 * self.dimZ)
            self.fc4 = nn.Linear(self.dimZ, self.n_cls)
            self.weight_keys = [['conv1.weight', 'conv1.bias'],
                                ['conv2.weight', 'conv2.bias'],
                                ['fc1.weight', 'fc1.bias'],
                                ['fc2.weight', 'fc2.bias'],
                                ['fc3.weight', 'fc3.bias'],
                                ['fc4.weight', 'fc4.bias']]

    def gaussian_noise(self, num_samples, K):
        # works with integers as well as tuples

        return torch.normal(torch.zeros(*num_samples, K), torch.ones(*num_samples, K)).to(self.device)

    def sample_prior_Z(self, num_samples):
        return self.gaussian_noise(num_samples=num_samples, K=self.dimZ)

    def encoder_result(self, encoder_output):
        mu = encoder_output[:, :self.dimZ]
        sigma = torch.nn.functional.softplus(encoder_output[:, self.dimZ:] - self.alpha)

        return mu, sigma

    def sample_encoder_Z(self, batch_size, encoder_Z_distr, num_samples):

        mu, sigma = encoder_Z_distr

        return mu + sigma * self.gaussian_noise(num_samples=(num_samples, batch_size), K=self.dimZ)

    def forward(self, batch_x, num_samples=1):

        if self.dataset == 'EMNIST':
            batch_size = batch_x.size()[0]
            # sample from encoder
            x = batch_x.view(-1, 1 * 28 * 28)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))

            encoder_output = self.fc3(x)
            encoder_Z_distr = self.encoder_result(encoder_output)
            to_decoder = self.sample_encoder_Z(batch_size=batch_size, encoder_Z_distr=encoder_Z_distr,
                                               num_samples=num_samples)
            decoder_logits = self.fc4(to_decoder)

            # batch should go first

        if self.dataset == 'FMNIST':
            batch_size = batch_x.size()[0]
            # sample from encoder
            x = self.pool(F.relu(self.conv1(batch_x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 12 * 4 * 4)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))

            encoder_output = self.fc3(x)
            encoder_Z_distr = self.encoder_result(encoder_output)
            to_decoder = self.sample_encoder_Z(batch_size=batch_size, encoder_Z_distr=encoder_Z_distr,
                                               num_samples=num_samples)
            decoder_logits = self.fc4(to_decoder)

        if self.dataset == 'CIFAR10':
            batch_size = batch_x.size()[0]
            x = self.pool(F.relu(self.conv1(batch_x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 64 * 5 * 5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))

            encoder_output = self.fc3(x)
            encoder_Z_distr = self.encoder_result(encoder_output)
            to_decoder = self.sample_encoder_Z(batch_size=batch_size, encoder_Z_distr=encoder_Z_distr,
                                               num_samples=num_samples)
            decoder_logits = self.fc4(to_decoder)

        if self.dataset == 'CIFAR100':
            batch_size = batch_x.size()[0]
            x = self.pool(F.relu(self.conv1(batch_x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 64 * 5 * 5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))

            encoder_output = self.fc3(x)
            encoder_Z_distr = self.encoder_result(encoder_output)
            to_decoder = self.sample_encoder_Z(batch_size=batch_size, encoder_Z_distr=encoder_Z_distr,
                                               num_samples=num_samples)
            decoder_logits = self.fc4(to_decoder)

        regL2R  = torch.norm(to_decoder)

        return encoder_Z_distr, decoder_logits, regL2R








#regL2R = to_decoder_mearn.norm(dim=1).mean()


'''
class Model_CMI(nn.Module):

    def __init__(self, args):
        self.probabilistic = True
        super(client_model_SR, self).__init__(args)

        self.dimZ = args.dimZ
        self.device = args.device
        self.dataset = args.dataset

        if self.dataset == 'EMNIST':
            self.n_cls = 10
        if self.dataset == 'FMNIST':
            self.n_cls = 10
        if self.dataset == 'CIFAR10':
            self.n_cls = 10
        if self.dataset == 'CIFAR100':
            self.n_cls = 100

        self.r_mu = nn.Parameter(torch.zeros(args.num_classes, args.z_dim))
        self.r_sigma = nn.Parameter(torch.ones(args.num_classes, args.z_dim))
        self.C = nn.Parameter(torch.ones([]))

        self.optim.add_param_group({'params':[self.r_mu,self.r_sigma,self.C],'lr':args.lr,'momentum':0.9})
'''

def KL_between_normals(q_distr, p_distr):
    mu_q, sigma_q = q_distr
    mu_p, sigma_p = p_distr    #Standard Deviation
    k = mu_q.size(1)

    mu_diff = mu_p - mu_q
    mu_diff_sq = torch.mul(mu_diff, mu_diff)
    logdet_sigma_q = torch.sum(2 * torch.log(torch.clamp(sigma_q, min=1e-8)), dim=1)
    logdet_sigma_p = torch.sum(2 * torch.log(torch.clamp(sigma_p, min=1e-8)), dim=1)

    fs = torch.sum(torch.div(sigma_q ** 2, sigma_p ** 2), dim=1) + torch.sum(torch.div(mu_diff_sq, sigma_p ** 2), dim=1)
    two_kl = fs - k + logdet_sigma_p - logdet_sigma_q
    return two_kl * 0.5


def xavier_init(ms):
    for m in ms :
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform(m.weight,gain=nn.init.calculate_gain('relu'))
            m.bias.data.zero_()
