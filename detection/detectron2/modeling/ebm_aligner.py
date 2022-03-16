import numpy as np
import matplotlib
matplotlib.use('Agg')
import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class EBMAligner:
    """Manages the lifecycle of the proposed Energy Based Latent Alignment.
    """
    def __init__(self,
                 ebm_latent_dim=2048,
                 ebm_n_layers=1,
                 ebm_n_hidden_units=64,
                 max_iter=1000,
                 ebm_lr=0.0001,
                 n_langevin_steps=30,
                 langevin_lr=0.1,
                 ema_decay=0.89,
                 log_iter=50,
                 batch_size=128):
        self.is_enabled = False

        # Configs of the EBM model
        self.ebm_latent_dim = ebm_latent_dim
        self.ebm_n_layers = ebm_n_layers
        self.ebm_n_hidden_units = ebm_n_hidden_units
        self.ebm_ema = None

        # EBM Learning configs
        self.max_iter = max_iter
        self.ebm_lr = ebm_lr
        self.n_langevin_steps = n_langevin_steps
        self.langevin_lr = langevin_lr
        self.ema_decay = ema_decay
        self.log_iter = log_iter
        self.batch_size = batch_size

    def ema(self, model1, model2, decay=0.999):
        par1 = dict(model1.named_parameters())
        par2 = dict(model2.named_parameters())
        for k in par1.keys():
            par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)

    def requires_grad(self, model, flag=True):
        for p in model.parameters():
            p.requires_grad = flag

    def sampler(self, ebm_model, x, langevin_steps=30, lr=0.1, return_seq=False):
        """The langevin sampler to sample from the ebm_model

        :param ebm_model: The source EBM model
        :param x: The data which is updated to minimize energy from EBM
        :param langevin_steps: The number of langevin steps
        :param lr: The langevin learning rate
        :param return_seq: Whether to return the sequence of updates to x
        :return: Sample(s) from EBM
        """
        x = x.clone().detach()
        x.requires_grad_(True)
        sgd = torch.optim.SGD([x], lr=lr)

        sequence = torch.zeros_like(x).unsqueeze(0).repeat(langevin_steps, 1, 1)
        for k in range(langevin_steps):
            sequence[k] = x.data
            ebm_model.zero_grad()
            sgd.zero_grad()
            energy = ebm_model(x).sum()

            (-energy).backward()
            sgd.step()

        if return_seq:
            return sequence
        else:
            return x.clone().detach()

    def learn_ebm(self, path):
        """Learn the EBM.

        current_task_data + prev_model acts as in-distribution data, and
        current_task_data + current_model acts as out-of-distribution data.
        This is used for learning the energy manifold.
        """
        ebm = EBM(latent_dim=self.ebm_latent_dim, n_layer=self.ebm_n_layers,
                       n_hidden=self.ebm_n_hidden_units).cuda()
        # if self.ebm_ema is None:
        self.ebm_ema = EBM(latent_dim=self.ebm_latent_dim, n_layer=self.ebm_n_layers,
                           n_hidden=self.ebm_n_hidden_units).cuda()
        # Initialize the exponential moving average of the EBM.
        self.ema(self.ebm_ema, ebm, decay=0.)

        ebm_optimizer = torch.optim.RMSprop(ebm.parameters(), lr=self.ebm_lr)

        iterations = 0

        feature_dateset = FeatureDataset(path)
        feature_loader = DataLoader(feature_dateset, batch_size=self.batch_size, drop_last=True)

        feature_iter = iter(feature_loader)

        print('Starting to learn the EBM.')
        while iterations < self.max_iter:
            ebm.zero_grad()
            ebm_optimizer.zero_grad()

            try:
                current_z, prev_z = next(feature_iter)
            except:
                feature_iter = iter(feature_loader)
                current_z, prev_z = next(feature_iter)

            prev_z = prev_z.cuda()
            current_z = current_z.cuda()

            self.requires_grad(ebm, False)
            sampled_z = self.sampler(ebm, current_z.clone().detach(), langevin_steps=self.n_langevin_steps, lr=self.langevin_lr)
            self.requires_grad(ebm, True)

            indistribution_energy = ebm(prev_z)
            oodistribution_energy = ebm(sampled_z)

            loss = -(indistribution_energy - oodistribution_energy).mean()

            loss.backward()
            ebm_optimizer.step()

            self.ema(self.ebm_ema, ebm, decay=self.ema_decay)

            if iterations == 0 or iterations % self.log_iter == 0:
                print("Iter: {:5d}, loss: {:7.5f}".format(iterations, loss))

            iterations += 1

        self.is_enabled = True

    def align_latents(self, z):
        self.requires_grad(self.ebm_ema, False)
        aligned_z = self.sampler(self.ebm_ema, z.clone().detach(), langevin_steps=self.n_langevin_steps, lr=self.langevin_lr)
        self.requires_grad(self.ebm_ema, True)
        return aligned_z


class EBM(nn.Module):
    """Defining the Energy Based Model.
    """
    def __init__(self, latent_dim=32, n_layer=1, n_hidden=64):
        super().__init__()

        mlp = nn.ModuleList()
        if n_layer == 0:
            mlp.append(nn.Linear(latent_dim, 1))
        else:
            mlp.append(nn.Linear(latent_dim, n_hidden))

            for _ in range(n_layer-1):
                mlp.append(nn.LeakyReLU(0.2))
                mlp.append(nn.Linear(n_hidden, n_hidden))

            mlp.append(nn.LeakyReLU(0.2))
            mlp.append(nn.Linear(n_hidden, 1))

        self.mlp = nn.Sequential(*mlp)

    def forward(self, x):
        return self.mlp(x)


class FeatureDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.file_names = os.listdir(path)

    def __getitem__(self, index):
        location = os.path.join(self.path, self.file_names[index])
        current_model_features, prev_model_features = torch.load(location)
        current_model_features = torch.Tensor(current_model_features)
        prev_model_features = torch.Tensor(prev_model_features)
        features = (current_model_features, prev_model_features)
        return features

    def __len__(self):
        return len(self.file_names)
