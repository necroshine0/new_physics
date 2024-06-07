import os
import math
import torch
import shutil
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from typing import Union

import torch.utils.data
import torch.optim as optim
from torch.nn.functional import mse_loss
from torch.utils.data import TensorDataset, DataLoader

from .modules import Encoder, Decoder
from ..training_assistant import TrainingAssistant


class CondVAE(nn.Module):
    def __init__(self,
                 var_dim,
                 cond_dim=None,
                 latent_dim=16,
                 hid_shapes=(64, 128, 256),
                 act=nn.Tanh(),
                 KL_weight=1e-3,
                 device="cpu"):
        super().__init__()
        self.var_dim = var_dim
        self.cond_dim = cond_dim
        self.latent_dim = latent_dim
        self.hid_shapes = hid_shapes
        self.act = act
        self.KL_weight = KL_weight
        self.device = device

        self.enc_in_dim = var_dim
        self.dec_in_dim = latent_dim
        if cond_dim is not None:
            self.enc_in_dim += cond_dim
            self.dec_in_dim += cond_dim

        self.encoder = Encoder(self.enc_in_dim, self.latent_dim, self.hid_shapes, self.act).to(self.device)
        self.decoder = Decoder(self.dec_in_dim, self.var_dim, self.hid_shapes, self.act).to(self.device)

    def sample_z(self, mu, log_sigma):
        eps = torch.randn(mu.shape).to(self.device)
        return mu + torch.exp(log_sigma / 2) * eps

    def custom_loss(self, x, rec_x, mu, log_sigma):
        KL = torch.mean(-0.5 * torch.sum(1 + log_sigma - mu ** 2 - log_sigma.exp(), dim=1), dim=0)
        recon_loss = mse_loss(x, rec_x, reduction="mean")
        return KL * self.KL_weight + recon_loss

    def compute_loss(self, x_batch, cond_batch):
        mu, log_sigma = self.encoder(x_batch, cond_batch)
        z_batch = self.sample_z(mu, log_sigma)
        x_batch_rec = self.decoder(z_batch, cond_batch)
        loss = self.custom_loss(x_batch, x_batch_rec, mu, log_sigma)
        return loss


# =========================== Wrapper ===========================


class CVAE(TrainingAssistant):
    def __init__(self,
                 var_dim,
                 cond_dim=None,
                 latent_dim=16,
                 hid_shapes=(64, 128, 256),
                 act=nn.Tanh(),
                 KL_weight=1e-3,
                 batch_size=64,
                 n_epochs=10,
                 checkpoint_dir=None,
                 device='cpu'):
        super().__init__(batch_size, checkpoint_dir)
        self.model = CondVAE(var_dim, cond_dim, latent_dim, hid_shapes, act, KL_weight, device)
        self.device = device
        self.n_epochs = n_epochs

    def run_epoch(self, loader, is_train=True):
        self.model.train() if is_train else self.model.eval()
        Y_size = 0
        epoch_loss = 0.0
        for data in loader:
            if is_train:
                self.optim.zero_grad()
            y0 = data[0].to(self.device)
            x_cond0 = data[1].to(self.device) if len(data) == 2 else None
            loss = self.model.compute_loss(y0, x_cond0)
            if is_train:
                loss.backward()
                self.optim.step()
            epoch_loss += loss.item() * y0.shape[0]
            Y_size += y0.shape[0]

        if self.scheduler is not None and is_train:
            self.scheduler.step()
        return epoch_loss / Y_size

    def sample(self, input: Union[torch.Tensor, np.ndarray, int], batch_size=None):
        N = None; X_cond = None
        if isinstance(input, int):
            N = input
            assert self.model.cond_dim is None
        else:
            X_cond = self._cast_to_tensor(input)
            assert self.model.cond_dim == X_cond.shape[1]

        if X_cond is not None:
            td = TensorDataset(X_cond)
            bs = X_cond.shape[1] if batch_size is None else batch_size
            batches = DataLoader(td, batch_size=bs, shuffle=False)
        else:
            if batch_size is not None:
                bs = batch_size
                n_batches = N // bs
                remains = N - bs * n_batches
                batches = [bs for _ in range(n_batches)]
                if remains > 0:
                    batches += [remains]
            else:
                batches = [N]

        Ys = []
        self.model.eval()
        with torch.no_grad():
            for data in tqdm(batches, 'batch'):
                if isinstance(data, int):
                    x_cond = None
                    size = data
                else:
                    x_cond = data[0].to(self.device)
                    size = x_cond.size(0)

                z = torch.randn((size, self.model.latent_dim)).to(self.device)
                y = self.model.decoder(z, x_cond).cpu().detach()
                Ys.append(y.cpu())
            Y = torch.cat(Ys, dim=0)
        return Y.numpy()
