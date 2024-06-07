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
from torch.utils.data import TensorDataset, DataLoader

from .modules import MAFBlock, BatchNormFlow
from ..training_assistant import TrainingAssistant


class MaskedAutoregressiveFlow(nn.Module):
    def __init__(self,
                 var_dim,
                 cond_dim=None,
                 hid_dim=128,
                 num_blocks=5,
                 momentum=1.0,
                 act=nn.Tanh()):
        super().__init__()
        self.var_dim = var_dim
        self.cond_dim = cond_dim
        self.hid_dim = hid_dim
        self.num_blocks = num_blocks
        self.momentum = momentum
        self.act = act

        self.blocks = nn.ModuleList([
            MAFBlock(self.var_dim, self.cond_dim,
                 self.hid_dim, self.momentum, self.act)
            for _ in range(self.num_blocks)
        ])

    def forward(self, inputs, cond=None, mode='forward', logdets=None):
        """ Performs a forward or backward pass for flow modules.
        Args:
            inputs: a tuple of inputs and logdets
            mode: to run direct computation or inverse
        """
        if logdets is None:
            logdets = torch.zeros(inputs.size(0), 1, device=inputs.device)

        if mode not in ["forward", "backward"]:
            raise ValueError(f"Unknown mode: {mode}")

        blocks = self.blocks if mode == "forward" else self.blocks[::-1]
        for block in blocks:
            inputs, logdet = block(inputs, cond, mode)
            logdets += logdet
        return inputs, logdets

    def log_probs(self, inputs, cond=None):
        u, log_jacob = self.forward(inputs, cond)
        log_probs = (-0.5 * u.pow(2) - 0.5 * math.log(2 * math.pi)).sum(
            -1, keepdim=True)
        return (log_probs + log_jacob).sum(-1, keepdim=True)


# =========================== Wrapper ===========================


class MAF(TrainingAssistant):
    def __init__(self,
                 var_dim,
                 cond_dim=None,
                 hid_dim=128,
                 num_blocks=5,
                 momentum=1.0,
                 act=nn.Tanh(),
                 batch_size=256,
                 n_epochs=100,
                 checkpoint_dir=None,
                 device='cpu'):
        super().__init__(batch_size, checkpoint_dir)
        self.model = MaskedAutoregressiveFlow(var_dim, cond_dim, hid_dim,
                                              num_blocks, momentum, act).to(device)
        self.n_epochs = n_epochs
        self.device = device

        # If data dim is (B, 1), then add noise dim
        self.noise = False

    def run_epoch(self, loader, is_train=True):
        self.model.train() if is_train else self.model.eval()
        Y_size = 0
        epoch_loss = 0.0
        for data in loader:
            if is_train:
                self.optim.zero_grad()
            y0 = data[0].to(self.device)
            x_cond0 = data[1].to(self.device) if len(data) == 2 else None

            # if y0.shape[1] == 1:
            #     y0 = np.hstack([y0, np.random.normal(0, 1, y0.shape)])
            #     self.noise = True
            # else:
            #     self.noise = False

            loss = -self.model.log_probs(y0, x_cond0).mean()
            if is_train:
                loss.backward()
                self.optim.step()
            epoch_loss += loss.item() * y0.shape[0]
            Y_size += y0.shape[0]

        if self.scheduler is not None and is_train:
            self.scheduler.step()

        if is_train:
            for module in self.model.modules():
                if isinstance(module, BatchNormFlow):
                    module.momentum = 0.0

            if x_cond0 is not None:
                with torch.no_grad():
                    self.model(loader.dataset.tensors[0].to(self.device),
                               loader.dataset.tensors[1].to(self.device).float())
            else:
                with torch.no_grad():
                    self.model(loaders.dataset.tensors[0].to(self.device))

            for module in self.model.modules():
                if isinstance(module, BatchNormFlow):
                    module.momentum = 1.0

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

                z = torch.randn((size, self.model.var_dim)).to(self.device)
                y = self.model.forward(z, x_cond, mode='backward')[0]
                y = y[:, 0] if self.noise else y
                Ys.append(y.cpu())
            Y = torch.cat(Ys, dim=0)
        return Y.numpy()
