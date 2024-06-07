import math
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

from typing import Union


class SinusoidalEmbedding(nn.Module):
    """Transformer sinusoidal position embedding"""
    def __init__(self, n_steps, steps_dim, requires_grad=False):
        super().__init__()
        self.n_steps = n_steps
        self.steps_dim = steps_dim

        position = torch.arange(self.n_steps).double().unsqueeze(1)
        emb = torch.arange(0, self.steps_dim, 2) / self.steps_dim * math.log(10000.0)
        div_term = torch.exp(-emb)
        self.pe = nn.Embedding(self.n_steps, self.steps_dim)
        self.pe.weight.data[:, 0::2] = torch.sin(position * div_term)
        self.pe.weight.data[:, 1::2] = torch.cos(position * div_term)
        self.pe.requires_grad_(requires_grad)

    def forward(self, t: Union[torch.tensor, int, np.ndarray]):
        if type(t) is int:
            t = torch.tensor([t])
        elif type(t) is np.ndarray:
            t = torch.from_numpy(t)
        return self.pe(t.view(-1))

class StepsEmbedding(nn.Module):
    def __init__(self, n_steps, steps_dim, steps_depth=2, act=nn.SiLU()):
        super().__init__()
        self.n_steps = n_steps
        self.steps_dim = steps_dim
        self.steps_depth = steps_depth

        sin_emb = SinusoidalEmbedding(self.n_steps, self.steps_dim, requires_grad=True)
        modules = nn.ModuleList([nn.Linear(self.steps_dim, self.steps_dim)])
        for i in range(self.steps_depth - 1):
            modules.append(act)
            modules.append(nn.Linear(self.steps_dim, self.steps_dim))

        self.net = nn.Sequential(sin_emb, *modules)

    def forward(self, t):
        return self.net(t)


class CondEmbedding(nn.Module):
    def __init__(self, in_dim, hid_dim, depth=2, act=nn.SiLU()):
        super().__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.depth = depth

        modules = nn.ModuleList([nn.Linear(self.in_dim, self.hid_dim), act])
        for i in range(self.depth - 1):
            modules.append(nn.Linear(self.hid_dim, self.hid_dim))
            modules.append(act)
        self.net = nn.Sequential(*modules)

    def forward(self, t):
        return self.net(t)


# ======================================= Backbones =======================================

class BaseBackbone(nn.Module):
    """
    Base class for noise recovering backbone (instead of U-Net in original paper)
    Every model inherited from this class must contain:
        - net logic
        - steps embedding
        - conditional embedding (optional)

    Backbone core (net) must return tensor shaped (B, 2 * var_dim)
    to predict noise and variance

    Input: [X, cond], var_dim + cond_dim
    Output: [X_samples], var_dim; [variances], var_dim
    """
    def __init__(self,
                 var_dim,
                 cond_dim=None,
                 hid_dim=64,
                 num_blocks=6,
                 n_steps=200,
                 steps_dim=32,
                 steps_depth=2,
                 act=nn.SiLU(),
                 dropout=0.0,
                 cond_include="sum",
                 cond_emb_dim=None,
                 device='cpu'):
        """
        Args:
            var_dim: size of target data
            cond_dim: input size of conditional data (None if no conditioning)
            hid_dim: hidden size of the net layers
            num_blocks: number of backbone layers
            n_steps: number of time steps embeddings (input size)
            steps_dim: hidden size of time steps embeddings
            act: activation function used in model
            dropout: dropout prob
            cond_include: how to include condition embeddings -- "sum" or "concat"
            cond_emb_dim: size of condition embedding
            device: device to run on ('cpu', 'cuda' or torch.cuda.device)
        NOTE:
            steps_out_dim is used when concat [y, cond, time] is used
            else output size of time is size of concat [y, cond] or [y] if unconditional
        """

        super().__init__()
        self.var_dim = var_dim
        self.cond_dim = cond_dim
        self.cond_input_dim = cond_dim
        self.hid_dim = hid_dim
        self.num_blocks = num_blocks
        self.n_steps = n_steps
        self.steps_dim = steps_dim
        self.steps_depth = steps_depth
        self.act = act
        self.dropout = dropout

        if self.cond_dim is not None:
            self.cond_include = cond_include
            if self.cond_include == "sum":
                self.cond_emb_dim = steps_dim
            else:
                self.cond_emb_dim = steps_dim if cond_emb_dim is None else cond_emb_dim
        else:
            self.cond_include = None
            self.cond_emb_dim = None

        self.proj = nn.Linear(self.var_dim, self.steps_dim)
        self.time_emb = StepsEmbedding(self.n_steps, self.steps_dim, self.steps_depth, act=nn.SiLU())
        if self.cond_dim is not None:
            self.cond_emb = CondEmbedding(self.cond_dim, self.cond_emb_dim, depth=2, act=nn.SiLU())

        self.net_in_dim = self.steps_dim
        if self.cond_include == "concat":
            self.net_in_dim += self.cond_emb_dim

        self.device = device
        # for example
        self.net = nn.Linear(self.net_in_dim, 2 * self.var_dim)

    def get_embs(self, x, t, cond=None):
        x_emb = self.proj(x)
        t_emb = self.time_emb(t)
        if cond is not None:
            cond_emb = self.cond_emb(cond)
        else:
            cond_emb = 0.0
        return x_emb, t_emb, cond_emb

    def get_x0(self, x, t, cond=None):
        x_emb, t_emb, cond_emb = self.get_embs(x, t, cond)
        x_input = x_emb + t_emb  # (B, steps_dim)
        if cond_emb is not None:
            if self.cond_include == "sum":
                x_input += cond_emb
            else:
                x_input = torch.cat([x_input, cond_emb], dim=1)  # (B, steps_dim + cond_emb_dim)
        return x_input

    def forward(self, x, t, cond=None):
        """
        Predicts eta_theta noise
        Args:
            x: input objects tensor of shape (B, var_dim)
            t: timestamp tensor of shape (B, 1)
            cond: input condition tensor of shape (B, cond_dim)
        Returns: predicted noise shaped (B, var_dim) and log-variance shaped (B, var_dim)
        """
        x_input = self.get_x0(x, t, cond)         # (B, net_in_dim)
        x_output = self.net(x_input)              # (B, 2 * var_dim)
        eta_noise, log_variances = torch.split(x_output, self.var_dim, dim=1)
        return eta_noise, log_variances

    def to(self, device):
        super().to(device)
        self.device = device
        return self


class MLPBlock(nn.Module):
    """
    Architecture from Revisiting Deep Learning Models for Tabular Data (NeurIPS 2021)
    MLPBlock(x) = Dropout(ReLU(Linear(x)))
    https://arxiv.org/pdf/2106.11959
    """
    def __init__(self,
                 in_dim,
                 out_dim=64,
                 dropout=0.0,
                 act=nn.ReLU()):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.act = act
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.act(self.linear(x)))


class MLPBackbone(BaseBackbone):
    """
    Architecture from Revisiting Deep Learning Models for Tabular Data (NeurIPS 2021)
    MLP(x) = Linear(MLPBlock(...(MLPBlock(x))))
    https://arxiv.org/pdf/2106.11959

    Input: [X, cond], var_dim + cond_dim
    Output: [X_samples], var_dim; [variances], var_dim
    """
    def __init__(self,
                 var_dim,
                 cond_dim=None,
                 hid_dim=128,
                 num_blocks=4,
                 n_steps=100,
                 steps_dim=32,
                 steps_depth=2,
                 act=nn.ReLU(),
                 dropout=0.0,
                 cond_include="sum",
                 cond_emb_dim=None,
                 device='cpu'):

        super().__init__(var_dim,
                 cond_dim,
                 hid_dim,
                 num_blocks,
                 n_steps,
                 steps_dim,
                 steps_depth,
                 act,
                 dropout,
                 cond_include,
                 cond_emb_dim,
                 device)

        input_block = MLPBlock(self.net_in_dim, self.hid_dim, self.dropout, self.act)
        blocks_list = [
                          MLPBlock(self.hid_dim, self.hid_dim, self.dropout, self.act)
                          for _ in range((self.num_blocks - 1))
                      ]
        head = nn.Linear(self.hid_dim, 2 * self.var_dim)

        blocks_list = [input_block] + blocks_list + [head]
        self.net = nn.Sequential(*nn.ModuleList(blocks_list))


class ResNetBlock(nn.Module):
    """
    Architecture from Revisiting Deep Learning Models for Tabular Data (NeurIPS 2021)
    ResNetBlock(x) = x + Dropout(Linear(Dropout(ReLU(Linear(BatchNorm(x))))))
    https://arxiv.org/pdf/2106.11959
    """
    def __init__(self,
                 in_dim,
                 hid_dim=64,
                 dropout=0.0,
                 act=nn.ReLU(),
                 device='cpu'):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(in_dim),
            nn.Linear(in_dim, hid_dim),
            act,
            nn.Dropout(dropout),
            nn.Linear(hid_dim, in_dim),
            nn.Dropout(dropout),
        )

        self.device = device

    def forward(self, x):
        return F.relu(self.block(x) + x)

    def to(self, device):
        super().to(device)
        self.device = device
        return self


class ResNetBackbone(BaseBackbone):
    """
    Architecture from Revisiting Deep Learning Models for Tabular Data (NeurIPS 2021)
    ResNet(x) = Prediction(ResNetBlock(...(ResNetBlock(Linear(x)))))
    Prediction(x) = Linear(ReLU(BatchNorm(x)))
    https://arxiv.org/pdf/2106.11959

    Input: [X, cond], var_dim + cond_dim
    Output: [X_samples], var_dim
    """
    def __init__(self,
                 var_dim,
                 cond_dim=None,
                 hid_dim=128,
                 num_blocks=4,
                 n_steps=100,
                 steps_dim=32,
                 steps_depth=2,
                 act=nn.ReLU(),
                 dropout=0.0,
                 device='cpu'
                 ):
        cond_include = "sum"
        cond_emb_dim = None
        super().__init__(var_dim,
                         cond_dim,
                         hid_dim,
                         num_blocks,
                         n_steps,
                         steps_dim,
                         steps_depth,
                         act,
                         dropout,
                         cond_include,
                         cond_emb_dim,
                         device)

        self.res_blocks = nn.ModuleList([
            ResNetBlock(self.steps_dim, self.hid_dim, self.dropout, act, self.device)
            for _ in range(self.num_blocks)
        ])

        self.prediction = nn.Sequential(
            nn.BatchNorm1d(self.steps_dim),
            nn.ReLU(),
            nn.Linear(self.steps_dim, 2 * self.var_dim)
        )

        # self.net = nn.Sequential(*self.res_blocks, self.prediction)

    def forward(self, x, t, cond=None):
        x_emb, t_emb, cond_emb = self.get_embs(x, t, cond)
        x = x_emb + cond_emb
        for block in self.res_blocks:
            x = block(x + t_emb + cond_emb)
        x_output = self.prediction(x)
        eta_noise, log_variances = torch.split(x_output, self.var_dim, dim=1)
        return eta_noise, log_variances
