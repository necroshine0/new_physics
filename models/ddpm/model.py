import os
import shutil
from tqdm import tqdm
from typing import Union

import math
import torch
from torch import nn
import numpy as np
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.nn.functional import mse_loss

from .modules import BaseBackbone
from ..training_assistant import TrainingAssistant


def get_betas(min_beta=1e-5, max_beta=2e-2, beta_grid='linear', n_steps=200):
    """
    Beta's initialization
    Args:
        min_beta: minimal beta in grid
        max_beta: maximum beta in grid
        beta_grid: grid initialization strategy one of ['linear', 'linear_invariant', 'square', 'sigmoid', 'cosine']
        n_steps: grid size
    Returns: beta grid

    NOTE: for cosine scheduler, see https://arxiv.org/pdf/2102.09672
    Also used in TabDDPM
    NOTE 2: 'linear_invariant' schedule means that you don't need to chose max/min beta,
    it is extended to work for any number of diffusion steps
    """

    if beta_grid == 'linear':
        betas = torch.linspace(min_beta, max_beta, n_steps)
    elif beta_grid == 'linear_invariant':
        scale = 1000 / n_steps
        beta_start, beta_end = scale * 0.0001, scale * 0.02
        betas = torch.linspace(beta_start, beta_end, n_steps)
    elif beta_grid == 'square':
        betas = torch.linspace(min_beta ** 0.5, max_beta ** 0.5, n_steps) ** 2
    elif beta_grid == 'sigmoid':
        betas = torch.linspace(-6, 6, n_steps)
        betas = torch.sigmoid(betas) * (max_beta - min_beta) + min_beta
    elif beta_grid == 'cosine':
        s = 0.008
        T = n_steps
        beta_boundary = 0.999
        ts = torch.arange(0, T + 1, 1)
        ts = (ts / T + s) / (1 + s)
        alphas_cumprod = torch.cos(math.pi * ts / 2) ** 2
        betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
        betas[betas > beta_boundary] = beta_boundary
    else:
        raise NotImplemented(f'Beta grid type "{beta_grid}" is not implemented')
    return betas


class DiffusionMLP(nn.Module):
    """DDPM multilayer model"""
    def __init__(self, backbone: BaseBackbone,
                 betas=(1e-4, 2e-2),
                 beta_grid='linear',
                 sigma_method='beta',
                 lambda_vlb=1e-3):
        """
        Args:
            backbone: backbone model to predict noise
                NOTE: you can use your own backbone based on BaseBackbone class
            betas: tuple of (min_beta, bax_beta)
            beta_grid: grid initialization strategy one of ['linear', 'linear_invariant', 'square', 'sigmoid', 'cosine']
            sigma_method: variance computation strategy, one of ['beta', 'beta_wave', 'learned']
        """
        super().__init__()
        self.device = 'cpu'
        self.backbone = backbone
        self.var_dim = self.backbone.var_dim
        self.cond_dim = self.backbone.cond_dim
        self.n_steps = self.backbone.n_steps

        # forward process variances \beta_t
        # paper, 3.1: fix as the constants
        self.sigma_method = sigma_method
        self.min_beta = betas[0]
        self.max_beta = betas[1]
        self.beta_grid = beta_grid
        self.betas = get_betas(self.min_beta, self.max_beta, beta_grid, self.n_steps)

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=-1)

        self.lambda_vlb = lambda_vlb

        self.to(backbone.device)
        assert self.device == backbone.device

    def get_beta_wave(self, t):
        num = self.betas[t] * (1 - self.alphas_cumprod[t - 1])
        denum = (1 - self.alphas_cumprod[t])
        beta_wave = num / denum
        return beta_wave

    def compute_q_posterior(self, x0, xt, t):
        """
        Computes posterior q(x_t-1 | x_t, x_0), see eq. (6)
        Returns \mu_t(x_t, x_0), \Sigma = beta_wave * I, see eq. (7)
        https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py#L208
        """
        beta_t = self.betas[t]
        alpha_t = self.alphas[t].view(-1, 1)
        alpha_bar_t = self.alphas_cumprod[t].view(-1, 1)
        alpha_bar_tm1 = self.alphas_cumprod[t - 1].view(-1, 1)

        mu_t = beta_t * alpha_bar_tm1.sqrt() / (1 - alpha_bar_t) * x0 \
                + alpha_t.sqrt() * (1 - alpha_bar_tm1) / (1 - alpha_bar_t) * xt

        log_var_t = torch.log(self.get_beta_wave(t))
        return mu_t, log_var_t

    def compute_p(self, xt, t, eta_theta):
        """
        Apply the model to get p_theta(x_t-1 | x_t), as well as a prediction of the initial x, x_0
        Returns \mu_theta(x_t, t), see eq. (11)
        https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py#L232
        """
        beta_t = self.betas[t]
        alpha_t = self.alphas[t].view(-1, 1)
        alpha_bar_t = self.alphas_cumprod[t].view(-1, 1)

        mu_theta = (xt - beta_t / (1 - alpha_bar_t).sqrt() * eta_theta) / alpha_t.sqrt()
        return mu_theta

    def compute_dkl(self, q_mu_t, q_log_var_t, p_mu_theta_t, p_log_var_theta_t):
        """
        Computes normal E[D_KL( q(x_t-1 | x_t, x_0) || p_theta(x_t-1 | x_t) )], or L_t-1 term
        https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py#L642
        Reference: https://statproofbook.github.io/P/norm-kl.html, eq. (2)
        """
        d_kl = (q_mu_t - p_mu_theta_t) ** 2 / torch.exp(-p_log_var_theta_t) \
                + torch.exp(q_log_var_t - p_log_var_theta_t)\
                - q_log_var_t + p_log_var_theta_t
        return d_kl.mean()

    def optimize_step(self, x0, cond=None):
        t = torch.randint(high=self.n_steps, size=(x0.shape[0], 1)).to(self.device)
        eta = torch.randn_like(x0)

        xt = self.forward_process(x0, t, eta)  # noised
        eta_theta, log_var_theta = self.reverse_process(xt, t, cond)
        L = mse_loss(eta, eta_theta, reduction="mean")  # L_simple term

        if self.sigma_method == "learned":
            q_mu_t, q_log_var_t = self.compute_q_posterior(x0, xt, t)
            p_mu_theta_t = self.compute_p(xt, t, eta_theta)
            p_log_var_theta_t = log_var_theta
            E_L_tm1 = self.compute_dkl(q_mu_t, q_log_var_t, p_mu_theta_t, p_log_var_theta_t)  # L_vlb = E[L_t-1] term
            L += self.lambda_vlb * E_L_tm1  # L_hybrid = L_simple + \lambda * L_vlb
        return L

    def get_sigma(self, t, model_vars=None):
        """
        Variance computation from betas, see paper, 2
        In a few words, \Sigma_\theta = \sigma^2 * I, where
        \sigma^2 = beta or \sigma^2 = beta_wave

        Options "learned" uses approach from https://arxiv.org/pdf/2102.09672,
        parameterizing \sigma^2 as log-scale interpolation between beta and beta_wave
        Requires model_vars as variances (no constraints)
        """
        if self.sigma_method == 'beta':
            # sigma_t^2 = beta_t, see paper, eq. (6)
            return self.betas[t].sqrt().view(-1, 1)
        elif self.sigma_method == 'beta_wave':
            # sigma_t^2 = beta_wave_t, see paper, eq. (7)
            return self.get_beta_wave(t).sqrt().view(-1, 1)
        elif self.sigma_method == 'learned':
            assert model_vars is not None
            betas_log = torch.log(self.betas[t]).view(-1, 1)
            betas_wave_log = torch.log(self.get_beta_wave(t)).view(-1, 1)
            return torch.exp(model_vars * betas_log + (1 - model_vars) * betas_wave_log)
        else:
            raise ValueError('Unknown method')

    def one_step_denoise(self, xt, t_val, cond=None):
        """
        Denoise xt-1 <-- xt, see paper, Algorithm 2 Sampling
        Args:
            xt:  objects to denoise of shape (B, var_dim)
            t_val: timestamp of xt
            cond: condition tensor of shape (B, cond_dim)

        Returns: denoised x_t-1
        """

        t = t_val * torch.ones(size=(xt.shape[0], 1), dtype=torch.long).to(self.device)

        # predict eta_theta
        eta_theta, log_var = self.reverse_process(xt, t, cond)

        # get x_t-1
        alpha_t = self.alphas[t].view(-1, 1)
        alpha_bar_t = self.alphas_cumprod[t].view(-1, 1)
        coef = (1 - alpha_t) / (1 - alpha_bar_t).sqrt()
        x = (xt - coef * eta_theta) / alpha_t.sqrt()

        if t_val > 1:
            z = torch.randn((xt.shape[0], self.backbone.var_dim)).to(self.device)
            sigma_t = self.get_sigma(t, torch.exp(log_var))
            x = x + sigma_t * z
        return x

    def reverse_process(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor = None):
        """
        Predict noise eta_theta from xt by backbone model
        See paper, eq. (11), Algorithm 1
        Args:
            x: input objects tensor of shape (B, var_dim)
            t: noising step (timestep)
            cond: condition tensor of shape (B, cond_dim)
        Returns: eta_theta of shape (B, var_dim)
        """
        eta_theta, variances = self.backbone(x, t, cond)
        return eta_theta, variances

    def forward_process(self, y: torch.Tensor, t: torch.Tensor, eta: torch.Tensor):
        """
        Noise the input objects by t steps, see paper, 3.2 and eq. (4)
        Args:
            y: input object tensor of shape (B, var_dim)
            t: noising step (timestep)
            eta: normal noise of shape (B, var_dim)
        Returns: noised objects in the close form of shape (B, var_dim)
        """
        alpha_bar_t = self.alphas_cumprod[t].view(-1, 1)
        mean = alpha_bar_t.sqrt() * y
        var = (1 - alpha_bar_t).sqrt()
        x_t_noised = mean + var * eta  # reparameterization trick
        return x_t_noised

    def to(self, device):
        super().to(device)
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.device = device
        return self


# =========================== Wrapper ===========================


class DDPM(TrainingAssistant):
    def __init__(self, backbone: BaseBackbone,
                 betas=(1e-4, 1e-2),
                 beta_grid='linear',
                 sigma_method='beta',
                 lambda_vlb=1e-3,
                 batch_size=64,
                 n_epochs=10,
                 checkpoint_dir=None,
                 device='cpu'):
        super().__init__(batch_size, checkpoint_dir)
        self.model = DiffusionMLP(backbone.to(device), betas, beta_grid, sigma_method, lambda_vlb)
        self.device = backbone.device
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
            loss = self.model.optimize_step(y0, x_cond0)
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
            assert self.model.backbone.cond_input_dim == X_cond.shape[1]

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

                y = torch.randn((size, self.model.backbone.var_dim)).to(self.device)
                for i in range(self.model.n_steps):
                    t_val = self.model.n_steps - i - 1
                    y = self.model.one_step_denoise(y, t_val, x_cond)
                Ys.append(y.cpu())

            Y = torch.cat(Ys, dim=0)
        return Y.numpy()
