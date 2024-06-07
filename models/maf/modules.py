import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_mask(in_features, out_features, in_flow_features, mask_type=None):
    """
    mask_type: input | None | output
    See Figure 1 for a better illustration:
    https://arxiv.org/pdf/1502.03509.pdf
    """
    if mask_type == 'input':
        in_degrees = torch.arange(in_features) % in_flow_features
    else:
        in_degrees = torch.arange(in_features) % (in_flow_features - 1)

    if mask_type == 'output':
        out_degrees = torch.arange(out_features) % in_flow_features - 1
    else:
        out_degrees = torch.arange(out_features) % (in_flow_features - 1)
    return (out_degrees.unsqueeze(-1) >= in_degrees.unsqueeze(0)).float()


class MaskedLinear(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 mask,
                 cond_dim=None,
                 bias=True):
        super(MaskedLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias)
        if cond_dim is not None:
            self.cond_emb = nn.Linear(
                cond_dim, out_features, bias=False)

        self.register_buffer('mask', mask)

    def forward(self, inputs, cond_inputs=None):
        output = F.linear(inputs, self.linear.weight * self.mask,
                          self.linear.bias)
        if cond_inputs is not None:
            output = output + self.cond_emb(cond_inputs)
        return output


class MADE(nn.Module):
    """
    An implementation of MADE
    (https://arxiv.org/abs/1502.03509).
    """
    def __init__(self,
                 var_dim,
                 hid_dim,
                 cond_dim=None,
                 act=nn.ReLU()):
        super(MADE, self).__init__()
        input_mask = get_mask(var_dim, hid_dim, var_dim, mask_type='input')
        hidden_mask = get_mask(hid_dim, hid_dim, var_dim)
        output_mask = get_mask(
            hid_dim, var_dim * 2, var_dim, mask_type='output')

        self.joiner = MaskedLinear(var_dim, hid_dim, input_mask, cond_dim)
        self.trunk = nn.Sequential(act,
                                   MaskedLinear(hid_dim, hid_dim,
                                                   hidden_mask), act,
                                   MaskedLinear(hid_dim, var_dim * 2,
                                                   output_mask))

    def forward(self, inputs, cond_inputs=None, mode='forward'):
        if mode not in ["forward", "backward"]:
            raise ValueError(f"Unknown mode: {mode}")
        if mode == 'forward':
            h = self.joiner(inputs, cond_inputs)
            m, a = self.trunk(h).chunk(2, 1)
            u = (inputs - m) * torch.exp(-a)
            return u, -a.sum(-1, keepdim=True)
        else:
            x = torch.zeros_like(inputs)
            for i_col in range(inputs.shape[1]):
                h = self.joiner(x, cond_inputs)
                m, a = self.trunk(h).chunk(2, 1)
                x[:, i_col] = inputs[:, i_col] * torch.exp(a[:, i_col]) + m[:, i_col]
            return x, -a.sum(-1, keepdim=True)


class BatchNormFlow(nn.Module):
    """
    An implementation of a batch normalization layer from
    Density estimation using Real NVP
    (https://arxiv.org/abs/1605.08803).
    """
    def __init__(self, var_dim, momentum=0.0, eps=1e-5):
        super().__init__()
        self.log_gamma = nn.Parameter(torch.zeros(var_dim))
        self.beta = nn.Parameter(torch.zeros(var_dim))
        self.momentum = momentum
        self.eps = eps

        self.register_buffer('running_mean', torch.zeros(var_dim))
        self.register_buffer('running_var', torch.ones(var_dim))

    def forward(self, inputs, cond_inputs=None, mode='forward'):
        if mode not in ["forward", "backward"]:
            raise ValueError(f"Unknown mode: {mode}")
        if mode == 'forward':
            if self.training:
                self.batch_mean = inputs.mean(0)
                self.batch_var = (inputs - self.batch_mean).pow(2).mean(0) + self.eps

                self.running_mean.mul_(self.momentum)
                self.running_var.mul_(self.momentum)
                self.running_mean.add_(self.batch_mean.data *
                                       (1 - self.momentum))
                self.running_var.add_(self.batch_var.data *
                                      (1 - self.momentum))

                mean = self.batch_mean
                var = self.batch_var
            else:
                mean = self.running_mean
                var = self.running_var

            x_hat = (inputs - mean) / var.sqrt()
            y = torch.exp(self.log_gamma) * x_hat + self.beta
            return y, (self.log_gamma - 0.5 * torch.log(var)).sum(
                -1, keepdim=True)
        else:
            if self.training:
                mean = self.batch_mean
                var = self.batch_var
            else:
                mean = self.running_mean
                var = self.running_var

            x_hat = (inputs - self.beta) / torch.exp(self.log_gamma)
            y = x_hat * var.sqrt() + mean
            return y, (-self.log_gamma + 0.5 * torch.log(var)).sum(
                -1, keepdim=True)


class Reverse(nn.Module):
    """
    An implementation of a reversing layer from
    Density estimation using Real NVP
    (https://arxiv.org/abs/1605.08803).
    """
    def __init__(self, var_dim):
        super(Reverse, self).__init__()
        self.perm = np.array(np.arange(0, var_dim)[::-1])
        self.inv_perm = np.argsort(self.perm)

    def forward(self, inputs, cond_inputs=None, mode='forward'):
        if mode not in ["forward", "backward"]:
            raise ValueError(f"Unknown mode: {mode}")

        inds = self.perm if mode == 'forward' else self.inv_perm
        return inputs[:, inds], torch.zeros(
                inputs.size(0), 1, device=inputs.device)


class MAFBlock(nn.Module):
    def __init__(self,
                 var_dim,
                 cond_dim=None,
                 hid_dim=128,
                 momentum=1.0,
                 act=nn.Tanh()):
        super().__init__()
        self.var_dim = var_dim
        self.cond_dim = cond_dim
        self.hid_dim = hid_dim
        self.momentum = momentum
        self.act = act

        self.layers = nn.ModuleList([
                MADE(self.var_dim, self.hid_dim, self.cond_dim, act=self.act),
                BatchNormFlow(self.var_dim, momentum=self.momentum),
                Reverse(self.var_dim)
            ])

        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                if hasattr(layer, 'bias') and layer.bias is not None:
                    layer.bias.data.fill_(0)

    def forward(self, inputs, cond_inputs=None, mode='forward', logdets=None):
        """
        Performs a forward or backward pass for flow modules.
        Args:
            inputs: a tuple of inputs and logdets
            mode: to run direct computation or inverse
        """
        if logdets is None:
            logdets = torch.zeros(inputs.size(0), 1, device=inputs.device)

        if mode not in ["forward", "backward"]:
            raise ValueError(f"Unknown mode: {mode}")

        layers = self.layers if mode == "forward" else self.layers[::-1]
        for layer in layers:
            inputs, logdet = layer(inputs, cond_inputs, mode)
            logdets += logdet
        return inputs, logdets
