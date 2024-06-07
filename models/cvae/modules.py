import math
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, n_inputs, lat_size, hidden=(10,), act=nn.Tanh()):
        super(Encoder, self).__init__()
        self.model = nn.Sequential()
        for i in range(len(hidden)):
            # add layer
            if i == 0:
                alayer = nn.Linear(n_inputs, hidden[i])
            else:
                alayer = nn.Linear(hidden[i - 1], hidden[i])
            self.model.append(alayer)
            self.model.append(act)

        self.mu = nn.Linear(hidden[-1], lat_size)
        self.log_sigma = nn.Linear(hidden[-1], lat_size)

    def forward(self, X, C=None):
        '''
        Implementation of encoding.

        Parameters:
        -----------
        X: torch.Tensor of shape [batch_size, var_size]
            Input sample to transform.
        C: torch.Tensor of shape [batch_size, cond_size] or None
            Condition values.

        Return:
        -------
        mu: torch.Tensor of shape [batch_size, lat_size]
            Transformed X.
        log_sigma: torch.Tensor of shape [batch_size, lat_size]
            Transformed X.
        '''
        if C is None:
            Z = X
        else:
            Z = torch.cat((X, C), dim=1)
        Z = self.model(Z)
        mu = self.mu(Z)
        log_sigma = self.log_sigma(Z)
        return mu, log_sigma


class Decoder(nn.Module):
    def __init__(self, n_inputs, n_outputs, hidden=(10,), act=nn.Tanh()):
        super(Decoder, self).__init__()
        self.model = nn.Sequential()
        for i in range(len(hidden)):
            # add layer
            if i == 0:
                alayer = nn.Linear(n_inputs, hidden[i])
            else:
                alayer = nn.Linear(hidden[i - 1], hidden[i])
            self.model.append(alayer)
            self.model.append(act)
        # output layer
        self.model.append(nn.Linear(hidden[-1], n_outputs))

    def forward(self, X, C=None):
        '''
        Implementation of decoding.

        Parameters:
        -----------
        X: torch.Tensor of shape [batch_size, lat_size]
            Input sample to transform.
        C: torch.Tensor of shape [batch_size, cond_size] or None
            Condition values.

        Return:
        -------
        X_rec: torch.Tensor of shape [lat_size, n_outputs]
            Transformed X.
        '''
        if C is None:
            Z = X
        else:
            Z = torch.cat((X, C), dim=1)
        X_rec = self.model(Z)
        return X_rec
