from typing import Callable
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from utils.misc import get_affine_params, swish

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20


class MLPNetwork(nn.Module):
    """
    MLP network (can be used as value or policy)
    When used as policy, it can output mean and std for Gaussian or deterministic action
    """

    def __init__(
        self,
        input_dim: int,
        out_dim: int,
        hidden_dim: int = 64,
        nonlin: Callable[[torch.Tensor], torch.Tensor] = F.relu,
        norm_in: bool = True,
    ):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(MLPNetwork, self).__init__()
        if norm_in:  # normalize inputs
            self.in_fn = nn.BatchNorm1d(input_dim)
            self.in_fn.weight.data.fill_(1)
            self.in_fn.bias.data.fill_(0)
        else:
            self.in_fn = lambda x: x
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.fc_log_std = nn.Linear(hidden_dim, out_dim)
        self.nonlin = nonlin
        self.out_fn = lambda x: x

    def forward(
        self,
        X: torch.Tensor,
    ):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): Output of network (actions, values, etc)
        """
        h1 = self.nonlin(self.fc1(self.in_fn(X)))
        h2 = self.nonlin(self.fc2(h1))
        h3 = self.fc3(h2)
        return self.out_fn(h3)


# modified from https://github.com/quanvuong/handful-of-trials-pytorch/blob/master/config/utils.py
class PtModel(nn.Module):
    def __init__(self, ensemble_size: int, dim_in: int, dim_out: int, hidden_dim: int):
        super().__init__()

        self.n_net = ensemble_size

        self.dim_in = dim_in
        self.dim_out = dim_out

        self.lin0_w, self.lin0_b = get_affine_params(ensemble_size, dim_in, hidden_dim)

        self.lin1_w, self.lin1_b = get_affine_params(
            ensemble_size, hidden_dim, hidden_dim
        )

        self.lin2_w, self.lin2_b = get_affine_params(
            ensemble_size, hidden_dim, hidden_dim
        )

        self.lin3_w, self.lin3_b = get_affine_params(ensemble_size, hidden_dim, dim_out)

        self.inputs_mu = nn.Parameter(torch.zeros(1, dim_in), requires_grad=False)
        self.inputs_sigma = nn.Parameter(torch.zeros(1, dim_in), requires_grad=False)

        self.max_logvar = nn.Parameter(
            torch.ones(1, dim_out // 2, dtype=torch.float32) / 2.0
        )
        self.min_logvar = nn.Parameter(
            -torch.ones(1, dim_out // 2, dtype=torch.float32) * 10.0
        )

    def compute_decays(self):

        lin0_decays = 0.0001 * (self.lin0_w**2).sum() / 2.0
        lin1_decays = 0.00025 * (self.lin1_w**2).sum() / 2.0
        lin2_decays = 0.00025 * (self.lin2_w**2).sum() / 2.0
        lin3_decays = 0.0005 * (self.lin3_w**2).sum() / 2.0

        return lin0_decays + lin1_decays + lin2_decays + lin3_decays

    def fit_input_stats(self, data: torch.Tensor):

        mu = torch.mean(data, 0, keepdims=True)
        sigma = torch.std(data, 0, keepdims=True)
        sigma[sigma < 1e-12] = 1.0

        self.inputs_mu.data = mu.to(next(self.parameters()).device).float()
        self.inputs_sigma.data = sigma.to(next(self.parameters()).device).float()

    def forward(self, inputs: torch.Tensor, ret_logvar: bool = False):

        # Transform inputs
        inputs = (inputs - self.inputs_mu) / self.inputs_sigma

        inputs = inputs.matmul(self.lin0_w) + self.lin0_b

        inputs = swish(inputs)

        inputs = inputs.matmul(self.lin1_w) + self.lin1_b
        inputs = swish(inputs)

        inputs = inputs.matmul(self.lin2_w) + self.lin2_b
        inputs = swish(inputs)

        inputs = inputs.matmul(self.lin3_w) + self.lin3_b

        mean = inputs[..., : self.dim_out // 2]

        logvar = inputs[..., self.dim_out // 2 :]

        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)

        if ret_logvar:
            return mean, logvar

        return mean, torch.exp(logvar)
