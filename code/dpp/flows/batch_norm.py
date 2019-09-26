import torch
import torch.nn as nn

from torch.distributions import constraints

from dpp.flows.base import Flow


class BatchNorm(Flow):
    """Batch normalization flow.

    Apply an affine transformation using estimated batch statistics
    such that the data has zero mean and unit variance.

    References:
        "Density estimation using Real NVP", Dinh et al., ICLR 2017
    """
    domain = constraints.real
    codomain = constraints.real

    def __init__(self, momentum=0.1, epsilon=1e-5):
        super().__init__()
        self.momentum = momentum
        self.epsilon = epsilon

        self.register_buffer('moving_mean', torch.zeros(1))
        self.register_buffer('moving_variance', torch.ones(1))

    def forward(self, x, **kwargs):
        return x * torch.sqrt(self.moving_variance + self.epsilon) + self.moving_mean

    def inverse(self, y, **kwargs):
        if self.training:
            mean, var = y.mean(), y.var()

            self.moving_mean.mul_(1 - self.momentum).add_(mean * self.momentum)
            self.moving_variance.mul_(1 - self.momentum).add_(var * self.momentum)
        else:
            mean, var = self.moving_mean, self.moving_variance

        std = torch.sqrt(var + self.epsilon)
        x = (y - mean) / std
        inv_log_det_jac = -torch.log(std).expand(y.shape)
        return x, inv_log_det_jac
