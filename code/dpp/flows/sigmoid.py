import torch
import torch.nn.functional as F

from torch.distributions import constraints

from .base import Flow


def _clipped_sigmoid(x):
    finfo = torch.finfo(x.dtype)
    return torch.clamp(torch.sigmoid(x), min=finfo.tiny, max=1. - finfo.eps)


class Sigmoid(Flow):
    """Convert samples as y = 1/(1 + exp(-x))."""
    domain = constraints.real
    codomain = constraints.unit_interval

    def forward(self, x, **kwargs):
        y = torch.sigmoid(x)
        log_det_jac = -F.softplus(-x) - F.softplus(x)
        return y, log_det_jac

    def inverse(self, y, **kwargs):
        x = torch.log(y) - torch.log1p(-y)
        inv_log_det_jac = -torch.log(y) - torch.log1p(-y)
        return x, inv_log_det_jac


def Logit():
    """Inverse of the sigmoid transformation."""
    return Sigmoid().get_inverse()
