import torch
import torch.nn.functional as F
from torch.distributions import constraints

from .base import Flow


class Softplus(Flow):
    """Convert samples as y = log(1 + exp(x))."""
    domain = constraints.real
    codomain = constraints.positive

    def __init__(self):
        super().__init__()
        self.epsilon = 1e-8

    def forward(self, x, **kwargs):
        y = F.softplus(x + self.epsilon)
        log_det_jac = -F.softplus(-x - self.epsilon)
        return y, log_det_jac

    def inverse(self, y, **kwargs):
        x = y + torch.log(-torch.expm1(-y - self.epsilon))
        inv_log_det_jac = -torch.log(-torch.expm1(-y - self.epsilon))
        return x, inv_log_det_jac


def InverseSoftplus():
    """Inverse of the softplus transformation."""
    return Softplus().get_inverse()
