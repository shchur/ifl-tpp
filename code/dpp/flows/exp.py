from torch.distributions import constraints
from .base import Flow


class Exp(Flow):
    """Convert samples as y = exp(x)."""
    domain = constraints.real
    codomain = constraints.positive

    def __init__(self):
        super().__init__()
        self.epsilon = 1e-8

    def forward(self, x, **kwargs):
        y = x.exp() - self.epsilon
        log_det_jac = x
        return y, log_det_jac

    def inverse(self, y, **kwargs):
        x = (y + self.epsilon).log()
        inv_log_det_jac = -x
        return x, inv_log_det_jac
