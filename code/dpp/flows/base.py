import torch
import torch.nn as nn
import torch.distributions as td

from dpp.nn import BaseModule


class Flow(BaseModule):
    """Base class for transforms with learnable parameters."""
    def __init__(self):
        super().__init__()

    def forward(self, x, **kwargs):
        """Compute f(x) and log_abs_det_jac(x)."""
        raise NotImplementedError

    def inverse(self, y, **kwargs):
        """Compute f^-1(y) and inv_log_abs_det_jac(y)."""
        raise NotImplementedError

    def get_inverse(self):
        """Get inverse transformation."""
        return Inverse(self)


class Inverse(Flow):
    def __init__(self, base_flow):
        super().__init__()
        self.base_flow = base_flow
        if hasattr(base_flow, 'domain'):
            self.codomain = base_flow.domain
        if hasattr(base_flow, 'codomain'):
            self.domain = base_flow.codomain

    def forward(self, x, **kwargs):
        return self.base_flow.inverse(x, **kwargs)

    def inverse(self, y, **kwargs):
        return self.base_flow.forward(y, **kwargs)
