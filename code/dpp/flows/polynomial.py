import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import constraints

from dpp.flows.base import Flow
from dpp.nn import Hypernet


class Polynomial(Flow):
    """Sum of squares polynomial flow layer.

    We parametrize the inverse transformation, since we are interested in
    density estimation.

    References:
        "Sum-of-Squares Polynomial Flow", Jaini et al., ICML 2019
    """
    domain = constraints.real
    codomain = constraints.real

    def __init__(self, config, max_degree=3, n_terms=4, hypernet_hidden_sizes=[64], min_clip=-5., max_clip=3.):
        super().__init__()
        self.max_degree = max_degree
        self.n_terms = n_terms
        powers = torch.arange(1, max_degree + 2, dtype=torch.get_default_dtype())
        self.register_buffer('powers', powers)

        # (degree + 1) x (degree + 1) matrix of degrees
        powers_mask = self.powers + torch.arange(max_degree + 1).unsqueeze(-1).type_as(powers)
        self.register_buffer('powers_mask', powers_mask)

        # Reciprocals of powers 1 / (p + q + 1)
        recip = powers_mask.reciprocal()
        self.register_buffer('recip', recip)

        self.shift_base = nn.Parameter(torch.zeros(1))

        self.use_history(config.use_history)
        self.use_embedding(config.use_embedding)
        self.min_clip = min_clip
        self.max_clip = max_clip
        self.hypernet = Hypernet(config,
                                 hidden_sizes=hypernet_hidden_sizes,
                                 param_sizes=[self.n_terms * (self.max_degree + 1)])

    def get_params(self, h, emb):
        A = self.hypernet(h, emb)
        A = A.view(-1, self.n_terms, self.max_degree + 1)
        c = self.shift_base
        return A, c

    def forward(self, x, h=None, emb=None):
        raise NotImplementedError

    def inverse(self, y, h=None, emb=None):
        A, c = self.get_params(h, emb)
        # coef has shape (batch_size, n_terms, max_degree + 1, max_degree + 1)
        coef = A.unsqueeze(-1) * A.unsqueeze(-2)
        # y_view has shape (batch_size, 1, 1, 1)
        y_view = y.view(-1, 1, 1, 1)
        # y_pow has shape (batch_size, 1, max_degree + 1, max_degree + 1)
        y_pow = y_view.pow(self.powers_mask)

        x = (y_pow * coef * self.recip).sum(dim=(-1, -2, -3)) + c
        # x has shape (batch_size), reshape to match shape of y
        x = x.view_as(y)

        # Compute inv_log_det_jac
        inv_det_jac = (y_view.pow(self.powers_mask - 1) * coef).sum(dim=(-1, -2, -3))
        inv_log_det_jac = torch.log(inv_det_jac + 1e-8).view_as(y)

        return x, inv_log_det_jac
