import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import constraints

from .base import Flow
from dpp.utils import clamp_preserve_gradients


class NonnegativeLinear(nn.Linear):
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

        # Make weight non-negative at initialization
        self.weight.data.abs_()
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        self.weight.data.clamp_(0.0)
        return F.linear(input, self.weight, self.bias)


class MLP(Flow):
    """Multilayer perceptron flow.

    We parametrize the inverse transformation, since we are interested in
    density estimation.

    References:
        "Fully Neural Network based Model for General Temporal Point Processes",
        Omi et al., NeurIPS 2019
    """
    domain = constraints.real
    codomain = constraints.real

    def __init__(self, config, n_layers=2, layer_size=64):
        super().__init__()
        self.use_history(config.use_history)
        self.use_embedding(config.use_embedding)
        if self.using_history:
            self.linear_rnn = nn.Linear(config.history_size, layer_size)
        if self.using_embedding:
            self.linear_emb = nn.Linear(config.embedding_size, layer_size)
        self.linear_time = NonnegativeLinear(1, layer_size)
        self.linear_layers = nn.ModuleList([
            NonnegativeLinear(layer_size, layer_size) for _ in range(n_layers - 1)
        ])
        self.final_layer = NonnegativeLinear(layer_size, 1)

    def forward(self, x, h=None, emb=None):
        raise NotImplementedError

    def inverse(self, y, h=None, emb=None):
        y = y.requires_grad_()
        x = self.cdf(y, h, emb)
        dx_dy = torch.autograd.grad(x, y, torch.ones_like(x), create_graph=True)[0]
        inv_log_det_jac = torch.log(dx_dy + 1e-8)
        y = y.detach()
        return x, inv_log_det_jac

    def cdf(self, y, h=None, emb=None):
        y = y.unsqueeze(-1)
        hidden = self.linear_time(y)
        if self.using_history:
            hidden += self.linear_rnn(h)
        if self.using_embedding:
            hidden += self.linear_emb(emb)
        hidden = torch.tanh(hidden)

        for linear in self.linear_layers:
            hidden = torch.tanh(linear(hidden))

        hidden = self.final_layer(hidden)

        return hidden.squeeze(-1)
