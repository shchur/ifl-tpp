import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td

from dpp.nn import BaseModule, Hypernet


class ExponentialDistribution(BaseModule):
    """Exponential distribution (a.k.a. constant intensity model).

    Same model was used in Upadhyay et al., NeurIPS 2018.
    """
    def __init__(self, config, hypernet_hidden_sizes=[]):
        super().__init__()
        self.use_history(config.use_history)
        self.use_embedding(config.use_embedding)

        self.hypernet = Hypernet(config, param_sizes=[1])
        self.reset_parameters()

    def reset_parameters(self):
        self.hypernet.reset_parameters()

    def get_params(self, h, emb):
        if not self.using_history:
            h = None
        if not self.using_embedding:
            emb = None
        lam = self.hypernet(h, emb)
        return F.softplus(lam)

    def log_prob(self, y, h=None, emb=None):
        y = y.unsqueeze(-1) + 1e-8
        lam = self.get_params(h, emb)
        log_p = lam.log() - lam * y
        return log_p.squeeze(-1)

    def log_cdf(self, y, h=None, emb=None):
        y = y.unsqueeze(-1) + 1e-8
        lam = self.get_params(h, emb)
        cdf = 1 - torch.exp(-lam * y) + 1e-8
        # return torch.log1p(-torch.exp(-lam * y)).squeeze(-1)
        # More numerically stable
        return torch.log(-torch.expm1(-lam * y)).squeeze(-1)

    def sample(self, n_samples, h=None, emb=None):
        lam = self.get_params(h, emb)
        dist = td.exponential.Exponential(lam)
        samples = dist.rsample([n_samples])

        if (h is not None):
            first_dims = h.shape[:-1]
        elif (emb is not None):
            first_dims = emb.shape[:-1]
        else:
            first_dims = torch.Size()
        shape = first_dims + torch.Size([n_samples])

        return samples.reshape(shape)
