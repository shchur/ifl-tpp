import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import constraints

from dpp.flows.base import Flow
from dpp.nn import Hypernet
from dpp.utils import clamp_preserve_gradients
from dpp.distributions.logistic_mixture import logistic_logcdf, logistic_logpdf, mixlogistic_logcdf, mixlogistic_logpdf


class LogisticMixtureFlow(Flow):
    """Deep Sigmoidal Flow layer.

    The transformation is based on the CDF of a logistic mixture distribution.

    We parametrize the inverse transformation, since we are interested in
    density estimation.

    References:
        "Neural Autoregressive Flows", Huang et al., ICML 2018
    """
    domain = constraints.unit_interval
    codomain = constraints.real

    def __init__(self, config, n_components=32, hypernet_hidden_sizes=[64], min_clip=-5., max_clip=3.):
        super().__init__()
        self.n_components = n_components

        self.use_history(config.use_history)
        self.use_embedding(config.use_embedding)
        self.min_clip = min_clip
        self.max_clip = max_clip
        self.hypernet = Hypernet(config,
                                 hidden_sizes=hypernet_hidden_sizes,
                                 param_sizes=[n_components, n_components, n_components])

    def get_params(self, h, emb):
        if not self.using_history:
            h = None
        if not self.using_embedding:
            emb = None
        prior_logits, means, log_scales = self.hypernet(h, emb)
        # Clamp values that go through exp for numerical stability
        prior_logits = clamp_preserve_gradients(prior_logits, self.min_clip, self.max_clip)
        log_scales = clamp_preserve_gradients(log_scales, self.min_clip, self.max_clip)
        return prior_logits, means, log_scales

    def forward(self, x, h=None, emb=None):
        raise NotImplementedError

    def inverse(self, y, h=None, emb=None):
        prior_logits, means, log_scales = self.get_params(h, emb)
        x = mixlogistic_logcdf(y, prior_logits, means, log_scales).exp()
        inv_log_det_jac = mixlogistic_logpdf(y, prior_logits, means, log_scales)
        return x, inv_log_det_jac
