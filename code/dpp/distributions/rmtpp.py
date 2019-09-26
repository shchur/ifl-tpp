import torch
import torch.nn as nn
import torch.nn.functional as F

from dpp.nn import BaseModule, Hypernet


class GompertzDistribution(BaseModule):
    """Gompertz distribution used in the RMTPP model.

    References:
        "Recurrent Marked Temporal Point Processes: Embedding
         Event History to Vector", Du et al., KDD 2016
    """
    def __init__(self, config, hypernet_hidden_sizes=[]):
        super().__init__()
        self.use_history(config.use_history)
        self.use_embedding(config.use_embedding)

        # w has to be positive for RMTPP to define a valid density
        # we use softplus as done in the reference implementation
        self.pre_w = nn.Parameter(torch.empty(1))

        self.hypernet = Hypernet(config, param_sizes=[1])
        self.reset_parameters()

    def reset_parameters(self):
        self.pre_w.data.fill_(-5.0)
        self.hypernet.reset_parameters()

    def get_params(self, h, emb):
        if not self.using_history:
            h = None
        if not self.using_embedding:
            emb = None
        bias = self.hypernet(h, emb)
        w = F.softplus(self.pre_w)
        # w = torch.exp(self.pre_w)
        return w, bias

    def log_prob(self, y, h=None, emb=None):
        """Compute log probability of the sample.

        Args:
            y: shape (*)
            h: shape (*, history_size)
            emb: shape (*, embedding_size)

        Returns:
            log_p: shape (*)
        """
        y = y.unsqueeze(-1) + 1e-8
        w, bias = self.get_params(h, emb)
        wt = y * w
        log_p = bias + wt + 1 / w * (bias.exp() - (wt + bias).exp())
        return log_p.squeeze(-1)

    def log_cdf(self, y, h=None, emb=None):
        # TODO: a numerically stable version?
        return torch.log(self.cdf(y, h, emb) + 1e-8)

    def cdf(self, y, h=None, emb=None):
        """Compute CDF of the sample.

        Args:
            y: shape (*)
            h: shape (*, history_size)
            emb: shape (*, embedding_size)

        Returns:
            cdf: shape (*)
        """
        y = y.unsqueeze(-1) + 1e-8
        w, bias = self.get_params(h, emb)
        wt = y * w
        cdf_ = 1 - torch.exp(1 / w * (bias.exp() - (wt + bias).exp()))
        return cdf_.squeeze(-1)

    def intensity(self, y, h=None, emb=None):
        w, bias = self.get_params(h, emb)
        wt = y * w
        return torch.exp(wt + bias)

    def sample(self, n_samples, h=None, emb=None):
        """Can be obtained with inverse CDF transform as

        z ~ U(0, 1)
        t = (log(exp(bias) - w*log(1 - z)) - b) / w
        """
        raise NotImplementedError
