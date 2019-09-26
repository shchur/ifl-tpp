import torch
import torch.nn as nn
import torch.distributions as td

from dpp.flows.base import Flow
from dpp.nn import BaseModule


class TransformedDistribution(BaseModule):
    """Distribution transformed with a series of normalizing flows.

    This is the class on which all normalizing flows models are based.
    """
    def __init__(self, transforms, base_dist=td.Uniform(0., 1.)):
        super().__init__()
        if isinstance(transforms, Flow):
            self.transforms = nn.ModuleList([transforms, ])
        elif isinstance(transforms, list):
            if not all(isinstance(t, Flow) for t in transforms):
                raise ValueError("transforms must be a Flow or a list of Flows")
            self.transforms = nn.ModuleList(transforms)
        else:
            raise ValueError(f"transforms must a Flow or a list, but was {type(transform)}")
        self.base_dist = base_dist

    def log_prob(self, y, h=None, emb=None):
        """Compute log probability of the samples.

        The probability is computed by going through the sequence of transformation
        in reversed order and using the change of variables formula.

        History embedding h and sequence embedding emb are used to generate
        parameters of the transformations / distributions.

        Args:
            y: Samples to score, shape (batch_size, seq_len)
            h: RNN encoding of the history, shape (batch_size, seq_len, rnn_hidden_size)
            emb: Sequence embedding, shape (batch_size, seq_len, embedding_size)

        Returns:
            log_p: shape (batch_size, seq_len)
        """
        log_p = 0.0
        for transform in reversed(self.transforms):
            x, inv_log_det_jacobian = transform.inverse(y, h=h, emb=emb)
            log_p += inv_log_det_jacobian
            y = x

        log_p += self.base_dist.log_prob(x, h=h, emb=emb)
        return log_p

    def log_cdf(self, y, h=None, emb=None):
        for transform in reversed(self.transforms):
            x, _ = transform.inverse(y, h=h, emb=emb)
            y = x
        return self.base_dist.log_cdf(y, h=h, emb=emb)

    def _sample(self, n_samples=1, h=None, emb=None, reparametrization=False):
        if reparametrization:
            x = self.base_dist.rsample(n_samples, h=h, emb=emb)
        else:
            x = self.base_dist.sample(n_samples, h=h, emb=emb)
        for transform in self.transforms:
            x = transform.forward(x, h=h, emb=emb)
            # Forward function outputs f(x), log_det_jac
            # Select only f(x)
            if isinstance(x, tuple):
                x = x[0]
        return x

    def sample(self, n_samples=1, h=None, emb=None):
        """Sample from the transformed distribution."""
        return self._sample(n_samples, h, emb, reparametrization=False)

    def rsample(self, n_samples=1, h=None, emb=None):
        """Sample from the transformed distribution with reparametrization."""
        return self._sample(n_samples, h, emb, reparametrization=True)
