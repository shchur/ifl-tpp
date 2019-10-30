import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td

from torch.distributions import constraints

from dpp.nn import BaseModule, Hypernet
from dpp.utils import clamp_preserve_gradients


def inverse_sigmoid(x):
    # Clamp tiny values (<1e-38 for float32)
    finfo = torch.finfo(x.dtype)
    x = x.clamp(min=finfo.tiny, max=1. - finfo.eps)
    return torch.log(x) - torch.log1p(-x)


def logistic_sample(means, log_scales):
    if means.shape != log_scales.shape:
        raise ValueError("Shapes of means and scales don't match.")
    z = torch.rand(means.shape)
    return torch.exp(log_scales) * inverse_sigmoid(z) + means


# Credit: https://github.com/aravindsrinivas/flowpp/blob/master/flows/logistic.py
def logistic_logpdf(x, mean, log_scale):
    z = (x - mean) * torch.exp(-log_scale)
    return z - log_scale - 2 * F.softplus(z)


def logistic_logcdf(x, mean, log_scale):
    z = (x - mean) * torch.exp(-log_scale)
    return F.logsigmoid(z)


def mixlogistic_logpdf(x, prior_logits, means, log_scales):
    log_prior = F.log_softmax(prior_logits, dim=-1)
    return torch.logsumexp(
        log_prior + logistic_logpdf(x.unsqueeze(-1), means, log_scales),
        dim=-1
    )

def mixlogistic_logcdf(x, prior_logits, means, log_scales):
    log_prior = F.log_softmax(prior_logits, dim=-1)
    return torch.logsumexp(
        log_prior + logistic_logcdf(x.unsqueeze(-1), means, log_scales),
        dim=-1
    )


class LogisticMixtureDistribution(BaseModule):
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
        """Generate model parameters based on the history and embeddings.

        Args:
            h: history embedding, shape [*, rnn_hidden_size]
            emb: sequence embedding, shape [*, embedding_size]

        Returns:
            prior_logits: shape [*, n_components]
            means: shape [*, n_components]
            log_scales: shape [*, n_components]
        """
        if not self.using_history:
            h = None
        if not self.using_embedding:
            emb = None
        prior_logits, means, log_scales = self.hypernet(h, emb)
        # Clamp values that go through exp for numerical stability
        prior_logits = clamp_preserve_gradients(prior_logits, self.min_clip, self.max_clip)
        log_scales = clamp_preserve_gradients(log_scales, self.min_clip, self.max_clip)
        return prior_logits, means, log_scales

    def log_prob(self, y, h=None, emb=None):
        prior_logits, means, log_scales = self.get_params(h, emb)
        return mixlogistic_logpdf(y, prior_logits, means, log_scales)

    def log_cdf(self, y, h=None, emb=None):
        prior_logits, means, log_scales = self.get_params(h, emb)
        return mixlogistic_logcdf(y, prior_logits, means, log_scales)

    def sample(self, n_samples, h=None, emb=None):
        """Draw samples from the model.

        Args:
            n_samples: number of samples to generate.
            h: hidden state, shape [*, rnn_hidden_size]
            emb: sequence embedding, shape [*, embedding_size]

        Returns:
            samples: shape [*, n_samples]
        """
        with torch.no_grad():
            prior_logits, means, log_scales = self.get_params(h, emb)
            # model parameters should have two dimensions for bmm to work
            # first dimensions will be restored later
            prior_logits = prior_logits.view(-1, self.n_components)
            means = means.view(-1, self.n_components)
            log_scales = log_scales.view(-1, self.n_components)

            categorical = td.Categorical(logits=prior_logits)
            z = categorical.sample([n_samples])
            # z has shape [n_samples, *], convert to [*, n_samples]
            dim_order = np.arange(len(prior_logits.shape))
            dim_order = tuple(np.concatenate([dim_order[1:], [0]]))
            z = z.permute(dim_order).contiguous()
            # z_oh has shape [*, n_samples, n_components]
            # convert it to [*, n_components, n_samples] for bmm to work
            z_oh = F.one_hot(z, num_classes=self.n_components).float().transpose(-2, -1)
            # add extra dim to means and log_scales for bmm to work
            means.unsqueeze_(-2)
            log_scales.unsqueeze_(-2)
            # select the correct component for each sample
            means_select = torch.bmm(means, z_oh)
            log_scales_select = torch.bmm(log_scales, z_oh)
            means_select.squeeze_(-2)
            log_scales_select.squeeze_(-2)
            # means_select and log_scales_select have shape [*, n_samples]
            samples = logistic_sample(means_select, log_scales_select)
            # reshape the samples back to the original shape
            if (h is not None):
                first_dims = h.shape[:-1]
            elif (emb is not None):
                first_dims = emb.shape[:-1]
            else:
                first_dims = torch.Size()
            shape = first_dims + torch.Size([n_samples])
        return samples.reshape(shape)
