import torch
import torch.distributions as td

__all__ = [
    'Exponential',
    'Uniform',
]

class Exponential(td.Exponential):
    """Exponential distribution with unit rate.

    Methods can take arbitrary kwargs (such as history embedding h).
    This is needed for compatibility with dpp.flows.TransformedDistribution.
    """
    def log_prob(self, value, **kwargs):
        return super().log_prob(value)

    def log_cdf(self, value, **kwargs):
        return torch.log(super().cdf(value) + 1e-8)


class Uniform(td.Uniform):
    """Uniform(0, 1) distribution.

    Methods can take arbitrary kwargs (such as history embedding h).
    This is needed for compatibility with dpp.flows.TransformedDistribution.
    """
    def log_prob(self, value, **kwargs):
        return torch.zeros_like(value)

    def log_cdf(self, value, **kwargs):
        result = torch.log(value)
        return result.clamp(max=0.0)
