"""
Different models for the distribution of inter-event times in a TPP.
All models produce parameters based on the history embedding `h` and
feature/sequence embedding `emb`.
"""
import torch.distributions as td
import dpp

import dpp.distributions as dist


__all__ = [
    'LogNormMix',
    'Exponential',
    'FullyNeuralNet',
    'RMTPP',
    'SOSPolynomial',
    'DeepSigmoidalFlow',
]


def LogNormMix(config, n_components=64, hypernet_hidden_sizes=[64], scale_init=1.0,
               shift_init=0.0, trainable_affine=False, use_sofplus=False, **kwargs):
    """Mixture of log-normal distributions.

    Denoted as 'LogNormMix' in our paper.
    Defined in Section 3.2, 3.3 and Appendix D.2.
    """
    base_dist = dist.NormalMixtureDistribution(config,
                                               n_components=n_components,
                                               hypernet_hidden_sizes=hypernet_hidden_sizes)
    transforms = [
        dpp.flows.FixedAffine(scale_init, shift_init, trainable=trainable_affine, use_shift=True),
        dpp.flows.Softplus() if use_sofplus else dpp.flows.Exp(),
    ]
    return dpp.flows.TransformedDistribution(transforms, base_dist)


def Exponential(config, n_components=64, hypernet_hidden_sizes=[64], scale_init=1.0,
                shift_init=0.0, trainable_affine=False, use_sofplus=False, **kwargs):
    """Constant intensity model (a.k.a. exponential distribution)

    References:
        "Deep reinforcement learning of marked temporal point processes",
        Upadhyay et al., NeurIPS 2018.
    """
    base_dist = dist.ExponentialDistribution(config, hypernet_hidden_sizes=hypernet_hidden_sizes)
    transforms = [
        dpp.flows.FixedAffine(scale_init=scale_init, trainable=trainable_affine),
    ]
    return dpp.flows.TransformedDistribution(transforms, base_dist)


def FullyNeuralNet(config, n_layers=2, layer_size=64, scale_init=1.0, trainable_affine=False, **kwargs):
    """Fully neural network intensity model.

    References:
        "Fully Neural Network based Model for General Temporal Point Processes",
        Omi et al., NeurIPS 2019
    """
    base_dist = dist.FullyNN(config, n_layers=n_layers, layer_size=layer_size)
    transforms = [
        dpp.flows.FixedAffine(scale_init=scale_init, trainable=trainable_affine),
    ]
    return dpp.flows.TransformedDistribution(transforms, base_dist)


def RMTPP(config, scale_init=1.0, trainable_affine=False, hypernet_hidden_sizes=[], **kwargs):
    """Exponential intensity (RMTPP) model.

    References:
        "Recurrent Marked Temporal Point Processes: Embedding
         Event History to Vector", Du et al., KDD 2016
    """
    base_dist = dist.GompertzDistribution(config,
                                          hypernet_hidden_sizes=hypernet_hidden_sizes)
    transforms = [
        dpp.flows.FixedAffine(scale_init=scale_init, trainable=trainable_affine),
    ]
    return dpp.flows.TransformedDistribution(transforms, base_dist)


def SOSPolynomial(config, scale_init=1.0, trainable_affine=False, hypernet_hidden_sizes=[],
                  n_layers=1, max_degree=3, n_terms=4, **kwargs):
    """Sum of squares polynomial flow model.

    Defined in Appendix D.5 of our paper.

    References:
        "Sum-of-Squares Polynomial Flow", Jaini et al., ICML 2019
    """
    base_dist = dist.Uniform(0., 1.)
    transforms = [dpp.flows.Logit()]
    for _ in range(n_layers):
        transforms.append(dpp.flows.Polynomial(config, max_degree=max_degree, n_terms=n_terms))
        transforms.append(dpp.flows.BatchNorm())

    transforms.append(dpp.flows.Exp())
    if scale_init != 1.0 or trainable_affine:
        transforms.append(dpp.flows.FixedAffine(scale_init=scale_init, trainable=trainable_affine))
    return dpp.flows.TransformedDistribution(transforms, base_dist)


def DeepSigmoidalFlow(config, n_layers=2, layer_size=64,
                      scale_init=1.0, trainable_affine=False, hypernet_hidden_sizes=[],
                      **kwargs):
    """Deep sigmoidal flow model.

    Defined in Appendix D.4 of our paper.

    References:
        "Neural Autoregressive Flows", Huang et al., ICML 2018
    """
    base_dist = dist.Uniform(0., 1.)
    transforms = [
        dpp.flows.LogisticMixtureFlow(config,
                                      n_components=layer_size,
                                      hypernet_hidden_sizes=hypernet_hidden_sizes),
        dpp.flows.BatchNorm()
    ]

    # Each DSF block g : [0, 1] -> R consists of (Logit + LogMix + BatchNorm)
    for _ in range(n_layers - 1):
        transforms.append(dpp.flows.Logit())
        transforms.append(dpp.flows.LogisticMixtureFlow(config,
                                                        n_components=layer_size,
                                                        hypernet_hidden_sizes=hypernet_hidden_sizes))
        transforms.append(dpp.flows.BatchNorm())

    transforms.append(dpp.flows.Exp())
    if scale_init != 1.0 or trainable_affine:
        transforms.append(dpp.flows.FixedAffine(scale_init, trainable=trainable_affine))
    return dpp.flows.TransformedDistribution(transforms, base_dist)
