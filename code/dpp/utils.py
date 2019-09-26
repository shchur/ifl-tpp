import numpy as np
import torch.nn as nn

def get_inter_times(arrival_times):
    """Convert arrival times to interevent times."""
    return arrival_times - np.concatenate([[0], arrival_times[:-1]])


def get_arrival_times(inter_times):
    """Convert interevent times to arrival times."""
    return inter_times.cumsum()


class DotDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def clamp_preserve_gradients(x, min, max):
    """Clamp the tensor while preserving gradients in the clamped region."""
    return x + (x.clamp(min, max) - x).detach()
