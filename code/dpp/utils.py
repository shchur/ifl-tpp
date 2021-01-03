import torch

from typing import Any, List, Optional


def _size_repr(key: str, item: Any) -> str:
    """String containing the size / shape of an object (e.g. a tensor, array)."""
    if isinstance(item, torch.Tensor) and item.dim() == 0:
        out = item.item()
    elif isinstance(item, torch.Tensor):
        out = str(list(item.size()))
    elif isinstance(item, list):
        out = f"[{len(item)}]"
    else:
        out = str(item)

    return f"{key}={out}"

class DotDict:
    """Dictionary where elements can be accessed as dict.entry."""

    def __getitem__(self, key):
        return getattr(self, key, None)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def keys(self):
        keys = [key for key in self.__dict__.keys() if self[key] is not None]
        keys = [key for key in keys if key[:2] != "__" and key[-2:] != "__"]
        return keys

    def __iter__(self):
        for key in sorted(self.keys()):
            yield key, self[key]

    def __contains__(self, key):
        return key in self.keys()

    def __repr__(self):
        info = [_size_repr(key, item) for key, item in self]
        return f"{self.__class__.__name__}({', '.join(info)})"


def clamp_preserve_gradients(x: torch.Tensor, min: float, max: float) -> torch.Tensor:
    """Clamp the tensor while preserving gradients in the clamped region."""
    return x + (x.clamp(min, max) - x).detach()


def pad_sequence(
        sequences: List[torch.Tensor],
        padding_value: float = 0,
        max_len: Optional[int] = None,
):
    r"""Pad a list of variable length Tensors with ``padding_value``"""
    dtype = sequences[0].dtype
    device = sequences[0].device
    seq_shape = sequences[0].shape
    trailing_dims = seq_shape[1:]
    if max_len is None:
        max_len = max([s.size(0) for s in sequences])
    out_dims = (len(sequences), max_len) + trailing_dims

    out_tensor = torch.empty(*out_dims, dtype=dtype, device=device).fill_(padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        # use index notation to prevent duplicate references to the tensor
        out_tensor[i, :length, ...] = tensor

    return out_tensor
