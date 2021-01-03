import torch

from typing import List, Optional

from dpp.utils import DotDict, pad_sequence
from .sequence import Sequence


class Batch(DotDict):
    """
    A batch consisting of padded sequences.

    Usually constructed using the from_list method.

    Args:
        inter_times: Padded inter-event times, shape (batch_size, seq_len)
        mask: Mask indicating which inter_times correspond to observed events
            (and not to padding), shape (batch_size, seq_len)
        marks: Padded marks associated with each event, shape (batch_size, seq_len)
    """
    def __init__(self, inter_times: torch.Tensor, mask: torch.Tensor, marks: Optional[torch.Tensor] = None, **kwargs):
        self.inter_times = inter_times
        self.mask = mask
        self.marks = marks

        for key, value in kwargs.items():
            self[key] = value

        self._validate_args()

    @property
    def size(self):
        """Number of sequences in the batch."""
        return self.inter_times.shape[0]

    @property
    def max_seq_len(self):
        """Length of the padded sequences."""
        return self.inter_times.shape[1]

    def _validate_args(self):
        """Check if all tensors have correct shapes."""
        if self.inter_times.ndim != 2:
            raise ValueError(
                f"inter_times must be a 2-d tensor (got {self.inter_times.ndim}-d)"
            )
        if self.mask.shape != (self.size, self.max_seq_len):
            raise ValueError(
                f"mask must be of shape (batch_size={self.size}, "
                f" max_seq_len={self.max_seq_len}), got {self.mask.shape}"
            )
        if self.marks is not None and self.marks.shape != (self.size, self.max_seq_len):
            raise ValueError(
                f"marks must be of shape (batch_size={self.size},"
                f" max_seq_len={self.max_seq_len}), got {self.marks.shape}"
            )

    @staticmethod
    def from_list(sequences: List[Sequence]):
        batch_size = len(sequences)
        # Remember that len(seq) = len(seq.inter_times) = len(seq.marks) + 1
        # since seq.inter_times also includes the survival time until t_end
        max_seq_len = max(len(seq) for seq in sequences)
        inter_times = pad_sequence([seq.inter_times for seq in sequences], max_len=max_seq_len)

        dtype = sequences[0].inter_times.dtype
        device = sequences[0].inter_times.device
        mask = torch.zeros(batch_size, max_seq_len, device=device, dtype=dtype)

        for i, seq in enumerate(sequences):
            mask[i, :len(seq) - 1] = 1

        if sequences[0].marks is not None:
            marks = pad_sequence([seq.marks for seq in sequences], max_len=max_seq_len)
        else:
            marks = None

        return Batch(inter_times, mask, marks)

    def get_sequence(self, idx: int) -> Sequence:
        length = int(self.mask[idx].sum(-1)) + 1
        inter_times = self.inter_times[idx, :length]
        if self.marks is not None:
            marks = self.marks[idx, :length - 1]
        else:
            marks = None
        # TODO: recover additional attributes (passed through kwargs) from the batch
        return Sequence(inter_times=inter_times, marks=marks)

    def to_list(self) -> List[Sequence]:
        """Convert a batch into a list of variable-length sequences."""
        return [self.get_sequence(idx) for idx in range(self.size)]
