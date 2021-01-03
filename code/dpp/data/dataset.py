from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import torch
import torch.utils.data

from .batch import Batch
from .sequence import Sequence


dataset_dir = Path(__file__).parents[3] / "data"


def list_datasets(folder: Union[Path, str] = dataset_dir):
    """List all datasets in the data folder."""
    files = sorted(file.stem for file in Path(folder).iterdir() if file.suffix == ".pkl")
    dirs = sorted(d for d in Path(folder).iterdir() if d.is_dir() and not d.name.startswith("_"))
    for d in dirs:
        if d.is_dir() and not d.stem.startswith("_"):
            for f in d.iterdir():
                if f.suffix == ".pkl":
                    files.append(d.stem + "/" + f.stem)
    return sorted(files)


def load_dataset(name: str, folder: Union[Path, str] = dataset_dir):
    if not name.endswith(".pkl"):
        name += ".pkl"
    path_to_file = Path(folder) / name
    dataset = torch.load(str(path_to_file))

    def get_inter_times(seq: dict):
        """Get inter-event times from a sequence."""
        return np.ediff1d(np.concatenate([[seq["t_start"]], seq["arrival_times"], [seq["t_end"]]]))

    sequences = [
        Sequence(
            inter_times=get_inter_times(seq),
            marks=seq.get("marks"),
            t_start=seq.get("t_start"),
            t_end=seq.get("t_end")
        )
        for seq in dataset["sequences"]
    ]
    return SequenceDataset(sequences=sequences, num_marks=dataset.get("num_marks", 1))


class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, sequences: List[Sequence], num_marks=1):
        self.sequences = sequences
        self.num_marks = num_marks

    def __getitem__(self, item):
        return self.sequences[item]

    def __len__(self):
        return len(self.sequences)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({len(self)})"

    def __add__(self, other: "SequenceDataset") -> "SequenceDataset":
        if not isinstance(other, SequenceDataset):
            raise ValueError(f"other must be a SequenceDataset (got {type(other)})")
        new_num_marks = max(self.num_marks, other.num_marks)
        new_sequences = self.sequences + other.sequences
        return SequenceDataset(new_sequences, num_marks=new_num_marks)

    def get_dataloader(
            self, batch_size: int = 32, shuffle: bool = True
    ) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self, batch_size=batch_size, shuffle=shuffle, collate_fn=Batch.from_list
        )

    def train_val_test_split(
            self, train_size=0.6, val_size=0.2, test_size=0.2, seed=None, shuffle=True,
    ) -> Tuple["SequenceDataset", "SequenceDataset", "SequenceDataset"]:
        """Split the sequences into train, validation and test subsets."""
        if train_size < 0 or val_size < 0 or test_size < 0:
            raise ValueError("train_size, val_size and test_size must be >= 0.")
        if train_size + val_size + test_size != 1.0:
            raise ValueError("train_size, val_size and test_size must add up to 1.")

        if seed is not None:
            np.random.seed(seed)

        all_idx = np.arange(len(self))
        if shuffle:
            np.random.shuffle(all_idx)

        train_end = int(train_size * len(self))  # idx of the last train sequence
        val_end = int((train_size + val_size) * len(self))  # idx of the last val seq

        train_idx = all_idx[:train_end]
        val_idx = all_idx[train_end:val_end]
        test_idx = all_idx[val_end:]

        train_sequences = [self.sequences[idx] for idx in train_idx]
        val_sequences = [self.sequences[idx] for idx in val_idx]
        test_sequences = [self.sequences[idx] for idx in test_idx]

        return (
            SequenceDataset(train_sequences, num_marks=self.num_marks),
            SequenceDataset(val_sequences, num_marks=self.num_marks),
            SequenceDataset(test_sequences, num_marks=self.num_marks),
        )

    def get_inter_time_statistics(self):
        """Get the mean and std of log(inter_time)."""
        all_inter_times = torch.cat([seq.inter_times[:-1] for seq in self.sequences])
        mean_log_inter_time = all_inter_times.log().mean()
        std_log_inter_time = all_inter_times.log().std()
        return mean_log_inter_time, std_log_inter_time

    @property
    def total_num_events(self):
        return sum(len(seq) - 1 for seq in self.sequences)
