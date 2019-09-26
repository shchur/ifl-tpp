import numpy as np
import torch
import torch.utils.data as data_utils

from pathlib import Path
from sklearn.model_selection import train_test_split

dataset_dir = Path(__file__).parents[2] / 'data'

class Batch():
    def __init__(self, in_time, out_time, length, index=None, in_mark=None, out_mark=None):
        self.in_time = in_time
        self.out_time = out_time
        self.length = length
        self.index = index
        self.in_mark = in_mark.long()
        self.out_mark = out_mark.long()

def load_dataset(name, normalize_min_max=False, log_mode=True):
    """Load dataset."""
    if not name.endswith('.npz'):
        name += '.npz'
    loader = dict(np.load(dataset_dir / name, allow_pickle=True))
    arrival_times = loader['arrival_times']
    marks = loader.get('marks', None)
    num_classes = len(set([x for s in marks for x in s])) if marks is not None else 1
    delta_times = [np.concatenate([[1.0], np.ediff1d(time)]) for time in arrival_times]
    return SequenceDataset(delta_times, marks=marks, num_classes=num_classes, log_mode=log_mode)


def list_datasets():
    check = lambda x: x.is_file() and x.suffix == '.npz'
    file_list = [x.stem for x in (dataset_dir).iterdir() if check(x)]
    file_list += ['synth/' + x.stem for x in (dataset_dir / 'synth').iterdir() if check(x)]
    return file_list


def collate(batch):
    batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)
    in_time = [item[0] for item in batch]
    out_time = [item[1] for item in batch]
    in_marks = [item[2] for item in batch]
    out_marks = [item[3] for item in batch]
    index = torch.tensor([item[4] for item in batch])
    length = torch.Tensor([len(item) for item in in_time])

    in_time = torch.nn.utils.rnn.pad_sequence(in_time, batch_first=True)
    out_time = torch.nn.utils.rnn.pad_sequence(out_time, batch_first=True)
    in_mark = torch.nn.utils.rnn.pad_sequence(in_marks, batch_first=True)
    out_mark = torch.nn.utils.rnn.pad_sequence(out_marks, batch_first=True)
    in_time[:,0] = 0 # set first to zeros

    return Batch(in_time, out_time, length, in_mark=in_mark, out_mark=out_mark, index=index)


class SequenceDataset(data_utils.Dataset):
    """Dataset class containing variable length sequences.

    Args:
        delta_times: Inter-arrival times between events. List of variable length sequences.

    """
    def __init__(self, delta_times=None, marks=None, in_times=None, out_times=None,
                 in_marks=None, out_marks=None, index=None, log_mode=True, num_classes=1):
        self.num_classes = num_classes

        if delta_times is not None:
            self.in_times = [torch.Tensor(t[:-1]) for t in delta_times]
            self.out_times = [torch.Tensor(t[1:]) for t in delta_times]
        else:
            if (not all(torch.is_tensor(t) for t in in_times) or
                not all(torch.is_tensor(t) for t in out_times)):
                raise ValueError("in and out times and marks must all be torch.Tensors")
            self.in_times = in_times
            self.out_times = out_times

        if marks is not None:
            self.in_marks = [torch.Tensor(m[:-1]) for m in marks]
            self.out_marks = [torch.Tensor(m[1:]) for m in marks]
        else:
            self.in_marks = in_marks
            self.out_marks = out_marks

        if index is None:
            index = torch.arange(len(self.in_times))
        if not torch.is_tensor(index):
            index = torch.tensor(index)

        if self.in_marks is None:
            self.in_marks = [torch.zeros_like(x) for x in self.in_times]
            self.out_marks = [torch.zeros_like(x) for x in self.out_times]

        self.index = index
        self.validate_times()
        # Apply log transformation to inputs
        if log_mode:
            self.in_times = [t.log_() for t in self.in_times]

    @property
    def num_series(self):
        return len(self.in_times)

    def validate_times(self):
        if len(self.in_times) != len(self.out_times):
            raise ValueError("in_times and out_times have different lengths.")

        if len(self.index) != len(self.in_times):
            raise ValueError("Length of index should match in_times/out_times")

        for s1, s2, s3, s4 in zip(self.in_times, self.out_times, self.in_marks, self.out_marks):
            if len(s1) != len(s2) or len(s3) != len(s4):
                raise ValueError("Some in/out series have different lengths.")
            if s3.max() >= self.num_classes or s4.max() >= self.num_classes:
                raise ValueError("Marks should not be larger than number of classes.")

    def break_down_long_sequences(self, max_seq_len):
        """Break down long sequences into shorter sub-sequences."""
        self.validate_times()
        new_in_times = []
        new_out_times = []
        new_in_marks = []
        new_out_marks = []
        new_index = []
        for idx in range(self.num_series):
            current_in = self.in_times[idx]
            current_out = self.out_times[idx]
            current_in_mark = self.in_marks[idx]
            current_out_mark = self.out_marks[idx]
            num_batches = int(np.ceil(len(current_in) / max_seq_len))
            for b in range(num_batches):
                new_in = current_in[b * max_seq_len : (b + 1) * max_seq_len]
                new_in_times.append(new_in)
                new_out = current_out[b * max_seq_len : (b + 1) * max_seq_len]
                new_out_times.append(new_out)

                new_in_mark = current_in_mark[b * max_seq_len : (b + 1) * max_seq_len]
                new_in_marks.append(new_in_mark)
                new_out_mark = current_out_mark[b * max_seq_len : (b + 1) * max_seq_len]
                new_out_marks.append(new_out_mark)

                new_index.append(self.index[idx])
        self.in_times = new_in_times
        self.out_times = new_out_times
        self.in_marks = new_in_marks
        self.out_marks = new_out_marks
        self.index = torch.tensor(new_index)
        self.validate_times()
        return self

    def train_val_test_split_whole(self, train_size=0.6, val_size=0.2, test_size=0.2, seed=123):
        """Split dataset into train, val and test parts."""
        np.random.seed(seed)
        all_idx = np.arange(self.num_series)
        train_idx, val_test_idx = train_test_split(all_idx, train_size=train_size, test_size=(val_size + test_size))
        if val_size == 0:
            val_idx = []
            test_idx = val_test_idx
        else:
            val_idx, test_idx = train_test_split(val_test_idx, train_size=(val_size / (val_size + test_size)),
                                                 test_size=(test_size / (val_size + test_size)))

        def get_dataset(ind):
            in_time, out_time = [], []
            in_mark, out_mark = [], []
            index = []
            for i in ind:
                in_time.append(self.in_times[i])
                out_time.append(self.out_times[i])
                in_mark.append(self.in_marks[i])
                out_mark.append(self.out_marks[i])
                index.append(self.index[i])
            return SequenceDataset(in_times=in_time, out_times=out_time, in_marks=in_mark, out_marks=out_mark,
                                   index=index, log_mode=False, num_classes=self.num_classes)

        data_train = get_dataset(train_idx)
        data_val = get_dataset(val_idx)
        data_test = get_dataset(test_idx)

        return data_train, data_val, data_test

    def train_val_test_split_each(self, train_size=0.6, val_size=0.2, test_size=0.2, seed=123):
        """Split each sequence in the dataset into train, val and test parts."""
        np.random.seed(seed)
        in_train, in_val, in_test = [], [], []
        out_train, out_val, out_test = [], [], []
        in_mark_train, in_mark_val, in_mark_test = [], [], []
        out_mark_train, out_mark_val, out_mark_test = [], [], []
        index_train, index_val, index_test = [], [], []

        for idx in range(self.num_series):
            n_elements = len(self.in_times[idx])
            n_train = int(train_size * n_elements)
            n_val = int(val_size * n_elements)

            if n_train == 0 or n_val == 0 or (n_elements - n_train - n_val) == 0:
                continue

            in_train.append(self.in_times[idx][:n_train])
            in_val.append(self.in_times[idx][n_train : (n_train + n_val)])
            in_test.append(self.in_times[idx][(n_train + n_val):])

            in_mark_train.append(self.in_marks[idx][:n_train])
            in_mark_val.append(self.in_marks[idx][n_train : (n_train + n_val)])
            in_mark_test.append(self.in_marks[idx][(n_train + n_val):])

            out_train.append(self.out_times[idx][:n_train])
            out_val.append(self.out_times[idx][n_train : (n_train + n_val)])
            out_test.append(self.out_times[idx][(n_train + n_val):])

            out_mark_train.append(self.out_marks[idx][:n_train])
            out_mark_val.append(self.out_marks[idx][n_train : (n_train + n_val)])
            out_mark_test.append(self.out_marks[idx][(n_train + n_val):])

            index_train.append(self.index[idx])
            index_val.append(self.index[idx])
            index_test.append(self.index[idx])

        data_train = SequenceDataset(in_times=in_train, out_times=out_train, in_marks=in_mark_train, out_marks=out_mark_train,
                                     index=index_train, log_mode=False, num_classes=self.num_classes)
        data_val = SequenceDataset(in_times=in_val, out_times=out_val, in_marks=in_mark_val, out_marks=out_mark_val,
                                   index=index_val, log_mode=False, num_classes=self.num_classes)
        data_test = SequenceDataset(in_times=in_test, out_times=out_test, in_marks=in_mark_test, out_marks=out_mark_test,
                                    index=index_test, log_mode=False, num_classes=self.num_classes)
        return data_train, data_val, data_test

    def normalize(self, mean_in=None, std_in=None, std_out=None):
        """Apply mean-std normalization to in_times."""
        if mean_in is None or std_in is None:
            mean_in, std_in = self.get_mean_std_in()
        self.in_times = [(t - mean_in) / std_in for t in self.in_times]
        if std_out is not None:
            # _, std_out = self.get_mean_std_out()
            self.out_times = [t / std_out for t in self.out_times]
        return self

    def get_mean_std_in(self):
        """Get mean and std of in_times."""
        flat_in_times = torch.cat(self.in_times)
        return flat_in_times.mean(), flat_in_times.std()

    def get_mean_std_out(self):
        """Get mean and std of out_times."""
        flat_out_times = torch.cat(self.out_times)
        return flat_out_times.mean(), flat_out_times.std()

    def get_log_mean_std_out(self):
        """Get mean and std of out_times."""
        flat_out_times = torch.cat(self.out_times).log()
        return flat_out_times.mean(), flat_out_times.std()

    def flatten(self):
        """Merge in_times and out_times to a single sequence."""
        flat_in_times = torch.cat(self.in_times)
        flat_out_times = torch.cat(self.out_times)
        return SequenceDataset(in_times=[flat_in_times], out_times=[flat_out_times], log_mode=False, num_classes=self.num_classes)

    def __add__(self, other):
        new_in_times = self.in_times + other.in_times
        new_out_times = self.out_times + other.out_times
        new_index = torch.cat([self.index, other.index + len(self.index)])
        return SequenceDataset(in_times=new_in_times, out_times=new_out_times, index=new_index,
                               num_classes=self.num_classes, log_mode=False)

    def __getitem__(self, key):
        return self.in_times[key], self.out_times[key], self.in_marks[key], self.out_marks[key], self.index[key]

    def __len__(self):
        return self.num_series

    def __repr__(self):
        return f"SequenceDataset({self.num_series})"
