import dpp
import torch
import torch.nn as nn

from torch.distributions import Categorical


class RecurrentTPP(nn.Module):
    """
    RNN-based TPP model for marked and unmarked event sequences.

    The marks are assumed to be conditionally independent of the inter-event times.

    Args:
        num_marks: Number of marks (i.e. classes / event types)
        mean_log_inter_time: Average log-inter-event-time, see dpp.data.dataset.get_inter_time_statistics
        std_log_inter_time: Std of log-inter-event-times, see dpp.data.dataset.get_inter_time_statistics
        context_size: Size of the context embedding (history embedding)
        mark_embedding_size: Size of the mark embedding (used as RNN input)
        rnn_type: Which RNN to use, possible choices {"RNN", "GRU", "LSTM"}

    """
    def __init__(
        self,
        num_marks: int,
        mean_log_inter_time: float = 0.0,
        std_log_inter_time: float = 1.0,
        context_size: int = 32,
        mark_embedding_size: int = 32,
        rnn_type: str = "GRU",
    ):
        super().__init__()
        self.num_marks = num_marks
        self.mean_log_inter_time = mean_log_inter_time
        self.std_log_inter_time = std_log_inter_time
        self.context_size = context_size
        self.mark_embedding_size = mark_embedding_size
        if self.num_marks > 1:
            self.num_features = 1 + self.mark_embedding_size
            self.mark_embedding = nn.Embedding(self.num_marks, self.mark_embedding_size)
            self.mark_linear = nn.Linear(self.context_size, self.num_marks)
        else:
            self.num_features = 1
        self.rnn_type = rnn_type
        self.rnn = getattr(nn, rnn_type)(input_size=self.num_features, hidden_size=self.context_size, batch_first=True)

    def get_features(self, batch: dpp.data.Batch) -> torch.Tensor:
        """
        Convert each event in a sequence into a feature vector.

        Args:
            batch: Batch of sequences in padded format (see dpp.data.batch).

        Returns:
            features: Feature vector corresponding to each event,
                shape (batch_size, seq_len, num_features)

        """
        features = torch.log(batch.inter_times + 1e-8).unsqueeze(-1)  # (batch_size, seq_len, 1)
        features = (features - self.mean_log_inter_time) / self.std_log_inter_time
        if self.num_marks > 1:
            mark_emb = self.mark_embedding(batch.marks)  # (batch_size, seq_len, mark_embedding_size)
            features = torch.cat([features, mark_emb], dim=-1)
        return features  # (batch_size, seq_len, num_features)

    def get_context(self, features: torch.Tensor) -> torch.Tensor:
        """
        Get the context (history) embedding from the sequence of events.

        Args:
            features: Feature vector corresponding to each event,
                shape (batch_size, seq_len, num_features)

        Returns:
            context: Context vector used to condition the distribution of each event,
                shape (batch_size, seq_len, context_size)

        """
        context = self.rnn(features)[0]
        batch_size, seq_len, context_size = context.shape
        # Shift the context by vectors by 1: context embedding after event i is used to predict event i + 1
        context = torch.cat([torch.zeros(batch_size, 1, context_size), context[:, 1:, :]], dim=1)
        return context

    def get_inter_time_dist(self, context: torch.Tensor) -> torch.distributions.Distribution:
        """
        Get the distribution over inter-event times given the context.

        Args:
            context: Context vector used to condition the distribution of each event,
                shape (batch_size, seq_len, context_size)

        Returns:
            dist: Distribution over inter-event times, has batch_shape (batch_size, seq_len)

        """
        raise NotImplementedError()

    def log_prob(self, batch: dpp.data.Batch) -> torch.Tensor:
        """Compute log-likelihood for a batch of sequences.

        Args:
            batch:

        Returns:
            log_p: shape (batch_size,)

        """
        features = self.get_features(batch)
        context = self.get_context(features)
        inter_time_dist = self.get_inter_time_dist(context)
        inter_times = batch.inter_times.clamp(1e-10)
        log_p = inter_time_dist.log_prob(inter_times)  # (batch_size, seq_len)

        # Survival probability of the last interval (from t_N to t_end).
        # You can comment this section of the code out if you don't want to implement the log_survival_function
        # for the distribution that you are using. This will make the likelihood computation slightly inaccurate,
        # but the difference shouldn't be significant if you are working with long sequences.
        last_event_idx = batch.mask.sum(-1, keepdim=True).long()  # (batch_size, 1)
        log_surv_all = inter_time_dist.log_survival_function(inter_times)  # (batch_size, seq_len)
        log_surv_last = torch.gather(log_surv_all, dim=-1, index=last_event_idx).squeeze(-1)  # (batch_size,)

        if self.num_marks > 1:
            mark_logits = torch.log_softmax(self.mark_linear(context), dim=-1)  # (batch_size, seq_len, num_marks)
            mark_dist = Categorical(logits=mark_logits)
            log_p += mark_dist.log_prob(batch.marks)  # (batch_size, seq_len)
        log_p *= batch.mask  # (batch_size, seq_len)
        return log_p.sum(-1) + log_surv_last  # (batch_size,)

    def sample(self, t_max: float, batch_size: int = 1, context_init: torch.Tensor = None) -> dpp.data.Batch:
        raise NotImplementedError()
