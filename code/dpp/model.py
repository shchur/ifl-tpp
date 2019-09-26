import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dpp

from dpp.utils import DotDict
from dpp.nn import BaseModule


class Model(BaseModule):
    """Base model class.

    Attributes:
        rnn: RNN for encoding the event history.
        embedding: Retrieve static embedding for each sequence.
        decoder: Compute log-likelihood of the inter-event times given hist and emb.

    Args:
        config: General model configuration (see dpp.model.ModelConfig).
        decoder: Model for computing log probability of t given history and embeddings.
            (see dpp.decoders for a list of possible choices)
    """
    def __init__(self, config, decoder):
        super().__init__()
        self.use_history(config.use_history)
        self.use_embedding(config.use_embedding)
        self.use_marks(config.use_marks)

        self.rnn = dpp.nn.RNNLayer(config)
        if self.using_embedding:
            self.embedding = nn.Embedding(config.num_embeddings, config.embedding_size)
            self.embedding.weight.data.fill_(0.0)

        if self.using_marks:
            self.num_classes = config.num_classes
            self.mark_layer = nn.Sequential(
                nn.Linear(config.history_size, config.history_size),
                nn.ReLU(),
                nn.Linear(config.history_size, self.num_classes)
            )

        self.decoder = decoder

    def mark_nll(self, h, y):
        """Compute log likelihood and accuracy of predicted marks

        Args:
            h: History vector
            y: Out marks, true label

        Returns:
            loss: Negative log-likelihood for marks, shape (batch_size, seq_len)
            accuracy: Percentage of correctly classified marks
        """
        x = self.mark_layer(h)
        x = F.log_softmax(x, dim=-1)
        loss = F.nll_loss(x.view(-1, self.num_classes), y.view(-1), reduction='none').view_as(y)
        accuracy = (y == x.argmax(-1)).float()
        return loss, accuracy

    def log_prob(self, input):
        """Compute log likelihood of the inter-event timesi in the batch.

        Args:
            input: Batch of data to score. See dpp.data.Input.

        Returns:
            time_log_prob: Log likelihood of each data point, shape (batch_size, seq_len)
            mark_nll: Negative log likelihood of marks, if using_marks is True
            accuracy: Accuracy of marks, if using_marks is True
        """
        # Encode the history with an RNN
        if self.using_history:
            h = self.rnn(input) # has shape (batch_size, seq_len, rnn_hidden_size)
        else:
            h = None
        # Get sequence embedding
        if self.using_embedding:
            # has shape (batch_size, seq_len, embedding_size)
            emb = self.embedding(input.index).unsqueeze(1).repeat(1, input.out_time.shape[1], 1)
        else:
            emb = None

        t = input.out_time  # has shape (batch_size, seq_len)
        time_log_prob = self.decoder.log_prob(t, h, emb)

        if self.using_marks:
            mark_nll, accuracy = self.mark_nll(h, input.out_mark)
            return time_log_prob, mark_nll, accuracy

        return time_log_prob

    def aggregate(self, values, lengths):
        """Calculate masked average of values.

        Sequences may have different lengths, so it's necessary to exclude
        the masked values in the padded sequence when computing the average.

        Arguments:
            values (list[tensor]): List of batches where each batch contains
                padded values, shape (batch size, sequence length)
            lengths (list[tensor]): List of batches where each batch contains
                lengths of sequences in a batch, shape (batch size)

        Returns:
            mean (float): Average value in values taking padding into account
        """

        if not isinstance(values, list):
            values = [values]
        if not isinstance(lengths, list):
            lengths = [lengths]

        total = 0.0
        for batch, length in zip(values, lengths):
            length = length.long()
            mask = torch.arange(batch.shape[1])[None, :] < length[:, None]
            mask = mask.float()

            batch[torch.isnan(batch)] = 0 # set NaNs to 0
            batch *= mask

            total += batch.sum()

        total_length = sum([x.sum() for x in lengths])

        return total / total_length


class ModelConfig(DotDict):
    """Configuration of the model.

    This config only contains parameters that need to be know by all the
    submodules. Submodule-specific parameters are passed to the respective
    constructors.

    Args:
        use_history: Should the model use the history embedding?
        history_size: Dimension of the history embedding.
        rnn_type: {'RNN', 'LSTM', 'GRU'}: RNN architecture to use.
        use_embedding: Should the model use the sequence embedding?
        embedding_size: Dimension of the sequence embedding.
        num_embeddings: Number of unique sequences in the dataset.
        use_marks: Should the model use the marks?
        mark_embedding_size: Dimension of the mark embedding.
        num_classes: Number of unique mark types, used as dimension of output
    """
    def __init__(self,
                 use_history=True,
                 history_size=32,
                 rnn_type='RNN',
                 use_embedding=False,
                 embedding_size=32,
                 num_embeddings=None,
                 use_marks=False,
                 mark_embedding_size=64,
                 num_classes=None):
        super().__init__()
        # RNN parameters
        self.use_history = use_history
        self.history_size = history_size
        self.rnn_type = rnn_type

        # Sequence embedding parameters
        self.use_embedding = use_embedding
        self.embedding_size = embedding_size
        if use_embedding and num_embeddings is None:
            raise ValueError("Number of embeddings has to be specified")
        self.num_embeddings = num_embeddings

        self.use_marks = use_marks
        self.mark_embedding_size = mark_embedding_size
        self.num_classes = num_classes
