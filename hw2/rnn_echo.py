"""HW2 - Recurrent Neural Networks (GRU echo task).

Skeleton (official TODOs).  The goal is an "echo" GRU: given a sequence of
letters, reproduce the same sequence delayed by ``DELAY`` steps.  A second,
harder variant conditions the model on a per-sequence variable delay.

Constraints from the assignment:
    * fixed-delay model: error rate below 1% (accuracy > 0.99)
    * variable-delay model: accuracy > 0.99 for delays in [0, 8]
    * each model must train within ~10 minutes
"""
import random
import string

import torch

# Max value of the generated integer. 26 == number of letters in the alphabet.
N = 26


def idx_to_onehot(x, k=N + 1):
    """Converts the generated integers to one-hot vectors."""
    ones = torch.eye(k)
    shape = x.shape
    res = ones.index_select(0, x.view(-1).type(torch.int64))
    return res.view(*shape, res.shape[-1])


class EchoDataset(torch.utils.data.IterableDataset):
    def __init__(self, delay=4, seq_length=15, size=1000):
        self.delay = delay
        self.seq_length = seq_length
        self.size = size

    def __len__(self):
        return self.size

    def __iter__(self):
        for _ in range(self.size):
            seq = torch.tensor(
                [random.choice(range(1, N + 1)) for _ in range(self.seq_length)],
                dtype=torch.int64,
            )
            result = torch.cat(
                (torch.zeros(self.delay), seq[: self.seq_length - self.delay])
            ).type(torch.int64)
            yield seq, result


class GRUMemory(torch.nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        # TODO: initialize your GRU + decoder
        raise NotImplementedError

    def forward(self, x):
        # TODO: return logits of shape (batch, seq_length, N+1)
        raise NotImplementedError

    @torch.no_grad()
    def test_run(self, s, device=torch.device("cpu")):
        # TODO: map string -> one-hot -> network -> decoded string
        raise NotImplementedError


class VariableDelayGRUMemory(torch.nn.Module):
    def __init__(self, hidden_size, max_delay):
        super().__init__()
        # TODO
        raise NotImplementedError

    def forward(self, x, delays):
        # TODO
        raise NotImplementedError

    @torch.no_grad()
    def test_run(self, s, delay):
        # TODO
        raise NotImplementedError
