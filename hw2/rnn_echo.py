"""HW2 - Recurrent Neural Networks (GRU echo task).

The "echo" task: given a sequence of letters (encoded 1..26, with 0 = blank),
reproduce the same sequence delayed by ``DELAY`` steps.  A GRU is well suited
because it must remember the last ``DELAY`` symbols.  The second variant makes
the delay an extra per-sequence input in [0, max_delay] so a single model must
handle any delay.

Run ``python train_rnn.py`` to train and verify both models.
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


# ---------------------------------------------------------------------------
# Fixed-delay echo
# ---------------------------------------------------------------------------
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
        self.hidden_size = hidden_size
        self.gru = torch.nn.GRU(
            input_size=N + 1,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.linear = torch.nn.Linear(hidden_size, N + 1)

    def forward(self, x):
        # x: (batch, seq_length, N+1) one-hot -> logits (batch, seq_length, N+1)
        out, hidden = self.gru(x)
        logits = self.linear(out)
        return logits

    @torch.no_grad()
    def test_run(self, s, device=torch.device("cpu")):
        """Map a lowercase string through the network and decode back to text.

        0 -> ' ', 1..26 -> 'a'..'z'.
        """
        self.eval()
        base = ord("a") - 1
        int_seq = torch.tensor([ord(c) - base for c in s], dtype=torch.int64)
        inp = idx_to_onehot(int_seq).unsqueeze(0).to(device)
        logits = self(inp).squeeze(0)
        pred = logits.argmax(dim=1)
        return "".join(" " if i == 0 else chr(int(i) + base) for i in pred)


# ---------------------------------------------------------------------------
# Variable-delay echo
# ---------------------------------------------------------------------------
class VariableDelayEchoDataset(torch.utils.data.IterableDataset):
    def __init__(self, max_delay=8, seq_length=20, size=1000):
        self.max_delay = max_delay
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
            delay = random.randint(0, self.max_delay)
            result = torch.cat(
                (torch.zeros(delay), seq[: self.seq_length - delay])
            ).type(torch.int64)
            yield seq, delay, result


class VariableDelayGRUMemory(torch.nn.Module):
    def __init__(self, hidden_size, max_delay, device=torch.device("cpu")):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_delay = max_delay
        self.device = device
        # +1 input feature carries the (normalized) desired delay.
        self.gru = torch.nn.GRU(
            input_size=N + 2,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
        )
        self.linear = torch.nn.Linear(hidden_size, N + 1)

    def forward(self, x, delays):
        # x: (batch, seq_length, N+1); delays: (batch,)
        seq_length = x.shape[1]
        d = (delays.float() / self.max_delay).view(-1, 1, 1).repeat(1, seq_length, 1)
        x = torch.cat((x, d.to(x.device)), dim=2)
        out, hidden = self.gru(x)
        logits = self.linear(out)
        return logits

    @torch.no_grad()
    def test_run(self, s, delay):
        self.eval()
        base = ord("a") - 1
        int_seq = torch.tensor([ord(c) - base for c in s], dtype=torch.int64)
        inp = idx_to_onehot(int_seq).unsqueeze(0).to(self.device)
        delays = torch.tensor([delay], device=self.device)
        logits = self(inp, delays).squeeze(0)
        pred = logits.argmax(dim=1)
        return "".join(" " if i == 0 else chr(int(i) + base) for i in pred)


# ---------------------------------------------------------------------------
# Test harnesses (from the assignment).
# ---------------------------------------------------------------------------
def test_model(model, device, delay, sequence_length=15):
    total = correct = 0
    for _ in range(500):
        s = "".join(random.choice(string.ascii_lowercase)
                    for _ in range(random.randint(15, 25)))
        result = model.test_run(s, device)
        assert delay > 0
        for c1, c2 in zip(s[:-delay], result[delay:]):
            correct += int(c1 == c2)
        total += len(s) - delay
    return correct / total


def test_variable_delay_model(model, seq_length=20):
    total = correct = 0
    for _ in range(500):
        s = "".join(random.choice(string.ascii_lowercase) for _ in range(seq_length))
        d = random.randint(0, model.max_delay)
        result = model.test_run(s, d)
        z = zip(s[:-d], result[d:]) if d > 0 else zip(s, result)
        for c1, c2 in z:
            correct += int(c1 == c2)
        total += len(s) - d
    return correct / total
