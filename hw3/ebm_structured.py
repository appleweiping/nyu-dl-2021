"""HW3 - Energy-Based Models and structured prediction.

Skeleton (official TODOs).  A CNN maps a strip image containing a short word to
a sequence of per-position energy vectors over a 27-symbol alphabet (a-z plus a
"between characters" symbol).  Structured prediction aligns the energy sequence
to a target word with a dynamic-program (Viterbi) that finds the minimum-energy
monotone path; the free energy is the loss driving training.
"""
import torch
from torch import nn

ALPHABET_SIZE = 27
BETWEEN = 26


class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: CNN producing (batch, width, ALPHABET_SIZE) energies
        raise NotImplementedError

    def forward(self, x):
        # TODO
        raise NotImplementedError


def build_path_matrix(energies, targets):
    # TODO: gather energies along the target symbols (vectorized, no python loop)
    raise NotImplementedError


def path_energy(pm, path):
    # TODO: sum of energies along a monotone path, else 2**30 for invalid paths
    raise NotImplementedError


def find_path(pm):
    # TODO: min-energy monotone alignment via dynamic programming
    raise NotImplementedError
