"""HW2 - Convolutional Neural Nets (cats vs dogs).

Skeleton (official TODOs).  Train a small CNN from scratch on the
``cats_and_dogs_filtered`` dataset and reach >= 75% validation accuracy, then
visualize the learned first-layer filters / intermediate activations.
"""
import torch
from torch import nn


class CatsDogsNet(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: build a CNN that reaches >= 75% validation accuracy
        raise NotImplementedError

    def forward(self, x):
        # TODO
        raise NotImplementedError
