"""HW2 - Convolutional Neural Nets (cats vs dogs).

A small CNN trained from scratch on ``cats_and_dogs_filtered`` (2000 train /
1000 validation images) to reach >= 75% validation accuracy on CPU.
"""
import torch
from torch import nn


class CatsDogsNet(nn.Module):
    """4 conv blocks (each: conv -> BN -> ReLU -> maxpool) + a small MLP head.

    Input images are resized to 128x128 RGB.  Each block halves the spatial
    size: 128 -> 64 -> 32 -> 16 -> 8, ending at 128 channels of 8x8.
    """

    def __init__(self):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=3, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(3, 16),    # 128 -> 64
            block(16, 32),   # 64  -> 32
            block(32, 64),   # 32  -> 16
            block(64, 128),  # 16  -> 8
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
