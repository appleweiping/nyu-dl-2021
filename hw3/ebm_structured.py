"""HW3 - Energy-Based Models and structured prediction.

We render short words (lowercase a-z) into 32-pixel-tall strip images and train
a CNN to read them.  The CNN outputs, for each horizontal position ``l``, an
energy vector over a 27-symbol alphabet (indices 0..25 = a..z, 26 = "between
characters" / blank).  Reading the word is a *structured* prediction problem:
we must align the length-``L`` energy sequence to the length-``T`` target using
a monotone path, and the free energy of that alignment is the training loss.

Key pieces
----------
* ``SimpleWordsDataset`` - on-the-fly word rendering with an open monospace font.
* ``SimpleNet``          - conv stack producing (batch, width, 27) energies.
* ``transform_word``     - target word -> symbol sequence with blanks inserted
                           between characters, e.g. "hi" -> [h, _, i, _].
* ``build_path_matrix``  - vectorized gather of energies along the target.
* ``find_path`` / ``free_energy`` - min-energy monotone DP alignment (Viterbi).

The whole thing runs on CPU.
"""
import os
import random
import string

import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
from torch import nn
from torch.nn.functional import cross_entropy

ALPHABET_SIZE = 27
BETWEEN = 26  # the "between characters" / blank symbol

# Use matplotlib's bundled DejaVu Sans Mono (open license) instead of the
# course's Anonymous.ttf so nothing copyrighted is committed or downloaded.
_MPL_FONT = None


def _font(size=20):
    global _MPL_FONT
    if _MPL_FONT is None:
        import matplotlib
        _MPL_FONT = os.path.join(
            os.path.dirname(matplotlib.__file__),
            "mpl-data", "fonts", "ttf", "DejaVuSansMono.ttf",
        )
    return ImageFont.truetype(_MPL_FONT, size)


class SimpleWordsDataset(torch.utils.data.IterableDataset):
    """Yields (image, text) pairs; images are 32-tall single-channel strips."""

    def __init__(self, max_length, len=100, jitter=False, noise=False):
        self.max_length = max_length
        self.transforms = transforms.ToTensor()
        self.len = len
        self.jitter = jitter
        self.noise = noise

    def __len__(self):
        return self.len

    def __iter__(self):
        for _ in range(self.len):
            text = "".join(random.choice(string.ascii_lowercase)
                           for _ in range(self.max_length))
            img = self.draw_text(text, jitter=self.jitter, noise=self.noise)
            yield img, text

    def draw_text(self, text, length=None, jitter=False, noise=False):
        if length is None:
            length = 18 * len(text)
        img = Image.new("L", (length, 32))
        fnt = _font(20)
        d = ImageDraw.Draw(img)
        pos = (random.randint(0, 7), 5) if jitter else (0, 5)
        d.text(pos, text, fill=1, font=fnt)
        img = self.transforms(img)
        img[img > 0] = 1
        if noise:
            img += torch.bernoulli(torch.ones_like(img) * 0.1)
            img = img.clamp(0, 1)
        return img[0]


class SimpleNet(nn.Module):
    """CNN mapping a (1 x 32 x W) strip to (W' x 27) energies."""

    def __init__(self):
        super().__init__()
        self.cnn_block = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2),                    # 16 x W/2
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2, 2),                    # 8 x W/4
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1)),          # 4 x W/4
            nn.Conv2d(256, ALPHABET_SIZE, 4, 1, 0),  # 1 x (W/4 - 3)
        )

    def forward(self, x):
        x = self.cnn_block(x)
        # (batch, alphabet, 1, width) -> (batch, width, alphabet)
        return x[:, :, 0, :].permute(0, 2, 1)


def transform_word(s):
    """'abc' -> tensor([a, _, b, _, c, _]) with '_' == BETWEEN (26)."""
    ss = "".join(c + "_" for c in s)
    t = [BETWEEN if c == "_" else ord(c) - ord("a") for c in ss]
    return torch.tensor(t)


def build_path_matrix(energies, targets):
    """energies (B, L, 27), targets (B, T) -> (B, L, T) with
    out[i, j, k] = energies[i, j, targets[i, k]].  Vectorized (no python loop).
    """
    L = energies.shape[1]
    idx = targets.unsqueeze(1).repeat_interleave(L, dim=1).long()
    return energies.gather(2, idx)


def build_ce_matrix(energies, targets):
    """Cross-entropy version: ce[i, j, k] = CE(energies[i, j], targets[i, k])."""
    T = targets.shape[1]
    L = energies.shape[1]
    e = energies.unsqueeze(2).repeat_interleave(T, dim=2).permute(0, 3, 1, 2)
    t = targets.unsqueeze(1).repeat_interleave(L, dim=1).long()
    return cross_entropy(e, t, reduction="none")


def path_energy(pm, path):
    """Sum of energies along a monotone path, or 2**30 if the path is invalid.

    A valid path starts at target index 0, ends at T-1, and at each step either
    stays or advances the target index by exactly one.
    """
    if path[0] != 0 or path[-1] != pm.shape[1] - 1:
        return torch.tensor(float(2 ** 30))
    for i in range(len(path) - 1):
        if path[i] != path[i + 1] and path[i] + 1 != path[i + 1]:
            return torch.tensor(float(2 ** 30))
    idx = torch.tensor(path).unsqueeze(1)
    return pm.gather(1, idx).sum()


def worst_path(pm):
    """Maximum-energy monotone path (used to illustrate the DP)."""
    L, T = pm.shape
    NEG = -(2 ** 30)
    dp = torch.full((L, T), 0.0)
    pre = torch.zeros((L, T), dtype=torch.bool)
    dp[0, 0] = pm[0, 0]
    for i in range(1, L):
        dp[i, 0] = dp[i - 1, 0] + pm[i, 0]
    for j in range(1, T):
        dp[0, j] = NEG
    for i in range(1, L):
        for j in range(1, T):
            if dp[i - 1, j] > dp[i - 1, j - 1]:
                dp[i, j] = dp[i - 1, j] + pm[i, j]
            else:
                dp[i, j] = dp[i - 1, j - 1] + pm[i, j]
                pre[i, j] = True
    path = [T - 1]
    while path[0] != 0:
        i = L - len(path)
        path.insert(0, path[0] - 1 if pre[i, path[0]] else path[0])
    return path


def find_path(pm):
    """Minimum-energy monotone alignment (Viterbi).

    Returns (free_energy, list_of_(l, t)_points, dp_array).
    """
    L, T = pm.shape
    POS = float(2 ** 30)
    dp = torch.zeros((L, T))
    pre = torch.zeros((L, T), dtype=torch.bool)
    dp[0, 0] = pm[0, 0]
    for i in range(1, L):
        dp[i, 0] = dp[i - 1, 0] + pm[i, 0]
    for j in range(1, T):
        dp[0, j] = POS
    for i in range(1, L):
        for j in range(1, T):
            if dp[i - 1, j] < dp[i - 1, j - 1]:
                dp[i, j] = dp[i - 1, j] + pm[i, j]
            else:
                dp[i, j] = dp[i - 1, j - 1] + pm[i, j]
                pre[i, j] = True
    path = [(L - 1, T - 1)]
    while path[0] != (0, 0):
        i, j = path[0]
        path.insert(0, (i - 1, j - 1) if pre[i, j] else (i - 1, j))
    return dp[-1, -1], path, dp


def free_energy_ce(energies_i, target_i):
    """Free energy of one sample = sum of cross-entropies along the best path.

    ``energies_i``: (L, 27) energies for one image.
    ``target_i``:   (T,) target symbols (already blank-augmented).
    """
    pm = build_path_matrix(energies_i.unsqueeze(0), target_i.unsqueeze(0))[0]
    _, path, _ = find_path(pm)
    ce = build_ce_matrix(energies_i.unsqueeze(0), target_i.unsqueeze(0))[0]
    total = energies_i.new_zeros(())
    for (l, t) in path:
        total = total + ce[l, t]
    return total
