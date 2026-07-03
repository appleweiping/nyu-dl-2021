# NYU Deep Learning Spring 2021 — Homework Solutions

> From-scratch backprop, CNNs, RNNs, and energy-based structured prediction — an
> independent, from-skeleton implementation of
> **NYU-DLSP21 — Deep Learning (Spring 2021)** (New York University, Yann LeCun &
> Alfredo Canziani), part of a [csdiy.wiki](https://csdiy.wiki/) full-catalog build.

![status](https://img.shields.io/badge/status-complete-brightgreen)
![language](https://img.shields.io/badge/python-3.11-informational)
![license](https://img.shields.io/badge/license-MIT-blue)

## Overview

[NYU-DLSP21](https://atcold.github.io/NYU-DLSP21/) is Yann LeCun and Alfredo
Canziani's graduate deep-learning course. This repo implements its three coding
homeworks, each built from the official starter skeleton and verified with the
course's own checks plus real train/inference runs on **CPU** (12-thread machine,
`OMP_NUM_THREADS=3`, `torch.set_num_threads(3)`):

- **HW1 — Backprop from scratch:** a two-layer MLP with hand-written forward and
  backward passes (no autograd), verified against `torch.autograd`.
- **HW2 — CNNs & RNNs:** a from-scratch CNN for cats-vs-dogs, and GRU "echo"
  networks (fixed and variable delay).
- **HW3 — Energy-based models & structured prediction:** a CNN reads rendered
  words; a hand-written Viterbi dynamic program aligns per-position energies to a
  target, and CTC (the marginalized form of that alignment) trains the reader.

## Results (measured on CPU, 3 threads)

| Assignment | What it does | Result (measured) |
|---|---|---|
| **HW1** backprop | 2-layer MLP, manual forward/backward + losses | **12/12** gradient checks match autograd (`<1e-3`); from-scratch training MSE **3.09 → 0.005** (616×) |
| **HW2** CNN | cats-vs-dogs, trained from scratch (106K params) | **val acc 0.779** (target ≥ 0.75), 30 epochs |
| **HW2** RNN fixed | GRU echo, delay = 4 | **acc 0.9922** (target > 0.99) |
| **HW2** RNN variable | GRU echo, delays 0–8 | **acc 0.9937** (target > 0.99) |
| **HW3** EBM | word reader + Viterbi alignment, CTC training | decode exact-match **1.000** (words len 1–4); single-char **1.000**; Viterbi + energy tables |

Figures (in `results/`): `hw1/loss_curve.png`, `hw2/cnn_curves.png`,
`hw2/cnn_filters.png`, `hw3/free_energy_curve.png`, `hw3/energy_table.png`.

### HW1 — hand-written backprop matches autograd

```
test1 (relu/identity + MSE): 4/4 True
test2 (sigmoid/sigmoid + BCE): 4/4 True
test3 (relu/relu + MSE): 4/4 True
```

### HW2 — RNN echo (delay = 4)

```
in : 'helloworldthisisanechotest'
out: '    helloworldthisisanccho'   # correctly shifted right by 4
```

## Implemented assignments

- [x] **HW1 — Fundamentals / backprop** (`hw1/`) — `mlp.py` implements a 2-layer
  MLP's forward/backward by hand for activations `{relu, sigmoid, identity}` and
  `mse_loss` / `bce_loss` with analytic gradients; `THEORY.md` covers the written
  part. Verified by the official `test1/2/3.py` autograders.
- [x] **HW2 — CNN** (`hw2/cnn_catsdogs.py`) — a 4-block conv net trained from
  scratch on `cats_and_dogs_filtered`, ≥ 75% validation accuracy, with
  first-layer filter visualization.
- [x] **HW2 — RNN** (`hw2/rnn_echo.py`) — `GRUMemory` (fixed delay) and
  `VariableDelayGRUMemory` (delay as input), both > 99% accuracy within the
  10-minute budget.
- [x] **HW3 — Energy-based structured prediction** (`hw3/ebm_structured.py`) —
  a CNN emits per-position energies over a 27-symbol alphabet; hand-written
  `find_path` (Viterbi) computes the min-energy monotone alignment / free energy;
  training uses CTC, the differentiable marginalization of that alignment.

## Project structure

```
nyu-dl-2021/
├── hw1/   mlp.py, test1/2/3.py (official), train_demo.py, THEORY.md
├── hw2/   cnn_catsdogs.py, train_cnn.py, download_data.py,
│          rnn_echo.py, train_rnn.py
├── hw3/   ebm_structured.py, train_ebm.py
├── results/  hw1/ hw2/ hw3/  (measured logs + figures)
├── requirements.txt
└── LICENSE
```

## How to run

```bash
# Python 3.11.  Reuse the shared csdiy env or install requirements:
python -m pip install -r requirements.txt

# HW1 — gradient checks + from-scratch training demo
cd hw1 && python test1.py && python test2.py && python test3.py
python train_demo.py

# HW2 — RNN echo (fixed + variable delay)
cd ../hw2 && python train_rnn.py
# HW2 — CNN cats vs dogs (downloads the dataset at runtime, ~68 MB)
python train_cnn.py

# HW3 — energy-based structured prediction (word reader)
cd ../hw3 && python train_ebm.py
```

All scripts set `torch.set_num_threads(3)`; export `OMP_NUM_THREADS=3` for shared
CPU builds.

## Verification

- **HW1:** the three official autograders (`test1/2/3.py`) compare our
  hand-derived gradients to `torch.autograd` for `{relu/identity+MSE,
  sigmoid/sigmoid+BCE, relu/relu+MSE}` — all 12 checks pass (`norm(diff) < 1e-3`).
  `train_demo.py` then trains a network using *only* our backprop + manual SGD
  (see `results/hw1/hw1_results.txt`).
- **HW2:** the assignment's own `test_model` / `test_variable_delay_model`
  harnesses report echo accuracy; the CNN reports held-out validation accuracy
  each epoch (`results/hw2/*`).
- **HW3:** exact-match word decoding on held-out rendered words, plus the
  Viterbi alignment / energy tables (`results/hw3/*`).

## Tech stack

Python 3.11, PyTorch 2.x (CPU), torchvision, NumPy, Matplotlib, Pillow, tqdm.

## Key ideas / what I learned

- Deriving and vectorizing the backward pass of an MLP by hand, and validating it
  against automatic differentiation.
- Why ReLU mitigates vanishing gradients versus sigmoid in deep nets.
- Training small CNNs from scratch on limited data (augmentation, BN, scheduling)
  and reading learned filters.
- Using GRUs as explicit memory to solve delayed-echo sequence tasks, including
  conditioning on a per-sequence control input (the delay).
- Energy-based structured prediction: per-position energies, monotone alignment
  via dynamic programming (Viterbi), free energy, and the connection to CTC as
  the soft/marginalized version of that alignment.

## Credits & license

Based on the homeworks of **NYU-DLSP21 — Deep Learning (Spring 2021)** by
Yann LeCun and Alfredo Canziani (New York University). This repository is an
independent educational reimplementation; all course materials, datasets, and
specifications belong to their original authors. The `cats_and_dogs_filtered`
dataset is downloaded at runtime and not redistributed here. Original code in
this repo is released under the [MIT License](LICENSE).
