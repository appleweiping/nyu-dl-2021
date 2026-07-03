"""HW1 evidence run: (1) re-run the three official gradient-check autograders and
(2) train the from-scratch MLP with hand-written backprop on a real regression
task, using *only* our own forward/backward + SGD (no autograd), and report the
loss curve.  Saves numbers + a figure to ``results/hw1/``.
"""
import os
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

torch.set_num_threads(3)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.abspath(os.path.join(HERE, "..", "results", "hw1"))
os.makedirs(RESULTS, exist_ok=True)

sys.path.insert(0, HERE)
from mlp import MLP, mse_loss  # noqa: E402


def run_autograders():
    lines = []
    for t in ("test1", "test2", "test3"):
        out = subprocess.run(
            [sys.executable, os.path.join(HERE, f"{t}.py")],
            capture_output=True, text=True, cwd=HERE,
        ).stdout.strip()
        passed = out.count("True")
        lines.append(f"{t}: {out.splitlines()} -> {passed}/4 gradient checks pass")
    return lines


def train_regression():
    """Learn y = sin(2 x0) + 0.5 x1^2 with a 2->64->1 MLP trained purely via our
    hand-written backprop and manual SGD."""
    torch.manual_seed(0)
    n = 512
    x = torch.rand(n, 2) * 2 - 1
    y = (torch.sin(2 * x[:, 0]) + 0.5 * x[:, 1] ** 2).unsqueeze(1)

    net = MLP(2, 64, "relu", 64, 1, "identity")
    # Small init so ReLU units don't die at the start.
    for k in net.parameters:
        net.parameters[k] = net.parameters[k] * 0.3

    lr = 0.05
    losses = []
    epochs = 400
    for _ in range(epochs):
        net.clear_grad_and_cache()
        y_hat = net.forward(x)
        J, dJdy_hat = mse_loss(y, y_hat)
        net.backward(dJdy_hat)
        # Manual SGD step using our own gradients.
        net.parameters["W1"] -= lr * net.grads["dJdW1"]
        net.parameters["b1"] -= lr * net.grads["dJdb1"]
        net.parameters["W2"] -= lr * net.grads["dJdW2"]
        net.parameters["b2"] -= lr * net.grads["dJdb2"]
        losses.append(J.item())
    return losses


def main():
    ag_lines = run_autograders()

    losses = train_regression()

    plt.figure(dpi=150)
    plt.plot(losses)
    plt.xlabel("epoch")
    plt.ylabel("MSE loss")
    plt.title("HW1: from-scratch MLP trained with hand-written backprop")
    plt.yscale("log")
    plt.grid(True, alpha=0.3)
    fig_path = os.path.join(RESULTS, "loss_curve.png")
    plt.savefig(fig_path, bbox_inches="tight")

    report = os.path.join(RESULTS, "hw1_results.txt")
    with open(report, "w", encoding="utf-8") as fh:
        fh.write("HW1 - Backprop from scratch: verification\n")
        fh.write("=" * 50 + "\n\n")
        fh.write("Gradient checks vs torch.autograd (norm(diff) < 1e-3):\n")
        for line in ag_lines:
            fh.write("  " + line + "\n")
        fh.write("\nReal training run (2->64->1 MLP, hand-written backprop + manual SGD):\n")
        fh.write(f"  task: regress y = sin(2*x0) + 0.5*x1^2\n")
        fh.write(f"  initial MSE: {losses[0]:.6f}\n")
        fh.write(f"  final   MSE: {losses[-1]:.6f}\n")
        fh.write(f"  reduction:   {losses[0] / losses[-1]:.1f}x\n")
        fh.write(f"  figure: results/hw1/loss_curve.png\n")

    print(open(report, encoding="utf-8").read())


if __name__ == "__main__":
    main()
