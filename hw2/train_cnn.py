"""Train the cats-vs-dogs CNN from scratch on CPU, verify >= 75% validation
accuracy, plot the loss/accuracy curves, and visualize learned first-layer
filters.  Evidence goes to ``results/hw2/``.
"""
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

torch.set_num_threads(3)

from cnn_catsdogs import CatsDogsNet  # noqa: E402
from download_data import download  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.abspath(os.path.join(HERE, "..", "results", "hw2"))
os.makedirs(RESULTS, exist_ok=True)
DEVICE = torch.device("cpu")
IMG = 128


def loaders(data_dir, bs=64):
    train_tf = transforms.Compose([
        transforms.Resize((IMG, IMG)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(8),
        transforms.ColorJitter(0.1, 0.1, 0.1),
        transforms.ToTensor(),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMG, IMG)),
        transforms.ToTensor(),
    ])
    train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), train_tf)
    val_ds = datasets.ImageFolder(os.path.join(data_dir, "validation"), val_tf)
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=0),
        DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=0),
    )


@torch.no_grad()
def evaluate(model, dl, lossf):
    model.eval()
    loss = correct = total = 0
    for x, y in dl:
        out = model(x)
        loss += lossf(out, y).item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    return loss / total, correct / total


def train(model, train_dl, val_dl, epochs=30, lr=1.5e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lossf = nn.CrossEntropyLoss()
    hist = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_acc = 0.0
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        run = n = 0
        for x, y in train_dl:
            out = model(x)
            loss = lossf(out, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            run += loss.item() * x.size(0)
            n += x.size(0)
        sched.step()
        tr_loss = run / n
        val_loss, val_acc = evaluate(model, val_dl, lossf)
        best_acc = max(best_acc, val_acc)
        hist["train_loss"].append(tr_loss)
        hist["val_loss"].append(val_loss)
        hist["val_acc"].append(val_acc)
        print(f"  epoch {ep + 1:2d}/{epochs}  train_loss={tr_loss:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
              f"t={time.time() - t0:.0f}s")
    return hist, best_acc, time.time() - t0


def plot_curves(hist):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), dpi=150)
    ax[0].plot(hist["train_loss"], label="train")
    ax[0].plot(hist["val_loss"], label="val")
    ax[0].set_title("Loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[0].grid(True, alpha=0.3)
    ax[1].plot(hist["val_acc"], color="green")
    ax[1].axhline(0.75, ls="--", color="red", label="target 0.75")
    ax[1].set_title("Validation accuracy"); ax[1].set_xlabel("epoch")
    ax[1].legend(); ax[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "cnn_curves.png"), bbox_inches="tight")


def plot_filters(model):
    w = model.features[0][0].weight.data.clone()  # (16, 3, 3, 3)
    w = (w - w.min()) / (w.max() - w.min() + 1e-8)
    fig, axes = plt.subplots(2, 8, figsize=(10, 3), dpi=150)
    for i, axi in enumerate(axes.flat):
        axi.imshow(w[i].permute(1, 2, 0).numpy())
        axi.axis("off")
    fig.suptitle("HW2 CNN: learned first-layer 3x3 filters")
    fig.savefig(os.path.join(RESULTS, "cnn_filters.png"), bbox_inches="tight")


def main():
    data_dir = download()
    torch.manual_seed(0)
    train_dl, val_dl = loaders(data_dir)
    model = CatsDogsNet().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Training CatsDogsNet ({n_params:,} params) from scratch ...")
    hist, best_acc, t = train(model, train_dl, val_dl)
    final_acc = hist["val_acc"][-1]

    plot_curves(hist)
    plot_filters(model)

    report = os.path.join(RESULTS, "hw2_cnn_results.txt")
    with open(report, "w", encoding="utf-8") as fh:
        fh.write("HW2 - CNN cats vs dogs (from scratch), CPU / 3 threads\n")
        fh.write("=" * 50 + "\n\n")
        fh.write(f"model params : {n_params:,}\n")
        fh.write(f"train time   : {t:.0f}s\n")
        fh.write(f"final val acc: {final_acc:.4f}\n")
        fh.write(f"best  val acc: {best_acc:.4f}  (target >= 0.75)  "
                 f"{'PASS' if best_acc >= 0.75 else 'FAIL'}\n\n")
        fh.write("per-epoch validation accuracy:\n")
        for i, a in enumerate(hist["val_acc"]):
            fh.write(f"  epoch {i + 1:2d}: {a:.4f}\n")
        fh.write("\nfigures: results/hw2/cnn_curves.png, results/hw2/cnn_filters.png\n")

    print("\n" + open(report, encoding="utf-8").read())


if __name__ == "__main__":
    main()
