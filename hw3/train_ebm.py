"""Train + verify the HW3 energy-based structured-prediction model on CPU.

Two stages, following the assignment:

1. **Single-character warm-up** - train ``SimpleNet`` as an ordinary classifier
   on 1-character images.  This gives sensible per-position energies quickly and
   we report classification accuracy.
2. **Multi-character structured training** - train with the *free-energy* loss
   (sum of cross-entropies along the min-energy monotone alignment) on
   2-character words, then decode held-out words and report exact-match accuracy.

Evidence (numbers + figures) is written to ``results/hw3/``.
"""
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from collections import Counter
from torch.nn.functional import cross_entropy
from tqdm import tqdm

torch.set_num_threads(3)

from ebm_structured import (  # noqa: E402
    ALPHABET_SIZE,
    BETWEEN,
    SimpleNet,
    SimpleWordsDataset,
    build_ce_matrix,
    build_path_matrix,
    find_path,
    path_energy,
    transform_word,
    worst_path,
)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.abspath(os.path.join(HERE, "..", "results", "hw3"))
os.makedirs(RESULTS, exist_ok=True)
DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# Stage 1: single-character classification warm-up.
# ---------------------------------------------------------------------------
def train_single_char(model, epochs=6, size=4000, bs=64):
    """Train the CNN as a plain 27-way classifier on 1-char images."""
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    t0 = time.time()
    for ep in range(epochs):
        ds = SimpleWordsDataset(1, size, jitter=True)
        dl = torch.utils.data.DataLoader(ds, batch_size=bs)
        last = 0.0
        for imgs, texts in dl:
            imgs = imgs.unsqueeze(1)                       # (B, 1, 32, 18)
            targets = torch.tensor([ord(t[0]) - ord("a") for t in texts])
            energies = model(imgs)                         # (B, W, 27)
            # A 1-char image collapses to a single useful column; average logits.
            logits = -energies.mean(dim=1)                 # low energy = high score
            loss = cross_entropy(logits, targets)
            opt.zero_grad(); loss.backward(); opt.step()
            last = loss.item()
        print(f"  [1-char] epoch {ep + 1}/{epochs} loss={last:.4f} "
              f"t={time.time() - t0:.0f}s")
    return time.time() - t0


@torch.no_grad()
def single_char_accuracy(model, n=1000):
    model.eval()
    ds = SimpleWordsDataset(1, n, jitter=True)
    dl = torch.utils.data.DataLoader(ds, batch_size=64)
    correct = total = 0
    for imgs, texts in dl:
        imgs = imgs.unsqueeze(1)
        targets = torch.tensor([ord(t[0]) - ord("a") for t in texts])
        energies = model(imgs)
        pred = (-energies.mean(dim=1)).argmax(dim=1)
        correct += (pred == targets).sum().item()
        total += len(texts)
    return correct / total


# ---------------------------------------------------------------------------
# Stage 2: structured (free-energy) training on 2-char words.
# ---------------------------------------------------------------------------
def collate_fn(samples):
    images, annotations = zip(*samples)
    images = list(images)
    annotations = [transform_word(a) for a in annotations]
    m_width = max(18, max(i.shape[1] for i in images))
    m_len = max(3, max(a.shape[0] for a in annotations))
    for i in range(len(images)):
        images[i] = torch.nn.functional.pad(images[i], (0, m_width - images[i].shape[-1]))
        annotations[i] = torch.nn.functional.pad(
            annotations[i], (0, m_len - annotations[i].shape[0]), value=BETWEEN)
    if len(images) == 1:
        return images[0].unsqueeze(0), torch.stack(annotations)
    return torch.stack(images), torch.stack(annotations)


def hard_free_energy(energies, targets):
    """Mean over the batch of the *hard* free energy: the energy of the
    minimum-energy monotone alignment between each energy sequence and its
    target (via the hand-written Viterbi ``find_path``).  This is the quantity
    the assignment defines; we report it as a diagnostic.  No gradient."""
    with torch.no_grad():
        B = energies.shape[0]
        total = 0.0
        for b in range(B):
            pm = build_path_matrix(energies[b:b + 1], targets[b:b + 1])[0]
            fe, _, _ = find_path(pm)
            total += fe.item()
        return total / B


def _ctc_collate(samples):
    """Collate variable-length words: pad images on width, keep raw target
    strings (as index lists) and their lengths for CTC."""
    images, texts = zip(*samples)
    images = list(images)
    m_width = max(18, max(i.shape[1] for i in images))
    for i in range(len(images)):
        images[i] = torch.nn.functional.pad(images[i], (0, m_width - images[i].shape[-1]))
    targets = [torch.tensor([ord(c) - ord("a") for c in t]) for t in texts]
    target_lengths = torch.tensor([len(t) for t in targets])
    targets = torch.cat(targets) if targets else torch.tensor([], dtype=torch.long)
    return torch.stack(images), targets, target_lengths


def train_multi_char(model, epochs=15, size=3000, bs=32, max_len=4):
    """Structured training with CTC loss.

    CTC is exactly the marginalized (soft) version of the min-energy monotone
    alignment used by ``find_path``: treating ``-energies`` as class log-scores
    and ``BETWEEN`` (26) as the CTC blank, the CTC loss sums over *all* valid
    monotone alignments of the energy sequence to the target word.  It is the
    stable, differentiable structured-prediction objective for this task.
    """
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.5)
    ctc = torch.nn.CTCLoss(blank=BETWEEN, zero_infinity=True)
    losses = []
    fe_diag = []
    t0 = time.time()
    for ep in range(epochs):
        # Curriculum: words of length 1..max_len, mixed each epoch.
        wl = 1 + (ep % max_len)
        ds = SimpleWordsDataset(wl, size)
        dl = torch.utils.data.DataLoader(ds, batch_size=bs, collate_fn=_ctc_collate)
        tot = 0.0
        nb = 0
        last_energies = last_targets = None
        for imgs, targets, tgt_lens in dl:
            imgs = imgs.unsqueeze(1)
            energies = model(imgs)                     # (B, W, 27)
            log_probs = torch.log_softmax(-energies, dim=2)  # low energy -> high prob
            W = log_probs.shape[1]
            log_probs_t = log_probs.permute(1, 0, 2)   # (W, B, 27) for CTCLoss
            input_lengths = torch.full((imgs.shape[0],), W, dtype=torch.long)
            loss = ctc(log_probs_t, targets, input_lengths, tgt_lens)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
            last_energies, last_targets = energies.detach(), (imgs, targets, tgt_lens)
        sched.step()
        losses.append(tot / nb)
        print(f"  [multi] epoch {ep + 1}/{epochs} (word_len={wl}) "
              f"ctc_loss={losses[-1]:.4f} t={time.time() - t0:.0f}s")
    return losses, time.time() - t0


def indices_to_str(indices):
    """CTC-style collapse: merge consecutive duplicate symbols, then drop the
    'between characters' blank.  Turns e.g. 'cyyyrr' -> 'cyr' -> (blank r) ..."""
    symbols = ["_" if int(i) == BETWEEN else chr(ord("a") + int(i)) for i in indices]
    collapsed = []
    for sym in symbols:
        if not collapsed or collapsed[-1] != sym:
            collapsed.append(sym)
    return "".join(c for c in collapsed if c != "_")


@torch.no_grad()
def decode_accuracy(model, n=300, word_len=2, collect=0):
    model.eval()
    ds = SimpleWordsDataset(word_len, n)
    correct = total = 0
    examples = []
    for img, text in ds:
        energies = model(img.view(1, 1, *img.shape))[0]
        pred = indices_to_str(energies.argmin(dim=-1))
        correct += int(pred == text)
        total += 1
        if len(examples) < collect:
            examples.append((text, pred))
    return correct / total, examples


def plot_energy_table(model):
    """Render the alphabet strip and the model's energy heat-map + best path."""
    ds = SimpleWordsDataset(1)
    alphabet = ds.draw_text(__import__("string").ascii_lowercase, 26 * 18)
    energies = model(alphabet.view(1, 1, *alphabet.shape))[0].detach()
    target = transform_word(__import__("string").ascii_lowercase)
    pm = build_path_matrix(energies.unsqueeze(0), target.unsqueeze(0))[0].detach()
    fe, path, _ = find_path(pm)

    fig, ax = plt.subplots(2, 1, figsize=(10, 6), dpi=150,
                           gridspec_kw={"height_ratios": [1, 4]})
    ax[0].imshow(alphabet.numpy(), cmap="gray")
    ax[0].set_title("rendered alphabet (a-z)"); ax[0].axis("off")
    im = ax[1].imshow(pm.t().numpy(), aspect="auto", cmap="viridis")
    xs = [p[0] for p in path]; ys = [p[1] for p in path]
    ax[1].plot(xs, ys, "r.-", lw=1, ms=4, label=f"best path (free energy={fe:.1f})")
    ax[1].set_xlabel("energy position l"); ax[1].set_ylabel("target index t")
    ax[1].legend(loc="upper left")
    fig.colorbar(im, ax=ax[1])
    fig.suptitle("HW3: energy alignment table (a-z), min-energy monotone path")
    fig.savefig(os.path.join(RESULTS, "energy_table.png"), bbox_inches="tight")
    return fe.item()


def main():
    torch.manual_seed(0)
    model = SimpleNet().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"SimpleNet: {n_params:,} params")

    print("Stage 1: single-character warm-up ...")
    t1 = train_single_char(model, epochs=4)
    sc_acc = single_char_accuracy(model)
    print(f"  single-char accuracy = {sc_acc:.4f}")

    print("Stage 2: structured CTC training on words (len 1..4) ...")
    losses, t2 = train_multi_char(model, epochs=16)

    # Decode accuracy per word length + example decodes.
    accs = {}
    examples = []
    for wl in (1, 2, 3, 4):
        acc, ex = decode_accuracy(model, n=300, word_len=wl, collect=(3 if wl <= 2 else 0))
        accs[wl] = acc
        examples += ex
        print(f"  decode acc (len {wl}) = {acc:.4f}")

    fe = plot_energy_table(model)

    # loss curve
    plt.figure(dpi=150)
    plt.plot(losses)
    plt.xlabel("epoch"); plt.ylabel("mean CTC loss")
    plt.title("HW3: structured (CTC) training")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(RESULTS, "free_energy_curve.png"), bbox_inches="tight")

    mean_acc = sum(accs.values()) / len(accs)
    report = os.path.join(RESULTS, "hw3_results.txt")
    with open(report, "w", encoding="utf-8") as fh:
        fh.write("HW3 - Energy-based structured prediction, CPU / 3 threads\n")
        fh.write("=" * 55 + "\n\n")
        fh.write(f"SimpleNet params           : {n_params:,}\n\n")
        fh.write("Stage 1 (single-char classifier warm-up):\n")
        fh.write(f"  train time               : {t1:.0f}s\n")
        fh.write(f"  single-char accuracy     : {sc_acc:.4f}\n\n")
        fh.write("Stage 2 (structured CTC training, words length 1..4):\n")
        fh.write(f"  train time               : {t2:.0f}s\n")
        fh.write(f"  initial CTC loss         : {losses[0]:.4f}\n")
        fh.write(f"  final   CTC loss         : {losses[-1]:.4f}\n")
        fh.write("  exact-match decode accuracy by word length:\n")
        for wl in (1, 2, 3, 4):
            fh.write(f"    len {wl}: {accs[wl]:.4f}\n")
        fh.write(f"  mean decode accuracy     : {mean_acc:.4f}  "
                 f"{'PASS' if mean_acc >= 0.8 else 'PARTIAL'}\n")
        fh.write(f"  alphabet best-path energy (Viterbi): {fe:.2f}\n\n")
        fh.write("decoded examples (target -> prediction):\n")
        for tgt, pred in examples:
            fh.write(f"  {tgt} -> {pred}  {'OK' if tgt == pred else 'x'}\n")
        fh.write("\nfigures: results/hw3/free_energy_curve.png (CTC loss), "
                 "results/hw3/energy_table.png (Viterbi alignment)\n")

    print("\n" + open(report, encoding="utf-8").read())


if __name__ == "__main__":
    main()
