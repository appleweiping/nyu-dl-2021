# HW3 Theory — Energy-Based Models & Structured Prediction

Concise notes for the written/conceptual part of NYU-DLSP21 HW3.

## Setup

A CNN reads a 32-pixel-tall strip image of a short word and outputs, for each of
`L` horizontal positions, an **energy vector** over a 27-symbol alphabet
(`a`..`z` = 0..25, plus a "between characters" / blank symbol = 26). Low energy =
the model believes that symbol is present at that position.

Reading the word is a **structured** prediction problem: we must align the
length-`L` energy sequence to the length-`T` target symbol sequence
`y = transform_word(word)` (which interleaves blanks, e.g. `hi` → `h _ i _`).

## Alignment as a minimum-energy monotone path

A valid alignment is a **monotone path** through the `L × T` "path matrix"
`pm[l, t] = energy[l, y[t]]`: it starts at `(0, 0)`, ends at `(L-1, T-1)`, and at
each step either stays in the same target index or advances it by exactly one.
Its energy is the sum of `pm` along the path.

- `find_path(pm)` computes the **minimum-energy** monotone path by dynamic
  programming (Viterbi): `dp[l, t] = pm[l, t] + min(dp[l-1, t], dp[l-1, t-1])`,
  with backpointers to recover the path. `dp[L-1, T-1]` is the **free energy**.
- `worst_path(pm)` is the same recurrence with `min → max` (used to illustrate
  the DP and bound the energy range).
- `path_energy(pm, path)` sums energies along a given path, returning `2³⁰` for
  any path that violates the monotonicity / endpoint constraints.

## Free energy and the loss

The **free energy** of an (image, word) pair is the energy of its best
alignment. Training should push the free energy of the *correct* word down. In
practice, the hard `min` over paths is replaced by a **soft minimization**
(log-sum-exp over all paths) — which is exactly the **CTC forward algorithm**.
Treating `−energies` as class log-scores and symbol 26 as the CTC blank, the CTC
loss marginalizes over *all* valid monotone alignments, giving a smooth,
differentiable objective. This is why this repo trains with `torch.nn.CTCLoss`
while keeping the hand-written Viterbi `find_path` for the hard-alignment
visualization and free-energy diagnostics.

## Decoding

At inference we take the per-position `argmin` over energies, then **collapse
consecutive duplicates and drop blanks** (standard CTC greedy decoding) to
recover the word. On held-out rendered words this reaches 100% exact-match for
lengths 1–4 (`results/hw3/hw3_results.txt`).

## Why energy-based / structured?

A plain per-position classifier can't handle that a character spans several
columns, that spacing varies, or that word length is unknown. Casting reading as
energy minimization over monotone alignments (a) makes the length-mismatch
between pixels and characters explicit, (b) yields a principled loss (free
energy / its soft CTC form), and (c) connects directly to the sequence-alignment
DPs used across speech and handwriting recognition.
