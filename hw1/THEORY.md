# HW1 Theory — Backpropagation

Concise solution notes for the written part of NYU-DLSP21 HW1. The network is

```
x  ->  Linear1 (W1, b1)  ->  f  ->  Linear2 (W2, b2)  ->  g  ->  y_hat  ->  loss ℓ(y_hat, y)
```

Let `s1 = W1 x + b1`, `a1 = f(s1)`, `s2 = W2 a1 + b2`, `y_hat = g(s2)`. All
activations `f, g ∈ {relu, sigmoid, identity}` are element-wise.

## 1. Training loop (one step)

1. **Forward** — feed `x` through the two linear layers and non-linearities to
   get `y_hat`.
2. **Loss** — evaluate `ℓ(y_hat, y)`.
3. **Zero grads** — clear accumulated gradients (autograd accumulates by
   default; our manual version also zeroes `self.grads`).
4. **Backward** — chain rule to get `∂ℓ/∂{W1,b1,W2,b2}`.
5. **Step** — `θ ← θ − η ∂ℓ/∂θ`.

## 2. Gradients (chain rule)

Write `δ2 = ∂ℓ/∂s2 = (∂ℓ/∂y_hat) ⊙ g'(s2)` and
`δ1 = ∂ℓ/∂s1 = (W2ᵀ δ2) ⊙ f'(s1)`. Then, for a single sample:

| Parameter | Gradient |
|---|---|
| `W2` | `δ2 · a1ᵀ` |
| `b2` | `δ2` |
| `W1` | `δ1 · xᵀ` |
| `b1` | `δ1` |

For a mini-batch the weight gradients are the sum of the outer products over the
batch (`dJdW2 = δ2ᵀ a1`, `dJdW1 = δ1ᵀ x`) and the bias gradients are the sum of
`δ` over the batch. This is exactly what `mlp.py::backward` computes.

## 3. Jacobians of the element-wise pieces

Because `f`, `g` act element-wise, `∂a1/∂s1` and `∂y_hat/∂s2` are **diagonal**
Jacobians, so the chain rule reduces to element-wise (Hadamard) multiplication
by the derivative:

- **ReLU**: `f'(s) = 1[s > 0]`.
- **Sigmoid**: `σ'(s) = σ(s)(1 − σ(s))`.
- **Identity**: derivative `= 1`.

## 4. Loss derivatives (`∂ℓ/∂y_hat`)

- **MSE** `ℓ = (1/n) Σ (y_hat − y)²`  ⇒  `∂ℓ/∂y_hat = (2/n)(y_hat − y)`.
- **BCE** `ℓ = −(1/n) Σ [y log y_hat + (1−y) log(1−y_hat)]`
  ⇒  `∂ℓ/∂y_hat = (1/n)[ −y/y_hat + (1−y)/(1−y_hat) ]`.

## 5. Swapping activations / losses

Changing `f`, `g`, or the loss does **not** change the *structure* of the
backward pass — only the diagonal derivative factors (`f'`, `g'`) and the loss
derivative `∂ℓ/∂y_hat` change. Dimensions are unchanged. This is why `mlp.py`
implements the derivatives as a lookup table keyed by the activation name.

## 6. Why ReLU over sigmoid in deep nets

Sigmoid saturates: `σ'(s) ≤ 0.25`, and repeatedly multiplying such factors
through many layers drives gradients toward zero (**vanishing gradients**). ReLU
has derivative exactly `1` on the active side, so it preserves gradient
magnitude and trains deep networks far more reliably.

All of the above is validated empirically: `test1/2/3.py` confirm the analytic
gradients match `torch.autograd` to `< 1e-3`, and `train_demo.py` shows the
network actually learning with these hand-derived gradients.
