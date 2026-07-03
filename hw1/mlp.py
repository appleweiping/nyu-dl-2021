"""HW1 - Fundamentals of neural nets: a two-layer MLP with backprop written by
hand (no ``autograd``).

Architecture::

    x  ->  Linear1 (W1, b1)  ->  f  ->  Linear2 (W2, b2)  ->  g  ->  y_hat

where ``f`` and ``g`` are each one of ``relu | sigmoid | identity``.

We keep the exact intermediate tensors from the forward pass in ``self.cache``
and reuse them in ``backward`` to compute closed-form gradients via the chain
rule.  The gradients are verified against ``torch.autograd`` in test1/2/3.py.

Internally everything is computed with the "features-on-rows" convention
(shape ``out_features x batch``) which mirrors the math notation
``z = W x + b``; inputs/outputs are transposed at the boundary so the public
API uses ``batch x features``.
"""
import torch


def _sigmoid(x):
    return 1.0 / (1.0 + torch.exp(-x))


# Element-wise activations and their derivatives (as functions of the pre-activation).
_ACTIVATIONS = {
    "relu": torch.relu,
    "sigmoid": _sigmoid,
    "identity": lambda x: x,
}
_ACTIVATION_GRADS = {
    "relu": lambda z: (z > 0).to(z.dtype),
    "sigmoid": lambda z: _sigmoid(z) * (1.0 - _sigmoid(z)),
    "identity": lambda z: torch.ones_like(z),
}


class MLP:
    def __init__(
        self,
        linear_1_in_features,
        linear_1_out_features,
        f_function,
        linear_2_in_features,
        linear_2_out_features,
        g_function,
    ):
        """
        Args:
            linear_1_in_features: the in features of first linear layer
            linear_1_out_features: the out features of first linear layer
            linear_2_in_features: the in features of second linear layer
            linear_2_out_features: the out features of second linear layer
            f_function: string for the f function: relu | sigmoid | identity
            g_function: string for the g function: relu | sigmoid | identity
        """
        self.f_function = f_function
        self.g_function = g_function

        self.parameters = dict(
            W1=torch.randn(linear_1_out_features, linear_1_in_features),
            b1=torch.randn(linear_1_out_features),
            W2=torch.randn(linear_2_out_features, linear_2_in_features),
            b2=torch.randn(linear_2_out_features),
        )
        self.grads = dict(
            dJdW1=torch.zeros(linear_1_out_features, linear_1_in_features),
            dJdb1=torch.zeros(linear_1_out_features),
            dJdW2=torch.zeros(linear_2_out_features, linear_2_in_features),
            dJdb2=torch.zeros(linear_2_out_features),
        )

        # put all the cache value you need in self.cache
        self.cache = dict()

    def forward(self, x):
        """
        Args:
            x: tensor shape (batch_size, linear_1_in_features)

        Returns:
            y_hat: tensor shape (batch_size, linear_2_out_features)
        """
        f = _ACTIVATIONS[self.f_function]
        g = _ACTIVATIONS[self.g_function]

        # Work with columns = samples so that z = W @ x + b matches the math.
        xT = x.t()                                              # (in1, batch)
        s1 = self.parameters["W1"] @ xT + self.parameters["b1"].unsqueeze(1)
        a1 = f(s1)                                              # (out1, batch)
        s2 = self.parameters["W2"] @ a1 + self.parameters["b2"].unsqueeze(1)
        y_hat = g(s2)                                           # (out2, batch)

        # Everything backward() needs.
        self.cache["x"] = x
        self.cache["s1"] = s1
        self.cache["a1"] = a1
        self.cache["s2"] = s2

        return y_hat.t()

    def backward(self, dJdy_hat):
        """
        Args:
            dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
        """
        f_grad = _ACTIVATION_GRADS[self.f_function]
        g_grad = _ACTIVATION_GRADS[self.g_function]

        x = self.cache["x"]                                    # (batch, in1)
        s1 = self.cache["s1"]                                  # (out1, batch)
        a1 = self.cache["a1"]                                  # (out1, batch)
        s2 = self.cache["s2"]                                  # (out2, batch)

        # dJ/ds2 = dJ/dy_hat * g'(s2)   -- both (batch, out2)
        dJds2 = dJdy_hat * g_grad(s2).t()                      # (batch, out2)

        # Linear2 params.  y_hat = g(W2 a1 + b2).
        # dJ/dW2 = sum_n dJds2_n outer a1_n ;  dJ/db2 = sum_n dJds2_n
        self.grads["dJdW2"] = dJds2.t() @ a1.t()              # (out2, out1)
        self.grads["dJdb2"] = dJds2.sum(dim=0)               # (out2,)

        # Propagate to a1, then through f.
        dJda1 = dJds2 @ self.parameters["W2"]                 # (batch, out1)
        dJds1 = dJda1 * f_grad(s1).t()                        # (batch, out1)

        # Linear1 params.
        self.grads["dJdW1"] = dJds1.t() @ x                  # (out1, in1)
        self.grads["dJdb1"] = dJds1.sum(dim=0)               # (out1,)

    def clear_grad_and_cache(self):
        for grad in self.grads:
            self.grads[grad].zero_()
        self.cache = dict()


def mse_loss(y, y_hat):
    """
    Mean-squared-error loss and its gradient wrt ``y_hat``.

    Args:
        y: the label tensor (batch_size, linear_2_out_features)
        y_hat: the prediction tensor (batch_size, linear_2_out_features)

    Return:
        J: scalar of loss
        dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """
    n = y.numel()
    diff = y_hat - y
    J = (diff ** 2).sum() / n
    dJdy_hat = (2.0 / n) * diff
    return J, dJdy_hat


def bce_loss(y, y_hat):
    """
    Binary-cross-entropy loss and its gradient wrt ``y_hat``.

    Args:
        y: the label tensor
        y_hat: the prediction tensor (already in (0, 1))

    Return:
        J: scalar of loss
        dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """
    eps = 1e-12
    n = y.numel()
    yh = y_hat.clamp(eps, 1.0 - eps)
    J = -(y * torch.log(yh) + (1.0 - y) * torch.log(1.0 - yh)).sum() / n
    dJdy_hat = (1.0 / n) * (-(y / yh) + (1.0 - y) / (1.0 - yh))
    return J, dJdy_hat
