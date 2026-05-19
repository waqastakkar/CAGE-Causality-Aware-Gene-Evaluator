"""CAGE pure-numpy deep-learning primitives for Phase II.

Implements the minimal neural-network building blocks required by the
sparse invariant autoencoder-classifier (step 4) without a deep-learning
framework. Every operator exposes a manual forward/backward contract:

    out = op.forward(x, training=True)
    grad_in = op.backward(grad_out)
    op.params   -> dict[str, ndarray]
    op.grads    -> dict[str, ndarray]  (populated by .backward)

An :class:`AdamW` optimizer consumes ``params`` + ``grads`` dicts to run
in-place decoupled-weight-decay Adam updates.

This module also houses:

* Losses: :func:`bce_with_logits`, :func:`cross_entropy_with_logits`,
  :func:`mse_loss`, each with ``*_grad`` companions.
* Tools: :func:`standardize_fit`, :func:`standardize_apply`,
  :func:`one_hot`, :func:`softmax`, :func:`sigmoid`,
  :func:`gradient_reversal` (a stateless scalar multiplier used between
  latent and adversary), :func:`save_npz_bundle`/:func:`load_npz_bundle`
  for checkpoints, and :class:`EnvironmentBatcher` / :class:`StratifiedBatcher`.

Design notes
------------
* All ops accept ``(n, d)`` float64 inputs and ``grad_out`` of the same
  shape unless documented otherwise.
* Parameters are stored and updated in-place, so training code simply
  collects ``(params, grads)`` dicts and hands them to the optimizer.
* The :class:`FeatureGate` supports both ``"l1"`` (sigmoid(alpha)
  multiplicative gate, L1-surrogate penalty) and ``"hard-concrete"``
  (stretched-sigmoid gate with L0 expectation penalty of Louizos et al.).
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger("cage.deep_model_utils")

__all__ = [
    # Activations / math
    "sigmoid",
    "softmax",
    "one_hot",
    # Losses
    "bce_with_logits",
    "bce_with_logits_grad",
    "cross_entropy_with_logits",
    "cross_entropy_with_logits_grad",
    "mse_loss",
    "mse_loss_grad",
    # Modules
    "Linear",
    "ReLU",
    "Dropout",
    "L2Normalize",
    "FeatureGate",
    # Optimizer
    "AdamW",
    # Utilities
    "standardize_fit",
    "standardize_apply",
    "gradient_reversal",
    "save_npz_bundle",
    "load_npz_bundle",
]


# ---------------------------------------------------------------------------
# Activation math
# ---------------------------------------------------------------------------


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable element-wise sigmoid."""
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax along ``axis``."""
    m = np.max(z, axis=axis, keepdims=True)
    e = np.exp(z - m)
    return e / np.sum(e, axis=axis, keepdims=True)


def one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    """Integer labels -> float one-hot matrix of shape ``(n, n_classes)``."""
    y = np.asarray(y).ravel().astype(np.int64)
    out = np.zeros((y.size, n_classes), dtype=np.float64)
    mask = (y >= 0) & (y < n_classes)
    out[np.arange(y.size)[mask], y[mask]] = 1.0
    return out


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------


def bce_with_logits(
    logits: np.ndarray,
    y: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
    eps: float = 1e-12,
) -> float:
    """Binary cross-entropy with logits (mean over batch).

    Numerically stable form: ``max(z,0) - z*y + log(1 + exp(-|z|))``.
    """
    z = np.asarray(logits, dtype=np.float64).ravel()
    t = np.asarray(y, dtype=np.float64).ravel()
    loss = np.maximum(z, 0.0) - z * t + np.log1p(np.exp(-np.abs(z)))
    if sample_weight is not None:
        w = np.asarray(sample_weight, dtype=np.float64).ravel()
        denom = max(float(w.sum()), eps)
        return float((w * loss).sum() / denom)
    return float(loss.mean())


def bce_with_logits_grad(
    logits: np.ndarray,
    y: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
    eps: float = 1e-12,
) -> np.ndarray:
    """Gradient of :func:`bce_with_logits` w.r.t. ``logits``."""
    z = np.asarray(logits, dtype=np.float64).ravel()
    t = np.asarray(y, dtype=np.float64).ravel()
    p = sigmoid(z)
    grad = p - t
    if sample_weight is not None:
        w = np.asarray(sample_weight, dtype=np.float64).ravel()
        denom = max(float(w.sum()), eps)
        grad = w * grad / denom
    else:
        grad = grad / max(1, grad.size)
    return grad.reshape(np.asarray(logits).shape)


def cross_entropy_with_logits(
    logits: np.ndarray,
    y: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
    eps: float = 1e-12,
) -> float:
    """Multi-class cross-entropy with logits (mean over batch)."""
    z = np.asarray(logits, dtype=np.float64)
    n = z.shape[0]
    if n == 0:
        return 0.0
    log_z = z - np.max(z, axis=1, keepdims=True)
    log_sum = np.log(np.sum(np.exp(log_z), axis=1, keepdims=True))
    log_probs = log_z - log_sum
    y = np.asarray(y, dtype=np.int64).ravel()
    losses = -log_probs[np.arange(n), y]
    if sample_weight is not None:
        w = np.asarray(sample_weight, dtype=np.float64).ravel()
        denom = max(float(w.sum()), eps)
        return float((w * losses).sum() / denom)
    return float(losses.mean())


def cross_entropy_with_logits_grad(
    logits: np.ndarray,
    y: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
    eps: float = 1e-12,
) -> np.ndarray:
    """Gradient of :func:`cross_entropy_with_logits` w.r.t. logits."""
    z = np.asarray(logits, dtype=np.float64)
    n = z.shape[0]
    if n == 0:
        return np.zeros_like(z)
    p = softmax(z, axis=1)
    y = np.asarray(y, dtype=np.int64).ravel()
    p[np.arange(n), y] -= 1.0
    if sample_weight is not None:
        w = np.asarray(sample_weight, dtype=np.float64).ravel()
        denom = max(float(w.sum()), eps)
        return p * (w[:, None] / denom)
    return p / n


def mse_loss(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean squared error over all elements of ``(pred - target)``."""
    return float(np.mean((pred - target) ** 2))


def mse_loss_grad(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Gradient of :func:`mse_loss` w.r.t. ``pred``."""
    diff = pred - target
    return 2.0 * diff / max(1, diff.size)


# ---------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------


class _Module:
    """Minimal base class; subclasses populate ``params`` and ``grads``."""

    def __init__(self, name: str = "module") -> None:
        self.name = name
        self.params: Dict[str, np.ndarray] = {}
        self.grads: Dict[str, np.ndarray] = {}

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def backward(self, grad_out: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def zero_grad(self) -> None:
        for k in self.grads:
            self.grads[k].fill(0.0)


class Linear(_Module):
    """Fully connected layer ``y = x W + b`` with He-style init."""

    def __init__(self, n_in: int, n_out: int, *, rng: np.random.Generator, name: str = "linear") -> None:
        super().__init__(name=name)
        scale = math.sqrt(2.0 / max(n_in, 1))
        W = rng.normal(0.0, scale, size=(n_in, n_out)).astype(np.float64)
        b = np.zeros(n_out, dtype=np.float64)
        self.params = {"W": W, "b": b}
        self.grads = {"W": np.zeros_like(W), "b": np.zeros_like(b)}
        self._cache: Dict[str, np.ndarray] = {}

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        self._cache["x"] = x
        return x @ self.params["W"] + self.params["b"]

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        x = self._cache["x"]
        # In-place accumulate so shared references (model.grads[key] is this
        # same array) stay consistent across the whole training step.
        self.grads["W"] += x.T @ grad_out
        self.grads["b"] += grad_out.sum(axis=0)
        return grad_out @ self.params["W"].T


class ReLU(_Module):
    """Element-wise ReLU; no trainable parameters."""

    def __init__(self, name: str = "relu") -> None:
        super().__init__(name=name)
        self._mask: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        self._mask = (x > 0).astype(np.float64)
        return x * self._mask

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        return grad_out * self._mask


class Dropout(_Module):
    """Inverted-dropout layer; identity at eval time."""

    def __init__(self, p: float = 0.2, *, rng: Optional[np.random.Generator] = None, name: str = "dropout") -> None:
        super().__init__(name=name)
        self.p = float(p)
        self._rng = rng or np.random.default_rng()
        self._mask: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        if not training or self.p <= 0.0:
            self._mask = None
            return x
        keep = 1.0 - self.p
        self._mask = (self._rng.uniform(0.0, 1.0, size=x.shape) < keep).astype(np.float64) / keep
        return x * self._mask

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        if self._mask is None:
            return grad_out
        return grad_out * self._mask


class L2Normalize(_Module):
    """Row-wise L2 normalization of the latent bottleneck (no trainable parameters).

    Constrains each sample's latent vector to unit L2 norm, preventing the
    adversary and encoder from entering an unconstrained magnitude escalation
    during GRL-based adversarial training. With bounded latent norms:
      - adversary logits are bounded by ||W_adv|| (regulated by weight decay)
      - invariance penalty ||mean(z|env) - mean(z)||^2 is bounded by 4
      - classification logits are bounded similarly
    """

    def __init__(self, eps: float = 1e-8, name: str = "l2norm") -> None:
        super().__init__(name=name)
        self.eps = float(eps)
        self._cache: Dict[str, np.ndarray] = {}

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        norms = np.sqrt((x * x).sum(axis=1, keepdims=True)) + self.eps
        z_norm = x / norms
        self._cache = {"z_norm": z_norm, "norms": norms}
        return z_norm

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        z_norm = self._cache["z_norm"]
        norms = self._cache["norms"]
        # Jacobian of x/||x||: d(z_norm_i)/d(x_j) = (delta_ij - z_norm_i*z_norm_j)/||x||
        dot = (grad_out * z_norm).sum(axis=1, keepdims=True)
        return (grad_out - dot * z_norm) / norms


class FeatureGate(_Module):
    """Per-feature sparse gate multiplicatively applied to inputs.

    Two sparsity regimes are supported:

    * ``"l1"`` : gate = sigmoid(alpha); penalty = mean(gate).
    * ``"hard-concrete"`` : stretched-sigmoid gate (Louizos et al., 2018)
      with L0-expectation penalty. During training, Gumbel noise is
      injected; during eval the noise is deterministic.

    The module exposes ``params['log_alpha']`` and ``grads['log_alpha']``,
    and supplies :meth:`sparsity_penalty` / :meth:`sparsity_penalty_grad`
    so the training loop can incorporate the gate's regularizer.
    """

    def __init__(
        self,
        n_features: int,
        *,
        sparsity_type: str = "l1",
        init_log_alpha: float = 2.0,
        beta: float = 2.0 / 3.0,
        gamma: float = -0.1,
        zeta: float = 1.1,
        rng: Optional[np.random.Generator] = None,
        name: str = "gate",
    ) -> None:
        super().__init__(name=name)
        if sparsity_type not in ("l1", "hard-concrete"):
            raise ValueError(f"Unknown sparsity_type {sparsity_type!r}")
        log_alpha = np.full(n_features, init_log_alpha, dtype=np.float64)
        self.params = {"log_alpha": log_alpha}
        self.grads = {"log_alpha": np.zeros_like(log_alpha)}
        self.sparsity_type = sparsity_type
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.zeta = float(zeta)
        self._rng = rng or np.random.default_rng()
        self._cache: Dict[str, Any] = {}

    # -- gate computation --------------------------------------------------

    def _gate_l1(self, training: bool) -> Tuple[np.ndarray, np.ndarray]:
        alpha = self.params["log_alpha"]
        gate = sigmoid(alpha)
        d_gate_d_alpha = gate * (1.0 - gate)
        return gate, d_gate_d_alpha

    def _gate_hc(self, training: bool) -> Tuple[np.ndarray, np.ndarray]:
        alpha = self.params["log_alpha"]
        if training:
            u = self._rng.uniform(1e-6, 1.0 - 1e-6, size=alpha.shape)
            s = sigmoid((np.log(u) - np.log(1.0 - u) + alpha) / self.beta)
        else:
            s = sigmoid(alpha / self.beta)
        s_stretched = s * (self.zeta - self.gamma) + self.gamma
        gate = np.clip(s_stretched, 0.0, 1.0)
        active = ((s_stretched > 0.0) & (s_stretched < 1.0)).astype(np.float64)
        d_gate_d_alpha = active * (s * (1.0 - s) * (self.zeta - self.gamma) / self.beta)
        return gate, d_gate_d_alpha

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        if self.sparsity_type == "l1":
            gate, d_gate_d_alpha = self._gate_l1(training)
        else:
            gate, d_gate_d_alpha = self._gate_hc(training)
        self._cache = {"x": x, "gate": gate, "d_gate_d_alpha": d_gate_d_alpha}
        return x * gate

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        x = self._cache["x"]
        gate = self._cache["gate"]
        d_gate_d_alpha = self._cache["d_gate_d_alpha"]
        # grad w.r.t. input
        grad_in = grad_out * gate
        # grad w.r.t. log_alpha accumulates across the batch; write in place
        # so the shared reference held by the parent model survives.
        grad_gate = (grad_out * x).sum(axis=0)
        self.grads["log_alpha"] += grad_gate * d_gate_d_alpha
        return grad_in

    # -- penalty -----------------------------------------------------------

    def sparsity_penalty(self) -> float:
        alpha = self.params["log_alpha"]
        if self.sparsity_type == "l1":
            return float(sigmoid(alpha).mean())
        shift = self.beta * math.log(-self.gamma / self.zeta)
        return float(sigmoid(alpha - shift).mean())

    def sparsity_penalty_grad(self, weight: float = 1.0) -> np.ndarray:
        alpha = self.params["log_alpha"]
        n = float(alpha.size)
        if self.sparsity_type == "l1":
            s = sigmoid(alpha)
            return weight * (s * (1.0 - s)) / n
        shift = self.beta * math.log(-self.gamma / self.zeta)
        s = sigmoid(alpha - shift)
        return weight * (s * (1.0 - s)) / n

    def eval_gate(self) -> np.ndarray:
        """Deterministic gate values used at inference time."""
        alpha = self.params["log_alpha"]
        if self.sparsity_type == "l1":
            return sigmoid(alpha)
        s = sigmoid(alpha / self.beta)
        s_stretched = s * (self.zeta - self.gamma) + self.gamma
        return np.clip(s_stretched, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------


class AdamW:
    """Decoupled weight-decay Adam (Loshchilov & Hutter, 2019).

    Parameters are tracked in a single flat name->array dict; each name
    may optionally opt out of weight decay (e.g., biases, gate log_alpha).
    """

    def __init__(
        self,
        params: Mapping[str, np.ndarray],
        *,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        no_decay_names: Sequence[str] = (),
    ) -> None:
        self.lr = float(lr)
        self.beta1, self.beta2 = float(betas[0]), float(betas[1])
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.no_decay_names = set(no_decay_names)
        self._m = {k: np.zeros_like(v) for k, v in params.items()}
        self._v = {k: np.zeros_like(v) for k, v in params.items()}
        self._t = 0

    def step(
        self,
        params: Dict[str, np.ndarray],
        grads: Mapping[str, np.ndarray],
        *,
        lr: Optional[float] = None,
    ) -> None:
        """Apply one Adam update in-place over ``params``.

        ``grads`` must share keys with ``params``. Missing grads are
        treated as zero (useful when some parameters are frozen on a
        particular step, e.g., the adversary during encoder updates).
        """
        self._t += 1
        lr_eff = self.lr if lr is None else float(lr)
        bc1 = 1.0 - self.beta1 ** self._t
        bc2 = 1.0 - self.beta2 ** self._t
        for name, p in params.items():
            g = grads.get(name)
            if g is None:
                continue
            # Decoupled weight decay
            if self.weight_decay > 0.0 and name not in self.no_decay_names:
                p *= (1.0 - lr_eff * self.weight_decay)
            m = self._m[name]
            v = self._v[name]
            m[...] = self.beta1 * m + (1.0 - self.beta1) * g
            v[...] = self.beta2 * v + (1.0 - self.beta2) * (g ** 2)
            mh = m / bc1
            vh = v / bc2
            p -= lr_eff * mh / (np.sqrt(vh) + self.eps)


# ---------------------------------------------------------------------------
# Standardization / utilities
# ---------------------------------------------------------------------------


def standardize_fit(X: np.ndarray, eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(mean, std)`` computed per feature across rows."""
    mean = X.mean(axis=0)
    std = X.std(axis=0, ddof=0)
    std = np.where(std < eps, 1.0, std)
    return mean, std


def standardize_apply(
    X: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    """Apply a previously fit ``(mean, std)`` standardization."""
    return (X - mean[None, :]) / std[None, :]


def gradient_reversal(grad: np.ndarray, lambda_: float) -> np.ndarray:
    """Multiply ``grad`` by ``-lambda_`` (the GRL pass)."""
    return -float(lambda_) * grad


def save_npz_bundle(path: str | Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Persist a dict of ndarrays into a ``.npz`` archive."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{k: np.asarray(v) for k, v in arrays.items()})


def load_npz_bundle(path: str | Path) -> Dict[str, np.ndarray]:
    """Load a previously saved ``.npz`` bundle into a plain dict."""
    with np.load(Path(path), allow_pickle=False) as data:
        return {k: np.asarray(data[k]) for k in data.files}
