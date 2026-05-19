"""CAGE Step 4 runner: sparse invariant autoencoder-classifier.

Trains the Phase II model over the patient-grouped outer folds emitted
by Step 2. The model stack is::

    x  ->  FeatureGate  ->  Encoder (2 hidden + ReLU) ->  latent
                                                       |
                                   +-------------------+----------------+
                                   |                                    |
                            Classifier head (BCE)           Confounder adversary (CE, GRL)
                                   |                                    |
                                   +-------------------+----------------+
                                                       |
                                              (optional) Decoder (MSE recon)

Losses (summed with user-supplied weights):

    L = L_cls + lambda_recon * L_recon + lambda_sparsity * L_gate
                                       + lambda_adv * L_adv_encoder
                                       + lambda_inv * L_inv

* ``L_cls``           : BCE-with-logits on tumor vs normal.
* ``L_recon``         : MSE between decoder output and the gated input.
* ``L_gate``          : L1 mean(sigmoid(alpha)) or L0 expectation (hard-concrete).
* ``L_adv_encoder``   : cross-entropy of the confounder adversary, reversed
                         via GRL so the encoder is pushed to confuse the
                         adversary (the adversary itself is trained
                         forward with the same loss, no reversal).
* ``L_inv``           : sum_k || mean(latent | env=k) - mean(latent) ||^2.

The adversary parameters are updated with the *non-reversed* gradient of
``L_adv_encoder`` on the same mini-batch (a "mini" two-player update
interleaved each step). The encoder receives the reversed gradient as
usual through the GRL node.

Outputs (see :func:`run_step4`):

    deep_oof_predictions.csv    per-sample tumor probability (OOF)
    gate_weights.csv            per-fold learned gate per gene
    latent_embeddings.csv       per-sample latent coordinates (OOF)
    deep_per_fold_metrics.csv   per-fold AUROC/AUPRC/BAC/Brier/...
    deep_summary_metrics.csv    overall-OOF + mean-of-folds with CIs
    deep_subgroup_metrics.csv   (optional) per-environment metrics
    deep_calibration.csv        (optional) reliability diagram
    deep_training_history.csv   per-epoch loss traces
    checkpoints/fold_<k>.npz    gate + encoder + classifier per fold
    phase2_summary.json         configuration + metrics + figure manifest
    figures/                    latent PCA, gate distribution, ROC/PR, calibration
"""

from __future__ import annotations

import csv
import json
import logging
import math
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from . import deep_model_utils as dm
from . import metrics as mx
from . import preprocess_esca as pp
from . import step3_runner as step3

logger = logging.getLogger("cage.step4.runner")

__all__ = [
    "SparseInvariantModel",
    "TrainingConfig",
    "build_model",
    "train_one_fold",
    "run_step4_cv",
    "generate_step4_figures",
]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class TrainingConfig:
    """Hyperparameters for one fold's training run."""
    latent_dim: int = 48
    hidden_dims: Sequence[int] = (256, 96)
    sparsity_type: str = "l1"      # or "hard-concrete"
    use_decoder: bool = True
    dropout: float = 0.1

    lr: float = 1e-3
    weight_decay: float = 1e-4
    n_epochs: int = 150
    batch_size: int = 64
    patience: int = 15

    lambda_recon: float = 0.1
    lambda_sparsity: float = 1e-3
    lambda_adv: float = 0.5
    lambda_inv: float = 0.1
    adv_ramp_epochs: int = 5   # linearly ramp lambda_adv from 0 -> target
    grad_clip_norm: float = 5.0  # L2 norm ceiling over all parameter grads

    seed: int = 2026

    def as_dict(self) -> Dict[str, Any]:
        return {
            "latent_dim": int(self.latent_dim),
            "hidden_dims": list(self.hidden_dims),
            "sparsity_type": self.sparsity_type,
            "use_decoder": bool(self.use_decoder),
            "dropout": float(self.dropout),
            "lr": float(self.lr),
            "weight_decay": float(self.weight_decay),
            "n_epochs": int(self.n_epochs),
            "batch_size": int(self.batch_size),
            "patience": int(self.patience),
            "lambda_recon": float(self.lambda_recon),
            "lambda_sparsity": float(self.lambda_sparsity),
            "lambda_adv": float(self.lambda_adv),
            "lambda_inv": float(self.lambda_inv),
            "adv_ramp_epochs": int(self.adv_ramp_epochs),
            "grad_clip_norm": float(self.grad_clip_norm),
            "seed": int(self.seed),
        }


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class SparseInvariantModel:
    """Numpy-only sparse invariant autoencoder-classifier.

    Layer graph::

        gate  ->  fc0 -> ReLU -> drop -> fc1 -> ReLU -> drop -> fc_latent
                          |                                           |
                          +------ (decoder: fc_dec -> ReLU -> fc_out) |
                                                                      |
                                           [GRL] -> adv_fc -> adv_logits
                                                |
                                                +-> clf_fc -> clf_logit
    """

    def __init__(
        self,
        n_features: int,
        n_confounders: int,
        config: TrainingConfig,
        *,
        rng: np.random.Generator,
    ) -> None:
        self.n_features = int(n_features)
        self.n_confounders = int(n_confounders)
        self.config = config
        self.rng = rng

        hidden = list(config.hidden_dims)
        latent = int(config.latent_dim)

        self.gate = dm.FeatureGate(
            n_features, sparsity_type=config.sparsity_type, init_log_alpha=2.0,
            rng=rng, name="gate",
        )
        self.fc0 = dm.Linear(n_features, hidden[0], rng=rng, name="fc0")
        self.relu0 = dm.ReLU(name="relu0")
        self.drop0 = dm.Dropout(p=config.dropout, rng=rng, name="drop0")
        self.fc1 = dm.Linear(hidden[0], hidden[1], rng=rng, name="fc1")
        self.relu1 = dm.ReLU(name="relu1")
        self.drop1 = dm.Dropout(p=config.dropout, rng=rng, name="drop1")
        self.fc_latent = dm.Linear(hidden[1], latent, rng=rng, name="fc_latent")
        self.l2norm = dm.L2Normalize(name="l2norm")

        # Classifier head (latent -> 1 logit)
        self.clf = dm.Linear(latent, 1, rng=rng, name="clf")

        # Adversary head (latent -> n_confounders logits, via GRL)
        self.adv = dm.Linear(latent, max(n_confounders, 2), rng=rng, name="adv")

        # Optional decoder (latent -> hidden[1] -> n_features)
        self.use_decoder = bool(config.use_decoder)
        if self.use_decoder:
            self.dec0 = dm.Linear(latent, hidden[1], rng=rng, name="dec0")
            self.relu_dec = dm.ReLU(name="relu_dec")
            self.dec1 = dm.Linear(hidden[1], n_features, rng=rng, name="dec1")

        # Consolidated parameter / grad dicts (stable keys)
        self._register_params()

    def _register_params(self) -> None:
        modules: List[Tuple[str, dm._Module]] = [
            ("gate", self.gate),
            ("fc0", self.fc0),
            ("fc1", self.fc1),
            ("fc_latent", self.fc_latent),
            ("clf", self.clf),
            ("adv", self.adv),
        ]
        if self.use_decoder:
            modules += [("dec0", self.dec0), ("dec1", self.dec1)]

        self._modules_by_name: "OrderedDict[str, dm._Module]" = OrderedDict(modules)
        self.params: Dict[str, np.ndarray] = {}
        self.grads: Dict[str, np.ndarray] = {}
        for mod_name, mod in self._modules_by_name.items():
            for pname, parr in mod.params.items():
                key = f"{mod_name}.{pname}"
                self.params[key] = parr
                self.grads[key] = mod.grads[pname]

    # ------------------------------------------------------------------
    # Forward passes
    # ------------------------------------------------------------------

    def _encode(self, x: np.ndarray, *, training: bool) -> Tuple[np.ndarray, np.ndarray]:
        gx = self.gate.forward(x, training=training)
        h0 = self.fc0.forward(gx)
        h0 = self.relu0.forward(h0)
        h0 = self.drop0.forward(h0, training=training)
        h1 = self.fc1.forward(h0)
        h1 = self.relu1.forward(h1)
        h1 = self.drop1.forward(h1, training=training)
        z = self.fc_latent.forward(h1)
        z = self.l2norm.forward(z, training=training)
        return gx, z

    def _decode(self, z: np.ndarray, *, training: bool) -> np.ndarray:
        h = self.dec0.forward(z)
        h = self.relu_dec.forward(h)
        x_rec = self.dec1.forward(h)
        return x_rec

    def forward(
        self, x: np.ndarray, *, training: bool = True
    ) -> Dict[str, np.ndarray]:
        gx, z = self._encode(x, training=training)
        clf_logits = self.clf.forward(z).ravel()
        adv_logits = self.adv.forward(z)
        out = {"gated_x": gx, "latent": z, "clf_logit": clf_logits, "adv_logits": adv_logits}
        if self.use_decoder:
            out["recon"] = self._decode(z, training=training)
        return out

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        logits = self.forward(x, training=False)["clf_logit"]
        return dm.sigmoid(logits)

    def encode(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x, training=False)["latent"]

    def eval_gate(self) -> np.ndarray:
        return self.gate.eval_gate()

    # ------------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------------

    def state_dict(self) -> Dict[str, np.ndarray]:
        return {k: v.copy() for k, v in self.params.items()}

    def load_state_dict(self, state: Mapping[str, np.ndarray]) -> None:
        for k, v in state.items():
            if k in self.params and self.params[k].shape == v.shape:
                self.params[k][...] = v


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def _compute_invariance_penalty(
    latent: np.ndarray,
    env_ids: np.ndarray,
) -> Tuple[float, np.ndarray]:
    """Return the env-invariance penalty and its gradient w.r.t. ``latent``.

    Penalty:  (1 / K) * sum_k || mean(latent | env=k) - mean(latent) ||^2.
    Only environments with valid (non-negative) ids contribute.
    """
    n, d = latent.shape
    if n == 0:
        return 0.0, np.zeros_like(latent)
    mask_valid = env_ids >= 0
    if mask_valid.sum() == 0:
        return 0.0, np.zeros_like(latent)

    z_valid = latent[mask_valid]
    ids = env_ids[mask_valid]
    uniq = np.unique(ids)
    K = int(uniq.size)
    if K < 2:
        return 0.0, np.zeros_like(latent)

    overall_mean = z_valid.mean(axis=0)
    grad = np.zeros_like(latent)
    loss = 0.0
    for k in uniq:
        mask_k = ids == k
        n_k = int(mask_k.sum())
        if n_k == 0:
            continue
        mu_k = z_valid[mask_k].mean(axis=0)
        diff = mu_k - overall_mean   # (d,)
        loss += float(diff @ diff)
        # d diff / d z_i = (1/n_k if i in k else 0) - (1/n_total if i valid else 0)
        n_valid_total = z_valid.shape[0]
        base_update = -(2.0 / n_valid_total) * diff
        # add to every valid sample
        grad_valid = np.broadcast_to(base_update, z_valid.shape).copy()
        # add (2/n_k * diff) to samples in class k
        grad_valid[mask_k] += (2.0 / n_k) * diff
        # accumulate back into the full grad at valid positions
        tmp = np.zeros_like(latent)
        tmp[mask_valid] = grad_valid
        grad += tmp

    loss = loss / K
    grad = grad / K
    return loss, grad


def _balanced_sample_weights(y: np.ndarray) -> np.ndarray:
    """Per-sample weights matching sklearn 'balanced' for binary labels."""
    y = np.asarray(y).ravel().astype(np.int64)
    n = y.size
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return np.ones(n, dtype=np.float64)
    w_pos = n / (2.0 * n_pos)
    w_neg = n / (2.0 * n_neg)
    return np.where(y == 1, w_pos, w_neg).astype(np.float64)


def _pick_inner_split(
    outer_fold: np.ndarray,
    inner_per_outer: Mapping[int, np.ndarray],
    fold_id: int,
) -> np.ndarray:
    """Return a boolean ``val_mask`` over all samples for inner_fold == 0."""
    arr = inner_per_outer.get(fold_id)
    if arr is None:
        return np.zeros_like(outer_fold, dtype=bool)
    # arr holds per-sample inner-fold strings ("" if sample in outer fold k);
    # pick arr == "0" as the validation split.
    return arr == "0"


def train_one_fold(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    conf_train: np.ndarray,
    env_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: TrainingConfig,
    n_confounders: int,
    logger_prefix: str = "",
    early_stopping: bool = True,
    restore_best: bool = True,
) -> Tuple[SparseInvariantModel, Dict[str, Any]]:
    """Train a single SparseInvariantModel and return the best-checkpoint copy.

    Uses inner-fold validation AUROC for early stopping; ties are broken by
    higher balanced accuracy, then lower validation loss. When ``restore_best``
    is false, the returned model is the final epoch state, which is useful for
    refitting a selected epoch count on all available outer-training samples.
    """
    rng = np.random.default_rng(config.seed)
    n_features = X_train.shape[1]

    model = SparseInvariantModel(
        n_features=n_features,
        n_confounders=n_confounders,
        config=config,
        rng=rng,
    )

    encoder_no_decay = {"gate.log_alpha", "fc0.b", "fc1.b", "fc_latent.b",
                       "clf.b", "adv.b"}
    if model.use_decoder:
        encoder_no_decay.update({"dec0.b", "dec1.b"})

    optimizer = dm.AdamW(
        model.params, lr=config.lr, weight_decay=config.weight_decay,
        no_decay_names=encoder_no_decay,
    )

    n_train = X_train.shape[0]
    sample_w = _balanced_sample_weights(y_train)
    best: Dict[str, Any] = {"auroc": -1.0, "bac": -1.0, "val_loss": math.inf,
                            "epoch": -1, "state": model.state_dict()}
    patience_left = config.patience
    history: List[Dict[str, float]] = []

    for epoch in range(1, config.n_epochs + 1):
        # Shuffle training indices
        perm = rng.permutation(n_train)
        losses_ep = {"cls": 0.0, "recon": 0.0, "adv": 0.0, "inv": 0.0,
                     "gate": 0.0, "total": 0.0, "n_batches": 0}
        adv_ramp = 1.0 if config.adv_ramp_epochs <= 0 else min(
            1.0, epoch / float(config.adv_ramp_epochs)
        )
        lam_adv_eff = config.lambda_adv * adv_ramp

        for start in range(0, n_train, config.batch_size):
            idx = perm[start:start + config.batch_size]
            if idx.size < 2:
                continue
            xb = X_train[idx]
            yb = y_train[idx].astype(np.float64)
            cb = conf_train[idx]
            eb = env_train[idx]
            wb = sample_w[idx]

            out = model.forward(xb, training=True)
            latent = out["latent"]
            clf_logit = out["clf_logit"]
            adv_logits = out["adv_logits"]

            # ----- losses -----
            # cls
            l_cls = dm.bce_with_logits(clf_logit, yb, sample_weight=wb)
            g_clf_logit = dm.bce_with_logits_grad(clf_logit, yb, sample_weight=wb)

            # adv (supervision: confounder class)
            # Only samples with valid confounder (>=0) contribute
            cb_valid = cb >= 0
            if cb_valid.any():
                l_adv = dm.cross_entropy_with_logits(adv_logits[cb_valid], cb[cb_valid])
                g_adv = np.zeros_like(adv_logits)
                g_adv[cb_valid] = dm.cross_entropy_with_logits_grad(
                    adv_logits[cb_valid], cb[cb_valid]
                )
            else:
                l_adv = 0.0
                g_adv = np.zeros_like(adv_logits)

            # recon
            if model.use_decoder:
                target = out["gated_x"]
                l_rec = dm.mse_loss(out["recon"], target)
                g_rec_out = dm.mse_loss_grad(out["recon"], target)
            else:
                l_rec = 0.0
                g_rec_out = None

            # invariance on latent
            l_inv, g_inv_latent = _compute_invariance_penalty(latent, eb)

            # sparsity
            l_gate = model.gate.sparsity_penalty()

            total = (
                l_cls
                + config.lambda_recon * l_rec
                + lam_adv_eff * l_adv
                + config.lambda_inv * l_inv
                + config.lambda_sparsity * l_gate
            )

            # ----- backward -----
            # reset grads
            for g in model.grads.values():
                g.fill(0.0)

            # classifier head: grad through clf -> latent
            g_z_from_clf = model.clf.backward(g_clf_logit.reshape(-1, 1))

            # adversary head:
            # - adversary's own gradients (train to MINIMIZE l_adv): use +g_adv
            # - encoder sees reversed gradient (-lambda_adv) w.r.t. latent
            g_z_from_adv_for_encoder = model.adv.backward(g_adv)
            # After adv.backward, model.adv.grads hold dL_adv/dW_adv and dL_adv/db_adv;
            # we want to scale them by +lam_adv_eff, and adversary's own portion is the normal step.
            # Scale adversary grads for adversary update (will be applied via optimizer step).
            # Scale the input-side gradient with -lam_adv_eff for encoder branch.
            model.adv.grads["W"] *= lam_adv_eff
            model.adv.grads["b"] *= lam_adv_eff
            g_z_from_adv_for_encoder = -lam_adv_eff * g_z_from_adv_for_encoder

            # invariance path: direct gradient into latent
            g_z_from_inv = config.lambda_inv * g_inv_latent

            # Optional decoder path
            g_gx_from_rec = None
            if model.use_decoder:
                # grad flows through decoder to latent, and through the gated_x target
                g_rec_out_scaled = config.lambda_recon * g_rec_out
                g_h_dec = model.dec1.backward(g_rec_out_scaled)
                g_h_dec = model.relu_dec.backward(g_h_dec)
                g_z_from_dec = model.dec0.backward(g_h_dec)
                # and the target of MSE is gated_x itself; d(L_rec)/d(gated_x) = -g_rec_out
                g_gx_from_rec = -config.lambda_recon * g_rec_out

            # Combine all latent-side gradients
            g_z = g_z_from_clf + g_z_from_adv_for_encoder + g_z_from_inv
            if model.use_decoder:
                g_z = g_z + g_z_from_dec

            # Encoder backward — pass through l2norm Jacobian before fc_latent
            g_h1 = model.fc_latent.backward(model.l2norm.backward(g_z))
            g_h1 = model.drop1.backward(g_h1)
            g_h1 = model.relu1.backward(g_h1)
            g_h0 = model.fc1.backward(g_h1)
            g_h0 = model.drop0.backward(g_h0)
            g_h0 = model.relu0.backward(g_h0)
            g_gx = model.fc0.backward(g_h0)

            # Add decoder's target-side gradient on gated_x
            if g_gx_from_rec is not None:
                g_gx = g_gx + g_gx_from_rec

            # Gate backward (accumulates grads['log_alpha'])
            _ = model.gate.backward(g_gx)

            # Add sparsity penalty gradient
            model.gate.grads["log_alpha"] += model.gate.sparsity_penalty_grad(
                weight=config.lambda_sparsity
            )

            # Global gradient clipping (L2 norm over all parameter grads)
            if config.grad_clip_norm > 0:
                total_sq = 0.0
                for g in model.grads.values():
                    total_sq += float((g * g).sum())
                total_norm = math.sqrt(total_sq) if total_sq > 0 else 0.0
                if total_norm > config.grad_clip_norm:
                    scale = config.grad_clip_norm / (total_norm + 1e-12)
                    for g in model.grads.values():
                        g *= scale

            # Optimizer step
            optimizer.step(model.params, model.grads)

            # accumulate losses for epoch summary
            losses_ep["cls"] += float(l_cls)
            losses_ep["recon"] += float(l_rec)
            losses_ep["adv"] += float(l_adv)
            losses_ep["inv"] += float(l_inv)
            losses_ep["gate"] += float(l_gate)
            losses_ep["total"] += float(total)
            losses_ep["n_batches"] += 1

        # --- Validation ---
        val_out = model.forward(X_val, training=False)
        val_logit = val_out["clf_logit"]
        val_probs = dm.sigmoid(val_logit)
        val_auc = mx.auroc(y_val, val_probs)
        val_bac = mx.balanced_accuracy(y_val, (val_probs >= 0.5).astype(np.int64))
        val_loss = dm.bce_with_logits(val_logit, y_val.astype(np.float64))

        nb = max(1, losses_ep["n_batches"])
        hist_row = {
            "epoch": int(epoch),
            "train_cls": losses_ep["cls"] / nb,
            "train_recon": losses_ep["recon"] / nb,
            "train_adv": losses_ep["adv"] / nb,
            "train_inv": losses_ep["inv"] / nb,
            "train_gate_penalty": losses_ep["gate"] / nb,
            "train_total": losses_ep["total"] / nb,
            "val_loss": float(val_loss),
            "val_auroc": float(val_auc),
            "val_bac": float(val_bac),
            "lambda_adv_effective": float(lam_adv_eff),
        }
        history.append(hist_row)

        improved = False
        if val_auc > best["auroc"] + 1e-6:
            improved = True
        elif abs(val_auc - best["auroc"]) <= 1e-6 and val_bac > best["bac"] + 1e-6:
            improved = True
        elif abs(val_auc - best["auroc"]) <= 1e-6 and val_loss < best["val_loss"]:
            improved = True

        if improved:
            best = {
                "auroc": float(val_auc),
                "bac": float(val_bac),
                "val_loss": float(val_loss),
                "epoch": int(epoch),
                "state": model.state_dict(),
            }
            patience_left = config.patience
        else:
            patience_left -= 1

        if epoch % 10 == 0 or epoch == 1 or improved:
            logger.info(
                "%s epoch %d/%d | cls=%.4f recon=%.4f adv=%.4f inv=%.4f gate=%.4f "
                "val_auc=%.4f val_bac=%.4f val_loss=%.4f %s",
                logger_prefix, epoch, config.n_epochs,
                hist_row["train_cls"], hist_row["train_recon"], hist_row["train_adv"],
                hist_row["train_inv"], hist_row["train_gate_penalty"],
                val_auc, val_bac, val_loss,
                "*" if improved else "",
            )

        if early_stopping and patience_left <= 0:
            logger.info("%s early stopping at epoch %d (best=%d)", logger_prefix, epoch, best["epoch"])
            break

    # Restore best weights unless the caller requested the final epoch state.
    if restore_best:
        model.load_state_dict(best["state"])

    # Post-training gate sparsity snapshot
    gate_vals = model.eval_gate()
    return model, {
        "history": history,
        "best_epoch": int(best["epoch"]),
        "best_val_auroc": float(best["auroc"]),
        "best_val_bac": float(best["bac"]),
        "best_val_loss": float(best["val_loss"]),
        "restore_best": bool(restore_best),
        "early_stopping": bool(early_stopping),
        "gate_mean": float(gate_vals.mean()),
        "gate_median": float(np.median(gate_vals)),
        "gate_frac_below_0_1": float((gate_vals < 0.1).mean()),
        "gate_frac_below_0_01": float((gate_vals < 0.01).mean()),
    }


# ---------------------------------------------------------------------------
# Confounder / environment encoding
# ---------------------------------------------------------------------------


def encode_confounder_column(
    master_rows: Sequence[Mapping[str, str]],
    column: str = "env_histology",
) -> Tuple[np.ndarray, int, List[str]]:
    """Turn a master-table column into integer labels (missing -> -1).

    Uses ``env_*`` binary strata if available; otherwise accepts plain
    categorical columns and assigns codes in alphabetical order of
    non-empty values. Returns ``(labels, n_classes, levels)``.
    """
    vals = [str(r.get(column, "")) for r in master_rows]
    uniq = sorted(set(v for v in vals if v != ""))
    code = {v: i for i, v in enumerate(uniq)}
    labels = np.array([code.get(v, -1) for v in vals], dtype=np.int64)
    return labels, len(uniq), uniq


def pick_env_column(master_rows: Sequence[Mapping[str, str]], candidates: Sequence[str]) -> str:
    """Return the first env column that has >=1 non-empty value in data."""
    for c in candidates:
        col = f"env_{c}" if not c.startswith("env_") else c
        if master_rows and col in master_rows[0]:
            if any(r.get(col, "") != "" for r in master_rows):
                return col
    # Fallback: any env_ column with variability
    for col in master_rows[0] if master_rows else []:
        if col.startswith("env_") and any(r.get(col, "") != "" for r in master_rows):
            return col
    return ""


# ---------------------------------------------------------------------------
# Top-level CV orchestration
# ---------------------------------------------------------------------------


def _extract_inner_folds(
    fold_rows: Sequence[Mapping[str, str]],
    n_outer: int,
) -> Dict[int, np.ndarray]:
    """Return ``{outer_fold_id: np.array([inner_fold_str, ...])}`` aligned to
    ``fold_rows`` order (empty string for samples in that outer test fold).
    """
    out: Dict[int, np.ndarray] = {}
    for k in range(n_outer):
        col = f"inner_fold_outer{k}"
        if not fold_rows or col not in fold_rows[0]:
            out[k] = np.array([""] * len(fold_rows), dtype=object)
            continue
        out[k] = np.array([r.get(col, "") for r in fold_rows], dtype=object)
    return out


def run_step4_cv(
    *,
    X: np.ndarray,
    y: np.ndarray,
    outer_fold: np.ndarray,
    groups: np.ndarray,
    master_rows: Sequence[Mapping[str, str]],
    fold_rows: Sequence[Mapping[str, str]],
    gene_names: Sequence[str],
    sample_barcodes: Sequence[str],
    config: TrainingConfig,
    confounder_column: str,
    environment_column: str,
    checkpoint_dir: Path,
) -> Dict[str, Any]:
    """Run patient-grouped outer-CV training of SparseInvariantModel.

    Returns
    -------
    dict with keys:
        oof_probs           (n,) OOF tumor probabilities
        oof_latents         (n, latent_dim) OOF latent vectors
        gate_weights_per_fold  (K, P) learned gate per fold
        per_fold_metrics    list of per-fold metric dicts
        training_history    per-fold per-epoch logs
        fold_info           per-fold best-epoch / gate stats
        folds_used          sorted list of fold ids
    """
    # Align and validate
    step3.assert_no_patient_leakage(groups, outer_fold)
    n_samples, n_features = X.shape
    fold_ids = sorted(set(int(f) for f in outer_fold))
    inner_folds_per_outer = _extract_inner_folds(fold_rows, max(fold_ids) + 1)

    # Confounder and env labels aligned to sample order
    conf_labels, n_conf_classes, conf_levels = encode_confounder_column(master_rows, confounder_column)
    if n_conf_classes < 2:
        logger.warning("Confounder column %s has <2 classes; adversarial loss disabled.",
                       confounder_column)
        config.lambda_adv = 0.0
        n_conf_classes = 2
    env_labels, n_env_classes, env_levels = encode_confounder_column(master_rows, environment_column)
    if n_env_classes < 2:
        logger.warning("Environment column %s has <2 classes; invariance loss disabled.",
                       environment_column)
        config.lambda_inv = 0.0

    oof_probs = np.full(n_samples, np.nan, dtype=np.float64)
    oof_latents = np.full((n_samples, config.latent_dim), np.nan, dtype=np.float64)
    gate_fold = np.zeros((len(fold_ids), n_features), dtype=np.float64)
    per_fold_metrics: List[Dict[str, Any]] = []
    training_history: List[Dict[str, Any]] = []
    fold_info: List[Dict[str, Any]] = []
    train_prediction_rows: List[Dict[str, Any]] = []

    for k_idx, fold_id in enumerate(fold_ids):
        test_mask = outer_fold == fold_id
        train_mask = ~test_mask

        val_inner_mask = _pick_inner_split(outer_fold, inner_folds_per_outer, fold_id)
        # Ensure val_inner_mask is a subset of training (non-test) samples.
        val_mask = val_inner_mask & train_mask
        tr_core_mask = train_mask & ~val_mask

        # If the inner split produced no validation samples, fall back to 20% random.
        if val_mask.sum() < 2:
            rng_fb = np.random.default_rng(config.seed + fold_id * 17)
            train_ids = np.where(train_mask)[0]
            val_size = max(2, int(0.2 * train_ids.size))
            val_pick = rng_fb.choice(train_ids, size=val_size, replace=False)
            val_mask = np.zeros_like(train_mask)
            val_mask[val_pick] = True
            tr_core_mask = train_mask & ~val_mask

        X_tr = X[tr_core_mask]
        y_tr = y[tr_core_mask]
        conf_tr = conf_labels[tr_core_mask]
        env_tr = env_labels[tr_core_mask]

        X_full_train = X[train_mask]
        y_full_train = y[train_mask]
        conf_full_train = conf_labels[train_mask]
        env_full_train = env_labels[train_mask]

        X_val = X[val_mask]
        y_val = y[val_mask]

        X_te = X[test_mask]
        y_te = y[test_mask]

        # Per-fold standardization (plane.md: per-fold z-score on top of global)
        mean_fold, std_fold = dm.standardize_fit(X_tr)
        X_tr_z = dm.standardize_apply(X_tr, mean_fold, std_fold)
        X_val_z = dm.standardize_apply(X_val, mean_fold, std_fold)
        X_te_z = dm.standardize_apply(X_te, mean_fold, std_fold)

        fold_config = TrainingConfig(**{**config.as_dict(), "seed": config.seed + 1000 * fold_id})
        logger.info(
            "Outer fold %d/%d (id=%d): train=%d  val=%d  test=%d | pos train=%d val=%d test=%d",
            k_idx + 1, len(fold_ids), fold_id,
            int(tr_core_mask.sum()), int(val_mask.sum()), int(test_mask.sum()),
            int((y_tr == 1).sum()), int((y_val == 1).sum()), int((y_te == 1).sum()),
        )

        model, info = train_one_fold(
            X_train=X_tr_z, y_train=y_tr, conf_train=conf_tr, env_train=env_tr,
            X_val=X_val_z, y_val=y_val, config=fold_config,
            n_confounders=n_conf_classes,
            logger_prefix=f"[fold {fold_id}]",
        )

        # Refit the selected architecture/epoch budget on the full outer
        # training partition. This gives the final OOF model the same training
        # sample count as the baselines while keeping the test fold untouched.
        refit_epochs = max(1, int(info["best_epoch"]))
        mean_full, std_full = dm.standardize_fit(X_full_train)
        X_full_train_z = dm.standardize_apply(X_full_train, mean_full, std_full)
        X_val_full_z = dm.standardize_apply(X_val, mean_full, std_full)
        X_te_z = dm.standardize_apply(X_te, mean_full, std_full)
        refit_config = TrainingConfig(**{
            **fold_config.as_dict(),
            "n_epochs": refit_epochs,
            "patience": max(refit_epochs + 1, fold_config.patience),
        })
        logger.info(
            "[fold %d] refitting final model for %d epoch(s) on full outer train=%d (pos=%d, neg=%d)",
            fold_id, refit_epochs, int(train_mask.sum()),
            int((y_full_train == 1).sum()), int((y_full_train == 0).sum()),
        )
        model, refit_info = train_one_fold(
            X_train=X_full_train_z,
            y_train=y_full_train,
            conf_train=conf_full_train,
            env_train=env_full_train,
            X_val=X_val_full_z,
            y_val=y_val,
            config=refit_config,
            n_confounders=n_conf_classes,
            logger_prefix=f"[fold {fold_id} refit]",
            early_stopping=False,
            restore_best=False,
        )
        info.update({
            "refit_epochs": int(refit_epochs),
            "refit_train_n": int(train_mask.sum()),
            "refit_train_pos": int((y_full_train == 1).sum()),
            "refit_train_neg": int((y_full_train == 0).sum()),
            "refit_gate_mean": float(refit_info["gate_mean"]),
            "refit_gate_frac_below_0_1": float(refit_info["gate_frac_below_0_1"]),
        })
        info["fold_id"] = int(fold_id)
        fold_info.append(info)

        # Predict on test samples (OOF)
        out_te = model.forward(X_te_z, training=False)
        probs_te = dm.sigmoid(out_te["clf_logit"])
        oof_probs[test_mask] = probs_te
        oof_latents[test_mask] = out_te["latent"]

        train_ids = np.where(train_mask)[0]
        train_probs = model.predict_proba(X_full_train_z)
        for sample_idx, y_i, p_i in zip(train_ids, y_full_train, train_probs):
            train_prediction_rows.append({
                "outer_fold": int(fold_id),
                "sample_index": int(sample_idx),
                "y_true": int(y_i),
                "deep_train_prob": float(p_i),
            })

        # Gate snapshot
        gate_fold[k_idx] = model.eval_gate()

        # Per-fold metrics
        y_pred_te = (probs_te >= 0.5).astype(np.int64)
        sens, spec = mx.sensitivity_specificity(y_te, y_pred_te)
        pf_row = {
            "fold_id": int(fold_id),
            "n_train": int(train_mask.sum()),
            "n_train_selection": int(tr_core_mask.sum()),
            "n_val": int(val_mask.sum()),
            "n_test": int(test_mask.sum()),
            "n_pos_train": int((y_full_train == 1).sum()),
            "n_pos_train_selection": int((y_tr == 1).sum()),
            "n_pos_val": int((y_val == 1).sum()),
            "n_pos_test": int((y_te == 1).sum()),
            "best_epoch": info["best_epoch"],
            "refit_epochs": int(refit_epochs),
            "auroc": mx.auroc(y_te, probs_te),
            "auprc": mx.auprc(y_te, probs_te),
            "balanced_accuracy": mx.balanced_accuracy(y_te, y_pred_te),
            "f1": mx.f1_score(y_te, y_pred_te),
            "brier": mx.brier_score(y_te, probs_te),
            "log_loss": mx.log_loss(y_te, probs_te),
            "sensitivity": sens,
            "specificity": spec,
            "gate_mean": info["gate_mean"],
            "gate_frac_below_0_1": info["gate_frac_below_0_1"],
        }
        per_fold_metrics.append(pf_row)

        # Per-fold training history
        for row in info["history"]:
            training_history.append({"fold_id": int(fold_id), **row})

        # Save checkpoint
        ckpt_path = checkpoint_dir / f"fold_{fold_id}.npz"
        state = model.state_dict()
        state["_meta_mean_fold"] = mean_full
        state["_meta_std_fold"] = std_full
        state["_meta_fold_id"] = np.array([fold_id], dtype=np.int64)
        dm.save_npz_bundle(ckpt_path, state)
        logger.info(
            "  fold %d OOF | AUROC=%.4f AUPRC=%.4f BAC=%.4f Brier=%.4f gate<0.1 frac=%.3f",
            fold_id, pf_row["auroc"], pf_row["auprc"], pf_row["balanced_accuracy"],
            pf_row["brier"], info["gate_frac_below_0_1"],
        )

    return {
        "oof_probs": oof_probs,
        "train_predictions": train_prediction_rows,
        "oof_latents": oof_latents,
        "gate_weights_per_fold": gate_fold,
        "per_fold_metrics": per_fold_metrics,
        "training_history": training_history,
        "fold_info": fold_info,
        "folds_used": fold_ids,
        "confounder_levels": conf_levels,
        "environment_levels": env_levels,
        "confounder_column": confounder_column,
        "environment_column": environment_column,
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _has_matplotlib() -> bool:
    try:
        import matplotlib  # noqa: F401
        return True
    except ImportError:
        return False


def _pca_2d(X: np.ndarray) -> np.ndarray:
    """Simple 2-D PCA via SVD on mean-centered data."""
    Xc = X - X.mean(axis=0, keepdims=True)
    # Use economy SVD on (n, d); U has shape (n, k)
    U, S, _ = np.linalg.svd(Xc, full_matrices=False)
    k = min(2, S.size)
    out = np.zeros((Xc.shape[0], 2), dtype=np.float64)
    out[:, :k] = U[:, :k] * S[:k]
    return out


def generate_step4_figures(
    *,
    y: np.ndarray,
    oof_probs: np.ndarray,
    train_predictions: Optional[Sequence[Mapping[str, Any]]] = None,
    oof_latents: np.ndarray,
    gate_weights_per_fold: np.ndarray,
    master_rows: Sequence[Mapping[str, str]],
    output_dir: Path,
    style: Any = None,
    formats: Sequence[str] = ("svg",),
    training_history: Optional[Sequence[Mapping[str, Any]]] = None,
    per_fold_metrics: Optional[Sequence[Mapping[str, Any]]] = None,
    gene_names: Optional[Sequence[str]] = None,
    oof_fold_ids: Optional[np.ndarray] = None,
    subgroup_rows: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Tuple[List[str], List[Tuple[str, str]]]:
    """Render Phase-II figures if matplotlib is available."""
    generated: List[str] = []
    skipped: List[Tuple[str, str]] = []
    fig_names = [
        "fig_latent_pca_label",
        "fig_latent_pca_histology",
        "fig_gate_distribution",
        "fig_deep_roc_pr",
        "fig_F3_deep_calibration",
    ]
    if training_history is not None:
        fig_names.append("fig_F1_training_curves")
    if per_fold_metrics is not None:
        fig_names.append("fig_F2_per_fold_metrics")
    if gene_names is not None:
        fig_names.append("fig_F5_gate_top_n")
    if oof_fold_ids is not None:
        fig_names.append("fig_F6_latent_pca_fold")
    if subgroup_rows is not None:
        fig_names.append("fig_F4_subgroup_heatmap")
    if not _has_matplotlib():
        for n in fig_names:
            skipped.append((n, "matplotlib not installed"))
        return generated, skipped

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from .publication_style import (
            apply_style, save_figure, semantic_color, categorical_colors,
        )
        if style is not None:
            apply_style(style)

        fig_dir = Path(output_dir) / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        valid = ~np.isnan(oof_probs)
        y_valid = y[valid]
        probs_valid = oof_probs[valid]
        Z = oof_latents[valid]
        Z2 = _pca_2d(Z)

        # --- Latent PCA by label ---
        try:
            fig, ax = plt.subplots(figsize=(3.6, 3.0))
            for cls, name in ((0, "Normal"), (1, "Tumor")):
                m = (y_valid == cls)
                ax.scatter(Z2[m, 0], Z2[m, 1],
                           s=14, alpha=0.85, edgecolor="black", linewidth=0.3,
                           color=semantic_color("normal" if cls == 0 else "tumor"),
                           label=f"{name} (n={int(m.sum())})")
            ax.set_xlabel("PC1 (latent)")
            ax.set_ylabel("PC2 (latent)")
            ax.set_title("Latent PCA - OOF")
            ax.legend(loc="best", fontsize=8)
            paths = save_figure(fig, fig_dir / "fig_latent_pca_label", style=style, formats=formats)
            if paths:
                generated.append("fig_latent_pca_label")
        except Exception as exc:
            skipped.append(("fig_latent_pca_label", str(exc)))

        # --- Latent PCA by histology (if available) ---
        try:
            hist = [r.get("histology", "") for r in master_rows]
            hist_valid = [h for h, v in zip(hist, valid) if v]
            uniq_h = sorted(set(h for h in hist_valid if h and h != "NA"))
            if len(uniq_h) >= 2:
                palette = categorical_colors(len(uniq_h))
                fig, ax = plt.subplots(figsize=(3.6, 3.0))
                for i, hh in enumerate(uniq_h):
                    m = np.array([h == hh for h in hist_valid])
                    ax.scatter(Z2[m, 0], Z2[m, 1], s=14, alpha=0.85,
                               edgecolor="black", linewidth=0.3,
                               color=palette[i], label=f"{hh} (n={int(m.sum())})")
                ax.set_xlabel("PC1 (latent)")
                ax.set_ylabel("PC2 (latent)")
                ax.set_title("Latent PCA colored by histology")
                ax.legend(loc="best", fontsize=8)
                paths = save_figure(fig, fig_dir / "fig_latent_pca_histology",
                                    style=style, formats=formats)
                if paths:
                    generated.append("fig_latent_pca_histology")
            else:
                skipped.append(("fig_latent_pca_histology", "histology column has <2 classes"))
        except Exception as exc:
            skipped.append(("fig_latent_pca_histology", str(exc)))

        # --- Gate distribution ---
        try:
            mean_gate = gate_weights_per_fold.mean(axis=0)
            fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.8))
            axes[0].hist(mean_gate, bins=40, color=semantic_color("highlight"),
                         edgecolor="black", linewidth=0.3)
            axes[0].set_xlabel("Mean gate weight (across folds)")
            axes[0].set_ylabel("Number of genes")
            axes[0].set_title("Gate weight distribution")
            # CV stability: std across folds vs mean
            fold_mean = gate_weights_per_fold.mean(axis=0)
            fold_std = gate_weights_per_fold.std(axis=0)
            axes[1].scatter(fold_mean, fold_std, s=4, alpha=0.4,
                            color=semantic_color("enriched"))
            axes[1].set_xlabel("Mean gate weight")
            axes[1].set_ylabel("Std across folds")
            axes[1].set_title("Cross-fold gate stability")
            fig.tight_layout()
            paths = save_figure(fig, fig_dir / "fig_gate_distribution",
                                style=style, formats=formats)
            if paths:
                generated.append("fig_gate_distribution")
        except Exception as exc:
            skipped.append(("fig_gate_distribution", str(exc)))

        # --- ROC/PR ---
        try:
            fpr, tpr, _ = mx.roc_curve(y_valid, probs_valid)
            prec, rec, _ = mx.precision_recall_curve(y_valid, probs_valid)
            fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.0))
            auc = mx.auroc(y_valid, probs_valid)
            ap = mx.auprc(y_valid, probs_valid)
            axes[0].plot(fpr, tpr, color=semantic_color("tumor"), linewidth=1.8,
                         label=f"Deep (AUROC={auc:.3f})")
            axes[0].plot([0, 1], [0, 1], "--", color="#999999", linewidth=1.0)
            axes[0].set_xlabel("False Positive Rate")
            axes[0].set_ylabel("True Positive Rate")
            axes[0].set_title("Deep OOF ROC")
            axes[0].legend(loc="lower right", fontsize=8)
            axes[1].plot(rec, prec, color=semantic_color("tumor"), linewidth=1.8,
                         label=f"Deep (AP={ap:.3f})")
            axes[1].set_xlabel("Recall")
            axes[1].set_ylabel("Precision")
            axes[1].set_title("Deep OOF PR")
            axes[1].legend(loc="lower left", fontsize=8)
            fig.tight_layout()
            paths = save_figure(fig, fig_dir / "fig_deep_roc_pr",
                                style=style, formats=formats)
            if paths:
                generated.append("fig_deep_roc_pr")
        except Exception as exc:
            skipped.append(("fig_deep_roc_pr", str(exc)))

        # ---- Figure F1: training curves (val AUROC + loss components) ----
        if training_history is not None:
            try:
                hist_rows = list(training_history)
                fold_ids_h = sorted(set(str(r.get("fold_id", r.get("outer_fold", 0)))
                                        for r in hist_rows))
                fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

                ax = axes[0]
                pal_h = categorical_colors(max(len(fold_ids_h), 2))
                max_ep = 0
                by_fold: dict = {}
                for r in hist_rows:
                    fid = str(r.get("fold_id", r.get("outer_fold", 0)))
                    ep  = int(r.get("epoch", 0))
                    aur = r.get("val_auroc")
                    if aur not in (None, "", "nan"):
                        by_fold.setdefault(fid, []).append((ep, float(aur)))
                        max_ep = max(max_ep, ep)
                for fi, fid in enumerate(fold_ids_h):
                    pts = sorted(by_fold.get(fid, []))
                    if pts:
                        ax.plot([p[0] for p in pts], [p[1] for p in pts],
                                alpha=0.6, linewidth=1.2, color=pal_h[fi],
                                label=f"Fold {fid}")
                if max_ep > 0:
                    epoch_range = range(1, max_ep + 1)
                    means_ep = []
                    for ep in epoch_range:
                        vals_ep = [float(r["val_auroc"]) for r in hist_rows
                                   if int(r.get("epoch", 0)) == ep
                                   and r.get("val_auroc") not in (None, "", "nan")]
                        means_ep.append(float(np.mean(vals_ep)) if vals_ep else np.nan)
                    ax.plot(list(epoch_range), means_ep, color="black", linewidth=2,
                            linestyle="--", label="Mean")
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Validation AUROC")
                ax.set_title("Validation AUROC per Epoch")
                ax.legend(fontsize=7, ncol=2)

                ax = axes[1]
                first_fold = fold_ids_h[0] if fold_ids_h else "0"
                fold0_rows = sorted(
                    [r for r in hist_rows
                     if str(r.get("fold_id", r.get("outer_fold", 0))) == first_fold],
                    key=lambda r: int(r.get("epoch", 0)),
                )
                eps_f0 = [int(r.get("epoch", 0)) for r in fold0_rows]
                loss_defs = [
                    ("train_cls",          "Classification",  semantic_color("tumor")),
                    ("train_adv",          "Adversarial",     semantic_color("normal")),
                    ("train_inv",          "Invariance",      semantic_color("enriched")),
                    ("train_gate_penalty", "Gate penalty",    semantic_color("highlight")),
                ]
                for col, lbl, col_color in loss_defs:
                    vals_l = [r.get(col) for r in fold0_rows]
                    vals_l_f = [float(v) for v in vals_l if v not in (None, "", "nan")]
                    if len(vals_l_f) == len(eps_f0):
                        ax.plot(eps_f0, vals_l_f, label=lbl, color=col_color, linewidth=1.4)
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Loss")
                ax.set_title(f"Training Loss Components (Fold {first_fold})")
                ax.legend(fontsize=7)

                fig.suptitle("Deep Model Training Dynamics", fontsize=11, fontweight="bold")
                fig.tight_layout()
                paths = save_figure(fig, fig_dir / "fig_F1_training_curves",
                                    style=style, formats=formats)
                if paths:
                    generated.append("fig_F1_training_curves")
            except Exception as exc:
                skipped.append(("fig_F1_training_curves", str(exc)))

        # ---- Figure F2: per-fold metrics strip ----
        if per_fold_metrics is not None:
            try:
                pfm = list(per_fold_metrics)
                metric_defs = [
                    ("auroc", "AUROC"), ("balanced_accuracy", "BAC"),
                    ("f1", "F1"), ("sensitivity", "Sensitivity"), ("specificity", "Specificity"),
                ]
                n_m = len(metric_defs)
                fig, axes = plt.subplots(1, n_m, figsize=(n_m * 2.2, 5))
                rng_f2 = np.random.default_rng(42)
                for ax, (col, lbl) in zip(axes, metric_defs):
                    vals_m = np.asarray(
                        [float(r[col]) for r in pfm if r.get(col) not in (None, "", "nan")],
                        dtype=np.float64,
                    )
                    vals_m = vals_m[np.isfinite(vals_m)]
                    if vals_m.size == 0:
                        ax.set_visible(False)
                        continue
                    jitter = (rng_f2.random(len(vals_m)) - 0.5) * 0.3
                    ax.scatter(jitter, vals_m, color=semantic_color("highlight"),
                               alpha=0.85, s=50, zorder=3)
                    med_m = float(np.median(vals_m))
                    ax.plot([-0.3, 0.3], [med_m, med_m],
                            color=semantic_color("highlight"), linewidth=2.5, zorder=4)
                    ax.text(0, med_m + 0.01, f"{med_m:.3f}", ha="center",
                            fontsize=8, fontweight="bold")
                    ax.set_xlim(-0.6, 0.6)
                    ax.set_xticks([])
                    ax.set_ylim(max(0.0, float(vals_m.min()) - 0.1),
                                min(1.12, float(vals_m.max()) + 0.12))
                    ax.set_title(lbl, fontsize=9, fontweight="bold")
                    ax.axhline(0.5, color="#999999", linewidth=0.7, linestyle="--", alpha=0.5)
                fig.suptitle("Deep Model Per-Fold Performance", fontsize=11, fontweight="bold")
                fig.tight_layout()
                paths = save_figure(fig, fig_dir / "fig_F2_per_fold_metrics",
                                    style=style, formats=formats)
                if paths:
                    generated.append("fig_F2_per_fold_metrics")
            except Exception as exc:
                skipped.append(("fig_F2_per_fold_metrics", str(exc)))

        # ---- Figure F3: deep model calibration curve ----
        try:
            bins_f3 = mx.calibration_curve(y_valid, probs_valid, n_bins=8)
            if bins_f3:
                fig, ax = plt.subplots(figsize=(3.5, 3.2))
                ax.plot([0, 1], [0, 1], linestyle="--", color="#999999", linewidth=1.0,
                        label="Perfect calibration")
                xs_b = [b["mean_predicted_prob"] for b in bins_f3]
                ys_b = [b["fraction_positives"] for b in bins_f3]
                ax.plot(xs_b, ys_b, marker="o", color=semantic_color("tumor"),
                        linewidth=1.4, markersize=4, label="Deep model")
                ax.set_xlabel("Mean predicted probability")
                ax.set_ylabel("Fraction of positives")
                ax.set_xlim(0.0, 1.02)
                ax.set_ylim(0.0, 1.02)
                ax.set_title("Deep Model OOF Calibration")
                ax.legend(loc="lower right", fontsize=8)
                fig.tight_layout()
                paths = save_figure(fig, fig_dir / "fig_F3_deep_calibration",
                                    style=style, formats=formats)
                if paths:
                    generated.append("fig_F3_deep_calibration")
        except Exception as exc:
            skipped.append(("fig_F3_deep_calibration", str(exc)))

        # ---- Figure F4: deep model subgroup AUROC heatmap ----
        if subgroup_rows is not None:
            try:
                subg_rows_f4 = list(subgroup_rows)
                model_names_f4 = sorted(set(str(r.get("model", "deep")) for r in subg_rows_f4))
                env_combos_f4: dict = {}
                for r in subg_rows_f4:
                    lbl = f"{r.get('environment', '')}={r.get('level', '')}"
                    env_combos_f4.setdefault(lbl, {})
                    try:
                        env_combos_f4[lbl][str(r.get("model", "deep"))] = float(r["auroc"])
                    except (TypeError, ValueError, KeyError):
                        pass
                combo_labels_f4 = sorted(env_combos_f4)
                if combo_labels_f4:
                    mat_f4 = np.full((len(model_names_f4), len(combo_labels_f4)), np.nan)
                    for mi, model in enumerate(model_names_f4):
                        for ci, lbl in enumerate(combo_labels_f4):
                            v = env_combos_f4.get(lbl, {}).get(model)
                            if v is not None:
                                mat_f4[mi, ci] = v
                    fig, ax = plt.subplots(
                        figsize=(max(7, len(combo_labels_f4) * 0.7 + 2),
                                 max(2.5, len(model_names_f4) * 0.8 + 2)),
                    )
                    im = ax.imshow(mat_f4, aspect="auto", cmap="cage_sequential", vmin=0.4, vmax=1.0)
                    fig.colorbar(im, ax=ax, label="AUROC")
                    ax.set_xticks(range(len(combo_labels_f4)))
                    ax.set_xticklabels(combo_labels_f4, rotation=40, ha="right", fontsize=7)
                    ax.set_yticks(range(len(model_names_f4)))
                    ax.set_yticklabels([m.capitalize() for m in model_names_f4], fontsize=9)
                    for i in range(len(model_names_f4)):
                        for j in range(len(combo_labels_f4)):
                            v = mat_f4[i, j]
                            if np.isfinite(v):
                                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                                        fontsize=6.5,
                                        color="white" if v >= 0.80 else "black")
                    ax.set_title("Subgroup AUROC — Deep Model", fontsize=11, fontweight="bold")
                    fig.tight_layout()
                    paths = save_figure(fig, fig_dir / "fig_F4_subgroup_heatmap",
                                        style=style, formats=formats)
                    if paths:
                        generated.append("fig_F4_subgroup_heatmap")
            except Exception as exc:
                skipped.append(("fig_F4_subgroup_heatmap", str(exc)))

        # ---- Figure F5: gate weights top-N bars (suppressed + activated) ----
        if gene_names is not None:
            try:
                gene_list_f5 = list(gene_names)
                mean_gate_f5 = gate_weights_per_fold.mean(axis=0)
                top_n_gate = 25
                sorted_idx = np.argsort(mean_gate_f5)
                sup_idx  = sorted_idx[:top_n_gate]
                act_idx  = sorted_idx[-top_n_gate:][::-1]

                fig, axes = plt.subplots(1, 2,
                                         figsize=(10, max(5, top_n_gate * 0.28 + 2)))
                for ax, idx, title, col in [
                    (axes[0], sup_idx, f"Top {top_n_gate} Suppressed\n(Lowest gate)",
                     semantic_color("normal")),
                    (axes[1], act_idx, f"Top {top_n_gate} Activated\n(Highest gate)",
                     semantic_color("tumor")),
                ]:
                    genes_g = [gene_list_f5[i] for i in idx]
                    vals_g  = [float(mean_gate_f5[i]) for i in idx]
                    y_pos_g = list(range(len(genes_g)))
                    ax.barh(y_pos_g, vals_g, color=col, edgecolor="white", height=0.7)
                    ax.set_yticks(y_pos_g)
                    ax.set_yticklabels(genes_g, fontsize=6)
                    ax.set_xlabel("Mean gate weight", fontsize=8)
                    ax.set_title(title, fontsize=9, fontweight="bold")
                fig.suptitle("Sparse Gate Weights — Deep Model Interpretability",
                             fontsize=11, fontweight="bold")
                fig.tight_layout()
                paths = save_figure(fig, fig_dir / "fig_F5_gate_top_n",
                                    style=style, formats=formats)
                if paths:
                    generated.append("fig_F5_gate_top_n")
            except Exception as exc:
                skipped.append(("fig_F5_gate_top_n", str(exc)))

        # ---- Figure F6: latent PCA coloured by outer fold ----
        if oof_fold_ids is not None:
            try:
                folds_valid = oof_fold_ids[valid]
                unique_folds_f6 = sorted(set(folds_valid.tolist()))
                palette_f6 = categorical_colors(max(len(unique_folds_f6), 2))
                fig, ax = plt.subplots(figsize=(4.2, 3.6))
                for fi, fold in enumerate(unique_folds_f6):
                    m = (folds_valid == fold)
                    ax.scatter(Z2[m, 0], Z2[m, 1], s=14, alpha=0.8,
                               edgecolor="black", linewidth=0.3,
                               color=palette_f6[fi],
                               label=f"Fold {fold} (n={int(m.sum())})")
                ax.set_xlabel("PC1 (latent)")
                ax.set_ylabel("PC2 (latent)")
                ax.set_title("Latent PCA coloured by outer fold")
                ax.legend(loc="best", fontsize=7, ncol=2)
                fig.tight_layout()
                paths = save_figure(fig, fig_dir / "fig_F6_latent_pca_fold",
                                    style=style, formats=formats)
                if paths:
                    generated.append("fig_F6_latent_pca_fold")
            except Exception as exc:
                skipped.append(("fig_F6_latent_pca_fold", str(exc)))

    except Exception as exc:  # pragma: no cover - environment-specific
        for n in fig_names:
            skipped.append((n, str(exc)))

    return generated, skipped
