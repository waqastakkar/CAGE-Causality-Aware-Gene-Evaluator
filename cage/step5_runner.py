"""CAGE Step 5 runner: Causality-Aware Priority Score (CDPS).

Produces the five component scores (attribution, gate, stability,
invariance, perturbation) consumed by the Phase-III CDPS weighting and
the final ranked gene tables. The runner is deliberately pure-numpy so
it reuses the Step 4 model primitives without any deep-learning
framework dependency.

Pipeline per fold ``k``:

    1. Load the fold's SparseInvariantModel from ``checkpoints/fold_k.npz``
       and restore the per-fold standardization (mean, std).
    2. Pick up to ``cap-per-class`` tumor + normal *test* samples for that
       fold (the out-of-fold samples the model never saw during training).
    3. Compute the chosen attribution (Integrated Gradients, Grad x Input,
       or the raw gate weight). Aggregate :math:`|attribution|` per gene,
       overall and per environment stratum.
    4. Collect a genome-wide gradient-based perturbation proxy using each
       fold's test samples and the per-fold tumor / normal centroids.
    5. Aggregate across folds into the four per-gene component arrays.

The gate component is read directly from ``gate_weights.csv`` emitted by
Step 4.
"""

from __future__ import annotations

import csv
import json
import logging
import math
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from . import deep_model_utils as dm
from . import metrics as mx
from . import preprocess_esca as pp
from . import step3_runner as step3
from .step4_runner import SparseInvariantModel, TrainingConfig

logger = logging.getLogger("cage.step5.runner")

__all__ = [
    "FoldArtifacts",
    "load_step4_bundle",
    "load_fold_models",
    "input_gradient",
    "integrated_gradients",
    "compute_attribution_per_fold",
    "compute_stability_scores",
    "compute_invariance_scores",
    "compute_perturbation_scores",
    "compute_gate_component",
    "normalize_scores",
    "build_cdps_ranking",
    "generate_step5_figures",
    "run_step5_cdps",
]


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class FoldArtifacts:
    """Restored deep-model state for a single CV fold."""
    fold_id: int
    model: SparseInvariantModel
    mean_fold: np.ndarray
    std_fold: np.ndarray
    test_mask: np.ndarray   # (n_samples,) bool - outer fold == fold_id


# ---------------------------------------------------------------------------
# Step-4 bundle loader
# ---------------------------------------------------------------------------


def load_step4_bundle(step4_dir: Path) -> Dict[str, Any]:
    """Load deep_oof_predictions, gate_weights, latent_embeddings, and config.

    Returns a dict with the parsed CSV records and the training config from
    ``phase2_summary.json`` so the SparseInvariantModel can be reconstructed.
    """
    step4_dir = Path(step4_dir)
    required = [
        "deep_oof_predictions.csv",
        "gate_weights.csv",
        "deep_summary_metrics.csv",
        "phase2_summary.json",
    ]
    missing = [n for n in required if not (step4_dir / n).exists()]
    if missing:
        raise FileNotFoundError(
            f"Step-4 outputs missing in {step4_dir}: {missing}. "
            "Run `python -m cage.step4_sparse_invariant_model` first."
        )

    with open(step4_dir / "phase2_summary.json", "r", encoding="utf-8") as fh:
        summary = json.load(fh)

    # OOF predictions (sample_barcode, patient_barcode, outer_fold, y_true, deep_prob)
    with open(step4_dir / "deep_oof_predictions.csv", "r", newline="", encoding="utf-8") as fh:
        oof_rows = [dict(r) for r in csv.DictReader(fh)]

    # Gate weights (gene, rank_by_mean, gate_mean_across_folds, gate_std_across_folds, gate_fold_*)
    with open(step4_dir / "gate_weights.csv", "r", newline="", encoding="utf-8") as fh:
        gate_rows = [dict(r) for r in csv.DictReader(fh)]

    # Latent embeddings (optional)
    latent_rows: List[Dict[str, str]] = []
    if (step4_dir / "latent_embeddings.csv").exists():
        with open(step4_dir / "latent_embeddings.csv", "r", newline="", encoding="utf-8") as fh:
            latent_rows = [dict(r) for r in csv.DictReader(fh)]

    checkpoint_dir = step4_dir / "checkpoints"
    checkpoint_files = sorted(checkpoint_dir.glob("fold_*.npz")) if checkpoint_dir.exists() else []

    return {
        "step4_dir": step4_dir,
        "summary": summary,
        "oof_rows": oof_rows,
        "gate_rows": gate_rows,
        "latent_rows": latent_rows,
        "checkpoint_files": checkpoint_files,
    }


# ---------------------------------------------------------------------------
# Model reconstruction
# ---------------------------------------------------------------------------


def _config_from_summary(summary: Mapping[str, Any]) -> TrainingConfig:
    mc = dict(summary.get("model_config", {}))
    # pick sensible fallbacks if anything is missing
    return TrainingConfig(
        latent_dim=int(mc.get("latent_dim", 48)),
        hidden_dims=tuple(int(h) for h in mc.get("hidden_dims", [256, 96])),
        sparsity_type=str(mc.get("sparsity_type", "l1")),
        use_decoder=bool(mc.get("use_decoder", True)),
        dropout=float(mc.get("dropout", 0.1)),
        lr=float(mc.get("lr", 1e-3)),
        weight_decay=float(mc.get("weight_decay", 1e-4)),
        n_epochs=int(mc.get("n_epochs", 150)),
        batch_size=int(mc.get("batch_size", 64)),
        patience=int(mc.get("patience", 15)),
        lambda_recon=float(mc.get("lambda_recon", 0.1)),
        lambda_sparsity=float(mc.get("lambda_sparsity", 1e-3)),
        lambda_adv=float(mc.get("lambda_adv", 0.5)),
        lambda_inv=float(mc.get("lambda_inv", 0.1)),
        adv_ramp_epochs=int(mc.get("adv_ramp_epochs", 5)),
        grad_clip_norm=float(mc.get("grad_clip_norm", 5.0)),
        seed=int(mc.get("seed", 2026)),
    )


def _infer_n_confounders(state: Mapping[str, np.ndarray]) -> int:
    if "adv.W" in state:
        return int(state["adv.W"].shape[1])
    return 2


def load_fold_models(
    *,
    checkpoint_files: Sequence[Path],
    summary: Mapping[str, Any],
    n_features: int,
    outer_fold: np.ndarray,
) -> List[FoldArtifacts]:
    """Restore SparseInvariantModel instances from Step-4 checkpoint bundles.

    Each ``.npz`` contains the flat parameter dict plus meta fields
    ``_meta_mean_fold`` (per-feature mean), ``_meta_std_fold`` (per-feature
    std) and ``_meta_fold_id``.
    """
    config = _config_from_summary(summary)
    fold_artifacts: List[FoldArtifacts] = []
    for path in checkpoint_files:
        data = np.load(path, allow_pickle=False)
        state: Dict[str, np.ndarray] = {k: data[k] for k in data.files}
        mean_fold = np.asarray(state.pop("_meta_mean_fold"), dtype=np.float64)
        std_fold = np.asarray(state.pop("_meta_std_fold"), dtype=np.float64)
        fold_id_arr = state.pop("_meta_fold_id", np.array([-1]))
        fold_id = int(np.asarray(fold_id_arr).ravel()[0])

        n_conf = _infer_n_confounders(state)
        rng = np.random.default_rng(config.seed + 2048 + fold_id)
        model = SparseInvariantModel(
            n_features=n_features, n_confounders=n_conf, config=config, rng=rng
        )
        # restore params
        for key, arr in state.items():
            if key in model.params and model.params[key].shape == arr.shape:
                model.params[key][...] = arr
        test_mask = (outer_fold == fold_id)
        fold_artifacts.append(
            FoldArtifacts(
                fold_id=fold_id, model=model,
                mean_fold=mean_fold, std_fold=std_fold, test_mask=test_mask,
            )
        )
        logger.info(
            "Loaded fold %d checkpoint | n_features=%d n_params=%d test_samples=%d",
            fold_id, n_features, sum(v.size for v in model.params.values()),
            int(test_mask.sum()),
        )
    return fold_artifacts


# ---------------------------------------------------------------------------
# Gradient / attribution primitives
# ---------------------------------------------------------------------------


def _zero_grads(model: SparseInvariantModel) -> None:
    for g in model.grads.values():
        g.fill(0.0)


def input_gradient(model: SparseInvariantModel, x_z: np.ndarray) -> np.ndarray:
    """Return :math:`d\\,logit/d\\,x` for each sample in ``x_z``.

    Shape: ``(N, P)``. Uses the (already-standardized) input and runs a
    single backward pass through the encoder and classifier head with
    ``training=False`` (dropout disabled, gate deterministic).
    """
    _zero_grads(model)
    _ = model.forward(x_z, training=False)
    grad_logit = np.ones((x_z.shape[0], 1), dtype=np.float64)
    g_z = model.clf.backward(grad_logit)
    g_h1 = model.fc_latent.backward(g_z)
    g_h1 = model.drop1.backward(g_h1)
    g_h1 = model.relu1.backward(g_h1)
    g_h0 = model.fc1.backward(g_h1)
    g_h0 = model.drop0.backward(g_h0)
    g_h0 = model.relu0.backward(g_h0)
    g_gx = model.fc0.backward(g_h0)
    g_x = model.gate.backward(g_gx)
    _zero_grads(model)
    return g_x


def integrated_gradients(
    model: SparseInvariantModel,
    x_z: np.ndarray,
    *,
    n_steps: int = 50,
    baseline: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Per-sample Integrated Gradients (Sundararajan et al., 2017).

    ``baseline`` defaults to zeros in standardized space, i.e. the per-fold
    training mean in raw space. The integral is approximated with the
    trapezoidal rule over ``n_steps`` points.
    """
    if baseline is None:
        baseline = np.zeros_like(x_z)
    if baseline.shape != x_z.shape:
        raise ValueError("baseline shape must match x_z")
    n_steps = max(2, int(n_steps))
    # Trapezoidal rule over alpha in [0, 1] with (n_steps+1) points.
    alphas = np.linspace(0.0, 1.0, n_steps + 1)
    accum = np.zeros_like(x_z)
    weights = np.ones_like(alphas)
    weights[0] = 0.5
    weights[-1] = 0.5
    for w, a in zip(weights, alphas):
        x_interp = baseline + float(a) * (x_z - baseline)
        g = input_gradient(model, x_interp)
        accum += float(w) * g
    mean_grad = accum / float(n_steps)
    return (x_z - baseline) * mean_grad


# ---------------------------------------------------------------------------
# Per-fold attribution pipeline
# ---------------------------------------------------------------------------


def _pick_capped(
    y: np.ndarray, indices: np.ndarray, cap_per_class: int, rng: np.random.Generator,
) -> np.ndarray:
    """Return up to ``cap_per_class`` tumor + cap_per_class normal indices
    from ``indices`` (which are sample positions, not labels)."""
    sel: List[int] = []
    for cls in (0, 1):
        cls_ix = indices[y[indices] == cls]
        if cls_ix.size == 0:
            continue
        if cls_ix.size <= cap_per_class:
            sel.extend(cls_ix.tolist())
        else:
            sel.extend(rng.choice(cls_ix, size=cap_per_class, replace=False).tolist())
    return np.asarray(sorted(sel), dtype=np.int64)


def compute_attribution_per_fold(
    *,
    fold_artifacts: Sequence[FoldArtifacts],
    X: np.ndarray,
    y: np.ndarray,
    method: str = "integrated-gradients",
    cap_per_class: int = 100,
    ig_steps: int = 50,
    seed: int = 2026,
) -> Dict[str, Any]:
    """Compute per-fold attribution and return aggregate / per-sample tables.

    Returns dict::

        per_fold_mean_abs     (K, P) - mean(|attribution|) per gene per fold
        per_fold_signed_mean  (K, P) - mean(attribution) per gene per fold
        sample_index          dict{fold_id -> np.ndarray of sample indices used}
        sample_attribution    dict{fold_id -> (n_used, P) attribution matrix}
        sample_labels         dict{fold_id -> (n_used,) label vector}
    """
    if method not in ("integrated-gradients", "grad-x-input", "gate-weight"):
        raise ValueError(f"Unknown attribution method: {method!r}")
    K = len(fold_artifacts)
    P = X.shape[1]
    per_fold_mean_abs = np.zeros((K, P), dtype=np.float64)
    per_fold_signed_mean = np.zeros((K, P), dtype=np.float64)
    rng_master = np.random.default_rng(seed)

    sample_index: Dict[int, np.ndarray] = {}
    sample_attribution: Dict[int, np.ndarray] = {}
    sample_labels: Dict[int, np.ndarray] = {}

    for k_idx, fa in enumerate(fold_artifacts):
        fold_seed = int(rng_master.integers(0, 2**31 - 1))
        fold_rng = np.random.default_rng(fold_seed)
        test_idx = np.where(fa.test_mask)[0]
        if test_idx.size == 0:
            logger.warning("Fold %d has no test samples; skipping attribution.", fa.fold_id)
            continue
        used = _pick_capped(y, test_idx, cap_per_class, fold_rng)
        if used.size == 0:
            continue

        X_used = X[used]
        X_z = dm.standardize_apply(X_used, fa.mean_fold, fa.std_fold)

        if method == "gate-weight":
            gate_vals = fa.model.eval_gate()
            attrib = np.broadcast_to(gate_vals[None, :], X_z.shape).astype(np.float64)
        elif method == "grad-x-input":
            g = input_gradient(fa.model, X_z)
            attrib = g * X_z
        else:  # integrated-gradients
            attrib = integrated_gradients(fa.model, X_z, n_steps=ig_steps)

        per_fold_mean_abs[k_idx] = np.mean(np.abs(attrib), axis=0)
        per_fold_signed_mean[k_idx] = np.mean(attrib, axis=0)
        sample_index[fa.fold_id] = used
        sample_attribution[fa.fold_id] = attrib
        sample_labels[fa.fold_id] = y[used].astype(np.int64)
        logger.info(
            "Attribution [%s] fold %d: used %d samples (%d tumor / %d normal) -> "
            "attrib[mean_abs]=%.4g attrib[top1]=%s",
            method, fa.fold_id, int(used.size),
            int((y[used] == 1).sum()), int((y[used] == 0).sum()),
            float(per_fold_mean_abs[k_idx].mean()),
            int(np.argmax(per_fold_mean_abs[k_idx])),
        )

    return {
        "method": method,
        "per_fold_mean_abs": per_fold_mean_abs,
        "per_fold_signed_mean": per_fold_signed_mean,
        "sample_index": sample_index,
        "sample_attribution": sample_attribution,
        "sample_labels": sample_labels,
    }


# ---------------------------------------------------------------------------
# Stability
# ---------------------------------------------------------------------------


def compute_stability_scores(
    per_fold_mean_abs: np.ndarray,
    *,
    top_frac: float = 0.05,
) -> Dict[str, np.ndarray]:
    """Return selection-frequency and cross-fold consistency metrics.

    * ``selection_frequency``: fraction of folds where a gene ranked in the
      top ``top_frac`` of that fold's ``|attribution|`` column.
    * ``attribution_std``: per-gene std across folds.
    * ``stability``: ``selection_frequency * (1 / (1 + rel_std))``.
    """
    K, P = per_fold_mean_abs.shape
    if K == 0:
        return {
            "selection_frequency": np.zeros(P, dtype=np.float64),
            "attribution_std": np.zeros(P, dtype=np.float64),
            "stability": np.zeros(P, dtype=np.float64),
        }

    top_n = max(1, int(math.ceil(top_frac * P)))
    selected = np.zeros((K, P), dtype=np.int64)
    for k in range(K):
        order = np.argsort(-per_fold_mean_abs[k], kind="mergesort")
        selected[k, order[:top_n]] = 1
    freq = selected.mean(axis=0)
    mean = per_fold_mean_abs.mean(axis=0)
    std = per_fold_mean_abs.std(axis=0, ddof=0)
    rel_std = std / (np.abs(mean) + 1e-12)
    stability = freq * (1.0 / (1.0 + rel_std))
    return {
        "selection_frequency": freq,
        "attribution_std": std,
        "attribution_mean": mean,
        "relative_std": rel_std,
        "stability": stability,
        "top_n_per_fold": top_n,
    }


# ---------------------------------------------------------------------------
# Environment invariance
# ---------------------------------------------------------------------------


def compute_invariance_scores(
    *,
    sample_attribution: Mapping[int, np.ndarray],
    sample_index: Mapping[int, np.ndarray],
    master_rows: Sequence[Mapping[str, str]],
    env_names: Sequence[str],
) -> Dict[str, Any]:
    """Invariance score = 1 / (1 + mean std of |attribution| across env levels).

    For each env column (``env_<name>``) we pool per-sample attributions
    across folds, split by the env level (``0``/``1``), compute the mean
    absolute attribution per gene per level, then take the std across
    levels (ignoring empty strata). Finally we average that std across
    the supplied env columns.
    """
    if not sample_attribution:
        return {"invariance": np.zeros(0, dtype=np.float64), "env_used": []}
    P = next(iter(sample_attribution.values())).shape[1]
    # Pool per-sample attribs into one big matrix plus row-wise sample index.
    rows: List[np.ndarray] = []
    idxs: List[int] = []
    for fid in sorted(sample_attribution.keys()):
        rows.append(sample_attribution[fid])
        idxs.extend(sample_index[fid].tolist())
    A = np.concatenate(rows, axis=0) if rows else np.zeros((0, P))
    A_abs = np.abs(A)
    idxs_arr = np.asarray(idxs, dtype=np.int64)

    env_used: List[str] = []
    per_env_std: List[np.ndarray] = []
    for env in env_names:
        col = f"env_{env}" if not env.startswith("env_") else env
        if not master_rows or col not in master_rows[0]:
            continue
        levels_seen: List[np.ndarray] = []
        for level in ("0", "1"):
            mask_level = np.array(
                [master_rows[int(i)].get(col, "") == level for i in idxs_arr], dtype=bool
            )
            if mask_level.sum() == 0:
                continue
            levels_seen.append(A_abs[mask_level].mean(axis=0))
        if len(levels_seen) >= 2:
            stacked = np.stack(levels_seen, axis=0)
            per_env_std.append(stacked.std(axis=0, ddof=0))
            env_used.append(env)

    if not per_env_std:
        # Fall back to zero std (and thus max invariance across all genes)
        avg_std = np.zeros(P, dtype=np.float64)
    else:
        avg_std = np.stack(per_env_std, axis=0).mean(axis=0)

    invariance = 1.0 / (1.0 + avg_std)
    return {
        "invariance": invariance,
        "avg_env_std": avg_std,
        "env_used": env_used,
    }


# ---------------------------------------------------------------------------
# Gate component
# ---------------------------------------------------------------------------


def compute_gate_component(
    *,
    gate_rows: Sequence[Mapping[str, str]],
    gene_names: Sequence[str],
) -> Dict[str, np.ndarray]:
    """Extract the gate-mean signal and per-fold gate matrix from the
    Step-4 ``gate_weights.csv``.

    Returns
    -------
    mean     (P,) gate_mean_across_folds aligned to ``gene_names``
    std      (P,) gate_std_across_folds
    per_fold (K, P) gate values per fold (columns ``gate_fold_*``)
    """
    if not gate_rows:
        P = len(gene_names)
        return {
            "mean": np.zeros(P, dtype=np.float64),
            "std": np.zeros(P, dtype=np.float64),
            "per_fold": np.zeros((0, P), dtype=np.float64),
        }

    first = gate_rows[0]
    fold_cols = sorted(
        (c for c in first.keys() if c.startswith("gate_fold_")),
        key=lambda c: int(c.rsplit("_", 1)[-1]),
    )
    gene_order = [r["gene"] for r in gate_rows]
    pos_by_gene = {g: i for i, g in enumerate(gene_names)}
    P = len(gene_names)
    mean = np.zeros(P, dtype=np.float64)
    std = np.zeros(P, dtype=np.float64)
    per_fold = np.zeros((len(fold_cols), P), dtype=np.float64)
    missing: List[str] = []
    for gi, gname in enumerate(gene_order):
        if gname not in pos_by_gene:
            missing.append(gname)
            continue
        p = pos_by_gene[gname]
        try:
            mean[p] = float(gate_rows[gi].get("gate_mean_across_folds", "0") or 0.0)
            std[p] = float(gate_rows[gi].get("gate_std_across_folds", "0") or 0.0)
        except ValueError:
            mean[p] = 0.0
            std[p] = 0.0
        for k, col in enumerate(fold_cols):
            try:
                per_fold[k, p] = float(gate_rows[gi].get(col, "0") or 0.0)
            except ValueError:
                per_fold[k, p] = 0.0
    if missing:
        logger.warning("Gate CSV contains %d genes not present in gene_names "
                       "(first: %s)", len(missing), missing[:3])
    return {"mean": mean, "std": std, "per_fold": per_fold}


# ---------------------------------------------------------------------------
# Perturbation (gradient-based, genome-wide)
# ---------------------------------------------------------------------------


def compute_perturbation_scores(
    *,
    fold_artifacts: Sequence[FoldArtifacts],
    X: np.ndarray,
    y: np.ndarray,
    cap_per_class: int = 100,
    seed: int = 2026,
) -> Dict[str, np.ndarray]:
    """Return a genome-wide per-gene perturbation proxy.

    The direct tumor <-> normal intervention of plane.md is approximated
    by a first-order Taylor expansion::

        delta_p_{i,j} = p(x_i)(1 - p(x_i)) * grad_j * delta_z_{i,j}

    where ``grad_j = d logit / d x_{z,j}`` for sample ``i`` and
    ``delta_z_{i,j}`` is the signed shift from the sample's value toward
    the opposing-class centroid in standardized space. The reported
    score pools the absolute per-sample predicted-probability changes.
    """
    K = len(fold_artifacts)
    P = X.shape[1]
    if K == 0:
        return {
            "perturbation": np.zeros(P, dtype=np.float64),
            "per_fold": np.zeros((0, P), dtype=np.float64),
        }
    rng_master = np.random.default_rng(seed)
    per_fold = np.zeros((K, P), dtype=np.float64)
    for k_idx, fa in enumerate(fold_artifacts):
        fold_rng = np.random.default_rng(int(rng_master.integers(0, 2**31 - 1)))
        test_idx = np.where(fa.test_mask)[0]
        if test_idx.size == 0:
            continue
        used = _pick_capped(y, test_idx, cap_per_class, fold_rng)
        if used.size == 0:
            continue
        X_used = X[used]
        y_used = y[used]
        X_z = dm.standardize_apply(X_used, fa.mean_fold, fa.std_fold)

        tumor_mask = y_used == 1
        normal_mask = y_used == 0
        if tumor_mask.sum() < 1 or normal_mask.sum() < 1:
            logger.warning(
                "Fold %d perturbation: only one class present (tumor=%d, normal=%d); "
                "skipping gene-wise shift.",
                fa.fold_id, int(tumor_mask.sum()), int(normal_mask.sum()),
            )
            per_fold[k_idx] = 0.0
            continue

        tumor_centroid_z = X_z[tumor_mask].mean(axis=0)
        normal_centroid_z = X_z[normal_mask].mean(axis=0)

        # Gradients of logit w.r.t. input (per sample).
        g_x = input_gradient(fa.model, X_z)  # (N, P)
        # Predicted probability (per sample) for sigmoid envelope term.
        probs = fa.model.predict_proba(X_z).ravel()  # (N,)
        envelope = (probs * (1.0 - probs)).reshape(-1, 1)  # (N, 1)

        # delta_z toward opposing centroid
        delta_z = np.empty_like(X_z)
        delta_z[tumor_mask] = normal_centroid_z[None, :] - X_z[tumor_mask]
        delta_z[normal_mask] = tumor_centroid_z[None, :] - X_z[normal_mask]

        # First-order Taylor estimate of per-sample probability change.
        dp = envelope * g_x * delta_z  # (N, P)
        per_fold[k_idx] = np.mean(np.abs(dp), axis=0)
        logger.info(
            "Perturbation fold %d: n_used=%d tumor=%d normal=%d mean|dp|=%.4g",
            fa.fold_id, int(used.size),
            int(tumor_mask.sum()), int(normal_mask.sum()),
            float(per_fold[k_idx].mean()),
        )

    return {
        "perturbation": per_fold.mean(axis=0),
        "per_fold": per_fold,
    }


# ---------------------------------------------------------------------------
# CDPS aggregation
# ---------------------------------------------------------------------------


def normalize_scores(
    scores: np.ndarray, *, method: str = "minmax", eps: float = 1e-12
) -> np.ndarray:
    """Normalize a per-gene score vector to [0, 1].

    * ``"minmax"``: (x - min) / (max - min)
    * ``"rank"``: average-rank transformation scaled to [0, 1]
    """
    x = np.asarray(scores, dtype=np.float64).ravel()
    if x.size == 0:
        return x
    if method == "rank":
        order = np.argsort(x, kind="mergesort")
        ranks = np.empty_like(order, dtype=np.float64)
        # average ranks on ties so the result has no flat plateau at 0/1.
        n = x.size
        i = 0
        while i < n:
            j = i
            while j + 1 < n and x[order[j + 1]] == x[order[i]]:
                j += 1
            avg = (i + j) / 2.0
            ranks[order[i:j + 1]] = avg
            i = j + 1
        return ranks / max(1.0, float(n - 1))
    # min-max
    lo = float(x.min())
    hi = float(x.max())
    if hi - lo < eps:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def build_cdps_ranking(
    *,
    gene_names: Sequence[str],
    attribution: np.ndarray,
    gate: np.ndarray,
    stability: np.ndarray,
    invariance: np.ndarray,
    perturbation: np.ndarray,
    weights: Mapping[str, float],
    normalization: str = "minmax",
    include_perturbation: bool = True,
) -> Dict[str, Any]:
    """Blend the five components into a CDPS-ranked gene table.

    Components are normalized individually, then linearly combined with
    user-supplied weights (which are re-normalized to sum to 1). Returns
    normalized arrays, the CDPS vector and a list of ranked-gene records.
    """
    P = len(gene_names)
    attribution_n = normalize_scores(attribution, method=normalization)
    gate_n = normalize_scores(gate, method=normalization)
    stability_n = normalize_scores(stability, method=normalization)
    invariance_n = normalize_scores(invariance, method=normalization)
    perturbation_n = normalize_scores(perturbation, method=normalization) \
        if include_perturbation else np.zeros(P, dtype=np.float64)

    raw_w = {
        "attribution": float(weights.get("attribution", 0.30)),
        "gate": float(weights.get("gate", 0.20)),
        "stability": float(weights.get("stability", 0.20)),
        "invariance": float(weights.get("invariance", 0.15)),
        "perturbation": float(weights.get("perturbation", 0.15)) if include_perturbation else 0.0,
    }
    total = sum(raw_w.values())
    if total <= 0:
        raise ValueError("Sum of CDPS weights is non-positive; provide at least one positive weight.")
    w = {k: v / total for k, v in raw_w.items()}

    cdps = (
        w["attribution"] * attribution_n
        + w["gate"] * gate_n
        + w["stability"] * stability_n
        + w["invariance"] * invariance_n
        + w["perturbation"] * perturbation_n
    )

    order = np.argsort(-cdps, kind="mergesort")
    records: List[Dict[str, Any]] = []
    for rank, idx in enumerate(order, start=1):
        records.append({
            "rank": int(rank),
            "gene": str(gene_names[int(idx)]),
            "cdps": float(cdps[idx]),
            "attribution_score": float(attribution[idx]),
            "attribution_norm": float(attribution_n[idx]),
            "gate_score": float(gate[idx]),
            "gate_norm": float(gate_n[idx]),
            "stability_score": float(stability[idx]),
            "stability_norm": float(stability_n[idx]),
            "invariance_score": float(invariance[idx]),
            "invariance_norm": float(invariance_n[idx]),
            "perturbation_score": float(perturbation[idx]),
            "perturbation_norm": float(perturbation_n[idx]),
        })
    return {
        "cdps": cdps,
        "order": order,
        "records": records,
        "weights_effective": w,
        "weights_raw": raw_w,
        "normalization": normalization,
        "components": {
            "attribution": attribution_n,
            "gate": gate_n,
            "stability": stability_n,
            "invariance": invariance_n,
            "perturbation": perturbation_n,
        },
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


def generate_step5_figures(
    *,
    ranking: Mapping[str, Any],
    per_fold_attribution: np.ndarray,
    gene_names: Sequence[str],
    output_dir: Path,
    style: Any = None,
    formats: Sequence[str] = ("svg",),
    top_k_heat: int = 25,
) -> Tuple[List[str], List[Tuple[str, str]]]:
    """Render Phase-III figures (stability heatmap, scatter, bar charts)."""
    generated: List[str] = []
    skipped: List[Tuple[str, str]] = []
    fig_names = [
        "fig_cdps_components_bar",
        "fig_attribution_vs_perturbation",
        "fig_stability_heatmap",
        "fig_G1_cdps_distribution",
        "fig_G2_top25_components",
        "fig_G3_score_scatter",
    ]
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
        records = list(ranking["records"])
        top = records[:top_k_heat]

        # --- CDPS component bar chart (top-K) ---
        try:
            n_top = len(top)
            components = ["attribution", "gate", "stability", "invariance", "perturbation"]
            palette = categorical_colors(len(components))
            fig, ax = plt.subplots(figsize=(min(9.0, 0.3 * n_top + 3.5), 3.6))
            bottom = np.zeros(n_top, dtype=np.float64)
            w = ranking["weights_effective"]
            xpos = np.arange(n_top)
            for i, comp in enumerate(components):
                key = f"{comp}_norm"
                vals = np.array([float(r[key]) * float(w[comp]) for r in top],
                                dtype=np.float64)
                ax.bar(xpos, vals, bottom=bottom,
                       color=palette[i], edgecolor="black", linewidth=0.4,
                       label=f"{comp} (w={w[comp]:.2f})")
                bottom += vals
            ax.set_xticks(xpos)
            ax.set_xticklabels([r["gene"] for r in top], rotation=90, fontsize=7)
            ax.set_ylabel("Weighted CDPS contribution")
            ax.set_title(f"Top-{n_top} CDPS components")
            ax.legend(loc="upper right", fontsize=7)
            fig.tight_layout()
            paths = save_figure(fig, fig_dir / "fig_cdps_components_bar",
                                style=style, formats=formats)
            if paths:
                generated.append("fig_cdps_components_bar")
        except Exception as exc:
            skipped.append(("fig_cdps_components_bar", str(exc)))

        # --- Attribution vs perturbation scatter (all genes) ---
        try:
            fig, ax = plt.subplots(figsize=(3.6, 3.2))
            attrib_norm = ranking["components"]["attribution"]
            pert_norm = ranking["components"]["perturbation"]
            ax.scatter(attrib_norm, pert_norm, s=4, alpha=0.25,
                       color=semantic_color("enriched"), edgecolor="none")
            # Highlight top-25 by CDPS
            for r in top[:25]:
                gi = list(gene_names).index(r["gene"])
                ax.scatter(attrib_norm[gi], pert_norm[gi], s=18,
                           color=semantic_color("tumor"), edgecolor="black",
                           linewidth=0.4, zorder=5)
            ax.set_xlabel("Attribution (normalized)")
            ax.set_ylabel("Perturbation (normalized)")
            ax.set_title("Attribution vs perturbation - all genes")
            fig.tight_layout()
            paths = save_figure(fig, fig_dir / "fig_attribution_vs_perturbation",
                                style=style, formats=formats)
            if paths:
                generated.append("fig_attribution_vs_perturbation")
        except Exception as exc:
            skipped.append(("fig_attribution_vs_perturbation", str(exc)))

        # --- Stability heatmap (top genes x folds) ---
        try:
            K = per_fold_attribution.shape[0]
            if K == 0:
                skipped.append(("fig_stability_heatmap", "no per-fold attribution"))
            else:
                n_top = min(top_k_heat, len(top))
                top_gene_names = [r["gene"] for r in top[:n_top]]
                gene_pos = [list(gene_names).index(g) for g in top_gene_names]
                heat = per_fold_attribution[:, gene_pos]
                # Row-wise (fold) min-max so differences in scale don't dominate
                row_min = heat.min(axis=1, keepdims=True)
                row_max = heat.max(axis=1, keepdims=True)
                heat_n = (heat - row_min) / np.where(
                    (row_max - row_min) < 1e-12, 1.0, (row_max - row_min)
                )
                fig, ax = plt.subplots(figsize=(min(9.0, 0.3 * n_top + 3.0), 2.6))
                im = ax.imshow(heat_n, aspect="auto", cmap="viridis")
                ax.set_yticks(range(K))
                ax.set_yticklabels([f"fold {k}" for k in range(K)])
                ax.set_xticks(range(n_top))
                ax.set_xticklabels(top_gene_names, rotation=90, fontsize=7)
                ax.set_title(f"Top-{n_top} gene |attribution| across folds")
                fig.colorbar(im, ax=ax, fraction=0.025, pad=0.04).set_label("row-normalized")
                fig.tight_layout()
                paths = save_figure(fig, fig_dir / "fig_stability_heatmap",
                                    style=style, formats=formats)
                if paths:
                    generated.append("fig_stability_heatmap")
        except Exception as exc:
            skipped.append(("fig_stability_heatmap", str(exc)))

        # ---- Figure G1: CDPS score distribution histogram ----
        try:
            all_recs = list(ranking["records"])
            all_scores = np.array([float(r["cdps"]) for r in all_recs], dtype=np.float64)
            all_scores = all_scores[np.isfinite(all_scores)]
            if all_scores.size > 0:
                top100_thresh = float(np.sort(all_scores)[-min(100, len(all_scores))])
                fig, ax = plt.subplots(figsize=(7, 4.5))
                ax.hist(all_scores, bins=80, color="#aaaaaa", edgecolor="white",
                        alpha=0.7, label="All genes")
                top_vals = all_scores[all_scores >= top100_thresh]
                ax.hist(top_vals, bins=20, color=semantic_color("enriched"),
                        edgecolor="white", alpha=0.9, label="Top 100")
                ax.axvline(top100_thresh, color="#d62728", linewidth=1.5, linestyle="--",
                           label=f"Top-100 threshold ({top100_thresh:.3f})")
                ax.set_xlabel("CDPS Score")
                ax.set_ylabel("Number of genes")
                ax.set_title("CDPS Score Distribution — All Genes")
                ax.legend(fontsize=8)
                ax.text(0.98, 0.95, f"n = {len(all_scores)} genes",
                        transform=ax.transAxes, ha="right", fontsize=8, color="#666666")
                fig.tight_layout()
                paths = save_figure(fig, fig_dir / "fig_G1_cdps_distribution",
                                    style=style, formats=formats)
                if paths:
                    generated.append("fig_G1_cdps_distribution")
        except Exception as exc:
            skipped.append(("fig_G1_cdps_distribution", str(exc)))

        # ---- Figure G2: top-25 raw component stacked bar ----
        try:
            top25_recs = list(ranking["records"])[:25]
            if top25_recs:
                comp_defs_g2 = [
                    ("attribution_norm",  "Attribution"),
                    ("gate_norm",         "Gate"),
                    ("stability_norm",    "Stability"),
                    ("invariance_norm",   "Invariance"),
                    ("perturbation_norm", "Perturbation"),
                ]
                n_g2 = len(top25_recs)
                palette_g2 = categorical_colors(len(comp_defs_g2))
                genes_g2 = [r["gene"] for r in top25_recs]
                x_g2 = np.arange(n_g2)

                fig, axes = plt.subplots(2, 1, figsize=(13, 9))

                # top: stacked bar of raw normalised components
                ax = axes[0]
                bottoms_g2 = np.zeros(n_g2, dtype=np.float64)
                for ci, (col, lbl) in enumerate(comp_defs_g2):
                    vals_g2 = np.array([float(r.get(col, 0)) for r in top25_recs],
                                       dtype=np.float64)
                    ax.bar(x_g2, vals_g2, bottom=bottoms_g2, label=lbl,
                           color=palette_g2[ci], edgecolor="white", linewidth=0.3, alpha=0.85)
                    bottoms_g2 += vals_g2
                ax.set_xticks(x_g2)
                ax.set_xticklabels(genes_g2, rotation=45, ha="right", fontsize=8)
                ax.set_ylabel("Normalised component score")
                ax.set_title("CDPS Component Scores — Top 25 Genes (Stacked)")
                ax.legend(fontsize=8, loc="upper right")

                # bottom: CDPS overall score bar
                ax = axes[1]
                cdps_g2 = np.array([float(r.get("cdps", 0)) for r in top25_recs],
                                    dtype=np.float64)
                bars_g2 = ax.bar(x_g2, cdps_g2, color=palette_g2[0],
                                 edgecolor="white", linewidth=0.3)
                ax.set_xticks(x_g2)
                ax.set_xticklabels(genes_g2, rotation=45, ha="right", fontsize=8)
                ax.set_ylabel("CDPS Score")
                ax.set_title("Composite CDPS Score — Top 25 Genes")
                for bar, v in zip(bars_g2, cdps_g2):
                    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.002,
                            f"{v:.3f}", ha="center", va="bottom", fontsize=6.5)

                fig.tight_layout()
                paths = save_figure(fig, fig_dir / "fig_G2_top25_components",
                                    style=style, formats=formats)
                if paths:
                    generated.append("fig_G2_top25_components")
        except Exception as exc:
            skipped.append(("fig_G2_top25_components", str(exc)))

        # ---- Figure G3: CDPS component score scatter matrix ----
        try:
            top_n_scatter = 200
            all_recs_g3 = sorted(ranking["records"], key=lambda r: -float(r["cdps"]))[:top_n_scatter]
            comp_defs = [
                ("attribution_norm",  "Attribution"),
                ("stability_norm",    "Stability"),
                ("invariance_norm",   "Invariance"),
                ("perturbation_norm", "Perturbation"),
            ]
            n_comp = len(comp_defs)
            cdps_vals_g3 = np.array([float(r["cdps"]) for r in all_recs_g3], dtype=np.float64)
            fig, axes = plt.subplots(n_comp, n_comp, figsize=(n_comp * 3, n_comp * 3))
            for i, (col_y, lbl_y) in enumerate(comp_defs):
                for j, (col_x, lbl_x) in enumerate(comp_defs):
                    ax = axes[i][j]
                    ys = np.array([float(r.get(col_y, 0)) for r in all_recs_g3], dtype=np.float64)
                    xs = np.array([float(r.get(col_x, 0)) for r in all_recs_g3], dtype=np.float64)
                    if i == j:
                        ax.hist(xs, bins=20, color=semantic_color("normal"),
                                edgecolor="white", alpha=0.8)
                        ax.set_title(lbl_x, fontsize=9, fontweight="bold")
                    else:
                        sc = ax.scatter(xs, ys, c=cdps_vals_g3, cmap="viridis",
                                        s=12, alpha=0.7, edgecolors="none")
                        if xs.std() > 0 and ys.std() > 0:
                            rho = float(np.corrcoef(xs, ys)[0, 1])
                            ax.text(0.05, 0.92, f"r={rho:.2f}", transform=ax.transAxes,
                                    fontsize=7, color="#333333")
                    if i == n_comp - 1:
                        ax.set_xlabel(lbl_x, fontsize=8)
                    if j == 0:
                        ax.set_ylabel(lbl_y, fontsize=8)
                    ax.tick_params(labelsize=7)
            fig.suptitle(
                f"CDPS Component Correlations — Top {top_n_scatter} Genes (colour = CDPS score)",
                fontsize=11, fontweight="bold",
            )
            fig.tight_layout()
            paths = save_figure(fig, fig_dir / "fig_G3_score_scatter",
                                style=style, formats=formats)
            if paths:
                generated.append("fig_G3_score_scatter")
        except Exception as exc:
            skipped.append(("fig_G3_score_scatter", str(exc)))

    except Exception as exc:  # pragma: no cover - environment-specific
        for n in fig_names:
            skipped.append((n, str(exc)))

    return generated, skipped


# ---------------------------------------------------------------------------
# High-level driver
# ---------------------------------------------------------------------------


def run_step5_cdps(
    *,
    step2_dir: Path,
    step4_dir: Path,
    output_dir: Path,
    attribution_method: str = "integrated-gradients",
    cap_per_class: int = 100,
    ig_steps: int = 50,
    run_perturbation: bool = True,
    env_names: Sequence[str] = ("smoking", "sex", "histology", "country", "stage"),
    weights: Optional[Mapping[str, float]] = None,
    normalization: str = "minmax",
    stability_top_frac: float = 0.05,
    top_ks: Sequence[int] = (25,),
    seed: int = 2026,
) -> Dict[str, Any]:
    """Execute the Step-5 pipeline and return a summary dict of outputs."""
    weights = dict(weights or {
        "attribution": 0.30, "gate": 0.20, "stability": 0.20,
        "invariance": 0.15, "perturbation": 0.15,
    })

    # Step 2 cohort
    artifacts = step3.load_step2_artifacts(step2_dir)
    aligned = step3.align_arrays(artifacts)
    X = aligned["X"]
    y = aligned["y"]
    groups = aligned["groups"]
    outer_fold = aligned["outer_fold"]
    master_rows = aligned["master_rows"]
    sample_barcodes = aligned["sample_barcodes"]
    gene_names = aligned["gene_names"]
    step3.assert_no_patient_leakage(groups, outer_fold)

    # Step 4 artifacts
    bundle = load_step4_bundle(step4_dir)
    fold_artifacts = load_fold_models(
        checkpoint_files=bundle["checkpoint_files"],
        summary=bundle["summary"],
        n_features=X.shape[1],
        outer_fold=outer_fold,
    )
    if not fold_artifacts:
        raise FileNotFoundError(f"No fold checkpoints found under {step4_dir}/checkpoints")

    # Component: attribution
    attribution_out = compute_attribution_per_fold(
        fold_artifacts=fold_artifacts, X=X, y=y,
        method=attribution_method, cap_per_class=cap_per_class,
        ig_steps=ig_steps, seed=seed,
    )
    attribution_mean = attribution_out["per_fold_mean_abs"].mean(axis=0)

    # Component: gate
    gate_out = compute_gate_component(
        gate_rows=bundle["gate_rows"], gene_names=gene_names,
    )

    # Component: stability
    stability_out = compute_stability_scores(
        attribution_out["per_fold_mean_abs"], top_frac=stability_top_frac,
    )

    # Component: invariance
    invariance_out = compute_invariance_scores(
        sample_attribution=attribution_out["sample_attribution"],
        sample_index=attribution_out["sample_index"],
        master_rows=master_rows,
        env_names=env_names,
    )

    # Component: perturbation (optional)
    if run_perturbation:
        perturbation_out = compute_perturbation_scores(
            fold_artifacts=fold_artifacts, X=X, y=y,
            cap_per_class=cap_per_class, seed=seed,
        )
        perturbation_vec = perturbation_out["perturbation"]
    else:
        perturbation_out = {
            "perturbation": np.zeros(X.shape[1], dtype=np.float64),
            "per_fold": np.zeros((0, X.shape[1]), dtype=np.float64),
        }
        perturbation_vec = perturbation_out["perturbation"]

    # CDPS
    ranking = build_cdps_ranking(
        gene_names=gene_names,
        attribution=attribution_mean,
        gate=gate_out["mean"],
        stability=stability_out["stability"],
        invariance=invariance_out["invariance"],
        perturbation=perturbation_vec,
        weights=weights, normalization=normalization,
        include_perturbation=run_perturbation,
    )

    return {
        "gene_names": list(gene_names),
        "sample_barcodes": list(sample_barcodes),
        "outer_fold": outer_fold,
        "fold_ids": [int(fa.fold_id) for fa in fold_artifacts],
        "attribution_out": attribution_out,
        "attribution_mean": attribution_mean,
        "gate_out": gate_out,
        "stability_out": stability_out,
        "invariance_out": invariance_out,
        "perturbation_out": perturbation_out,
        "ranking": ranking,
        "config": {
            "attribution_method": attribution_method,
            "cap_per_class": int(cap_per_class),
            "ig_steps": int(ig_steps),
            "run_perturbation": bool(run_perturbation),
            "env_names": list(env_names),
            "weights": weights,
            "normalization": normalization,
            "stability_top_frac": float(stability_top_frac),
            "top_ks": list(top_ks),
            "seed": int(seed),
            "step2_dir": str(step2_dir),
            "step4_dir": str(step4_dir),
        },
    }
