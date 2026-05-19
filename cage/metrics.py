"""CAGE binary-classification metrics (numpy-only).

Implements the scoring utilities required by Step 3 (grouped baselines) and
Step 4 (deep invariant model) without depending on scikit-learn. Every
function accepts plain numpy arrays / python sequences, returns plain
floats or ``dict``/``list`` objects (JSON-friendly), and is deterministic
for identical inputs.

Metrics provided
----------------
auroc(y_true, y_score)                 tie-aware Mann-Whitney ROC-AUC
auprc(y_true, y_score)                 average precision (trapezoidal)
balanced_accuracy(y_true, y_pred)      mean of per-class recall
brier_score(y_true, y_prob)            mean-squared-error calibration
log_loss(y_true, y_prob, eps=1e-12)    binary cross-entropy
confusion_counts(y_true, y_pred)       (tp, fp, tn, fn)
sensitivity_specificity(y_true, y_pred)  (sensitivity, specificity)
f1_score(y_true, y_pred)               harmonic mean of precision/recall
calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")
                                        reliability diagram bin table
roc_curve(y_true, y_score)             (fpr, tpr, thresholds)
precision_recall_curve(y_true, y_score)  (precision, recall, thresholds)
summarize_binary(y_true, y_score,      single-call scorecard dict
                 threshold=0.5)
bootstrap_ci(metric_fn, y_true, y_score, n_boot, seed, level)
                                        percentile-CI for any scalar metric
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

__all__ = [
    "auroc",
    "auprc",
    "balanced_accuracy",
    "brier_score",
    "log_loss",
    "confusion_counts",
    "sensitivity_specificity",
    "f1_score",
    "calibration_curve",
    "roc_curve",
    "precision_recall_curve",
    "summarize_binary",
    "bootstrap_ci",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _as_1d_float(a: Sequence[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(a, dtype=np.float64).ravel()
    return arr


def _as_1d_int(a: Sequence[int] | np.ndarray) -> np.ndarray:
    arr = np.asarray(a).ravel()
    # Coerce to int; any non-0/1 values raise later when we check labels.
    return arr.astype(np.int64)


def _validate_binary(y_true: np.ndarray) -> None:
    uniq = np.unique(y_true)
    bad = [int(v) for v in uniq if int(v) not in (0, 1)]
    if bad:
        raise ValueError(f"y_true must contain only 0/1; found {bad}")


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------


def auroc(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    """Return the ROC AUC for binary labels with tie handling.

    Uses the Mann-Whitney U formulation:
        AUC = (U - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    where U is computed from average ranks (so tied scores contribute 0.5).
    Returns ``float('nan')`` if a class is missing.
    """
    yt = _as_1d_int(y_true)
    ys = _as_1d_float(y_score)
    if yt.shape != ys.shape or yt.size == 0:
        return float("nan")
    _validate_binary(yt)
    n_pos = int((yt == 1).sum())
    n_neg = int((yt == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(ys, kind="mergesort")
    ranks = np.empty_like(ys, dtype=np.float64)
    i = 0
    n = ys.size
    sorted_scores = ys[order]
    while i < n:
        j = i + 1
        while j < n and sorted_scores[j] == sorted_scores[i]:
            j += 1
        # Average rank for the tied block (1-indexed ranks)
        avg_rank = (i + 1 + j) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j

    sum_ranks_pos = float(ranks[yt == 1].sum())
    u = sum_ranks_pos - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)


def auprc(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    """Return average precision (AUC of precision-recall curve).

    Uses the step-function definition
        AP = sum_k (R_k - R_{k-1}) * P_k
    matching scikit-learn's ``average_precision_score`` (tie-robust).
    """
    yt = _as_1d_int(y_true)
    ys = _as_1d_float(y_score)
    if yt.shape != ys.shape or yt.size == 0:
        return float("nan")
    _validate_binary(yt)
    n_pos = int((yt == 1).sum())
    if n_pos == 0:
        return float("nan")

    order = np.argsort(-ys, kind="mergesort")
    yt_sorted = yt[order]
    ys_sorted = ys[order]

    # Collapse ties: aggregate TP/FP counts at each distinct threshold
    distinct = np.concatenate(([True], ys_sorted[1:] != ys_sorted[:-1]))
    tp_cum = np.cumsum(yt_sorted == 1)
    fp_cum = np.cumsum(yt_sorted == 0)
    idx = np.nonzero(distinct)[0]
    # For tied blocks, use the last index so precision/recall reflect all ties.
    # Compute boundary indices: end of each tied block
    end_idx = np.empty(idx.size, dtype=np.int64)
    end_idx[:-1] = idx[1:] - 1
    end_idx[-1] = yt_sorted.size - 1

    tp = tp_cum[end_idx].astype(np.float64)
    fp = fp_cum[end_idx].astype(np.float64)
    precision = np.where((tp + fp) > 0, tp / (tp + fp), 1.0)
    recall = tp / n_pos
    # Prepend a recall of 0 so the first step contributes precision * recall_0.
    recall_prev = np.concatenate(([0.0], recall[:-1]))
    return float(np.sum((recall - recall_prev) * precision))


def balanced_accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Mean of per-class recall; equals 0.5 * (TPR + TNR) for binary."""
    yt = _as_1d_int(y_true)
    yp = _as_1d_int(y_pred)
    if yt.shape != yp.shape or yt.size == 0:
        return float("nan")
    _validate_binary(yt)
    _validate_binary(yp)
    n_pos = int((yt == 1).sum())
    n_neg = int((yt == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    tpr = float(((yp == 1) & (yt == 1)).sum()) / n_pos
    tnr = float(((yp == 0) & (yt == 0)).sum()) / n_neg
    return 0.5 * (tpr + tnr)


def brier_score(y_true: Sequence[int], y_prob: Sequence[float]) -> float:
    """Binary Brier score: mean((y_true - y_prob)^2)."""
    yt = _as_1d_float(y_true)
    yp = _as_1d_float(y_prob)
    if yt.shape != yp.shape or yt.size == 0:
        return float("nan")
    return float(np.mean((yt - yp) ** 2))


def log_loss(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    eps: float = 1e-12,
) -> float:
    """Binary cross-entropy (average, natural log)."""
    yt = _as_1d_float(y_true)
    yp = _as_1d_float(y_prob)
    if yt.shape != yp.shape or yt.size == 0:
        return float("nan")
    yp = np.clip(yp, eps, 1.0 - eps)
    return float(-np.mean(yt * np.log(yp) + (1.0 - yt) * np.log(1.0 - yp)))


def confusion_counts(
    y_true: Sequence[int], y_pred: Sequence[int]
) -> Tuple[int, int, int, int]:
    """Return ``(tp, fp, tn, fn)`` for binary labels."""
    yt = _as_1d_int(y_true)
    yp = _as_1d_int(y_pred)
    _validate_binary(yt)
    _validate_binary(yp)
    tp = int(((yp == 1) & (yt == 1)).sum())
    fp = int(((yp == 1) & (yt == 0)).sum())
    tn = int(((yp == 0) & (yt == 0)).sum())
    fn = int(((yp == 0) & (yt == 1)).sum())
    return tp, fp, tn, fn


def sensitivity_specificity(
    y_true: Sequence[int], y_pred: Sequence[int]
) -> Tuple[float, float]:
    """Return ``(sensitivity, specificity)``."""
    tp, fp, tn, fn = confusion_counts(y_true, y_pred)
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    return sens, spec


def f1_score(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Binary F1 score; returns NaN if both precision and recall are 0."""
    tp, fp, tn, fn = confusion_counts(y_true, y_pred)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    if precision == 0.0 and recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Curves
# ---------------------------------------------------------------------------


def roc_curve(
    y_true: Sequence[int], y_score: Sequence[float]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute ``(fpr, tpr, thresholds)`` with the (1, 1) sentinel appended.

    Sorted by descending score so plotting goes bottom-left -> top-right.
    """
    yt = _as_1d_int(y_true)
    ys = _as_1d_float(y_score)
    _validate_binary(yt)
    n_pos = int((yt == 1).sum())
    n_neg = int((yt == 0).sum())
    order = np.argsort(-ys, kind="mergesort")
    yt_sorted = yt[order]
    ys_sorted = ys[order]

    distinct = np.concatenate((ys_sorted[1:] != ys_sorted[:-1], [True]))
    tp_cum = np.cumsum(yt_sorted == 1)
    fp_cum = np.cumsum(yt_sorted == 0)
    idx = np.nonzero(distinct)[0]

    tps = tp_cum[idx]
    fps = fp_cum[idx]
    thresholds = ys_sorted[idx]

    # Prepend the (0, 0) origin at +inf threshold so curves start at (0,0).
    tps = np.concatenate(([0], tps))
    fps = np.concatenate(([0], fps))
    thresholds = np.concatenate(([np.inf], thresholds))

    fpr = fps / n_neg if n_neg else np.zeros_like(fps, dtype=np.float64)
    tpr = tps / n_pos if n_pos else np.zeros_like(tps, dtype=np.float64)
    return fpr.astype(np.float64), tpr.astype(np.float64), thresholds.astype(np.float64)


def precision_recall_curve(
    y_true: Sequence[int], y_score: Sequence[float]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute ``(precision, recall, thresholds)`` sorted by threshold desc.

    A terminal ``(precision=1, recall=0)`` point is appended (matches sklearn).
    """
    yt = _as_1d_int(y_true)
    ys = _as_1d_float(y_score)
    _validate_binary(yt)
    n_pos = int((yt == 1).sum())
    order = np.argsort(-ys, kind="mergesort")
    yt_sorted = yt[order]
    ys_sorted = ys[order]

    distinct = np.concatenate((ys_sorted[1:] != ys_sorted[:-1], [True]))
    tp_cum = np.cumsum(yt_sorted == 1)
    fp_cum = np.cumsum(yt_sorted == 0)
    idx = np.nonzero(distinct)[0]

    tps = tp_cum[idx].astype(np.float64)
    fps = fp_cum[idx].astype(np.float64)
    thresholds = ys_sorted[idx]

    precision = np.where((tps + fps) > 0, tps / (tps + fps), 1.0)
    recall = tps / n_pos if n_pos else np.zeros_like(tps, dtype=np.float64)

    # Terminal point
    precision = np.concatenate((precision, [1.0]))
    recall = np.concatenate((recall, [0.0]))
    thresholds = np.concatenate((thresholds, [np.inf]))
    return precision, recall, thresholds


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def calibration_curve(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    n_bins: int = 10,
    strategy: str = "uniform",
) -> List[Dict[str, float]]:
    """Reliability-diagram bins for a binary predictor.

    Each dict contains: ``bin_index``, ``n``, ``mean_predicted_prob``,
    ``fraction_positives``, ``lower``, ``upper``. Empty bins are omitted.

    Parameters
    ----------
    strategy:
        ``"uniform"`` -> equal-width bins on ``[0, 1]``.
        ``"quantile"`` -> bins with approximately equal sample counts.
    """
    yt = _as_1d_float(y_true)
    yp = _as_1d_float(y_prob)
    if yt.shape != yp.shape or yt.size == 0 or n_bins < 1:
        return []

    if strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    elif strategy == "quantile":
        qs = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.quantile(yp, qs)
        edges[0] = 0.0
        edges[-1] = 1.0
        # dedup edges in case of many tied probabilities
        edges = np.unique(edges)
        if edges.size < 2:
            edges = np.array([0.0, 1.0])
    else:
        raise ValueError(f"Unknown calibration strategy {strategy!r}")

    bins = np.clip(np.digitize(yp, edges[1:-1], right=False), 0, edges.size - 2)
    out: List[Dict[str, float]] = []
    for b in range(edges.size - 1):
        mask = bins == b
        n = int(mask.sum())
        if n == 0:
            continue
        out.append(
            {
                "bin_index": b,
                "n": n,
                "mean_predicted_prob": float(yp[mask].mean()),
                "fraction_positives": float(yt[mask].mean()),
                "lower": float(edges[b]),
                "upper": float(edges[b + 1]),
            }
        )
    return out


# ---------------------------------------------------------------------------
# High-level scorecard
# ---------------------------------------------------------------------------


def summarize_binary(
    y_true: Sequence[int],
    y_score: Sequence[float],
    threshold: float = 0.5,
) -> Dict[str, float]:
    """One-shot scorecard for step-3/4 reporting.

    Computes AUROC, AUPRC, balanced accuracy, F1, sensitivity, specificity,
    Brier, and log loss at the supplied decision threshold. NaNs propagate
    where a class is missing so downstream aggregations stay honest.
    """
    yt = _as_1d_int(y_true)
    ys = _as_1d_float(y_score)
    yp = (ys >= threshold).astype(np.int64)
    tp, fp, tn, fn = confusion_counts(yt, yp) if yt.size else (0, 0, 0, 0)
    sens, spec = sensitivity_specificity(yt, yp) if yt.size else (float("nan"), float("nan"))
    return {
        "n": int(yt.size),
        "n_positive": int((yt == 1).sum()),
        "n_negative": int((yt == 0).sum()),
        "threshold": float(threshold),
        "auroc": auroc(yt, ys),
        "auprc": auprc(yt, ys),
        "balanced_accuracy": balanced_accuracy(yt, yp),
        "f1": f1_score(yt, yp),
        "sensitivity": sens,
        "specificity": spec,
        "brier": brier_score(yt, ys),
        "log_loss": log_loss(yt, ys),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


# ---------------------------------------------------------------------------
# Uncertainty
# ---------------------------------------------------------------------------


def bootstrap_ci(
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    y_true: Sequence[int],
    y_score: Sequence[float],
    n_boot: int = 1000,
    seed: int = 2026,
    level: float = 0.95,
) -> Dict[str, float]:
    """Percentile bootstrap CI for any scalar metric.

    Resamples ``n_boot`` times with replacement, drops NaN resamples
    (caused by single-class bootstraps), and returns a dict with
    ``estimate``, ``mean``, ``lower``, ``upper``, ``level``, ``n_boot``,
    ``n_valid``. NaN-safe for rare-class cohorts.
    """
    yt = _as_1d_int(y_true)
    ys = _as_1d_float(y_score)
    if yt.size == 0:
        return {
            "estimate": float("nan"),
            "mean": float("nan"),
            "lower": float("nan"),
            "upper": float("nan"),
            "level": float(level),
            "n_boot": int(n_boot),
            "n_valid": 0,
        }
    rng = np.random.default_rng(seed)
    estimate = float(metric_fn(yt, ys))
    replicates = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, yt.size, size=yt.size)
        try:
            replicates[i] = float(metric_fn(yt[idx], ys[idx]))
        except Exception:
            replicates[i] = float("nan")
    valid = replicates[~np.isnan(replicates)]
    if valid.size == 0:
        return {
            "estimate": estimate,
            "mean": float("nan"),
            "lower": float("nan"),
            "upper": float("nan"),
            "level": float(level),
            "n_boot": int(n_boot),
            "n_valid": 0,
        }
    alpha = (1.0 - level) / 2.0
    lower = float(np.quantile(valid, alpha))
    upper = float(np.quantile(valid, 1.0 - alpha))
    return {
        "estimate": estimate,
        "mean": float(valid.mean()),
        "lower": lower,
        "upper": upper,
        "level": float(level),
        "n_boot": int(n_boot),
        "n_valid": int(valid.size),
    }
