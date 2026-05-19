"""CAGE Step 3 runner helpers (orchestration for grouped baselines).

This module keeps step3_grouped_baselines.py focused on argparse/CLI while
the numerical pipeline lives here:

* :func:`load_step2_artifacts` - reads the normalized matrix, master table,
  and grouped CV folds emitted by step 2.
* :func:`align_arrays` - aligns the three tables on the common sample order
  and returns numpy arrays ready for modelling.
* :func:`run_grouped_cv` - trains each requested model on every outer fold,
  collects out-of-fold probabilities, and records per-fold metrics.
* :func:`aggregate_feature_importances` - averages per-fold importances and
  produces a ranked table.
* :func:`compute_subgroup_metrics` - stratifies OOF predictions by the
  environment columns (env_smoking, env_sex, env_histology, env_country,
  env_stage) and reports per-subgroup AUROC / AUPRC / BAC / Brier.
* :func:`generate_step3_figures` - renders ROC, PR, calibration, and a
  cross-model comparison bar chart when matplotlib is available;
  otherwise returns a list of skipped names.
"""

from __future__ import annotations

import csv
import logging
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from . import deep_model_utils as dm
from . import metrics as mx
from . import preprocess_esca as pp
from .baseline_models import BaselineModel, build_model

logger = logging.getLogger("cage.step3.runner")

__all__ = [
    "load_step2_artifacts",
    "align_arrays",
    "run_grouped_cv",
    "aggregate_feature_importances",
    "compute_subgroup_metrics",
    "generate_step3_figures",
]


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def _read_dict_rows(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with open(path, "r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fields = list(reader.fieldnames or [])
        rows = [dict(r) for r in reader]
    return fields, rows


def load_step2_artifacts(step2_dir: Path) -> Dict[str, Any]:
    """Load the step-2 deliverables required by step 3.

    Expects the directory to contain:
        master_samples_primary.csv
        normalized_primary_matrix.csv   (samples x genes)
        grouped_outer_folds.csv
    """
    step2_dir = Path(step2_dir)
    required = [
        "master_samples_primary.csv",
        "normalized_primary_matrix.csv",
        "grouped_outer_folds.csv",
    ]
    missing = [n for n in required if not (step2_dir / n).exists()]
    if missing:
        raise FileNotFoundError(
            f"Step-2 outputs missing in {step2_dir}: {missing}. "
            "Run `python -m cage.step2_build_cohort` first."
        )

    master_fields, master_rows = _read_dict_rows(step2_dir / "master_samples_primary.csv")
    fold_fields, fold_rows = _read_dict_rows(step2_dir / "grouped_outer_folds.csv")

    # Normalized matrix is samples x genes; first column = sample_barcode
    sample_barcodes: list[str] = []
    gene_names: list[str] = []
    rows: list[list[float]] = []
    with open(step2_dir / "normalized_primary_matrix.csv", "r", newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        gene_names = header[1:]
        for row in reader:
            sample_barcodes.append(row[0])
            rows.append([float(v) if v not in ("", "NA", "NaN", "nan") else 0.0
                         for v in row[1:]])
    X = np.asarray(rows, dtype=np.float64)
    logger.info(
        "Loaded step-2 artifacts: %d samples, %d genes, %d master cols, %d fold cols",
        X.shape[0], X.shape[1], len(master_fields), len(fold_fields),
    )
    return {
        "master_fields": master_fields,
        "master_rows": master_rows,
        "fold_fields": fold_fields,
        "fold_rows": fold_rows,
        "X": X,
        "sample_barcodes": sample_barcodes,
        "gene_names": gene_names,
    }


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------


def align_arrays(artifacts: Mapping[str, Any]) -> Dict[str, Any]:
    """Align master, folds, and expression on a shared sample ordering.

    The normalized matrix order is treated as authoritative; master and
    fold tables are indexed by ``sample_barcode``. Returns numpy arrays:

        X, shape (n, p) float64
        y, shape (n,) int64              1 = Tumor, 0 = Normal
        groups, shape (n,) object        patient_barcode for leakage checks
        outer_fold, shape (n,) int64
        master_rows (aligned list[dict])
        gene_names (list[str])
        sample_barcodes (list[str])
    """
    sample_barcodes: list[str] = list(artifacts["sample_barcodes"])
    master_rows: list[Dict[str, str]] = list(artifacts["master_rows"])
    fold_rows: list[Dict[str, str]] = list(artifacts["fold_rows"])
    X: np.ndarray = np.asarray(artifacts["X"], dtype=np.float64)
    gene_names: list[str] = list(artifacts["gene_names"])

    master_by_bc = {r["sample_barcode"]: r for r in master_rows}
    fold_by_bc = {r["sample_barcode"]: r for r in fold_rows}

    missing_master = [bc for bc in sample_barcodes if bc not in master_by_bc]
    missing_folds = [bc for bc in sample_barcodes if bc not in fold_by_bc]
    if missing_master:
        raise KeyError(f"{len(missing_master)} sample barcodes missing from master table; first: {missing_master[:3]}")
    if missing_folds:
        raise KeyError(f"{len(missing_folds)} sample barcodes missing from folds table; first: {missing_folds[:3]}")

    aligned_master: list[Dict[str, str]] = []
    aligned_folds: list[Dict[str, str]] = []
    for bc in sample_barcodes:
        aligned_master.append(master_by_bc[bc])
        aligned_folds.append(fold_by_bc[bc])

    y = np.array([int(r["label_int"]) for r in aligned_master], dtype=np.int64)
    groups = np.array([r["patient_barcode"] for r in aligned_master], dtype=object)
    outer_fold = np.array([int(r["outer_fold"]) for r in aligned_folds], dtype=np.int64)

    return {
        "X": X,
        "y": y,
        "groups": groups,
        "outer_fold": outer_fold,
        "master_rows": aligned_master,
        "fold_rows": aligned_folds,
        "gene_names": gene_names,
        "sample_barcodes": sample_barcodes,
    }


# ---------------------------------------------------------------------------
# Leakage checks
# ---------------------------------------------------------------------------


def assert_no_patient_leakage(groups: np.ndarray, outer_fold: np.ndarray) -> None:
    """Raise if any patient appears in more than one outer fold."""
    by_patient: dict[str, set[int]] = {}
    for g, f in zip(groups, outer_fold):
        by_patient.setdefault(str(g), set()).add(int(f))
    leaked = {g: sorted(fs) for g, fs in by_patient.items() if len(fs) > 1}
    if leaked:
        raise RuntimeError(
            f"Patient-grouped CV violated: {len(leaked)} patients appear in >1 outer fold: "
            f"{dict(list(leaked.items())[:3])}"
        )


# ---------------------------------------------------------------------------
# CV loop
# ---------------------------------------------------------------------------


def run_grouped_cv(
    X: np.ndarray,
    y: np.ndarray,
    outer_fold: np.ndarray,
    groups: np.ndarray,
    model_names: Sequence[str],
    *,
    seed: int = 2026,
    decision_threshold: float = 0.5,
    model_overrides: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """Fit each model over patient-grouped outer folds and collect OOF preds.

    Returns a dict with:
        "oof_probs"        : {model_name -> (n,) float64 OOF probabilities}
        "per_fold_metrics" : list of dicts (one row per model/fold)
        "feature_importances": {model_name -> (p,) float64 mean importance}
        "model_configs"    : {model_name -> config dict}
        "folds_used"       : sorted list of fold ids actually encountered
    """
    assert_no_patient_leakage(groups, outer_fold)
    overrides = dict(model_overrides or {})
    fold_ids = sorted(set(int(f) for f in outer_fold.tolist()))
    n_samples, n_features = X.shape

    oof_probs: Dict[str, np.ndarray] = {
        name: np.full(n_samples, np.nan, dtype=np.float64) for name in model_names
    }
    train_prediction_rows: list[Dict[str, Any]] = []
    per_fold_rows: list[Dict[str, Any]] = []
    importances_accum: Dict[str, np.ndarray] = {
        name: np.zeros(n_features, dtype=np.float64) for name in model_names
    }
    importances_count: Dict[str, int] = {name: 0 for name in model_names}
    model_configs: Dict[str, Dict[str, Any]] = {}

    # Per-fold scaler statistics recorded for leakage audit manifest.
    fold_scaler_manifest: list[Dict[str, Any]] = []

    for k, fold_id in enumerate(fold_ids):
        test_mask = outer_fold == fold_id
        train_mask = ~test_mask
        X_tr_raw, X_te_raw = X[train_mask], X[test_mask]
        y_tr, y_te = y[train_mask], y[test_mask]
        train_ids = np.where(train_mask)[0]
        n_tr, n_te = X_tr_raw.shape[0], X_te_raw.shape[0]
        n_pos_tr = int((y_tr == 1).sum())
        n_pos_te = int((y_te == 1).sum())

        # ── Fold-local standardisation (LEAKAGE PREVENTION) ─────────────────
        # Fit mean/std exclusively on the TRAINING partition; apply to both.
        # This ensures no test-fold expression statistics influence training.
        mean_fold, std_fold = dm.standardize_fit(X_tr_raw)
        X_tr = dm.standardize_apply(X_tr_raw, mean_fold, std_fold)
        X_te = dm.standardize_apply(X_te_raw, mean_fold, std_fold)
        fold_scaler_manifest.append({
            "outer_fold": int(fold_id),
            "n_train": int(n_tr),
            "n_test": int(n_te),
            "scaler_mean_global": float(np.mean(mean_fold)),
            "scaler_std_global": float(np.mean(std_fold)),
        })
        # ────────────────────────────────────────────────────────────────────

        logger.info(
            "Outer fold %d/%d (id=%d): train=%d (pos=%d), test=%d (pos=%d) "
            "| scaler fit on %d training samples",
            k + 1, len(fold_ids), fold_id, n_tr, n_pos_tr, n_te, n_pos_te, n_tr,
        )

        for name in model_names:
            model_kwargs = dict(overrides.get(name, {}))
            # Vary seed per fold for RF, keep logistic/elasticnet deterministic
            fold_seed = seed + fold_id * 1000
            model = build_model(name, seed=fold_seed, **model_kwargs)
            model.fit(X_tr, y_tr)
            train_probs = model.predict_proba(X_tr)
            probs = model.predict_proba(X_te)
            oof_probs[name][test_mask] = probs

            for sample_idx, y_i, p_i in zip(train_ids, y_tr, train_probs):
                train_prediction_rows.append({
                    "model": name,
                    "outer_fold": int(fold_id),
                    "sample_index": int(sample_idx),
                    "y_true": int(y_i),
                    "train_prob": float(p_i),
                })

            # Per-fold metrics (NaN-safe: single-class test folds allowed)
            y_pred = (probs >= decision_threshold).astype(np.int64)
            row = {
                "model": name,
                "outer_fold": int(fold_id),
                "n_train": int(n_tr),
                "n_test": int(n_te),
                "n_pos_train": int(n_pos_tr),
                "n_pos_test": int(n_pos_te),
                "auroc": mx.auroc(y_te, probs),
                "auprc": mx.auprc(y_te, probs),
                "balanced_accuracy": mx.balanced_accuracy(y_te, y_pred),
                "f1": mx.f1_score(y_te, y_pred),
                "brier": mx.brier_score(y_te, probs),
                "log_loss": mx.log_loss(y_te, probs),
            }
            sens, spec = mx.sensitivity_specificity(y_te, y_pred)
            row["sensitivity"] = sens
            row["specificity"] = spec
            per_fold_rows.append(row)

            # Accumulate feature importances
            try:
                imp = model.feature_importances()
                importances_accum[name] += imp
                importances_count[name] += 1
            except Exception as exc:  # pragma: no cover
                logger.warning("[%s] fold %d: importance unavailable (%s)", name, fold_id, exc)

            model_configs.setdefault(name, model.get_config())

    # Mean feature importances
    feature_importances: Dict[str, np.ndarray] = {}
    for name in model_names:
        c = max(importances_count[name], 1)
        feature_importances[name] = importances_accum[name] / c

    return {
        "oof_probs": oof_probs,
        "train_predictions": train_prediction_rows,
        "per_fold_metrics": per_fold_rows,
        "feature_importances": feature_importances,
        "model_configs": model_configs,
        "folds_used": fold_ids,
        "decision_threshold": float(decision_threshold),
        "fold_scaler_manifest": fold_scaler_manifest,  # leakage audit trail
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate_feature_importances(
    gene_names: Sequence[str],
    importances: Mapping[str, np.ndarray],
    top_n: int = 50,
) -> List[Dict[str, Any]]:
    """Return a ranked long-form table of the top features per model."""
    rows: list[Dict[str, Any]] = []
    for name, imp in importances.items():
        order = np.argsort(-imp, kind="mergesort")
        rank = 0
        for idx in order[:top_n]:
            rank += 1
            rows.append(
                {
                    "model": name,
                    "rank": rank,
                    "gene": gene_names[int(idx)],
                    "importance": float(imp[int(idx)]),
                }
            )
    return rows


def summarize_oof_metrics(
    y: np.ndarray,
    oof_probs: Mapping[str, np.ndarray],
    per_fold_rows: Sequence[Mapping[str, Any]],
    *,
    decision_threshold: float = 0.5,
    bootstrap_ci_n: int = 500,
    seed: int = 2026,
) -> List[Dict[str, Any]]:
    """Collapse per-fold + overall OOF metrics into a summary table.

    For each model we emit:
      * one "overall_oof" row (single pooled AUROC/AUPRC/... with bootstrap CI)
      * one "mean_of_folds" row (mean +- std of per-fold metrics)
    """
    rows: list[Dict[str, Any]] = []
    models = sorted(set(str(r["model"]) for r in per_fold_rows))

    for name in models:
        probs = oof_probs[name]
        valid = ~np.isnan(probs)
        y_valid = y[valid]
        p_valid = probs[valid]
        y_pred = (p_valid >= decision_threshold).astype(np.int64)

        auc_ci = mx.bootstrap_ci(mx.auroc, y_valid, p_valid, n_boot=bootstrap_ci_n, seed=seed)
        ap_ci = mx.bootstrap_ci(mx.auprc, y_valid, p_valid, n_boot=bootstrap_ci_n, seed=seed + 1)

        overall = {
            "model": name,
            "aggregation": "overall_oof",
            "n": int(valid.sum()),
            "n_positive": int((y_valid == 1).sum()),
            "n_negative": int((y_valid == 0).sum()),
            "threshold": float(decision_threshold),
            "auroc": mx.auroc(y_valid, p_valid),
            "auroc_ci_lower": auc_ci["lower"],
            "auroc_ci_upper": auc_ci["upper"],
            "auprc": mx.auprc(y_valid, p_valid),
            "auprc_ci_lower": ap_ci["lower"],
            "auprc_ci_upper": ap_ci["upper"],
            "balanced_accuracy": mx.balanced_accuracy(y_valid, y_pred),
            "f1": mx.f1_score(y_valid, y_pred),
            "brier": mx.brier_score(y_valid, p_valid),
            "log_loss": mx.log_loss(y_valid, p_valid),
        }
        sens, spec = mx.sensitivity_specificity(y_valid, y_pred)
        overall["sensitivity"] = sens
        overall["specificity"] = spec
        rows.append(overall)

        # Mean of folds
        fold_rows = [r for r in per_fold_rows if r["model"] == name]
        metric_keys = ["auroc", "auprc", "balanced_accuracy", "f1",
                       "brier", "log_loss", "sensitivity", "specificity"]
        mean_row: Dict[str, Any] = {
            "model": name,
            "aggregation": "mean_of_folds",
            "n_folds": len(fold_rows),
        }
        for k in metric_keys:
            vals = [float(r.get(k, float("nan"))) for r in fold_rows]
            vals_arr = np.asarray(vals, dtype=np.float64)
            finite = vals_arr[np.isfinite(vals_arr)]
            if finite.size:
                mean_row[k + "_mean"] = float(finite.mean())
                mean_row[k + "_std"] = float(finite.std(ddof=0))
                mean_row[k + "_n_valid_folds"] = int(finite.size)
            else:
                mean_row[k + "_mean"] = float("nan")
                mean_row[k + "_std"] = float("nan")
                mean_row[k + "_n_valid_folds"] = 0
        rows.append(mean_row)

    return rows


# ---------------------------------------------------------------------------
# Subgroup sensitivity
# ---------------------------------------------------------------------------


def compute_subgroup_metrics(
    master_rows: Sequence[Mapping[str, str]],
    y: np.ndarray,
    oof_probs: Mapping[str, np.ndarray],
    env_names: Sequence[str],
    *,
    decision_threshold: float = 0.5,
) -> List[Dict[str, Any]]:
    """Return per-model, per-environment-stratum metrics for OOF predictions."""
    rows: list[Dict[str, Any]] = []
    for name, probs in oof_probs.items():
        for env in env_names:
            col = f"env_{env}"
            if master_rows and col not in master_rows[0]:
                continue
            for level in ("0", "1"):
                mask = np.array(
                    [r.get(col, "") == level for r in master_rows], dtype=bool
                )
                mask &= ~np.isnan(probs)
                if mask.sum() == 0:
                    continue
                y_sub = y[mask]
                p_sub = probs[mask]
                y_pred = (p_sub >= decision_threshold).astype(np.int64)
                sens, spec = mx.sensitivity_specificity(y_sub, y_pred)
                rows.append(
                    {
                        "model": name,
                        "environment": env,
                        "level": level,
                        "n": int(mask.sum()),
                        "n_positive": int((y_sub == 1).sum()),
                        "n_negative": int((y_sub == 0).sum()),
                        "auroc": mx.auroc(y_sub, p_sub),
                        "auprc": mx.auprc(y_sub, p_sub),
                        "balanced_accuracy": mx.balanced_accuracy(y_sub, y_pred),
                        "brier": mx.brier_score(y_sub, p_sub),
                        "sensitivity": sens,
                        "specificity": spec,
                    }
                )
    return rows


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def compute_calibration(
    y: np.ndarray,
    oof_probs: Mapping[str, np.ndarray],
    *,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> List[Dict[str, Any]]:
    """Flattened calibration table for all requested models."""
    rows: list[Dict[str, Any]] = []
    for name, probs in oof_probs.items():
        valid = ~np.isnan(probs)
        if valid.sum() == 0:
            continue
        bins = mx.calibration_curve(y[valid], probs[valid], n_bins=n_bins, strategy=strategy)
        for b in bins:
            rows.append({"model": name, **b})
    return rows


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _has_matplotlib() -> bool:
    try:
        import matplotlib  # noqa: F401
        return True
    except ImportError:
        return False


def generate_step3_figures(
    *,
    y: np.ndarray,
    oof_probs: Mapping[str, np.ndarray],
    train_predictions: Optional[Sequence[Mapping[str, Any]]] = None,
    per_fold_metrics: Sequence[Mapping[str, Any]],
    output_dir: Path,
    style: Any = None,
    formats: Sequence[str] = ("svg",),
    run_calibration: bool = False,
    feature_importances: Optional[Mapping[str, np.ndarray]] = None,
    gene_names: Optional[Sequence[str]] = None,
    subgroup_rows: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Tuple[List[str], List[Tuple[str, str]]]:
    """Render step-3 publication figures if matplotlib is available."""
    generated: list[str] = []
    skipped: list[tuple[str, str]] = []

    fig_names = [
        "fig_baseline_roc",
        "fig_baseline_pr",
        "fig_baseline_comparison",
        "fig_E2_per_fold_auroc",
    ]
    if run_calibration:
        fig_names.append("fig_baseline_calibration")
    if feature_importances is not None and gene_names is not None:
        fig_names.append("fig_E4_feature_importance")
    if subgroup_rows is not None:
        fig_names.append("fig_E5_subgroup_heatmap")

    if not _has_matplotlib():
        for n in fig_names:
            skipped.append((n, "matplotlib not installed"))
        return generated, skipped

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from .publication_style import (
            apply_style, save_figure, categorical_colors,
        )
        if style is not None:
            apply_style(style)

        fig_dir = Path(output_dir) / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        model_names = sorted(oof_probs.keys())
        palette = categorical_colors(max(len(model_names), 3))

        # ---- Figure: ROC curves ----
        try:
            fig, ax = plt.subplots(figsize=(3.8, 3.2))
            for i, name in enumerate(model_names):
                probs = np.asarray(oof_probs[name])
                valid = ~np.isnan(probs)
                fpr, tpr, _ = mx.roc_curve(y[valid], probs[valid])
                auc = mx.auroc(y[valid], probs[valid])
                ax.plot(fpr, tpr, color=palette[i], linewidth=1.6,
                        label=f"{name} (AUROC={auc:.3f})")
            ax.plot([0, 1], [0, 1], linestyle="--", color="#999999", linewidth=1.0)
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.02, 1.02)
            ax.set_title("Out-of-fold ROC (baselines)")
            ax.legend(loc="lower right", fontsize=8)
            paths = save_figure(fig, fig_dir / "fig_baseline_roc",
                                style=style, formats=formats)
            if paths:
                generated.append("fig_baseline_roc")
        except Exception as exc:
            skipped.append(("fig_baseline_roc", str(exc)))

        # ---- Figure: Precision-Recall curves ----
        try:
            fig, ax = plt.subplots(figsize=(3.8, 3.2))
            for i, name in enumerate(model_names):
                probs = np.asarray(oof_probs[name])
                valid = ~np.isnan(probs)
                prec, rec, _ = mx.precision_recall_curve(y[valid], probs[valid])
                ap = mx.auprc(y[valid], probs[valid])
                ax.plot(rec, prec, color=palette[i], linewidth=1.6,
                        label=f"{name} (AP={ap:.3f})")
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(0.0, 1.02)
            ax.set_title("Out-of-fold PR (baselines)")
            ax.legend(loc="lower left", fontsize=8)
            paths = save_figure(fig, fig_dir / "fig_baseline_pr",
                                style=style, formats=formats)
            if paths:
                generated.append("fig_baseline_pr")
        except Exception as exc:
            skipped.append(("fig_baseline_pr", str(exc)))

        # ---- Figure: Model-comparison bar chart ----
        try:
            metrics_to_plot = ["auroc", "auprc", "balanced_accuracy", "brier"]
            per_fold = [dict(r) for r in per_fold_metrics]
            fig, axes = plt.subplots(1, len(metrics_to_plot),
                                     figsize=(2.4 * len(metrics_to_plot), 3.0))
            if len(metrics_to_plot) == 1:
                axes = [axes]
            xs = np.arange(len(model_names))
            for ax, metric in zip(axes, metrics_to_plot):
                means = []
                stds = []
                for name in model_names:
                    vals = [float(r.get(metric, float("nan")))
                            for r in per_fold if r["model"] == name]
                    vals = np.asarray(vals, dtype=np.float64)
                    vals = vals[np.isfinite(vals)]
                    means.append(float(vals.mean()) if vals.size else float("nan"))
                    stds.append(float(vals.std(ddof=0)) if vals.size else 0.0)
                ax.bar(xs, means, yerr=stds, color=palette[:len(model_names)],
                       edgecolor="black", linewidth=0.6, capsize=3)
                ax.set_xticks(xs)
                ax.set_xticklabels(model_names, rotation=30, ha="right", fontsize=8)
                ax.set_title(metric, fontsize=10)
                if metric != "brier":
                    ax.set_ylim(0.0, 1.05)
            fig.suptitle("Baseline CV metrics (mean +- std)", fontsize=11, fontweight="bold")
            fig.tight_layout(rect=[0, 0, 1, 0.94])
            paths = save_figure(fig, fig_dir / "fig_baseline_comparison",
                                style=style, formats=formats)
            if paths:
                generated.append("fig_baseline_comparison")
        except Exception as exc:
            skipped.append(("fig_baseline_comparison", str(exc)))

        # ---- Figure: Calibration curve ----
        if run_calibration:
            try:
                fig, ax = plt.subplots(figsize=(3.5, 3.2))
                ax.plot([0, 1], [0, 1], linestyle="--", color="#999999", linewidth=1.0,
                        label="Perfect calibration")
                for i, name in enumerate(model_names):
                    probs = np.asarray(oof_probs[name])
                    valid = ~np.isnan(probs)
                    bins = mx.calibration_curve(y[valid], probs[valid], n_bins=8)
                    if not bins:
                        continue
                    xs_b = [b["mean_predicted_prob"] for b in bins]
                    ys_b = [b["fraction_positives"] for b in bins]
                    ax.plot(xs_b, ys_b, marker="o", color=palette[i],
                            linewidth=1.4, markersize=4, label=name)
                ax.set_xlabel("Mean predicted probability")
                ax.set_ylabel("Fraction of positives")
                ax.set_xlim(0.0, 1.02)
                ax.set_ylim(0.0, 1.02)
                ax.set_title("OOF Calibration")
                ax.legend(loc="lower right", fontsize=8)
                paths = save_figure(fig, fig_dir / "fig_baseline_calibration",
                                    style=style, formats=formats)
                if paths:
                    generated.append("fig_baseline_calibration")
            except Exception as exc:
                skipped.append(("fig_baseline_calibration", str(exc)))

        # ---- Figure E2: per-fold AUROC strip plot ----
        try:
            model_names_e2 = sorted(set(r["model"] for r in per_fold_metrics))
            palette_e2 = categorical_colors(max(len(model_names_e2), 3))
            fig, ax = plt.subplots(figsize=(max(5, len(model_names_e2) * 1.5 + 1), 4.5))
            rng_j = np.random.default_rng(42)
            for mi, model in enumerate(model_names_e2):
                aurocs = [float(r["auroc"]) for r in per_fold_metrics
                          if r["model"] == model and r.get("auroc") not in (None, "", "nan")]
                aurocs_arr = np.asarray(aurocs, dtype=np.float64)
                aurocs_arr = aurocs_arr[np.isfinite(aurocs_arr)]
                if aurocs_arr.size == 0:
                    continue
                jitter = (rng_j.random(len(aurocs_arr)) - 0.5) * 0.25
                ax.scatter([mi] * len(aurocs_arr) + jitter, aurocs_arr,
                           color=palette_e2[mi], alpha=0.85, s=40, zorder=3)
                med = float(np.median(aurocs_arr))
                ax.plot([mi - 0.22, mi + 0.22], [med, med],
                        color=palette_e2[mi], linewidth=2.5, zorder=4)
                ax.text(mi, med + 0.01, f"{med:.3f}", ha="center", fontsize=8, fontweight="bold")
            ax.set_xticks(range(len(model_names_e2)))
            ax.set_xticklabels([m.capitalize() for m in model_names_e2], fontsize=9)
            ax.set_ylabel("AUROC")
            ax.set_ylim(0, 1.12)
            ax.axhline(0.5, color="#999999", linewidth=0.8, linestyle="--", alpha=0.5)
            ax.set_title("Per-Fold AUROC — Baseline Models")
            ax.text(0.98, 0.02, "Bar = median | Points = individual folds",
                    transform=ax.transAxes, ha="right", fontsize=7, color="#aaaaaa")
            fig.tight_layout()
            paths = save_figure(fig, fig_dir / "fig_E2_per_fold_auroc",
                                style=style, formats=formats)
            if paths:
                generated.append("fig_E2_per_fold_auroc")
        except Exception as exc:
            skipped.append(("fig_E2_per_fold_auroc", str(exc)))

        # ---- Figure E4: feature importance horizontal bars ----
        if feature_importances is not None and gene_names is not None:
            try:
                gene_list = list(gene_names)
                model_names_e4 = sorted(feature_importances.keys())
                n_models_e4 = max(len(model_names_e4), 1)
                top_n = 20
                fig, axes = plt.subplots(
                    1, n_models_e4,
                    figsize=(n_models_e4 * 4.5, max(6, top_n * 0.35 + 2)),
                    sharey=False,
                )
                if n_models_e4 == 1:
                    axes = [axes]
                palette_e4 = categorical_colors(n_models_e4)
                for ax, model in zip(axes, model_names_e4):
                    imp = np.asarray(feature_importances[model], dtype=np.float64)
                    top_idx = np.argsort(imp)[-top_n:][::-1]
                    top_genes = [gene_list[i] for i in top_idx]
                    top_imp   = imp[top_idx]
                    y_pos = list(range(len(top_genes)))
                    ax.barh(y_pos, top_imp[::-1], color=palette_e4[model_names_e4.index(model)],
                            edgecolor="white", linewidth=0.3)
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels(top_genes[::-1], fontsize=7)
                    ax.set_xlabel("Importance", fontsize=8)
                    ax.set_title(model.capitalize(), fontsize=10)
                fig.suptitle("Top-20 Feature Importance — Baseline Models",
                             fontsize=11, fontweight="bold")
                fig.tight_layout()
                paths = save_figure(fig, fig_dir / "fig_E4_feature_importance",
                                    style=style, formats=formats)
                if paths:
                    generated.append("fig_E4_feature_importance")
            except Exception as exc:
                skipped.append(("fig_E4_feature_importance", str(exc)))

        # ---- Figure E5: subgroup AUROC heatmap ----
        if subgroup_rows is not None:
            try:
                subg_rows = list(subgroup_rows)
                model_names_e5 = sorted(set(str(r["model"]) for r in subg_rows))
                env_combos: dict[str, dict[str, float]] = {}
                for r in subg_rows:
                    lbl = f"{r.get('environment', '')}={r.get('level', '')}"
                    env_combos.setdefault(lbl, {})
                    auroc_v = r.get("auroc")
                    try:
                        env_combos[lbl][str(r["model"])] = float(auroc_v)
                    except (TypeError, ValueError):
                        pass
                combo_labels = sorted(env_combos)
                mat = np.full((len(model_names_e5), len(combo_labels)), np.nan)
                for mi, model in enumerate(model_names_e5):
                    for ci, lbl in enumerate(combo_labels):
                        v = env_combos.get(lbl, {}).get(model)
                        if v is not None:
                            mat[mi, ci] = v
                fig, ax = plt.subplots(
                    figsize=(max(8, len(combo_labels) * 0.7 + 2),
                             max(3, len(model_names_e5) * 0.8 + 2)),
                )
                im = ax.imshow(mat, aspect="auto", cmap="cage_sequential", vmin=0.4, vmax=1.0)
                fig.colorbar(im, ax=ax, label="AUROC")
                ax.set_xticks(range(len(combo_labels)))
                ax.set_xticklabels(combo_labels, rotation=40, ha="right", fontsize=7)
                ax.set_yticks(range(len(model_names_e5)))
                ax.set_yticklabels([m.capitalize() for m in model_names_e5], fontsize=9)
                for i in range(len(model_names_e5)):
                    for j in range(len(combo_labels)):
                        v = mat[i, j]
                        if np.isfinite(v):
                            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                                    color="white" if v >= 0.80 else "black")
                ax.set_title("Subgroup AUROC — Baseline Models", fontsize=11, fontweight="bold")
                fig.tight_layout()
                paths = save_figure(fig, fig_dir / "fig_E5_subgroup_heatmap",
                                    style=style, formats=formats)
                if paths:
                    generated.append("fig_E5_subgroup_heatmap")
            except Exception as exc:
                skipped.append(("fig_E5_subgroup_heatmap", str(exc)))

    except Exception as exc:  # pragma: no cover - environment-specific
        for n in fig_names:
            skipped.append((n, str(exc)))

    return generated, skipped
