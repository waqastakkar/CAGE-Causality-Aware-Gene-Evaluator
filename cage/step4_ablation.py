"""CAGE Step 4: Ablation study framework.

Evaluates the contribution of each architectural component by running
the full nested-CV pipeline with specific components disabled or replaced.
Each ablation variant produces per-fold and OOF metrics that can be directly
compared with the full model via the CDPS ranking.

Ablation variants
-----------------
full_model          The default configuration (reference).
no_adversary        lambda_adv=0 (no confounder removal).
no_invariance       lambda_inv=0 (no environment-invariance penalty).
no_gate             lambda_sparsity=0 + gate weights set to 1.0 after training
                    (effectively no learned sparsity; gate still present
                    but its penalty term is removed).
no_decoder          use_decoder=False (no reconstruction loss).
hard_concrete_gate  sparsity_type='hard-concrete' instead of 'l1'.
no_adversary_no_inv lambda_adv=0 AND lambda_inv=0 (no invariance/deconf).

Outputs (written by the caller)
--------------------------------
ablation_summary_metrics.csv     mean-of-folds AUROC/AUPRC/BAC/Brier per variant
ablation_per_fold_metrics.csv    per-fold metrics for every variant
ablation_cdps_rank_stability.csv Spearman rank correlation of CDPS vs full model

Usage
-----
from cage.step4_ablation import ABLATION_VARIANTS, run_ablation_study
results = run_ablation_study(
    X=X, y=y, outer_fold=outer_fold, groups=groups,
    fold_rows=fold_rows, master_rows=master_rows,
    base_config=best_config,  # from hptuner or default
    variants=ABLATION_VARIANTS,
    seed=2026,
)
"""

from __future__ import annotations

import logging
import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from . import deep_model_utils as dm
from . import metrics as mx
from . import step3_runner as step3
from .step4_runner import (
    TrainingConfig,
    SparseInvariantModel,
    train_one_fold,
    encode_confounder_column,
    _extract_inner_folds,
    _pick_inner_split,
    _balanced_sample_weights,
)

logger = logging.getLogger("cage.step4.ablation")

__all__ = [
    "AblationVariant",
    "ABLATION_VARIANTS",
    "run_single_variant",
    "run_ablation_study",
    "compute_ablation_summary",
    "compute_rank_correlations",
]


# ---------------------------------------------------------------------------
# Ablation variant definitions
# ---------------------------------------------------------------------------

@dataclass
class AblationVariant:
    """A named override of TrainingConfig fields."""
    name: str
    description: str
    overrides: Dict[str, Any]

    def apply(self, base: TrainingConfig) -> TrainingConfig:
        """Return a new TrainingConfig with this variant's overrides applied."""
        d = base.as_dict()
        d.update(self.overrides)
        return TrainingConfig(**d)


ABLATION_VARIANTS: List[AblationVariant] = [
    AblationVariant(
        name="full_model",
        description="Full model with all components (reference).",
        overrides={},
    ),
    AblationVariant(
        name="no_adversary",
        description="Adversarial confounder removal disabled (lambda_adv=0).",
        overrides={"lambda_adv": 0.0},
    ),
    AblationVariant(
        name="no_invariance",
        description="Environment-invariance penalty disabled (lambda_inv=0).",
        overrides={"lambda_inv": 0.0},
    ),
    AblationVariant(
        name="no_gate",
        description="Feature gate sparsity penalty removed (lambda_sparsity=0).",
        overrides={"lambda_sparsity": 0.0},
    ),
    AblationVariant(
        name="no_decoder",
        description="Reconstruction decoder disabled (use_decoder=False).",
        overrides={"use_decoder": False, "lambda_recon": 0.0},
    ),
    AblationVariant(
        name="hard_concrete_gate",
        description="Hard-concrete L0 gate instead of L1 sigmoid gate.",
        overrides={"sparsity_type": "hard-concrete"},
    ),
    AblationVariant(
        name="no_adversary_no_inv",
        description="Both adversary and invariance disabled (baseline encoder).",
        overrides={"lambda_adv": 0.0, "lambda_inv": 0.0},
    ),
]


# ---------------------------------------------------------------------------
# Single variant run
# ---------------------------------------------------------------------------

def _run_ablation_cv(
    *,
    X: np.ndarray,
    y: np.ndarray,
    outer_fold: np.ndarray,
    groups: np.ndarray,
    fold_rows: List[Dict[str, Any]],
    master_rows: List[Dict[str, Any]],
    confounder_column: str,
    environment_column: str,
    config: TrainingConfig,
) -> Dict[str, Any]:
    """Lightweight outer-CV loop for ablation variants (no checkpoint I/O).

    Mirrors the logic in run_step4_cv but skips checkpoint writing and
    latent-embedding collection for speed. Returns oof_probs, gate weights,
    and per-fold metrics.
    """
    step3.assert_no_patient_leakage(groups, outer_fold)
    n_samples, n_features = X.shape
    fold_ids = sorted(set(int(f) for f in outer_fold))
    n_outer = max(fold_ids) + 1
    inner_folds_per_outer = _extract_inner_folds(fold_rows, n_outer)

    conf_labels, n_conf_classes, _ = encode_confounder_column(master_rows, confounder_column)
    env_labels, n_env_classes, _ = encode_confounder_column(master_rows, environment_column)
    eff_n_conf = max(n_conf_classes, 2)
    if n_conf_classes < 2:
        config = TrainingConfig(**{**config.as_dict(), "lambda_adv": 0.0})
    if n_env_classes < 2:
        config = TrainingConfig(**{**config.as_dict(), "lambda_inv": 0.0})

    oof_probs = np.full(n_samples, np.nan, dtype=np.float64)
    gate_weights: List[np.ndarray] = []
    per_fold_metrics: List[Dict[str, Any]] = []

    for k_idx, fold_id in enumerate(fold_ids):
        test_mask = outer_fold == fold_id
        train_mask = ~test_mask

        val_mask = _pick_inner_split(outer_fold, inner_folds_per_outer, fold_id) & train_mask
        if val_mask.sum() < 2:
            rng_fb = np.random.default_rng(config.seed + fold_id * 17)
            tr_idx = np.where(train_mask)[0]
            val_size = max(2, int(0.2 * tr_idx.size))
            val_pick = rng_fb.choice(tr_idx, size=val_size, replace=False)
            val_mask = np.zeros(n_samples, dtype=bool)
            val_mask[val_pick] = True

        tr_core = train_mask & ~val_mask

        X_tr_raw, X_val_raw, X_te_raw = X[tr_core], X[val_mask], X[test_mask]
        y_tr, y_val, y_te = y[tr_core], y[val_mask], y[test_mask]
        conf_tr = conf_labels[tr_core]
        env_tr = env_labels[tr_core]

        mean_f, std_f = dm.standardize_fit(X_tr_raw)
        X_tr_z = dm.standardize_apply(X_tr_raw, mean_f, std_f)
        X_val_z = dm.standardize_apply(X_val_raw, mean_f, std_f)
        X_te_z = dm.standardize_apply(X_te_raw, mean_f, std_f)

        fold_cfg = TrainingConfig(**{**config.as_dict(), "seed": config.seed + 1000 * fold_id})
        model, info = train_one_fold(
            X_train=X_tr_z, y_train=y_tr,
            conf_train=conf_tr, env_train=env_tr,
            X_val=X_val_z, y_val=y_val,
            config=fold_cfg, n_confounders=eff_n_conf,
            logger_prefix=f"[ablation fold {fold_id}]",
        )

        out_te = model.forward(X_te_z, training=False)
        oof_probs[test_mask] = dm.sigmoid(out_te["clf_logit"])

        gate_w = dm.sigmoid(model.gate.params["log_alpha"])
        gate_weights.append(gate_w)

        y_pred_te = (oof_probs[test_mask] >= 0.5).astype(np.int64)
        if len(np.unique(y_te)) > 1:
            per_fold_metrics.append({
                "fold_id": int(fold_id),
                "auroc": float(mx.auroc(y_te, oof_probs[test_mask])),
                "auprc": float(mx.auprc(y_te, oof_probs[test_mask])),
                "balanced_accuracy": float(mx.balanced_accuracy(y_te, y_pred_te)),
                "brier": float(mx.brier_score(y_te, oof_probs[test_mask])),
                "n_train": int(tr_core.sum()),
                "n_test": int(test_mask.sum()),
            })

    gate_matrix = np.stack(gate_weights, axis=0) if gate_weights else np.zeros((0, n_features))
    return {
        "oof_probs": oof_probs,
        "gate_weights_per_fold": gate_matrix,
        "per_fold_metrics": per_fold_metrics,
    }


def run_single_variant(
    *,
    variant: AblationVariant,
    X: np.ndarray,
    y: np.ndarray,
    outer_fold: np.ndarray,
    groups: np.ndarray,
    fold_rows: List[Dict[str, Any]],
    master_rows: List[Dict[str, Any]],
    confounder_column: str = "gender_encoded",
    environment_column: str = "env_smoking",
    base_config: Optional[TrainingConfig] = None,
    seed: int = 2026,
) -> Dict[str, Any]:
    """Run the full outer-CV pipeline for a single ablation variant.

    Returns a dict with keys:
        variant_name        str
        oof_probs           (n,) OOF tumor probabilities
        gate_weights_per_fold (K, P) per-fold gate weights
        per_fold_metrics    list of per-fold metric dicts
        oof_auroc, oof_auprc, oof_bac, oof_brier  float
    """
    base = base_config if base_config is not None else TrainingConfig()
    cfg = variant.apply(base)
    cfg.seed = seed

    logger.info("Ablation variant: %s | %s", variant.name, variant.description)

    result = _run_ablation_cv(
        X=X, y=y, outer_fold=outer_fold, groups=groups,
        fold_rows=fold_rows, master_rows=master_rows,
        confounder_column=confounder_column,
        environment_column=environment_column,
        config=cfg,
    )

    oof_probs = result["oof_probs"]
    oof_y = y.astype(np.int64)
    valid = ~np.isnan(oof_probs)

    if valid.sum() > 0 and len(np.unique(oof_y[valid])) > 1:
        y_pred = (oof_probs[valid] >= 0.5).astype(np.int64)
        oof_metrics: Dict[str, float] = {
            "oof_auroc": float(mx.auroc(oof_y[valid], oof_probs[valid])),
            "oof_auprc": float(mx.auprc(oof_y[valid], oof_probs[valid])),
            "oof_bac": float(mx.balanced_accuracy(oof_y[valid], y_pred)),
            "oof_brier": float(mx.brier_score(oof_y[valid], oof_probs[valid])),
        }
    else:
        oof_metrics = {k: float("nan") for k in ("oof_auroc", "oof_auprc", "oof_bac", "oof_brier")}

    return {
        "variant_name": variant.name,
        "oof_probs": oof_probs,
        "gate_weights_per_fold": result["gate_weights_per_fold"],
        "per_fold_metrics": result["per_fold_metrics"],
        **oof_metrics,
    }


# ---------------------------------------------------------------------------
# Full ablation study
# ---------------------------------------------------------------------------

def run_ablation_study(
    *,
    X: np.ndarray,
    y: np.ndarray,
    outer_fold: np.ndarray,
    groups: np.ndarray,
    fold_rows: List[Dict[str, Any]],
    master_rows: List[Dict[str, Any]],
    confounder_column: str = "gender_encoded",
    environment_column: str = "env_smoking",
    base_config: Optional[TrainingConfig] = None,
    variants: Optional[List[AblationVariant]] = None,
    seed: int = 2026,
) -> Dict[str, Any]:
    """Run all ablation variants and collect results for comparison.

    Returns
    -------
    dict with keys:
        variant_results   : {variant_name -> run_single_variant output}
        summary_rows      : list of per-variant summary dicts (for CSV)
        per_fold_rows     : list of per-variant × per-fold dicts (for CSV)
    """
    variant_list = variants if variants is not None else ABLATION_VARIANTS
    variant_results: Dict[str, Any] = {}
    summary_rows: List[Dict[str, Any]] = []
    per_fold_rows: List[Dict[str, Any]] = []

    for v in variant_list:
        logger.info("=== Ablation: %s ===", v.name)
        try:
            res = run_single_variant(
                variant=v,
                X=X, y=y, outer_fold=outer_fold, groups=groups,
                fold_rows=fold_rows, master_rows=master_rows,
                confounder_column=confounder_column,
                environment_column=environment_column,
                base_config=base_config,
                seed=seed,
            )
        except Exception as exc:
            logger.error("Variant %s failed: %s", v.name, exc)
            res = {
                "variant_name": v.name,
                "oof_probs": np.full(y.shape[0], np.nan),
                "gate_weights_per_fold": np.array([]),
                "per_fold_metrics": [],
                "oof_auroc": float("nan"),
                "oof_auprc": float("nan"),
                "oof_bac": float("nan"),
                "oof_brier": float("nan"),
            }

        variant_results[v.name] = res
        summary_rows.append({
            "variant": v.name,
            "description": v.description,
            "oof_auroc": res["oof_auroc"],
            "oof_auprc": res["oof_auprc"],
            "oof_bac": res["oof_bac"],
            "oof_brier": res["oof_brier"],
        })

        for row in res.get("per_fold_metrics", []):
            per_fold_rows.append({"variant": v.name, **row})

    return {
        "variant_results": variant_results,
        "summary_rows": summary_rows,
        "per_fold_rows": per_fold_rows,
    }


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def compute_ablation_summary(per_fold_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Compute mean ± std of per-fold metrics grouped by variant.

    Each row in the output has: variant, mean_auroc, std_auroc,
    mean_auprc, std_auprc, mean_bac, std_bac, mean_brier, std_brier.
    """
    from collections import defaultdict
    accum: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    for row in per_fold_rows:
        v = row.get("variant", "unknown")
        for key in ("auroc", "auprc", "balanced_accuracy", "brier"):
            val = row.get(key, float("nan"))
            if not math.isnan(float(val)):
                accum[v][key].append(float(val))

    summary = []
    for variant, metrics in sorted(accum.items()):
        entry: Dict[str, Any] = {"variant": variant}
        for key, vals in metrics.items():
            arr = np.array(vals)
            entry[f"mean_{key}"] = float(arr.mean())
            entry[f"std_{key}"] = float(arr.std())
            entry[f"n_folds_{key}"] = len(vals)
        summary.append(entry)
    return summary


def compute_rank_correlations(
    variant_results: Dict[str, Any],
    reference_variant: str = "full_model",
) -> List[Dict[str, Any]]:
    """Compute Spearman rank correlation of mean gate weights vs reference.

    For each non-reference variant, computes the Spearman correlation between
    its mean gate-weight vector and the reference model's mean gate vector.
    This measures how much the feature ranking changes when a component is
    removed.

    Returns list of dicts: {variant, spearman_rho, n_genes}.
    """
    ref = variant_results.get(reference_variant)
    if ref is None:
        return []

    ref_gates = ref.get("gate_weights_per_fold")
    if ref_gates is None or len(ref_gates) == 0:
        return []

    ref_mean = np.nanmean(np.atleast_2d(ref_gates), axis=0)
    if ref_mean.size == 0:
        return []

    rows = []
    for name, res in variant_results.items():
        if name == reference_variant:
            continue
        gates = res.get("gate_weights_per_fold")
        if gates is None or len(gates) == 0:
            rows.append({"variant": name, "spearman_rho": float("nan"), "n_genes": 0})
            continue
        var_mean = np.nanmean(np.atleast_2d(gates), axis=0)
        if var_mean.shape != ref_mean.shape:
            rows.append({"variant": name, "spearman_rho": float("nan"),
                         "n_genes": int(ref_mean.size)})
            continue

        n = ref_mean.size
        ref_rank = np.argsort(np.argsort(-ref_mean)).astype(np.float64)
        var_rank = np.argsort(np.argsort(-var_mean)).astype(np.float64)
        d2 = np.sum((ref_rank - var_rank) ** 2)
        rho = 1.0 - 6.0 * d2 / (n * (n * n - 1.0)) if n > 1 else float("nan")
        rows.append({"variant": name, "spearman_rho": float(rho), "n_genes": int(n)})

    return rows
