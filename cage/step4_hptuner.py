"""CAGE Step 4: Nested hyperparameter tuning for SparseInvariantModel.

Implements randomized search over the deep-model's hyperparameter space using
the inner CV folds defined in grouped_outer_folds.csv. For each outer fold,
the best configuration is selected by mean validation AUROC across inner folds.

Outputs
-------
nested_tuning_results.csv         all trial × inner-fold results (for diagnostics)
best_hyperparams_per_outer_fold.json  best config per outer fold

Usage
-----
from cage.step4_hptuner import run_nested_hp_tuning, DEFAULT_SEARCH_SPACE

results = run_nested_hp_tuning(
    X=X,                       # (n, p) expression matrix, already aligned
    y=y,                       # (n,) binary labels
    outer_fold=outer_fold,     # (n,) outer fold ids
    groups=groups,             # (n,) patient barcodes (for leakage checks)
    fold_rows=fold_rows,       # list of dicts from grouped_outer_folds.csv
    master_rows=master_rows,   # list of dicts from master_samples_primary.csv
    n_conf_classes=n_conf_classes,
    n_trials=30,
    seed=2026,
    n_epochs_hpo=60,           # reduced epoch budget for HPO speed
)
"""

from __future__ import annotations

import logging
import math
import random
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from . import deep_model_utils as dm
from . import metrics as mx
from . import step3_runner as step3
from .step4_runner import (
    TrainingConfig,
    SparseInvariantModel,
    train_one_fold,
    _extract_inner_folds,
    _pick_inner_split,
    encode_confounder_column,
    _balanced_sample_weights,
)

logger = logging.getLogger("cage.step4.hptuner")

__all__ = [
    "DEFAULT_SEARCH_SPACE",
    "sample_config",
    "run_one_trial_inner_fold",
    "run_nested_hp_tuning",
]


# ---------------------------------------------------------------------------
# Default hyperparameter search space
# ---------------------------------------------------------------------------

DEFAULT_SEARCH_SPACE: Dict[str, List[Any]] = {
    "latent_dim":        [32, 48, 64, 96],
    "hidden_dims":       [(128, 64), (256, 96), (256, 128), (512, 128)],
    "sparsity_type":     ["l1", "hard-concrete"],
    "dropout":           [0.0, 0.05, 0.1, 0.2, 0.3],
    "lr":                [5e-4, 1e-3, 2e-3, 3e-3],
    "weight_decay":      [0.0, 1e-4, 5e-4, 1e-3],
    "lambda_sparsity":   [1e-4, 5e-4, 1e-3, 5e-3],
    "lambda_adv":        [0.1, 0.3, 0.5, 1.0],
    "lambda_inv":        [0.0, 0.05, 0.1, 0.3],
    "lambda_recon":      [0.0, 0.05, 0.1, 0.2],
    "patience":          [10, 15, 20],
}


def sample_config(
    search_space: Dict[str, List[Any]],
    base_config: TrainingConfig,
    rng_py: random.Random,
    n_epochs_hpo: int = 60,
) -> TrainingConfig:
    """Sample one TrainingConfig uniformly from *search_space*.

    Parameters not in *search_space* are inherited from *base_config*.
    ``n_epochs`` is always overridden to ``n_epochs_hpo`` so that HPO runs
    complete faster than a full training run.
    """
    sampled: Dict[str, Any] = base_config.as_dict()
    for key, choices in search_space.items():
        sampled[key] = rng_py.choice(choices)
    sampled["n_epochs"] = n_epochs_hpo
    return TrainingConfig(**sampled)


# ---------------------------------------------------------------------------
# Single-trial inner-fold evaluation
# ---------------------------------------------------------------------------

def run_one_trial_inner_fold(
    *,
    X_inner_train: np.ndarray,
    y_inner_train: np.ndarray,
    conf_inner_train: np.ndarray,
    env_inner_train: np.ndarray,
    X_inner_val: np.ndarray,
    y_inner_val: np.ndarray,
    config: TrainingConfig,
    n_conf_classes: int,
    trial_idx: int,
    outer_fold_id: int,
    inner_fold_id: int,
) -> Dict[str, Any]:
    """Train one trial on a single inner fold and return val metrics.

    Returns a dict with keys: trial, outer_fold, inner_fold, val_auroc,
    val_auprc, val_bac, val_brier, best_epoch, plus all hyperparameter values.
    """
    if len(np.unique(y_inner_val)) < 2:
        logger.warning(
            "Trial %d outer=%d inner=%d: single-class val fold; AUROC=NaN",
            trial_idx, outer_fold_id, inner_fold_id,
        )
        row = {"trial": trial_idx, "outer_fold": outer_fold_id,
               "inner_fold": inner_fold_id, "val_auroc": float("nan"),
               "val_auprc": float("nan"), "val_bac": float("nan"),
               "val_brier": float("nan"), "best_epoch": -1}
        row.update(config.as_dict())
        return row

    _, info = train_one_fold(
        X_train=X_inner_train,
        y_train=y_inner_train,
        conf_train=conf_inner_train,
        env_train=env_inner_train,
        X_val=X_inner_val,
        y_val=y_inner_val,
        config=config,
        n_confounders=n_conf_classes,
        logger_prefix=f"[hpo t{trial_idx} o{outer_fold_id} i{inner_fold_id}]",
    )

    row: Dict[str, Any] = {
        "trial": trial_idx,
        "outer_fold": outer_fold_id,
        "inner_fold": inner_fold_id,
        "val_auroc": float(info.get("best_val_auroc", float("nan"))),
        "val_bac": float(info.get("best_val_bac", float("nan"))),
        "val_loss": float(info.get("best_val_loss", float("nan"))),
        "best_epoch": int(info.get("best_epoch", -1)),
    }
    row.update(config.as_dict())
    return row


# ---------------------------------------------------------------------------
# Main tuning loop
# ---------------------------------------------------------------------------

def run_nested_hp_tuning(
    *,
    X: np.ndarray,
    y: np.ndarray,
    outer_fold: np.ndarray,
    groups: np.ndarray,
    fold_rows: List[Dict[str, Any]],
    master_rows: List[Dict[str, Any]],
    confounder_column: str = "gender_encoded",
    environment_column: str = "env_smoking",
    n_conf_classes: int = 2,
    search_space: Optional[Dict[str, List[Any]]] = None,
    base_config: Optional[TrainingConfig] = None,
    n_trials: int = 30,
    n_epochs_hpo: int = 60,
    selection_metric: str = "val_auroc",
    seed: int = 2026,
) -> Dict[str, Any]:
    """Run nested hyperparameter tuning over all outer folds.

    For each outer fold:
      1. Extracts the outer training set (train + inner-val samples).
      2. Iterates *n_trials* sampled configurations.
      3. For each trial, evaluates on each inner fold using the inner-train /
         inner-val split defined in *fold_rows*.
      4. Picks the trial with the highest mean ``selection_metric`` across
         inner folds (NaN inner folds are excluded from the mean).

    Parameters
    ----------
    X : (n, p) expression matrix, already fold-locally standardised externally
        OR raw VST values — this function performs its own per-inner-fold
        standardisation internally.
    n_trials : int
        Number of random configurations to evaluate per outer fold.
    n_epochs_hpo : int
        Epoch budget for each HPO trial (should be smaller than the final run).
    selection_metric : str
        Inner-fold metric to maximise (default: ``val_auroc``).

    Returns
    -------
    dict with keys:
        all_trial_rows        : list of per-trial × per-inner-fold dicts
        best_per_outer_fold   : {outer_fold_id -> best TrainingConfig}
        summary               : list of per-outer-fold selection summaries
    """
    sp = search_space if search_space is not None else DEFAULT_SEARCH_SPACE
    base = base_config if base_config is not None else TrainingConfig()
    rng_py = random.Random(seed)

    step3.assert_no_patient_leakage(groups, outer_fold)

    fold_ids = sorted(set(int(f) for f in outer_fold))
    n_outer = max(fold_ids) + 1
    inner_folds_per_outer = _extract_inner_folds(fold_rows, n_outer)

    conf_labels, resolved_n_conf, _ = encode_confounder_column(master_rows, confounder_column)
    env_labels, resolved_n_env, _ = encode_confounder_column(master_rows, environment_column)
    eff_n_conf = max(n_conf_classes, resolved_n_conf, 2)

    all_trial_rows: List[Dict[str, Any]] = []
    best_per_outer_fold: Dict[int, TrainingConfig] = {}
    summary: List[Dict[str, Any]] = []

    for outer_id in fold_ids:
        test_mask = outer_fold == outer_id
        train_mask = ~test_mask

        logger.info(
            "HP tuning outer fold %d | %d train samples | %d trials",
            outer_id, int(train_mask.sum()), n_trials,
        )

        # Pre-sample all n_trials configs for this outer fold
        trial_configs = [
            sample_config(sp, base, rng_py, n_epochs_hpo=n_epochs_hpo)
            for _ in range(n_trials)
        ]
        # Give each trial a unique seed
        for t_idx, cfg in enumerate(trial_configs):
            cfg.seed = seed + outer_id * 10000 + t_idx * 100

        # Collect inner-fold ids for this outer fold training set
        inner_col = f"inner_fold_outer{outer_id}"
        inner_fold_arr = np.full(X.shape[0], -1, dtype=np.int64)
        for i, rec in enumerate(fold_rows):
            bc = rec.get("sample_barcode", "")
            # fold_rows indices should correspond to X rows — assume same order
            if int(rec["outer_fold"]) != outer_id:
                inner_val = rec.get(inner_col, "")
                if inner_val not in ("", "nan", None):
                    try:
                        inner_fold_arr[i] = int(inner_val)
                    except (ValueError, TypeError):
                        pass

        inner_ids = sorted(set(int(v) for v in inner_fold_arr if v >= 0))
        if not inner_ids:
            logger.warning(
                "Outer fold %d: no inner folds found in fold_rows; "
                "falling back to single 20%% validation split.", outer_id
            )
            rng_fb = np.random.default_rng(seed + outer_id)
            tr_idx = np.where(train_mask)[0]
            val_size = max(2, int(0.2 * tr_idx.size))
            val_pick = rng_fb.choice(tr_idx, size=val_size, replace=False)
            inner_fold_arr[val_pick] = 0
            inner_ids = [0]

        for t_idx, trial_cfg in enumerate(trial_configs):
            inner_aurocs: List[float] = []

            for inner_id in inner_ids:
                inner_val_mask = (inner_fold_arr == inner_id) & train_mask
                inner_tr_mask = train_mask & ~inner_val_mask

                if inner_val_mask.sum() < 2 or inner_tr_mask.sum() < 4:
                    continue

                X_itr_raw = X[inner_tr_mask]
                X_ival_raw = X[inner_val_mask]
                y_itr = y[inner_tr_mask]
                y_ival = y[inner_val_mask]
                conf_itr = conf_labels[inner_tr_mask]
                env_itr = env_labels[inner_tr_mask]

                # Inner-fold standardisation (prevents inner leakage)
                m, s = dm.standardize_fit(X_itr_raw)
                X_itr = dm.standardize_apply(X_itr_raw, m, s)
                X_ival = dm.standardize_apply(X_ival_raw, m, s)

                row = run_one_trial_inner_fold(
                    X_inner_train=X_itr,
                    y_inner_train=y_itr,
                    conf_inner_train=conf_itr,
                    env_inner_train=env_itr,
                    X_inner_val=X_ival,
                    y_inner_val=y_ival,
                    config=trial_cfg,
                    n_conf_classes=eff_n_conf,
                    trial_idx=t_idx,
                    outer_fold_id=outer_id,
                    inner_fold_id=inner_id,
                )
                all_trial_rows.append(row)
                val = row.get(selection_metric, float("nan"))
                if not math.isnan(float(val)):
                    inner_aurocs.append(float(val))

            mean_metric = float(np.mean(inner_aurocs)) if inner_aurocs else float("nan")
            logger.debug(
                "Outer %d trial %d: mean_%s=%.4f (%d inner folds)",
                outer_id, t_idx, selection_metric, mean_metric, len(inner_aurocs),
            )
            # Attach summary metric to last rows for this trial
            for row in all_trial_rows:
                if row["trial"] == t_idx and row["outer_fold"] == outer_id:
                    row["mean_inner_metric"] = mean_metric

        # Select best trial (max mean inner metric; NaN trials are excluded)
        trial_means: Dict[int, float] = {}
        for row in all_trial_rows:
            if row["outer_fold"] != outer_id:
                continue
            t = row["trial"]
            v = row.get("mean_inner_metric", float("nan"))
            if not math.isnan(float(v)):
                trial_means[t] = max(trial_means.get(t, float("-inf")), float(v))

        if not trial_means:
            logger.warning("Outer fold %d: no valid trials; using base config.", outer_id)
            best_per_outer_fold[outer_id] = base
            summary.append({
                "outer_fold": outer_id,
                "best_trial": -1,
                "best_mean_metric": float("nan"),
                "selection_metric": selection_metric,
                "n_valid_trials": 0,
            })
            continue

        best_trial_idx = max(trial_means, key=lambda k: trial_means[k])
        best_cfg = trial_configs[best_trial_idx]
        best_per_outer_fold[outer_id] = best_cfg

        logger.info(
            "Outer fold %d: best trial %d | mean_%s=%.4f | latent_dim=%d lr=%.4g",
            outer_id, best_trial_idx, selection_metric, trial_means[best_trial_idx],
            best_cfg.latent_dim, best_cfg.lr,
        )
        summary.append({
            "outer_fold": outer_id,
            "best_trial": best_trial_idx,
            "best_mean_metric": trial_means[best_trial_idx],
            "selection_metric": selection_metric,
            "n_valid_trials": len(trial_means),
            **{f"best_{k}": v for k, v in best_cfg.as_dict().items()},
        })

    return {
        "all_trial_rows": all_trial_rows,
        "best_per_outer_fold": best_per_outer_fold,
        "summary": summary,
    }
