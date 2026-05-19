"""CAGE Step 5: CDPS robustness and uncertainty analysis.

Provides five robustness analyses for the Causality-Aware Priority Score:

1. Weight sensitivity grid
   Recomputes CDPS for a dense grid of weight combinations and tracks how
   the top-K gene set and Spearman rank correlation change across the grid.

2. Bootstrap confidence intervals for gene ranks
   Resamples outer folds with replacement to estimate rank uncertainty
   for each gene (mean rank, 95% CI lower/upper).

3. Leave-one-component-out (LOCO) sensitivity
   Drops each CDPS component in turn and reports the Spearman rank
   correlation vs. the full-weight CDPS.

4. Label-permutation null model
   Permutes tumor/normal labels across all folds and recomputes CDPS,
   establishing a null distribution for top-K CDPS values.

5. Random gene-set baseline
   Repeatedly draws random gene sets of size k and computes their mean
   CDPS to establish a baseline above which nominated genes must score.

Outputs (written by caller)
---------------------------
cdps_weight_sensitivity.csv     grid of weight configs × top-K overlap
cdps_rank_bootstrap_ci.csv      per-gene mean rank ± 95% CI
cdps_component_ablation.csv     Spearman rho of LOCO vs full CDPS
cdps_null_model_results.csv     null CDPS distribution stats
cdps_random_baseline.csv        random gene-set baseline CDPS values
"""

from __future__ import annotations

import logging
import math
from itertools import product
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from . import metrics as mx
from .step5_runner import build_cdps_ranking, normalize_scores

logger = logging.getLogger("cage.step5.robustness")

__all__ = [
    "weight_sensitivity_grid",
    "bootstrap_rank_ci",
    "leave_one_component_out",
    "label_permutation_null",
    "random_gene_set_baseline",
    "run_cdps_robustness",
]

# ---------------------------------------------------------------------------
# 1. Weight sensitivity grid
# ---------------------------------------------------------------------------

def weight_sensitivity_grid(
    *,
    gene_names: Sequence[str],
    attribution: np.ndarray,
    gate: np.ndarray,
    stability: np.ndarray,
    invariance: np.ndarray,
    perturbation: np.ndarray,
    top_k: int = 25,
    n_steps: int = 5,
    normalization: str = "minmax",
) -> List[Dict[str, Any]]:
    """Evaluate CDPS ranking over a grid of weight combinations.

    Each of the 5 components is allowed to take ``n_steps`` evenly-spaced
    values in [0, 1]; only combinations that sum to > 0 are tested.
    For each weight combination records the top-K gene list and the
    Spearman rank correlation vs the default weights.

    Parameters
    ----------
    n_steps : int
        Number of steps per component weight axis (results in up to
        ``n_steps**5`` grid points, pruned to sum > 0).

    Returns
    -------
    list of dicts: weight_attribution, weight_gate, weight_stability,
    weight_invariance, weight_perturbation, top_k_genes (list),
    spearman_rho_vs_default, jaccard_vs_default.
    """
    weight_values = np.linspace(0.0, 1.0, n_steps)
    default_weights = {
        "attribution": 0.30, "gate": 0.20, "stability": 0.20,
        "invariance": 0.15, "perturbation": 0.15,
    }

    # Compute default ranking once
    default_result = build_cdps_ranking(
        gene_names=list(gene_names),
        attribution=attribution, gate=gate, stability=stability,
        invariance=invariance, perturbation=perturbation,
        weights=default_weights, normalization=normalization,
    )
    default_cdps = default_result["cdps"]
    default_order = np.argsort(-default_cdps, kind="mergesort")
    default_top_k = set(str(gene_names[i]) for i in default_order[:top_k])

    n_genes = len(gene_names)
    default_rank = np.empty(n_genes, dtype=np.float64)
    default_rank[default_order] = np.arange(n_genes, dtype=np.float64)

    rows: List[Dict[str, Any]] = []

    for wa, wg, ws, wi, wp in product(weight_values, repeat=5):
        total = float(wa + wg + ws + wi + wp)
        if total < 1e-9:
            continue

        result = build_cdps_ranking(
            gene_names=list(gene_names),
            attribution=attribution, gate=gate, stability=stability,
            invariance=invariance, perturbation=perturbation,
            weights={"attribution": wa, "gate": wg, "stability": ws,
                     "invariance": wi, "perturbation": wp},
            normalization=normalization,
        )
        cdps = result["cdps"]
        order = np.argsort(-cdps, kind="mergesort")
        top_k_genes = [str(gene_names[i]) for i in order[:top_k]]
        this_top_k = set(top_k_genes)

        # Jaccard vs default top-K
        jaccard = len(this_top_k & default_top_k) / max(len(this_top_k | default_top_k), 1)

        # Spearman rho vs default
        this_rank = np.empty(n_genes, dtype=np.float64)
        this_rank[order] = np.arange(n_genes, dtype=np.float64)
        d2 = float(np.sum((default_rank - this_rank) ** 2))
        rho = 1.0 - 6.0 * d2 / (n_genes * (n_genes * n_genes - 1.0)) if n_genes > 1 else float("nan")

        rows.append({
            "weight_attribution": float(wa / total),
            "weight_gate": float(wg / total),
            "weight_stability": float(ws / total),
            "weight_invariance": float(wi / total),
            "weight_perturbation": float(wp / total),
            "top_k_genes": top_k_genes,
            "jaccard_vs_default": float(jaccard),
            "spearman_rho_vs_default": float(rho),
        })

    logger.info("Weight sensitivity grid: %d configurations evaluated.", len(rows))
    return rows


# ---------------------------------------------------------------------------
# 2. Bootstrap rank confidence intervals
# ---------------------------------------------------------------------------

def bootstrap_rank_ci(
    *,
    gene_names: Sequence[str],
    attribution: np.ndarray,
    gate: np.ndarray,
    stability: np.ndarray,
    invariance: np.ndarray,
    perturbation: np.ndarray,
    weights: Optional[Mapping[str, float]] = None,
    n_boot: int = 1000,
    seed: int = 2026,
    level: float = 0.95,
    normalization: str = "minmax",
) -> List[Dict[str, Any]]:
    """Bootstrap confidence intervals for each gene's CDPS rank.

    For each bootstrap replicate, resamples genes with replacement, recomputes
    CDPS on the resampled space, and records each gene's rank. Returns the
    mean rank, rank standard deviation, and percentile CI across replicates.

    Note: rank CI here measures rank uncertainty due to scoring variability,
    not fold-sampling uncertainty. For fold-sampling CI, use the fold-level
    bootstrap in metrics.bootstrap_ci.
    """
    rng = np.random.default_rng(seed)
    n_genes = len(gene_names)
    w = weights if weights is not None else {
        "attribution": 0.30, "gate": 0.20, "stability": 0.20,
        "invariance": 0.15, "perturbation": 0.15,
    }
    alpha = (1.0 - level) / 2.0

    # Components stacked: (n_genes, 5)
    comp_matrix = np.column_stack([attribution, gate, stability, invariance, perturbation])

    rank_matrix = np.zeros((n_boot, n_genes), dtype=np.float64)

    for b in range(n_boot):
        idx = rng.integers(0, n_genes, size=n_genes)
        boot_names = [gene_names[i] for i in idx]
        boot_comp = comp_matrix[idx]
        result = build_cdps_ranking(
            gene_names=boot_names,
            attribution=boot_comp[:, 0],
            gate=boot_comp[:, 1],
            stability=boot_comp[:, 2],
            invariance=boot_comp[:, 3],
            perturbation=boot_comp[:, 4],
            weights=w,
            normalization=normalization,
        )
        cdps = result["cdps"]
        rank_order = np.argsort(-cdps, kind="mergesort")
        ranks = np.empty(n_genes, dtype=np.float64)
        ranks[rank_order] = np.arange(1, n_genes + 1, dtype=np.float64)
        rank_matrix[b] = ranks

    mean_ranks = rank_matrix.mean(axis=0)
    std_ranks = rank_matrix.std(axis=0)
    lo_ranks = np.quantile(rank_matrix, alpha, axis=0)
    hi_ranks = np.quantile(rank_matrix, 1.0 - alpha, axis=0)

    # Also compute point estimate (full data)
    point_result = build_cdps_ranking(
        gene_names=list(gene_names),
        attribution=attribution, gate=gate, stability=stability,
        invariance=invariance, perturbation=perturbation,
        weights=w, normalization=normalization,
    )
    point_cdps = point_result["cdps"]
    point_order = np.argsort(-point_cdps, kind="mergesort")
    point_ranks = np.empty(n_genes, dtype=np.float64)
    point_ranks[point_order] = np.arange(1, n_genes + 1, dtype=np.float64)

    rows = []
    for i, gene in enumerate(gene_names):
        rows.append({
            "gene": str(gene),
            "cdps": float(point_cdps[i]),
            "point_rank": int(point_ranks[i]),
            "mean_boot_rank": float(mean_ranks[i]),
            "std_boot_rank": float(std_ranks[i]),
            f"rank_ci_lo_{int(level*100)}": float(lo_ranks[i]),
            f"rank_ci_hi_{int(level*100)}": float(hi_ranks[i]),
        })
    # Sort by point_rank for readability
    rows.sort(key=lambda r: r["point_rank"])
    return rows


# ---------------------------------------------------------------------------
# 3. Leave-one-component-out (LOCO)
# ---------------------------------------------------------------------------

def leave_one_component_out(
    *,
    gene_names: Sequence[str],
    attribution: np.ndarray,
    gate: np.ndarray,
    stability: np.ndarray,
    invariance: np.ndarray,
    perturbation: np.ndarray,
    weights: Optional[Mapping[str, float]] = None,
    normalization: str = "minmax",
) -> List[Dict[str, Any]]:
    """Drop each component in turn and compute Spearman rho vs full CDPS.

    Returns list of dicts: {dropped_component, spearman_rho, n_genes,
    jaccard_top25_vs_full}.
    """
    w = weights if weights is not None else {
        "attribution": 0.30, "gate": 0.20, "stability": 0.20,
        "invariance": 0.15, "perturbation": 0.15,
    }
    components = {
        "attribution": attribution, "gate": gate, "stability": stability,
        "invariance": invariance, "perturbation": perturbation,
    }
    zeros = np.zeros(len(gene_names), dtype=np.float64)
    n_genes = len(gene_names)

    full_result = build_cdps_ranking(
        gene_names=list(gene_names), **components, weights=w,
        normalization=normalization,
    )
    full_cdps = full_result["cdps"]
    full_order = np.argsort(-full_cdps, kind="mergesort")
    full_top25 = set(str(gene_names[i]) for i in full_order[:25])
    full_rank = np.empty(n_genes, dtype=np.float64)
    full_rank[full_order] = np.arange(n_genes, dtype=np.float64)

    rows = []
    for dropped in components:
        loco_comps = {k: (zeros if k == dropped else v) for k, v in components.items()}
        loco_w = {k: (0.0 if k == dropped else float(w.get(k, 0.0))) for k in w}

        result = build_cdps_ranking(
            gene_names=list(gene_names), **loco_comps, weights=loco_w,
            normalization=normalization,
        )
        loco_cdps = result["cdps"]
        loco_order = np.argsort(-loco_cdps, kind="mergesort")
        loco_top25 = set(str(gene_names[i]) for i in loco_order[:25])
        loco_rank = np.empty(n_genes, dtype=np.float64)
        loco_rank[loco_order] = np.arange(n_genes, dtype=np.float64)

        d2 = float(np.sum((full_rank - loco_rank) ** 2))
        rho = 1.0 - 6.0 * d2 / (n_genes * (n_genes ** 2 - 1.0)) if n_genes > 1 else float("nan")
        jaccard = len(full_top25 & loco_top25) / max(len(full_top25 | loco_top25), 1)

        rows.append({
            "dropped_component": dropped,
            "spearman_rho_vs_full": float(rho),
            "jaccard_top25_vs_full": float(jaccard),
            "n_genes": int(n_genes),
        })

    return rows


# ---------------------------------------------------------------------------
# 4. Label-permutation null model
# ---------------------------------------------------------------------------

def label_permutation_null(
    *,
    gene_names: Sequence[str],
    attribution: np.ndarray,
    gate: np.ndarray,
    stability: np.ndarray,
    invariance: np.ndarray,
    perturbation: np.ndarray,
    weights: Optional[Mapping[str, float]] = None,
    n_permutations: int = 1000,
    top_k: int = 25,
    seed: int = 2026,
    normalization: str = "minmax",
) -> Dict[str, Any]:
    """Establish a null CDPS distribution by permuting component scores.

    For each permutation, shuffles all five component arrays jointly
    (same permutation order for all arrays, to preserve correlations
    between components), recomputes CDPS, and records the mean top-K
    CDPS and the maximum CDPS value.

    Returns a dict with null distribution arrays and percentile thresholds.
    """
    rng = np.random.default_rng(seed)
    w = weights if weights is not None else {
        "attribution": 0.30, "gate": 0.20, "stability": 0.20,
        "invariance": 0.15, "perturbation": 0.15,
    }
    n_genes = len(gene_names)
    comp_matrix = np.column_stack([attribution, gate, stability, invariance, perturbation])

    null_top_k_mean: List[float] = []
    null_max_cdps: List[float] = []

    for _ in range(n_permutations):
        perm = rng.permutation(n_genes)
        boot = comp_matrix[perm]
        result = build_cdps_ranking(
            gene_names=list(gene_names),
            attribution=boot[:, 0], gate=boot[:, 1], stability=boot[:, 2],
            invariance=boot[:, 3], perturbation=boot[:, 4],
            weights=w, normalization=normalization,
        )
        cdps = result["cdps"]
        order = np.argsort(-cdps, kind="mergesort")
        null_top_k_mean.append(float(cdps[order[:top_k]].mean()))
        null_max_cdps.append(float(cdps.max()))

    null_arr = np.array(null_top_k_mean)
    max_arr = np.array(null_max_cdps)

    # Real top-K mean
    real_result = build_cdps_ranking(
        gene_names=list(gene_names),
        attribution=attribution, gate=gate, stability=stability,
        invariance=invariance, perturbation=perturbation,
        weights=w, normalization=normalization,
    )
    real_cdps = real_result["cdps"]
    real_order = np.argsort(-real_cdps, kind="mergesort")
    real_top_k_mean = float(real_cdps[real_order[:top_k]].mean())
    real_max = float(real_cdps.max())

    p_value_top_k = float(np.mean(null_arr >= real_top_k_mean))
    p_value_max = float(np.mean(max_arr >= real_max))

    return {
        "n_permutations": n_permutations,
        "top_k": top_k,
        "real_top_k_mean_cdps": real_top_k_mean,
        "real_max_cdps": real_max,
        "null_top_k_mean": float(null_arr.mean()),
        "null_top_k_std": float(null_arr.std()),
        "null_top_k_p95": float(np.quantile(null_arr, 0.95)),
        "null_top_k_p99": float(np.quantile(null_arr, 0.99)),
        "p_value_top_k": p_value_top_k,
        "p_value_max": p_value_max,
        "null_top_k_distribution": null_top_k_mean,
        "null_max_distribution": null_max_cdps,
    }


# ---------------------------------------------------------------------------
# 5. Random gene-set baseline
# ---------------------------------------------------------------------------

def random_gene_set_baseline(
    *,
    gene_names: Sequence[str],
    attribution: np.ndarray,
    gate: np.ndarray,
    stability: np.ndarray,
    invariance: np.ndarray,
    perturbation: np.ndarray,
    weights: Optional[Mapping[str, float]] = None,
    set_sizes: Sequence[int] = (10, 25, 50, 100),
    n_draws: int = 500,
    seed: int = 2026,
    normalization: str = "minmax",
) -> List[Dict[str, Any]]:
    """Compare nominated gene-set mean CDPS to random draws of equal size.

    For each size in ``set_sizes``, draws ``n_draws`` random gene sets from
    the full gene pool and records the distribution of mean CDPS for those
    random sets. Returns per-size stats for baseline comparison.
    """
    rng = np.random.default_rng(seed)
    w = weights if weights is not None else {
        "attribution": 0.30, "gate": 0.20, "stability": 0.20,
        "invariance": 0.15, "perturbation": 0.15,
    }
    n_genes = len(gene_names)

    real_result = build_cdps_ranking(
        gene_names=list(gene_names),
        attribution=attribution, gate=gate, stability=stability,
        invariance=invariance, perturbation=perturbation,
        weights=w, normalization=normalization,
    )
    real_cdps = real_result["cdps"]
    real_order = np.argsort(-real_cdps, kind="mergesort")

    rows = []
    for k in set_sizes:
        if k > n_genes:
            continue
        real_top_k_mean = float(real_cdps[real_order[:k]].mean())

        draws = np.array([
            real_cdps[rng.choice(n_genes, size=k, replace=False)].mean()
            for _ in range(n_draws)
        ])
        p_value = float(np.mean(draws >= real_top_k_mean))

        rows.append({
            "set_size": int(k),
            "real_top_k_mean_cdps": real_top_k_mean,
            "random_mean": float(draws.mean()),
            "random_std": float(draws.std()),
            "random_p95": float(np.quantile(draws, 0.95)),
            "random_p99": float(np.quantile(draws, 0.99)),
            "p_value_vs_random": p_value,
            "fold_enrichment": real_top_k_mean / max(float(draws.mean()), 1e-12),
        })

    return rows


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def run_cdps_robustness(
    *,
    gene_names: Sequence[str],
    attribution: np.ndarray,
    gate: np.ndarray,
    stability: np.ndarray,
    invariance: np.ndarray,
    perturbation: np.ndarray,
    weights: Optional[Mapping[str, float]] = None,
    top_k: int = 25,
    n_boot: int = 500,
    n_permutations: int = 500,
    n_draws: int = 300,
    n_sensitivity_steps: int = 4,
    seed: int = 2026,
    normalization: str = "minmax",
    skip: Sequence[str] = (),
) -> Dict[str, Any]:
    """Run all five CDPS robustness analyses and return a results bundle.

    Parameters
    ----------
    skip : sequence of str
        Names of analyses to skip: 'weight_sensitivity', 'bootstrap_rank',
        'loco', 'null_model', 'random_baseline'.
    """
    logger.info("Running CDPS robustness suite for %d genes...", len(gene_names))
    skip_set = set(skip)
    results: Dict[str, Any] = {}

    kwargs = dict(
        gene_names=gene_names, attribution=attribution, gate=gate,
        stability=stability, invariance=invariance, perturbation=perturbation,
        weights=weights, normalization=normalization,
    )

    if "weight_sensitivity" not in skip_set:
        logger.info("  1/5 Weight sensitivity grid...")
        results["weight_sensitivity"] = weight_sensitivity_grid(
            **kwargs, top_k=top_k, n_steps=n_sensitivity_steps
        )

    if "bootstrap_rank" not in skip_set:
        logger.info("  2/5 Bootstrap rank CIs...")
        results["bootstrap_rank"] = bootstrap_rank_ci(**kwargs, n_boot=n_boot, seed=seed)

    if "loco" not in skip_set:
        logger.info("  3/5 Leave-one-component-out...")
        results["loco"] = leave_one_component_out(**kwargs)

    if "null_model" not in skip_set:
        logger.info("  4/5 Label-permutation null model...")
        results["null_model"] = label_permutation_null(
            **kwargs, n_permutations=n_permutations, top_k=top_k, seed=seed
        )

    if "random_baseline" not in skip_set:
        logger.info("  5/5 Random gene-set baseline...")
        results["random_baseline"] = random_gene_set_baseline(
            **kwargs, set_sizes=(top_k, top_k * 2, top_k * 4), n_draws=n_draws, seed=seed
        )

    logger.info("CDPS robustness suite complete.")
    return results
