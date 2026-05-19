"""CAGE Step 6 runner: biological & statistical validation (Phase IV).

Pure-numpy implementation of the Phase-IV pipeline:

    1. Differential expression (tumor vs normal) using Welch t or Mann-Whitney
       U on the VST + z-scored 5000-gene matrix from Step 2, with Benjamini-
       Hochberg FDR control.
    2. Over-representation enrichment (hypergeometric) across optional
       Hallmark / Reactome / KEGG-GO GMT files on the top-K CDPS genes.
    3. PPI network support (degree / induced subgraph / connected component)
       using an optional user-supplied edge list.
    4. Clinico-pathologic association: tumor-only Welch across environment
       strata + log-rank survival (median split).
    5. Subgroup robustness: tumor-vs-normal effect direction consistency
       across environment strata.
    6. Integrated validation score (weighted per plane.md default
       0.35/0.20/0.15/0.15/0.10/0.05) and final validated gene ranking.

Every statistical helper is implemented from scratch with ``math.lgamma`` /
``math.erf`` so the module has no scipy / pandas / sklearn dependency.
"""

from __future__ import annotations

import csv
import json
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from . import preprocess_esca as pp

logger = logging.getLogger("cage.step6.runner")

__all__ = [
    "load_step2_matrices",
    "load_cdps_ranking",
    "welch_t_test",
    "mann_whitney_u_test",
    "log_rank_test",
    "hypergeometric_log_sf",
    "bh_fdr",
    "run_differential_expression",
    "run_enrichment",
    "run_network_support",
    "run_clinical_association",
    "run_survival_analysis",
    "run_subgroup_robustness",
    "compute_integrated_validation_score",
    "generate_step6_figures",
    "run_step6_validation",
]


# ============================================================================
# Basic statistical primitives (numpy/math only)
# ============================================================================


def _betacf(a: float, b: float, x: float, *, max_iter: int = 200, eps: float = 3e-14) -> float:
    """Continued-fraction evaluation of the incomplete beta (Numerical Recipes).

    Requires ``0 < x < 1``. Used inside :func:`_regularized_incomplete_beta`.
    """
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            return h
    return h


def _regularized_incomplete_beta(x: float, a: float, b: float) -> float:
    """Regularized incomplete beta function I_x(a, b) for 0 <= x <= 1."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_bt = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log(1.0 - x)
    )
    bt = math.exp(log_bt)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def student_t_sf_two_sided(t: float, df: float) -> float:
    """Two-sided survival function for Student-t with df degrees of freedom.

    Returns ``P(|T| >= |t|)`` = ``I_{df/(df+t^2)}(df/2, 1/2)``.
    NaN-safe: returns 1.0 if ``t`` is NaN or ``df <= 0``.
    """
    if not math.isfinite(t) or not math.isfinite(df) or df <= 0:
        return 1.0
    t2 = t * t
    x = df / (df + t2)
    p = _regularized_incomplete_beta(x, 0.5 * df, 0.5)
    # Clip for numerical safety
    return max(min(float(p), 1.0), 0.0)


def _normal_sf(z: float) -> float:
    """One-sided upper-tail survival for standard normal (uses math.erf)."""
    if not math.isfinite(z):
        return 0.5
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def welch_t_test(
    x_a: np.ndarray, x_b: np.ndarray
) -> Dict[str, float]:
    """Welch's unequal-variance two-sided t-test between ``x_a`` and ``x_b``.

    Returns dict with mean_a, mean_b, mean_diff, var_a, var_b, t_stat, df,
    p_value. NaN-safe; returns NaNs if either array has <2 non-NaN values.
    """
    a = np.asarray(x_a, dtype=np.float64).ravel()
    b = np.asarray(x_b, dtype=np.float64).ravel()
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    n_a = int(a.size)
    n_b = int(b.size)
    if n_a < 2 or n_b < 2:
        return {
            "mean_a": float(a.mean()) if n_a else float("nan"),
            "mean_b": float(b.mean()) if n_b else float("nan"),
            "mean_diff": float("nan"),
            "var_a": float("nan"),
            "var_b": float("nan"),
            "t_stat": float("nan"),
            "df": float("nan"),
            "p_value": float("nan"),
            "n_a": n_a, "n_b": n_b,
        }
    m_a = float(a.mean())
    m_b = float(b.mean())
    v_a = float(a.var(ddof=1))
    v_b = float(b.var(ddof=1))
    se2 = v_a / n_a + v_b / n_b
    if se2 <= 0.0:
        return {
            "mean_a": m_a, "mean_b": m_b, "mean_diff": m_a - m_b,
            "var_a": v_a, "var_b": v_b,
            "t_stat": float("nan"), "df": float("nan"),
            "p_value": float("nan"),
            "n_a": n_a, "n_b": n_b,
        }
    t_stat = (m_a - m_b) / math.sqrt(se2)
    num = se2 * se2
    den = ((v_a / n_a) ** 2) / max(n_a - 1, 1) + ((v_b / n_b) ** 2) / max(n_b - 1, 1)
    df = num / den if den > 0 else 0.0
    p = student_t_sf_two_sided(t_stat, df)
    return {
        "mean_a": m_a, "mean_b": m_b, "mean_diff": m_a - m_b,
        "var_a": v_a, "var_b": v_b,
        "t_stat": t_stat, "df": df, "p_value": p,
        "n_a": n_a, "n_b": n_b,
    }


def mann_whitney_u_test(x_a: np.ndarray, x_b: np.ndarray) -> Dict[str, float]:
    """Mann-Whitney U two-sided test (normal approximation w/ tie correction).

    Matches scipy's ``mannwhitneyu(..., alternative="two-sided",
    use_continuity=True)`` asymptotic behaviour.
    """
    a = np.asarray(x_a, dtype=np.float64).ravel()
    b = np.asarray(x_b, dtype=np.float64).ravel()
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    n_a = int(a.size)
    n_b = int(b.size)
    if n_a == 0 or n_b == 0:
        return {"u_stat": float("nan"), "z_stat": float("nan"),
                "p_value": float("nan"), "n_a": n_a, "n_b": n_b}
    combined = np.concatenate([a, b])
    order = np.argsort(combined, kind="mergesort")
    # Average ranks with tie handling
    n = combined.size
    ranks = np.empty(n, dtype=np.float64)
    sorted_vals = combined[order]
    i = 0
    tie_correction = 0.0
    while i < n:
        j = i + 1
        while j < n and sorted_vals[j] == sorted_vals[i]:
            j += 1
        avg = (i + 1 + j) / 2.0  # 1-indexed ranks
        ranks[order[i:j]] = avg
        t = j - i
        if t > 1:
            tie_correction += (t ** 3 - t)
        i = j
    rank_sum_a = float(ranks[:n_a].sum())
    u_a = rank_sum_a - n_a * (n_a + 1) / 2.0
    u_b = n_a * n_b - u_a
    u = min(u_a, u_b)
    mean_u = n_a * n_b / 2.0
    sigma2 = (n_a * n_b / 12.0) * ((n_a + n_b + 1.0) - tie_correction / ((n_a + n_b) * (n_a + n_b - 1.0)))
    if sigma2 <= 0.0:
        return {"u_stat": float(u_a), "z_stat": float("nan"),
                "p_value": float("nan"), "n_a": n_a, "n_b": n_b}
    z = (abs(u_a - mean_u) - 0.5) / math.sqrt(sigma2)  # continuity correction
    p = 2.0 * _normal_sf(z)
    p = min(max(p, 0.0), 1.0)
    signed_z = (u_a - mean_u) / math.sqrt(sigma2)
    return {"u_stat": float(u_a), "z_stat": float(signed_z),
            "p_value": float(p), "n_a": n_a, "n_b": n_b}


def bh_fdr(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg step-up FDR adjustment."""
    p = np.asarray(p, dtype=np.float64).ravel()
    n = p.size
    out = np.full(n, float("nan"), dtype=np.float64)
    if n == 0:
        return out
    valid = ~np.isnan(p)
    if not valid.any():
        return out
    pv = p[valid]
    order = np.argsort(pv, kind="mergesort")
    ranked = pv[order]
    m = ranked.size
    raw_q = ranked * m / (np.arange(1, m + 1, dtype=np.float64))
    # Enforce monotone non-decreasing from the top
    for i in range(m - 2, -1, -1):
        if raw_q[i] > raw_q[i + 1]:
            raw_q[i] = raw_q[i + 1]
    raw_q = np.clip(raw_q, 0.0, 1.0)
    adj = np.empty(m, dtype=np.float64)
    adj[order] = raw_q
    out_valid = out.copy()
    idx_valid = np.where(valid)[0]
    out[idx_valid] = adj
    return out


def hypergeometric_log_sf(k: int, K: int, n: int, N: int) -> float:
    """Log P(X >= k) for X ~ Hypergeometric(N, K, n) using log-gamma.

    Arguments mirror scipy's ``hypergeom.sf(k-1, N, K, n)``: ``N`` population,
    ``K`` successes in population, ``n`` draws, ``k`` observed successes.
    """
    if k <= 0:
        return 0.0  # log(1)
    if K > N or n > N or K < 0 or n < 0 or k > min(K, n):
        return float("-inf")
    lgN = math.lgamma(N + 1)

    def _log_choose(a: int, b: int) -> float:
        if b < 0 or b > a:
            return float("-inf")
        return math.lgamma(a + 1) - math.lgamma(b + 1) - math.lgamma(a - b + 1)

    log_denom = _log_choose(N, n)
    total = float("-inf")
    k_max = min(K, n)
    for x in range(k, k_max + 1):
        log_p = _log_choose(K, x) + _log_choose(N - K, n - x) - log_denom
        if total == float("-inf"):
            total = log_p
        else:
            m = max(total, log_p)
            total = m + math.log(math.exp(total - m) + math.exp(log_p - m))
    return total


def hypergeometric_sf(k: int, K: int, n: int, N: int) -> float:
    """P(X >= k) for Hypergeometric(N, K, n)."""
    lp = hypergeometric_log_sf(k, K, n, N)
    if lp == float("-inf"):
        return 0.0
    return min(max(math.exp(lp), 0.0), 1.0)


def log_rank_test(
    times_a: np.ndarray, events_a: np.ndarray,
    times_b: np.ndarray, events_b: np.ndarray,
) -> Dict[str, float]:
    """Two-sample log-rank test.

    Returns ``chi2``, ``p_value``, ``observed_a``, ``expected_a``,
    ``hazard_direction`` (+1 if group-a hazard higher, -1 otherwise).
    NaN-safe: returns NaNs if either arm has no events.
    """
    times = np.concatenate([times_a, times_b]).astype(np.float64)
    events = np.concatenate([events_a, events_b]).astype(np.int64)
    group = np.concatenate([
        np.ones(times_a.size, dtype=np.int64),
        np.zeros(times_b.size, dtype=np.int64),
    ])
    valid = np.isfinite(times) & (times >= 0)
    times, events, group = times[valid], events[valid], group[valid]
    if times.size == 0:
        return {"chi2": float("nan"), "p_value": float("nan"),
                "observed_a": 0.0, "expected_a": 0.0, "hazard_direction": 0}
    order = np.argsort(times, kind="mergesort")
    times = times[order]; events = events[order]; group = group[order]
    unique_times = np.unique(times[events == 1])
    at_risk_a = int((group == 1).sum())
    at_risk_b = int((group == 0).sum())
    sum_o_a = 0.0
    sum_e_a = 0.0
    sum_var = 0.0
    for t in unique_times:
        in_t = times == t
        d_a = int(((events == 1) & (group == 1) & in_t).sum())
        d_b = int(((events == 1) & (group == 0) & in_t).sum())
        d = d_a + d_b
        if at_risk_a == 0 or at_risk_b == 0 or d == 0:
            drop_now = int(in_t.sum())
            # Remove everyone observed at this time from the risk sets
            lost_a = int(((group == 1) & in_t).sum())
            lost_b = int(((group == 0) & in_t).sum())
            at_risk_a -= lost_a
            at_risk_b -= lost_b
            continue
        n = at_risk_a + at_risk_b
        e_a = d * (at_risk_a / n)
        v = (
            at_risk_a * at_risk_b * d * (n - d)
            / (n * n * (n - 1))
        ) if n > 1 else 0.0
        sum_o_a += d_a
        sum_e_a += e_a
        sum_var += v
        lost_a = int(((group == 1) & in_t).sum())
        lost_b = int(((group == 0) & in_t).sum())
        at_risk_a -= lost_a
        at_risk_b -= lost_b
    if sum_var <= 0:
        return {"chi2": float("nan"), "p_value": float("nan"),
                "observed_a": sum_o_a, "expected_a": sum_e_a,
                "hazard_direction": 0}
    chi2 = (sum_o_a - sum_e_a) ** 2 / sum_var
    # p = P(chi2 >= x) for 1 df = 2*(1 - Phi(sqrt(x))) = erfc(sqrt(x/2))
    p = math.erfc(math.sqrt(chi2 / 2.0))
    direction = 1 if sum_o_a > sum_e_a else (-1 if sum_o_a < sum_e_a else 0)
    return {
        "chi2": float(chi2),
        "p_value": float(p),
        "observed_a": float(sum_o_a),
        "expected_a": float(sum_e_a),
        "hazard_direction": int(direction),
    }


# ============================================================================
# Loaders
# ============================================================================


@dataclass
class CohortBundle:
    gene_names: List[str]
    sample_barcodes: List[str]
    X: np.ndarray            # (n_samples, n_genes), normalized z-scored
    y: np.ndarray            # (n_samples,) 0/1 normal/tumor
    master_rows: List[Dict[str, str]]


def load_step2_matrices(step2_dir: Path) -> CohortBundle:
    """Load the normalized expression matrix + master metadata for DE / clinical."""
    step2_dir = Path(step2_dir)
    norm_path = step2_dir / "normalized_primary_matrix.csv"
    master_path = step2_dir / "master_samples_primary.csv"
    if not norm_path.exists():
        raise FileNotFoundError(f"Missing {norm_path}")
    if not master_path.exists():
        raise FileNotFoundError(f"Missing {master_path}")

    # Normalized matrix: samples × genes (first column = sample_barcode)
    with open(norm_path, "r", newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        gene_names = list(header[1:])
        sample_barcodes: List[str] = []
        data_rows: List[List[float]] = []
        for row in reader:
            sample_barcodes.append(row[0])
            data_rows.append([float(v) if v not in ("", "NA", "NaN", "nan") else float("nan")
                              for v in row[1:]])
    X = np.asarray(data_rows, dtype=np.float64)

    # Master metadata
    with open(master_path, "r", newline="", encoding="utf-8") as fh:
        master_rows = [dict(r) for r in csv.DictReader(fh)]

    # Build y aligned to sample barcodes via master table
    master_by_bc: Dict[str, Dict[str, str]] = {}
    for r in master_rows:
        master_by_bc[r["sample_barcode"]] = r
    y = np.zeros(len(sample_barcodes), dtype=np.int64)
    aligned_rows: List[Dict[str, str]] = []
    for i, bc in enumerate(sample_barcodes):
        r = master_by_bc.get(bc, {})
        aligned_rows.append(r)
        y[i] = int(r.get("label_int", "0") or 0)

    return CohortBundle(
        gene_names=gene_names, sample_barcodes=sample_barcodes,
        X=X, y=y, master_rows=aligned_rows,
    )


def load_cdps_ranking(step5_dir: Path) -> Dict[str, Any]:
    """Load ranked_genes_cdps.csv and top25/top100 if present."""
    step5_dir = Path(step5_dir)
    ranked_path = step5_dir / "ranked_genes_cdps.csv"
    if not ranked_path.exists():
        raise FileNotFoundError(f"Missing {ranked_path}")
    with open(ranked_path, "r", newline="", encoding="utf-8") as fh:
        rows = [dict(r) for r in csv.DictReader(fh)]

    # Parse the summary for config echo
    summary: Dict[str, Any] = {}
    summary_path = step5_dir / "phase3_summary.json"
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as fh:
            summary = json.load(fh)

    return {
        "ranked_rows": rows,
        "ranked_path": str(ranked_path),
        "summary": summary,
    }


# ============================================================================
# Step 2: Differential expression
# ============================================================================


def run_differential_expression(
    *,
    cohort: CohortBundle,
    method: str = "welch",
    fdr_threshold: float = 0.05,
    lfc_threshold: float = 1.0,
) -> Dict[str, Any]:
    """Gene-wise tumor-vs-normal DE on the normalized matrix."""
    method = method.lower()
    if method not in ("welch", "ranksum", "nb"):
        raise ValueError(f"Unknown DE method: {method!r}")
    if method == "nb":
        logger.warning(
            "NB DE test requires statsmodels; falling back to Welch t-test."
        )
        method = "welch"

    X = cohort.X
    y = cohort.y
    tumor_mask = (y == 1)
    normal_mask = (y == 0)
    n_tumor = int(tumor_mask.sum())
    n_normal = int(normal_mask.sum())
    P = X.shape[1]

    stats = np.full(P, float("nan"), dtype=np.float64)
    pvals = np.full(P, float("nan"), dtype=np.float64)
    dfs = np.full(P, float("nan"), dtype=np.float64)
    effect = np.full(P, float("nan"), dtype=np.float64)
    mean_t = np.full(P, float("nan"), dtype=np.float64)
    mean_n = np.full(P, float("nan"), dtype=np.float64)

    if method == "welch":
        for j in range(P):
            a = X[tumor_mask, j]
            b = X[normal_mask, j]
            res = welch_t_test(a, b)
            stats[j] = res["t_stat"]
            pvals[j] = res["p_value"]
            dfs[j] = res["df"]
            mean_t[j] = res["mean_a"]
            mean_n[j] = res["mean_b"]
            effect[j] = res["mean_diff"]
    else:  # ranksum
        for j in range(P):
            a = X[tumor_mask, j]
            b = X[normal_mask, j]
            res = mann_whitney_u_test(a, b)
            stats[j] = res["z_stat"]
            pvals[j] = res["p_value"]
            mean_t[j] = float(a[~np.isnan(a)].mean()) if np.any(~np.isnan(a)) else float("nan")
            mean_n[j] = float(b[~np.isnan(b)].mean()) if np.any(~np.isnan(b)) else float("nan")
            effect[j] = mean_t[j] - mean_n[j]

    fdr = bh_fdr(pvals)
    sig_fdr = (fdr < fdr_threshold).astype(np.int64)
    sig_both = ((fdr < fdr_threshold) & (np.abs(effect) >= lfc_threshold)).astype(np.int64)

    rows: List[Dict[str, Any]] = []
    for j, g in enumerate(cohort.gene_names):
        rows.append({
            "gene": g,
            "n_tumor": n_tumor,
            "n_normal": n_normal,
            "mean_tumor_norm": float(mean_t[j]),
            "mean_normal_norm": float(mean_n[j]),
            "effect_size_norm": float(effect[j]),
            "test_stat": float(stats[j]),
            "df": float(dfs[j]) if method == "welch" else float("nan"),
            "p_value": float(pvals[j]),
            "fdr_bh": float(fdr[j]),
            "sig_fdr": int(sig_fdr[j]),
            "sig_fdr_and_effect": int(sig_both[j]),
        })

    qc = {
        "method": method,
        "n_samples": int(X.shape[0]),
        "n_tumor": n_tumor,
        "n_normal": n_normal,
        "n_genes_tested": P,
        "n_sig_fdr": int(sig_fdr.sum()),
        "n_sig_fdr_and_effect": int(sig_both.sum()),
        "fdr_threshold": float(fdr_threshold),
        "effect_threshold_abs": float(lfc_threshold),
    }

    logger.info(
        "DE [%s] tumor(n=%d) vs normal(n=%d): %d/%d FDR<%.3f, %d pass |effect|>=%.2f",
        method, n_tumor, n_normal, qc["n_sig_fdr"], P,
        fdr_threshold, qc["n_sig_fdr_and_effect"], lfc_threshold,
    )

    return {
        "rows": rows,
        "effect": effect,
        "fdr": fdr,
        "pvalue": pvals,
        "sig_fdr_and_effect": sig_both,
        "qc": qc,
    }


def build_top_cdps_de_support(
    *,
    cdps_rows: Sequence[Mapping[str, str]],
    de_rows: Sequence[Mapping[str, Any]],
    top_k: int = 100,
) -> List[Dict[str, Any]]:
    """Merge DE stats into the top-K CDPS table."""
    de_by_gene: Dict[str, Dict[str, Any]] = {r["gene"]: dict(r) for r in de_rows}
    out: List[Dict[str, Any]] = []
    for rec in list(cdps_rows)[: top_k]:
        gene = rec["gene"]
        de = de_by_gene.get(gene, {})
        out.append({
            "rank_cdps": int(rec.get("rank", -1) or -1),
            "gene": gene,
            "cdps": float(rec.get("cdps", 0.0) or 0.0),
            "mean_tumor_norm": float(de.get("mean_tumor_norm", float("nan"))),
            "mean_normal_norm": float(de.get("mean_normal_norm", float("nan"))),
            "effect_size_norm": float(de.get("effect_size_norm", float("nan"))),
            "test_stat": float(de.get("test_stat", float("nan"))),
            "p_value": float(de.get("p_value", float("nan"))),
            "fdr_bh": float(de.get("fdr_bh", float("nan"))),
            "sig_fdr": int(de.get("sig_fdr", 0) or 0),
            "sig_fdr_and_effect": int(de.get("sig_fdr_and_effect", 0) or 0),
        })
    return out


# ============================================================================
# Step 3: Enrichment (over-representation)
# ============================================================================


def _parse_gmt(path: Path) -> List[Dict[str, Any]]:
    """Parse an MSigDB-style .gmt file into a list of gene-set dicts."""
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n\r")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            name = parts[0]
            desc = parts[1]
            genes = [g.strip() for g in parts[2:] if g.strip()]
            if not genes:
                continue
            out.append({"name": name, "description": desc, "genes": genes})
    return out


def _enrichment_for_sets(
    *,
    gene_sets: Sequence[Mapping[str, Any]],
    foreground: Sequence[str],
    universe: Sequence[str],
    source: str,
) -> List[Dict[str, Any]]:
    fg_set = set(foreground)
    uni_set = set(universe)
    fg_in_uni = fg_set & uni_set
    n_fg = len(fg_in_uni)
    N = len(uni_set)

    rows: List[Dict[str, Any]] = []
    pvals: List[float] = []
    for gs in gene_sets:
        genes = set(gs["genes"]) & uni_set
        K = len(genes)
        if K == 0 or n_fg == 0 or N == 0:
            continue
        overlap = genes & fg_in_uni
        k = len(overlap)
        if k == 0:
            p = 1.0
        else:
            p = hypergeometric_sf(k, K, n_fg, N)
        expected = n_fg * (K / N)
        fold_enrich = (k / expected) if expected > 0 else float("nan")
        rows.append({
            "source": source,
            "pathway": gs["name"],
            "description": gs.get("description", ""),
            "K_pathway_size": K,
            "n_foreground": n_fg,
            "N_universe": N,
            "overlap": k,
            "expected": float(expected),
            "fold_enrichment": float(fold_enrich),
            "p_value": float(p),
            "overlap_genes": ";".join(sorted(overlap)),
        })
        pvals.append(p)
    if not rows:
        return []
    qvals = bh_fdr(np.asarray(pvals, dtype=np.float64))
    for r, q in zip(rows, qvals):
        r["fdr_bh"] = float(q)
    # Sort by ascending p_value
    rows.sort(key=lambda r: (r["p_value"], -r["overlap"]))
    return rows


def run_enrichment(
    *,
    foreground_genes: Sequence[str],
    universe_genes: Sequence[str],
    gmt_paths: Mapping[str, Optional[Path]],
) -> Dict[str, Any]:
    """Over-representation enrichment (hypergeometric) across GMT sources.

    ``gmt_paths`` maps source name (e.g. ``"hallmark"``) to an optional
    .gmt path. Any ``None``/missing path is skipped with a log message.
    """
    all_rows: Dict[str, List[Dict[str, Any]]] = {}
    skipped: List[Tuple[str, str]] = []
    gene_to_pathways: Dict[str, List[str]] = defaultdict(list)
    for source, path in gmt_paths.items():
        if path is None:
            skipped.append((source, "no GMT file supplied"))
            continue
        path = Path(path)
        if not path.exists():
            skipped.append((source, f"missing GMT: {path}"))
            continue
        try:
            sets = _parse_gmt(path)
            if not sets:
                skipped.append((source, f"empty GMT: {path}"))
                continue
        except Exception as exc:  # pragma: no cover
            skipped.append((source, f"parse error: {exc}"))
            continue
        logger.info("Enrichment [%s] loaded %d gene sets from %s", source, len(sets), path.name)
        rows = _enrichment_for_sets(
            gene_sets=sets, foreground=foreground_genes,
            universe=universe_genes, source=source,
        )
        all_rows[source] = rows
        # Build gene -> pathway membership (top hits only)
        for r in rows:
            for g in r["overlap_genes"].split(";"):
                if g:
                    gene_to_pathways[g].append(f"{source}:{r['pathway']}")

    # Flat summary table across sources, top-10 each
    summary_rows: List[Dict[str, Any]] = []
    for source, rows in all_rows.items():
        for r in rows[:10]:
            summary_rows.append({
                "source": source,
                "pathway": r["pathway"],
                "overlap": r["overlap"],
                "K_pathway_size": r["K_pathway_size"],
                "fold_enrichment": r["fold_enrichment"],
                "p_value": r["p_value"],
                "fdr_bh": r["fdr_bh"],
            })

    # Gene → pathway membership rows (top-K foreground only)
    membership_rows: List[Dict[str, Any]] = []
    for gene in foreground_genes:
        paths = gene_to_pathways.get(gene, [])
        if not paths:
            continue
        membership_rows.append({
            "gene": gene,
            "n_pathways": len(paths),
            "pathways": ";".join(paths),
        })

    return {
        "per_source_rows": all_rows,
        "summary_rows": summary_rows,
        "membership_rows": membership_rows,
        "skipped_sources": skipped,
    }


# ============================================================================
# Step 4: PPI network support
# ============================================================================


def _parse_edge_list(path: Path) -> List[Tuple[str, str]]:
    edges: List[Tuple[str, str]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", "\t").split("\t")
            if len(parts) < 2:
                continue
            a, b = parts[0].strip(), parts[1].strip()
            if not a or not b or a == b:
                continue
            edges.append((a, b))
    return edges


def run_network_support(
    *,
    top_genes: Sequence[str],
    universe_genes: Sequence[str],
    edge_list_path: Optional[Path],
) -> Dict[str, Any]:
    """PPI support: degree in global graph + induced subgraph + component id."""
    if edge_list_path is None or not Path(edge_list_path).exists():
        return {
            "skipped_reason": "no PPI edge list supplied" if edge_list_path is None
            else f"missing edge list: {edge_list_path}",
            "rows": [],
            "edges_top": [],
            "summary": {},
        }
    edges = _parse_edge_list(Path(edge_list_path))
    uni = set(universe_genes)
    # Keep only edges whose endpoints are in the universe
    adj: Dict[str, set] = defaultdict(set)
    for a, b in edges:
        if a in uni and b in uni:
            adj[a].add(b)
            adj[b].add(a)
    degree_global = {g: len(adj[g]) for g in universe_genes}
    top_set = set(top_genes)
    # Induced subgraph on top genes
    induced_adj: Dict[str, set] = {g: adj[g] & top_set for g in top_genes}
    degree_induced = {g: len(induced_adj[g]) for g in top_genes}

    # Connected components on induced subgraph (BFS)
    comp_id: Dict[str, int] = {}
    comp_sizes: Dict[int, int] = {}
    current = 0
    for g in top_genes:
        if g in comp_id:
            continue
        # BFS
        stack = [g]
        current += 1
        size = 0
        while stack:
            node = stack.pop()
            if node in comp_id:
                continue
            comp_id[node] = current
            size += 1
            for nb in induced_adj[node]:
                if nb not in comp_id:
                    stack.append(nb)
        comp_sizes[current] = size

    largest_comp = max(comp_sizes.values()) if comp_sizes else 0
    rows: List[Dict[str, Any]] = []
    for g in top_genes:
        rows.append({
            "gene": g,
            "degree_global": int(degree_global.get(g, 0)),
            "degree_in_top": int(degree_induced.get(g, 0)),
            "component_id": int(comp_id.get(g, -1)),
            "component_size": int(comp_sizes.get(comp_id.get(g, -1), 0)),
            "in_largest_component": int(
                comp_sizes.get(comp_id.get(g, -1), 0) == largest_comp and largest_comp > 0
            ),
        })

    edges_top: List[Dict[str, str]] = []
    seen: set = set()
    for g in top_genes:
        for nb in induced_adj[g]:
            key = tuple(sorted((g, nb)))
            if key in seen:
                continue
            seen.add(key)
            edges_top.append({"gene_a": key[0], "gene_b": key[1]})

    summary = {
        "n_top_genes": len(top_genes),
        "n_edges_top": len(edges_top),
        "n_components_top": len(comp_sizes),
        "largest_component_size": int(largest_comp),
        "global_graph_n_nodes": len({n for pair in edges for n in pair if n in uni}),
        "global_graph_n_edges": sum(len(v) for v in adj.values()) // 2,
    }

    return {
        "skipped_reason": None,
        "rows": rows,
        "edges_top": edges_top,
        "summary": summary,
    }


# ============================================================================
# Step 5a: Clinical association
# ============================================================================


def _env_levels_for_tumors(
    master_rows: Sequence[Mapping[str, str]],
    y: np.ndarray,
    env_col: str,
) -> np.ndarray:
    """Return an int array with env level or -1 for missing/non-tumor."""
    out = np.full(len(master_rows), -1, dtype=np.int64)
    for i, r in enumerate(master_rows):
        if int(y[i]) != 1:
            continue
        v = r.get(env_col, "")
        if v in ("", "NA"):
            continue
        try:
            out[i] = int(v)
        except ValueError:
            continue
    return out


def run_clinical_association(
    *,
    cohort: CohortBundle,
    top_genes: Sequence[str],
    env_names: Sequence[str],
) -> Dict[str, Any]:
    """For each top gene, per env_* column, Welch t-test across the two levels
    among tumor-only samples. Report min-p across environments + BH FDR."""
    gene_to_idx = {g: i for i, g in enumerate(cohort.gene_names)}
    env_cols = [f"env_{n}" if not n.startswith("env_") else n for n in env_names]

    rows: List[Dict[str, Any]] = []
    all_pvals: List[float] = []
    for gene in top_genes:
        j = gene_to_idx.get(gene)
        if j is None:
            continue
        row: Dict[str, Any] = {"gene": gene}
        min_p = float("inf")
        best_env = ""
        for env_name, env_col in zip(env_names, env_cols):
            levels = _env_levels_for_tumors(cohort.master_rows, cohort.y, env_col)
            idx0 = np.where(levels == 0)[0]
            idx1 = np.where(levels == 1)[0]
            if idx0.size < 2 or idx1.size < 2:
                row[f"{env_name}_p_value"] = float("nan")
                row[f"{env_name}_effect"] = float("nan")
                row[f"{env_name}_n0"] = int(idx0.size)
                row[f"{env_name}_n1"] = int(idx1.size)
                continue
            a = cohort.X[idx0, j]
            b = cohort.X[idx1, j]
            res = welch_t_test(a, b)
            row[f"{env_name}_p_value"] = float(res["p_value"])
            row[f"{env_name}_effect"] = float(res["mean_diff"])
            row[f"{env_name}_n0"] = int(idx0.size)
            row[f"{env_name}_n1"] = int(idx1.size)
            if math.isfinite(res["p_value"]) and res["p_value"] < min_p:
                min_p = float(res["p_value"])
                best_env = env_name
        row["min_p_value"] = float(min_p) if math.isfinite(min_p) else float("nan")
        row["best_env"] = best_env
        all_pvals.append(row["min_p_value"])
        rows.append(row)
    # BH over min-p across genes
    q = bh_fdr(np.asarray(all_pvals, dtype=np.float64))
    for r, qv in zip(rows, q):
        r["min_p_fdr_bh"] = float(qv)
    return {"rows": rows}


# ============================================================================
# Step 5b: Survival analysis (median split log-rank)
# ============================================================================


def _build_survival_arrays(
    master_rows: Sequence[Mapping[str, str]],
    y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (time, event, tumor_mask) from master metadata; NaN-safe."""
    n = len(master_rows)
    time = np.full(n, float("nan"), dtype=np.float64)
    event = np.zeros(n, dtype=np.int64)
    tumor = (y == 1)
    for i, r in enumerate(master_rows):
        if not tumor[i]:
            continue
        vital = (r.get("vital_status", "") or "").strip().lower()
        d_death = (r.get("days_to_death", "") or "").strip()
        d_last = (r.get("days_to_last_follow_up", "") or "").strip()
        ev = 1 if vital in ("dead", "deceased", "1", "true") else 0
        try:
            td = float(d_death) if d_death not in ("", "NA", "NaN") else float("nan")
        except ValueError:
            td = float("nan")
        try:
            tl = float(d_last) if d_last not in ("", "NA", "NaN") else float("nan")
        except ValueError:
            tl = float("nan")
        if ev == 1 and math.isfinite(td) and td > 0:
            time[i] = td
            event[i] = 1
        elif math.isfinite(tl) and tl > 0:
            time[i] = tl
            event[i] = 0
    return time, event, tumor


def run_survival_analysis(
    *,
    cohort: CohortBundle,
    top_genes: Sequence[str],
) -> Dict[str, Any]:
    time, event, tumor = _build_survival_arrays(cohort.master_rows, cohort.y)
    gene_to_idx = {g: i for i, g in enumerate(cohort.gene_names)}
    tumor_idx = np.where(tumor & np.isfinite(time))[0]
    rows: List[Dict[str, Any]] = []
    if tumor_idx.size < 10:
        logger.warning("Survival analysis skipped: only %d tumor samples with time.",
                       int(tumor_idx.size))
        return {"rows": rows, "n_analyzed": int(tumor_idx.size),
                "skipped_reason": "too few tumor samples with valid time"}

    t = time[tumor_idx]
    e = event[tumor_idx]
    n_events = int(e.sum())
    for gene in top_genes:
        j = gene_to_idx.get(gene)
        if j is None:
            continue
        expr = cohort.X[tumor_idx, j]
        valid = ~np.isnan(expr)
        if valid.sum() < 10:
            continue
        median = float(np.median(expr[valid]))
        high = expr >= median
        low = ~high
        if high.sum() < 3 or low.sum() < 3:
            continue
        res = log_rank_test(t[high], e[high], t[low], e[low])
        rows.append({
            "gene": gene,
            "n_tumor_with_time": int(valid.sum()),
            "n_events": n_events,
            "expr_median_split": median,
            "chi2": float(res["chi2"]),
            "p_value": float(res["p_value"]),
            "hazard_direction_high_vs_low": int(res["hazard_direction"]),
        })
    if rows:
        q = bh_fdr(np.asarray([r["p_value"] for r in rows], dtype=np.float64))
        for r, qv in zip(rows, q):
            r["fdr_bh"] = float(qv)
    return {"rows": rows, "n_analyzed": int(tumor_idx.size),
            "skipped_reason": None}


# ============================================================================
# Step 5c: Subgroup robustness
# ============================================================================


def run_subgroup_robustness(
    *,
    cohort: CohortBundle,
    top_genes: Sequence[str],
    env_names: Sequence[str],
) -> Dict[str, Any]:
    """For each gene, compute tumor-vs-normal direction/effect within each
    env stratum and summarise cross-stratum agreement."""
    gene_to_idx = {g: i for i, g in enumerate(cohort.gene_names)}
    env_cols = [f"env_{n}" if not n.startswith("env_") else n for n in env_names]
    rows: List[Dict[str, Any]] = []

    for gene in top_genes:
        j = gene_to_idx.get(gene)
        if j is None:
            continue
        row: Dict[str, Any] = {"gene": gene}
        directions: List[int] = []
        effects: List[float] = []
        for env_name, env_col in zip(env_names, env_cols):
            for level in ("0", "1"):
                sel_tumor: List[int] = []
                sel_normal: List[int] = []
                for i, r in enumerate(cohort.master_rows):
                    if r.get(env_col, "") != level:
                        continue
                    if int(cohort.y[i]) == 1:
                        sel_tumor.append(i)
                    else:
                        sel_normal.append(i)
                if len(sel_tumor) < 2 or len(sel_normal) < 2:
                    row[f"{env_name}_{level}_effect"] = float("nan")
                    row[f"{env_name}_{level}_p"] = float("nan")
                    row[f"{env_name}_{level}_dir"] = 0
                    continue
                a = cohort.X[np.asarray(sel_tumor, dtype=np.int64), j]
                b = cohort.X[np.asarray(sel_normal, dtype=np.int64), j]
                res = welch_t_test(a, b)
                eff = float(res["mean_diff"])
                row[f"{env_name}_{level}_effect"] = eff
                row[f"{env_name}_{level}_p"] = float(res["p_value"])
                direction = 1 if eff > 0 else (-1 if eff < 0 else 0)
                row[f"{env_name}_{level}_dir"] = direction
                if direction != 0:
                    directions.append(direction)
                    effects.append(eff)
        if directions:
            pos = sum(1 for d in directions if d > 0)
            neg = len(directions) - pos
            consistency = max(pos, neg) / len(directions)
            row["n_strata_evaluated"] = len(directions)
            row["majority_direction"] = 1 if pos >= neg else -1
            row["agreement_fraction"] = float(consistency)
            row["effect_mean"] = float(np.mean(effects))
            row["effect_std"] = float(np.std(effects, ddof=0))
        else:
            row["n_strata_evaluated"] = 0
            row["majority_direction"] = 0
            row["agreement_fraction"] = float("nan")
            row["effect_mean"] = float("nan")
            row["effect_std"] = float("nan")
        rows.append(row)
    return {"rows": rows}


# ============================================================================
# Step 6: Integrated validation score + final ranking
# ============================================================================


def _normalize_0_1(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size == 0:
        return x
    mask = np.isfinite(x)
    if not mask.any():
        return np.zeros_like(x)
    lo = float(np.min(x[mask]))
    hi = float(np.max(x[mask]))
    if hi - lo < 1e-12:
        out = np.zeros_like(x)
        out[~mask] = 0.0
        return out
    y = (x - lo) / (hi - lo)
    y = np.where(mask, y, 0.0)
    return np.clip(y, 0.0, 1.0)


def compute_integrated_validation_score(
    *,
    gene_names: Sequence[str],
    cdps_rows: Sequence[Mapping[str, Any]],
    de_rows: Sequence[Mapping[str, Any]],
    enrichment_membership: Sequence[Mapping[str, Any]],
    network_rows: Sequence[Mapping[str, Any]],
    clinical_rows: Sequence[Mapping[str, Any]],
    survival_rows: Sequence[Mapping[str, Any]],
    subgroup_rows: Sequence[Mapping[str, Any]],
    weights: Mapping[str, float],
) -> Dict[str, Any]:
    """Combine the validation components into a final ranked table."""
    gene_idx = {g: i for i, g in enumerate(gene_names)}
    P = len(gene_names)

    # DE support: -log10(fdr) * sign(effect), min-max'd
    de_score = np.zeros(P, dtype=np.float64)
    de_map = {r["gene"]: r for r in de_rows}
    for g, i in gene_idx.items():
        r = de_map.get(g)
        if r is None:
            continue
        fdr = float(r.get("fdr_bh", float("nan")))
        eff = float(r.get("effect_size_norm", 0.0))
        if math.isfinite(fdr) and fdr > 0:
            de_score[i] = -math.log10(max(fdr, 1e-300)) * (1 if eff >= 0 else 1)
            # Direction doesn't down-weight; both up/down regulation are valid signal
    de_norm = _normalize_0_1(de_score)

    # Enrichment: number of enriched pathways the gene belongs to
    enrich_count = np.zeros(P, dtype=np.float64)
    for m in enrichment_membership:
        g = m.get("gene", "")
        if g in gene_idx:
            enrich_count[gene_idx[g]] = float(m.get("n_pathways", 0) or 0)
    enrich_norm = _normalize_0_1(enrich_count)

    # Network: degree_global
    net_score = np.zeros(P, dtype=np.float64)
    for r in network_rows:
        g = r.get("gene", "")
        if g in gene_idx:
            net_score[gene_idx[g]] = float(r.get("degree_global", 0) or 0)
    net_norm = _normalize_0_1(net_score)

    # Clinical: -log10(min_p)
    clin_score = np.zeros(P, dtype=np.float64)
    for r in clinical_rows:
        g = r.get("gene", "")
        if g not in gene_idx:
            continue
        p = r.get("min_p_value", float("nan"))
        try:
            p_f = float(p) if p not in ("", None) else float("nan")
        except (TypeError, ValueError):
            p_f = float("nan")
        if math.isfinite(p_f) and p_f > 0:
            clin_score[gene_idx[g]] = -math.log10(max(p_f, 1e-300))
    # Survival adds into clinical if available
    for r in survival_rows:
        g = r.get("gene", "")
        if g not in gene_idx:
            continue
        p = r.get("p_value", float("nan"))
        try:
            p_f = float(p) if p not in ("", None) else float("nan")
        except (TypeError, ValueError):
            p_f = float("nan")
        if math.isfinite(p_f) and p_f > 0:
            v = -math.log10(max(p_f, 1e-300))
            clin_score[gene_idx[g]] = max(clin_score[gene_idx[g]], v)
    clin_norm = _normalize_0_1(clin_score)

    # Subgroup robustness: agreement_fraction
    sub_score = np.zeros(P, dtype=np.float64)
    for r in subgroup_rows:
        g = r.get("gene", "")
        if g not in gene_idx:
            continue
        v = r.get("agreement_fraction", 0.0)
        try:
            v_f = float(v) if v not in ("", None) else 0.0
        except (TypeError, ValueError):
            v_f = 0.0
        if math.isfinite(v_f):
            sub_score[gene_idx[g]] = v_f
    sub_norm = sub_score  # already 0..1 by definition

    # External: placeholder (0 by default)
    ext_norm = np.zeros(P, dtype=np.float64)

    raw_w = {
        "de": float(weights.get("de", 0.35)),
        "enrichment": float(weights.get("enrichment", 0.20)),
        "network": float(weights.get("network", 0.15)),
        "clinical": float(weights.get("clinical", 0.15)),
        "subgroup": float(weights.get("subgroup", 0.10)),
        "external": float(weights.get("external", 0.05)),
    }
    total = sum(raw_w.values())
    if total <= 0:
        raise ValueError("Sum of validation weights must be positive")
    w = {k: v / total for k, v in raw_w.items()}

    validation = (
        w["de"] * de_norm
        + w["enrichment"] * enrich_norm
        + w["network"] * net_norm
        + w["clinical"] * clin_norm
        + w["subgroup"] * sub_norm
        + w["external"] * ext_norm
    )

    # Pull CDPS values aligned to the gene universe
    cdps_map = {r["gene"]: r for r in cdps_rows}
    cdps_vals = np.zeros(P, dtype=np.float64)
    cdps_ranks = np.full(P, len(gene_names) + 1, dtype=np.int64)
    for g, i in gene_idx.items():
        r = cdps_map.get(g)
        if r is None:
            continue
        try:
            cdps_vals[i] = float(r.get("cdps", 0.0) or 0.0)
        except (TypeError, ValueError):
            cdps_vals[i] = 0.0
        try:
            cdps_ranks[i] = int(r.get("rank", P + 1) or P + 1)
        except (TypeError, ValueError):
            cdps_ranks[i] = P + 1
    cdps_norm = _normalize_0_1(cdps_vals)
    final_score = 0.5 * cdps_norm + 0.5 * validation

    order = np.argsort(-final_score, kind="mergesort")
    records: List[Dict[str, Any]] = []
    for rank, i in enumerate(order, start=1):
        records.append({
            "rank_final": int(rank),
            "gene": gene_names[int(i)],
            "final_score": float(final_score[int(i)]),
            "cdps_rank": int(cdps_ranks[int(i)]),
            "cdps": float(cdps_vals[int(i)]),
            "validation_score": float(validation[int(i)]),
            "de_norm": float(de_norm[int(i)]),
            "enrichment_norm": float(enrich_norm[int(i)]),
            "network_norm": float(net_norm[int(i)]),
            "clinical_norm": float(clin_norm[int(i)]),
            "subgroup_norm": float(sub_norm[int(i)]),
            "external_norm": float(ext_norm[int(i)]),
        })

    return {
        "records": records,
        "weights_effective": w,
        "weights_raw": raw_w,
        "components": {
            "de": de_norm,
            "enrichment": enrich_norm,
            "network": net_norm,
            "clinical": clin_norm,
            "subgroup": sub_norm,
            "external": ext_norm,
            "cdps": cdps_norm,
            "validation": validation,
            "final": final_score,
        },
    }


# ============================================================================
# Figures (graceful skip without matplotlib)
# ============================================================================


def _has_matplotlib() -> bool:
    try:
        import matplotlib  # noqa: F401
        return True
    except ImportError:
        return False


def generate_step6_figures(
    *,
    de_rows: Sequence[Mapping[str, Any]],
    cdps_top_genes: Sequence[str],
    enrichment_summary_rows: Sequence[Mapping[str, Any]],
    network_summary: Mapping[str, Any],
    final_records: Sequence[Mapping[str, Any]],
    output_dir: Path,
    style: Any = None,
    formats: Sequence[str] = ("svg",),
    fdr_threshold: float = 0.05,
    lfc_threshold: float = 1.0,
    top_k_bar: int = 25,
    clinical_assoc_rows: Optional[Sequence[Mapping[str, Any]]] = None,
    hallmark_rows: Optional[Sequence[Mapping[str, Any]]] = None,
    reactome_rows: Optional[Sequence[Mapping[str, Any]]] = None,
    immune_rows: Optional[Sequence[Mapping[str, Any]]] = None,
    survival_rows: Optional[Sequence[Mapping[str, Any]]] = None,
    subgroup_sensitivity_rows: Optional[Sequence[Mapping[str, Any]]] = None,
    pathway_membership_rows: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Tuple[List[str], List[Tuple[str, str]]]:
    generated: List[str] = []
    skipped: List[Tuple[str, str]] = []
    fig_names = [
        "fig_de_volcano",
        "fig_cdps_vs_de",
        "fig_enrichment_summary",
        "fig_final_rank_components",
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

        eff = np.asarray([float(r.get("effect_size_norm", 0.0)) for r in de_rows])
        fdr = np.asarray([float(r.get("fdr_bh", 1.0)) for r in de_rows])
        genes = [r["gene"] for r in de_rows]

        # --- Volcano ---
        try:
            fig, ax = plt.subplots(figsize=(3.8, 3.4))
            nlog = -np.log10(np.clip(fdr, 1e-300, 1.0))
            sig_mask = (fdr < fdr_threshold) & (np.abs(eff) >= lfc_threshold)
            ax.scatter(eff[~sig_mask], nlog[~sig_mask], s=5, alpha=0.25,
                       color="#888888", edgecolor="none")
            ax.scatter(eff[sig_mask], nlog[sig_mask], s=8, alpha=0.85,
                       color=semantic_color("tumor"), edgecolor="none",
                       label=f"FDR<{fdr_threshold:g} & |eff|≥{lfc_threshold:g}")
            # Highlight top CDPS genes
            top_set = set(cdps_top_genes)
            for i, g in enumerate(genes):
                if g in top_set:
                    ax.scatter(eff[i], nlog[i], s=20,
                               color=semantic_color("enriched"),
                               edgecolor="black", linewidth=0.4, zorder=5)
            ax.axhline(-math.log10(fdr_threshold), color="black", lw=0.7, ls="--")
            ax.axvline(lfc_threshold, color="black", lw=0.7, ls="--")
            ax.axvline(-lfc_threshold, color="black", lw=0.7, ls="--")
            ax.set_xlabel("Effect size (normalized)")
            ax.set_ylabel("-log10(FDR)")
            ax.set_title("Tumor vs normal DE")
            ax.legend(loc="upper left", fontsize=7)
            fig.tight_layout()
            paths = save_figure(fig, fig_dir / "fig_de_volcano", style=style, formats=formats)
            if paths:
                generated.append("fig_de_volcano")
        except Exception as exc:
            skipped.append(("fig_de_volcano", str(exc)))

        # --- CDPS vs DE effect scatter ---
        try:
            fig, ax = plt.subplots(figsize=(3.6, 3.2))
            de_map = {r["gene"]: float(r.get("effect_size_norm", 0.0)) for r in de_rows}
            fdr_map = {r["gene"]: float(r.get("fdr_bh", 1.0)) for r in de_rows}
            xs, ys, cs = [], [], []
            for rec in final_records:
                g = rec["gene"]
                if g not in de_map:
                    continue
                xs.append(float(rec["cdps"]))
                ys.append(de_map[g])
                cs.append(-math.log10(max(fdr_map.get(g, 1.0), 1e-300)))
            sc = ax.scatter(xs, ys, s=5, c=cs, cmap="viridis", alpha=0.6, edgecolor="none")
            fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.04).set_label("-log10(FDR)")
            ax.axhline(0, color="black", lw=0.6, ls=":")
            ax.set_xlabel("CDPS")
            ax.set_ylabel("DE effect size (norm)")
            ax.set_title("CDPS vs DE effect")
            fig.tight_layout()
            paths = save_figure(fig, fig_dir / "fig_cdps_vs_de", style=style, formats=formats)
            if paths:
                generated.append("fig_cdps_vs_de")
        except Exception as exc:
            skipped.append(("fig_cdps_vs_de", str(exc)))

        # --- Enrichment summary dot plot ---
        try:
            if enrichment_summary_rows:
                fig, ax = plt.subplots(figsize=(6.0, min(6.0, 0.3 * len(enrichment_summary_rows) + 1.0)))
                pathways = [r["pathway"] for r in enrichment_summary_rows]
                folds = [float(r.get("fold_enrichment", 0.0) or 0.0) for r in enrichment_summary_rows]
                qs = [-math.log10(max(float(r.get("fdr_bh", 1.0) or 1.0), 1e-300))
                      for r in enrichment_summary_rows]
                ns = [int(r.get("overlap", 0) or 0) for r in enrichment_summary_rows]
                ypos = np.arange(len(pathways))[::-1]
                sc = ax.scatter(folds, ypos, s=[10 + 4 * n for n in ns], c=qs,
                                cmap="viridis", edgecolor="black", linewidth=0.3)
                ax.set_yticks(ypos)
                ax.set_yticklabels(pathways, fontsize=7)
                ax.set_xlabel("Fold enrichment")
                fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.04).set_label("-log10(FDR)")
                ax.set_title("Top enriched pathways")
                fig.tight_layout()
                paths = save_figure(fig, fig_dir / "fig_enrichment_summary",
                                    style=style, formats=formats)
                if paths:
                    generated.append("fig_enrichment_summary")
            else:
                skipped.append(("fig_enrichment_summary", "no enrichment rows"))
        except Exception as exc:
            skipped.append(("fig_enrichment_summary", str(exc)))

        # --- Final rank components stacked bar ---
        try:
            top = list(final_records)[:top_k_bar]
            if not top:
                skipped.append(("fig_final_rank_components", "no final records"))
            else:
                components = ["de_norm", "enrichment_norm", "network_norm",
                              "clinical_norm", "subgroup_norm", "external_norm"]
                palette = categorical_colors(len(components))
                fig, ax = plt.subplots(figsize=(min(9.0, 0.3 * len(top) + 3.5), 3.6))
                bottom = np.zeros(len(top), dtype=np.float64)
                xpos = np.arange(len(top))
                for i, comp in enumerate(components):
                    vals = np.array([float(r.get(comp, 0.0)) for r in top])
                    ax.bar(xpos, vals, bottom=bottom, color=palette[i],
                           edgecolor="black", linewidth=0.4, label=comp.replace("_norm", ""))
                    bottom += vals
                ax.set_xticks(xpos)
                ax.set_xticklabels([r["gene"] for r in top], rotation=90, fontsize=7)
                ax.set_ylabel("Validation component value")
                ax.set_title(f"Top-{len(top)} validation components")
                ax.legend(loc="upper right", fontsize=7)
                fig.tight_layout()
                paths = save_figure(fig, fig_dir / "fig_final_rank_components",
                                    style=style, formats=formats)
                if paths:
                    generated.append("fig_final_rank_components")
        except Exception as exc:
            skipped.append(("fig_final_rank_components", str(exc)))

        # ---- Figure H2: clinical association heatmap ----
        if clinical_assoc_rows is not None:
            try:
                covariates = ["smoking", "sex", "histology", "country", "stage"]
                top_set_h2 = set(cdps_top_genes)
                gene_order_h2 = [r.get("gene", "") for r in clinical_assoc_rows
                                  if r.get("gene") in top_set_h2][:50]
                if not gene_order_h2:
                    gene_order_h2 = [r.get("gene", "") for r in clinical_assoc_rows][:50]
                assoc_by_gene = {r.get("gene", ""): r for r in clinical_assoc_rows}
                mat_h2 = []
                for gene in gene_order_h2:
                    r = assoc_by_gene.get(gene, {})
                    row_v = []
                    for cov in covariates:
                        p_v = r.get(f"{cov}_p_value")
                        try:
                            row_v.append(-math.log10(float(p_v)) if p_v and float(p_v) > 0 else 0.0)
                        except (ValueError, TypeError):
                            row_v.append(0.0)
                    mat_h2.append(row_v)
                mat_h2_np = np.array(mat_h2)
                n_g_h2 = len(gene_order_h2)
                fig, ax = plt.subplots(figsize=(7, max(5, n_g_h2 * 0.25 + 2)))
                im = ax.imshow(mat_h2_np, aspect="auto", cmap="cage_sequential", vmin=0)
                fig.colorbar(im, ax=ax, label="-log10(p-value)")
                ax.set_xticks(range(len(covariates)))
                ax.set_xticklabels([c.capitalize() for c in covariates], fontsize=9)
                ax.set_yticks(range(n_g_h2))
                ax.set_yticklabels(gene_order_h2, fontsize=6.5)
                sig_thresh = -math.log10(0.05)
                for ii in range(n_g_h2):
                    for jj in range(len(covariates)):
                        if mat_h2_np[ii, jj] > sig_thresh:
                            import matplotlib.patches as mpatches
                            ax.add_patch(plt.Rectangle((jj - 0.5, ii - 0.5), 1, 1,
                                                        fill=False, edgecolor="black",
                                                        linewidth=1.2))
                ax.set_title("Clinical Association — Top CDPS Genes\n(border = p < 0.05)")
                fig.tight_layout()
                paths = save_figure(fig, fig_dir / "fig_H2_clinical_assoc_heatmap",
                                    style=style, formats=formats)
                if paths:
                    generated.append("fig_H2_clinical_assoc_heatmap")
            except Exception as exc:
                skipped.append(("fig_H2_clinical_assoc_heatmap", str(exc)))

        # ---- Figure H3: pathway enrichment lollipop ----
        if hallmark_rows is not None or reactome_rows is not None:
            try:
                import matplotlib.patches as mpatches_h3
                all_path_rows = list(hallmark_rows or []) + list(reactome_rows or [])
                all_path_rows = [r for r in all_path_rows
                                  if int(r.get("overlap", 0) or 0) >= 2]
                top_path = sorted(all_path_rows, key=lambda r: float(r.get("p_value", 1) or 1))[:20]
                if top_path:
                    path_labels = []
                    for r in top_path:
                        src = str(r.get("source", ""))
                        pth = str(r.get("pathway", ""))
                        pth = pth.replace("HALLMARK_", "").replace("REACTOME_", "")
                        path_labels.append(f"[{src[:3].upper()}] {pth[:50]}")
                    fold_enr = [float(r.get("fold_enrichment", 1) or 1) for r in top_path]
                    pvals_h3 = [float(r.get("p_value", 1) or 1) for r in top_path]
                    overlaps_h3 = [int(r.get("overlap", 0) or 0) for r in top_path]
                    c_hallmark = semantic_color("highlight")
                    c_reactome = semantic_color("normal")
                    colors_h3 = [c_hallmark if r.get("source") == "hallmark" else c_reactome
                                  for r in top_path]
                    fig, ax = plt.subplots(
                        figsize=(9, max(5, len(path_labels) * 0.45 + 1.5)),
                    )
                    y_pos = list(range(len(path_labels)))[::-1]
                    ax.barh(y_pos, fold_enr, height=0.6,
                            color=[c + "55" for c in colors_h3], edgecolor="none")
                    for yi, (fe, p, n_ol, col_h) in enumerate(
                        zip(fold_enr, pvals_h3, overlaps_h3, colors_h3)
                    ):
                        ax.plot(fe, y_pos[yi], "o", color=col_h,
                                markersize=max(4, min(12, n_ol * 1.5)), zorder=3)
                        ax.text(fe + 0.05, y_pos[yi], f"n={n_ol}, p={p:.3f}",
                                va="center", fontsize=7.5, color="#333333")
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels(path_labels, fontsize=8)
                    ax.set_xlabel("Fold enrichment")
                    ax.set_title("Pathway Enrichment — Top 20 (Hallmark + Reactome)")
                    patches_h3 = [
                        mpatches_h3.Patch(color=c_hallmark, label="Hallmark"),
                        mpatches_h3.Patch(color=c_reactome, label="Reactome"),
                    ]
                    ax.legend(handles=patches_h3, fontsize=8, loc="lower right")
                    fig.tight_layout()
                    paths = save_figure(fig, fig_dir / "fig_H3_pathway_enrichment",
                                        style=style, formats=formats)
                    if paths:
                        generated.append("fig_H3_pathway_enrichment")
            except Exception as exc:
                skipped.append(("fig_H3_pathway_enrichment", str(exc)))

        # ---- Figure H4: survival summary lollipop ----
        if survival_rows is not None:
            try:
                import matplotlib.patches as mpatches_h4
                cdps_set_h4 = set(cdps_top_genes)
                surv = [r for r in survival_rows if r.get("gene") in cdps_set_h4]
                surv = sorted(surv, key=lambda r: float(r.get("p_value", 1) or 1))
                if surv:
                    genes_h4  = [r["gene"] for r in surv]
                    pvals_h4  = [float(r.get("p_value", 1) or 1) for r in surv]
                    haz_dir   = [int(r.get("hazard_direction_high_vs_low", 0) or 0)
                                  for r in surv]
                    log_p_h4  = [-math.log10(p) if p > 0 else 0 for p in pvals_h4]
                    c_worse   = semantic_color("tumor")
                    c_better  = semantic_color("normal")
                    c_na      = "#aaaaaa"
                    cols_h4   = [c_worse if d == 1 else (c_better if d == -1 else c_na)
                                  for d in haz_dir]
                    n_h4      = len(genes_h4)
                    fig, ax   = plt.subplots(figsize=(7, max(4, n_h4 * 0.22 + 2)))
                    y_pos_h4  = list(range(n_h4))
                    ax.barh(y_pos_h4, log_p_h4, height=0.5,
                            color=[c + "44" for c in cols_h4], edgecolor="none")
                    ax.scatter(log_p_h4, y_pos_h4, color=cols_h4, s=30, zorder=3)
                    p_thresh_h4 = -math.log10(0.05)
                    ax.axvline(p_thresh_h4, color="#d62728", linewidth=0.9,
                               linestyle="--", alpha=0.6, label="p = 0.05")
                    ax.set_yticks(y_pos_h4)
                    ax.set_yticklabels(genes_h4, fontsize=6.5)
                    ax.set_xlabel("-log10(p-value)")
                    ax.set_title("Survival Association — Top CDPS Genes (TCGA ESCC, OS)")
                    patches_h4 = [
                        mpatches_h4.Patch(color=c_worse,  label="High expr = worse OS"),
                        mpatches_h4.Patch(color=c_better, label="High expr = better OS"),
                    ]
                    ax.legend(handles=patches_h4, fontsize=8)
                    fig.tight_layout()
                    paths = save_figure(fig, fig_dir / "fig_H4_survival_summary",
                                        style=style, formats=formats)
                    if paths:
                        generated.append("fig_H4_survival_summary")
            except Exception as exc:
                skipped.append(("fig_H4_survival_summary", str(exc)))

        # ---- Figure H5: subgroup sensitivity heatmap ----
        if subgroup_sensitivity_rows is not None:
            try:
                sg_rows = list(subgroup_sensitivity_rows)
                sg_envs = ["smoking", "sex", "histology", "country", "stage"]
                cdps_set_h5 = set(cdps_top_genes)
                gene_order_h5 = [r.get("gene", "") for r in sg_rows
                                   if r.get("gene") in cdps_set_h5][:25]
                if not gene_order_h5:
                    gene_order_h5 = [r.get("gene", "") for r in sg_rows][:25]
                by_gene_h5 = {r.get("gene", ""): r for r in sg_rows}
                mat_h5 = np.full((len(gene_order_h5), len(sg_envs)), np.nan)
                for gi, gene in enumerate(gene_order_h5):
                    r = by_gene_h5.get(gene, {})
                    for ei, env in enumerate(sg_envs):
                        v = r.get("agreement_fraction")
                        try:
                            mat_h5[gi, ei] = float(v) if v is not None else np.nan
                        except (ValueError, TypeError):
                            pass
                n_g_h5 = len(gene_order_h5)
                fig, ax = plt.subplots(
                    figsize=(max(6, len(sg_envs) * 1.2 + 2), max(5, n_g_h5 * 0.25 + 2)),
                )
                im = ax.imshow(mat_h5, aspect="auto", cmap="cage_sequential", vmin=0, vmax=1)
                fig.colorbar(im, ax=ax, label="Agreement fraction")
                ax.set_xticks(range(len(sg_envs)))
                ax.set_xticklabels([s.capitalize() for s in sg_envs], fontsize=9)
                ax.set_yticks(range(n_g_h5))
                ax.set_yticklabels(gene_order_h5, fontsize=6.5)
                ax.set_title("Subgroup Sensitivity — Top CDPS Genes")
                fig.tight_layout()
                paths = save_figure(fig, fig_dir / "fig_H5_subgroup_sensitivity",
                                    style=style, formats=formats)
                if paths:
                    generated.append("fig_H5_subgroup_sensitivity")
            except Exception as exc:
                skipped.append(("fig_H5_subgroup_sensitivity", str(exc)))

        # ---- Figure H6: final ranking waterfall ----
        try:
            top_h6 = list(final_records)[:25]
            if top_h6:
                waterfall_comps = [
                    ("de_norm",        "DE support",       "#2166ac"),
                    ("enrichment_norm","Pathway enrich.",  "#4dac26"),
                    ("subgroup_norm",  "Subgroup robust.", "#f1a340"),
                    ("external_norm",  "External valid.",  "#d7191c"),
                    ("cdps",           "CDPS base",        "#aaaaaa"),
                ]
                n_h6 = len(top_h6)
                fig, ax = plt.subplots(figsize=(13, max(4, n_h6 * 0.38 + 2)))
                xp = list(range(n_h6))
                bottoms_h6 = [0.0] * n_h6
                for col_w, lbl_w, col_c in waterfall_comps:
                    vals_w = [float(r.get(col_w, 0) or 0) for r in top_h6]
                    ax.bar(xp, vals_w, bottom=bottoms_h6, width=0.65, label=lbl_w,
                           color=col_c, edgecolor="white", linewidth=0.3, alpha=0.9)
                    bottoms_h6 = [b + v for b, v in zip(bottoms_h6, vals_w)]
                final_scores_h6 = [float(r.get("final_score", 0) or 0) for r in top_h6]
                ax.plot(xp, final_scores_h6, "k.", markersize=5, zorder=5, label="Final score")
                ax.set_xticks(xp)
                ax.set_xticklabels([r["gene"] for r in top_h6], rotation=45, ha="right", fontsize=8)
                ax.set_ylabel("Score (stacked components)")
                ax.set_title(f"Final Validated Gene Ranking — Top {n_h6} Genes")
                ax.legend(fontsize=8, loc="upper right")
                fig.tight_layout()
                paths = save_figure(fig, fig_dir / "fig_H6_final_ranking_waterfall",
                                    style=style, formats=formats)
                if paths:
                    generated.append("fig_H6_final_ranking_waterfall")
        except Exception as exc:
            skipped.append(("fig_H6_final_ranking_waterfall", str(exc)))

        # ---- Figure H7: gene-pathway bubble chart ----
        if pathway_membership_rows is not None:
            try:
                pm_rows = list(pathway_membership_rows)
                cdps_set_h7 = set(cdps_top_genes)
                pm_by_gene = {r.get("gene", ""): r for r in pm_rows}
                gene_order_h7 = [g for g in cdps_top_genes
                                   if g in pm_by_gene
                                   and int(pm_by_gene[g].get("n_pathways", 0) or 0) > 0][:25]
                all_paths_h7: set[str] = set()
                for gene in gene_order_h7:
                    pth_str = pm_by_gene[gene].get("pathways", "") or ""
                    for p in pth_str.split(";"):
                        pp = p.strip().split(":")[-1][:40]
                        if pp:
                            all_paths_h7.add(pp)
                path_list_h7 = sorted(all_paths_h7)[:30]
                if gene_order_h7 and path_list_h7:
                    mat_h7 = np.zeros((len(gene_order_h7), len(path_list_h7)))
                    for gi, gene in enumerate(gene_order_h7):
                        pth_str = pm_by_gene[gene].get("pathways", "") or ""
                        for p in pth_str.split(";"):
                            pname = p.strip().split(":")[-1][:40]
                            if pname in path_list_h7:
                                mat_h7[gi, path_list_h7.index(pname)] = 1
                    fig_h7_h = max(4, len(gene_order_h7) * 0.35 + 1.5)
                    fig_h7_w = max(8, len(path_list_h7) * 0.6 + 2)
                    fig, ax = plt.subplots(figsize=(fig_h7_w, fig_h7_h))
                    for gi in range(len(gene_order_h7)):
                        for pi in range(len(path_list_h7)):
                            col_h7 = semantic_color("normal") if mat_h7[gi, pi] > 0 else "#dddddd"
                            sz_h7  = 60 if mat_h7[gi, pi] > 0 else 5
                            ax.scatter(pi, gi, s=sz_h7, color=col_h7,
                                       alpha=0.8 if mat_h7[gi, pi] > 0 else 0.2,
                                       edgecolors="white", linewidth=0.5, zorder=2)
                    ax.set_xticks(range(len(path_list_h7)))
                    ax.set_xticklabels(path_list_h7, rotation=45, ha="right", fontsize=6.5)
                    ax.set_yticks(range(len(gene_order_h7)))
                    ax.set_yticklabels(gene_order_h7, fontsize=8)
                    ax.set_title("Gene-Pathway Membership — Top CDPS Genes")
                    fig.tight_layout()
                    paths = save_figure(fig, fig_dir / "fig_H7_gene_pathway_bubble",
                                        style=style, formats=formats)
                    if paths:
                        generated.append("fig_H7_gene_pathway_bubble")
            except Exception as exc:
                skipped.append(("fig_H7_gene_pathway_bubble", str(exc)))

        # ----------------------------------------------------------------
        # fig_H8_immune_enrichment
        # Lollipop: top significant C7 ImmuneSigDB gene sets enriched in
        # top CDPS genes.  Sets are grouped by broad immune category inferred
        # from the pathway name prefix (T_CELL, B_CELL, NK_CELL, DC,
        # NEUTROPHIL, CYTOKINE, CHECKPOINT, OTHER).  Bars coloured by
        # category; dot = -log10(FDR); shown only if immune_rows supplied.
        # ----------------------------------------------------------------
        try:
            if immune_rows is not None and len(immune_rows) > 0:
                _IMMUNE_CATEGORIES = [
                    ("T_CELL",      ["T_CELL", "CD8", "CD4", "TH1", "TH2", "TH17", "TREG", "TCR"]),
                    ("B_CELL",      ["B_CELL", "BCR", "GERMINAL", "PLASMA"]),
                    ("NK_CELL",     ["NK_CELL", "NKT", "NK ", "NATURAL_KILLER"]),
                    ("DC",          ["DC", "DENDRITIC"]),
                    ("NEUTROPHIL",  ["NEUTROPHIL", "GRANULOCYTE"]),
                    ("MACROPHAGE",  ["MACROPHAGE", "MONOCYTE", "MYELOID", "M1_", "M2_"]),
                    ("CYTOKINE",    ["CYTOKINE", "INTERFERON", "IFN", "TNF", "IL", "INTERLEUKIN"]),
                    ("CHECKPOINT",  ["CHECKPOINT", "PD1", "PDL1", "CTLA4", "LAG3", "TIM3"]),
                ]

                def _immune_category(name: str) -> str:
                    name_u = name.upper()
                    for cat, tokens in _IMMUNE_CATEGORIES:
                        if any(tok in name_u for tok in tokens):
                            return cat
                    return "OTHER"

                sig_immune = [
                    r for r in immune_rows
                    if float(r.get("fdr_bh", 1.0)) < 0.25 and int(r.get("overlap", 0)) >= 2
                ][:30]

                if sig_immune:
                    all_cats = sorted(set(_immune_category(r["pathway"]) for r in sig_immune))
                    cat_colors = dict(zip(all_cats, _ccolors(len(all_cats))))

                    labels_h8 = []
                    for r in sig_immune:
                        pth = r["pathway"]
                        # strip common ImmuneSigDB prefixes (GSExxxxx_, LIxx_, etc.)
                        import re as _re
                        pth = _re.sub(r'^(GSE\d+_|LI\d+_|BENOTMANE_|GOLDRATH_|KAECH_|SHEN_TOLGA_)', '', pth)
                        labels_h8.append(pth[:52] + "…" if len(pth) > 52 else pth)

                    vals_h8 = [-float(_re.sub(r'[^0-9e\.\-]', '', str(r.get("fdr_bh", 1.0))) or "1")
                               for r in sig_immune]
                    # use -log10(fdr_bh)
                    import math as _math
                    vals_h8 = [
                        -_math.log10(max(float(r.get("fdr_bh", 1.0)), 1e-10))
                        for r in sig_immune
                    ]
                    bar_colors_h8 = [
                        cat_colors[_immune_category(r["pathway"])] for r in sig_immune
                    ]

                    fig_h, ax_h = plt.subplots(figsize=(8, max(4, len(sig_immune) * 0.32 + 1)))
                    apply_style(fig_h, style)
                    y_pos = list(range(len(sig_immune) - 1, -1, -1))
                    ax_h.barh(y_pos, vals_h8, color=bar_colors_h8, alpha=0.75, height=0.55)
                    ax_h.scatter(vals_h8, y_pos, color=bar_colors_h8, s=38, zorder=3,
                                 edgecolors="white", linewidths=0.5)
                    ax_h.set_yticks(y_pos)
                    ax_h.set_yticklabels(labels_h8, fontsize=6.5)
                    ax_h.axvline(-_math.log10(0.05), color="#888", linewidth=0.8,
                                 linestyle="--", label="FDR = 0.05")
                    ax_h.axvline(-_math.log10(0.25), color="#bbb", linewidth=0.6,
                                 linestyle=":", label="FDR = 0.25")
                    ax_h.set_xlabel("−log₁₀(FDR)", fontsize=style.axis_label_font_size if style else 10)
                    ax_h.set_title("H8 — Immune Gene-Set Enrichment (C7 ImmuneSigDB)",
                                   fontsize=style.title_font_size if style else 12)
                    # Legend for categories
                    from matplotlib.patches import Patch
                    legend_patches = [
                        Patch(color=cat_colors[c], label=c.replace("_", " ").title())
                        for c in all_cats
                    ]
                    ax_h.legend(handles=legend_patches, fontsize=7, loc="lower right",
                                title="Immune category", title_fontsize=7,
                                framealpha=0.7, ncol=max(1, len(all_cats) // 4))
                    fig_h.tight_layout()
                    paths_h8 = save_figure(
                        fig_h, fig_dir / "fig_H8_immune_enrichment",
                        style=style, formats=list(formats),
                    )
                    if paths_h8:
                        generated.append("fig_H8_immune_enrichment")
                else:
                    skipped.append(("fig_H8_immune_enrichment",
                                    "no immune gene sets passed FDR<0.25 & overlap>=2"))
            else:
                skipped.append(("fig_H8_immune_enrichment", "no immune_rows supplied"))
        except Exception as exc:
            skipped.append(("fig_H8_immune_enrichment", str(exc)))

    except Exception as exc:  # pragma: no cover
        for n in fig_names:
            skipped.append((n, str(exc)))
    return generated, skipped


# ============================================================================
# Top-level driver
# ============================================================================


def run_step6_validation(
    *,
    step2_dir: Path,
    step5_dir: Path,
    output_dir: Path,
    de_method: str = "welch",
    fdr_threshold: float = 0.05,
    lfc_threshold: float = 1.0,
    top_k_for_enrichment: int = 100,
    top_k_for_clinical: int = 100,
    gmt_hallmark: Optional[Path] = None,
    gmt_reactome: Optional[Path] = None,
    gmt_kegg_go: Optional[Path] = None,
    gmt_immune: Optional[Path] = None,
    ppi_edge_list: Optional[Path] = None,
    env_names: Sequence[str] = ("smoking", "sex", "histology", "country", "stage"),
    run_enrichment_flag: bool = False,
    run_network_flag: bool = False,
    run_survival_flag: bool = False,
    run_subgroup_flag: bool = False,
    weights: Optional[Mapping[str, float]] = None,
    seed: int = 2026,
) -> Dict[str, Any]:
    weights = dict(weights or {
        "de": 0.35, "enrichment": 0.20, "network": 0.15,
        "clinical": 0.15, "subgroup": 0.10, "external": 0.05,
    })
    cohort = load_step2_matrices(step2_dir)
    bundle = load_cdps_ranking(step5_dir)
    cdps_rows = bundle["ranked_rows"]

    logger.info(
        "Step 6 input | genes=%d samples=%d (tumor=%d / normal=%d) cdps_rows=%d",
        len(cohort.gene_names), int(cohort.X.shape[0]),
        int((cohort.y == 1).sum()), int((cohort.y == 0).sum()),
        len(cdps_rows),
    )

    # 1. DE
    de_out = run_differential_expression(
        cohort=cohort, method=de_method,
        fdr_threshold=fdr_threshold, lfc_threshold=lfc_threshold,
    )

    # CDPS x DE merge
    cdps_de_support = build_top_cdps_de_support(
        cdps_rows=cdps_rows, de_rows=de_out["rows"], top_k=top_k_for_enrichment,
    )

    # 2. Enrichment
    enrichment_out: Dict[str, Any] = {
        "per_source_rows": {}, "summary_rows": [], "membership_rows": [],
        "skipped_sources": [],
    }
    if run_enrichment_flag:
        top_genes_for_enrich = [r["gene"] for r in cdps_rows[: top_k_for_enrichment]]
        enrichment_out = run_enrichment(
            foreground_genes=top_genes_for_enrich,
            universe_genes=cohort.gene_names,
            gmt_paths={
                "hallmark": gmt_hallmark,
                "reactome": gmt_reactome,
                "kegg_go": gmt_kegg_go,
                "immune": gmt_immune,
            },
        )

    # 3. Network
    top_genes_for_net = [r["gene"] for r in cdps_rows[: max(top_k_for_clinical, 100)]]
    network_out: Dict[str, Any] = {"skipped_reason": None, "rows": [],
                                   "edges_top": [], "summary": {}}
    if run_network_flag:
        network_out = run_network_support(
            top_genes=top_genes_for_net,
            universe_genes=cohort.gene_names,
            edge_list_path=ppi_edge_list,
        )

    # 4. Clinical association on top CDPS genes
    top_genes_for_clin = [r["gene"] for r in cdps_rows[: top_k_for_clinical]]
    clinical_out = run_clinical_association(
        cohort=cohort, top_genes=top_genes_for_clin, env_names=env_names,
    )

    # 5. Survival (optional)
    survival_out: Dict[str, Any] = {"rows": [], "n_analyzed": 0, "skipped_reason": "not requested"}
    if run_survival_flag:
        survival_out = run_survival_analysis(
            cohort=cohort, top_genes=top_genes_for_clin,
        )

    # 6. Subgroup robustness (optional)
    subgroup_out: Dict[str, Any] = {"rows": []}
    if run_subgroup_flag:
        subgroup_out = run_subgroup_robustness(
            cohort=cohort, top_genes=top_genes_for_clin, env_names=env_names,
        )

    # 7. Integrated validation score
    integrated = compute_integrated_validation_score(
        gene_names=cohort.gene_names,
        cdps_rows=cdps_rows,
        de_rows=de_out["rows"],
        enrichment_membership=enrichment_out["membership_rows"],
        network_rows=network_out["rows"],
        clinical_rows=clinical_out["rows"],
        survival_rows=survival_out["rows"],
        subgroup_rows=subgroup_out["rows"],
        weights=weights,
    )

    return {
        "cohort": {
            "n_samples": int(cohort.X.shape[0]),
            "n_tumor": int((cohort.y == 1).sum()),
            "n_normal": int((cohort.y == 0).sum()),
            "n_genes": int(len(cohort.gene_names)),
        },
        "cdps_top_genes": [r["gene"] for r in cdps_rows[:25]],
        "de": de_out,
        "cdps_de_support": cdps_de_support,
        "enrichment": enrichment_out,
        "network": network_out,
        "clinical": clinical_out,
        "survival": survival_out,
        "subgroup": subgroup_out,
        "integrated": integrated,
        "gene_names": cohort.gene_names,
        "config": {
            "de_method": de_method,
            "fdr_threshold": float(fdr_threshold),
            "lfc_threshold": float(lfc_threshold),
            "top_k_for_enrichment": int(top_k_for_enrichment),
            "top_k_for_clinical": int(top_k_for_clinical),
            "env_names": list(env_names),
            "weights": weights,
            "run_enrichment": bool(run_enrichment_flag),
            "run_network": bool(run_network_flag),
            "run_survival": bool(run_survival_flag),
            "run_subgroup_sensitivity": bool(run_subgroup_flag),
            "gmt_hallmark": str(gmt_hallmark) if gmt_hallmark else None,
            "gmt_reactome": str(gmt_reactome) if gmt_reactome else None,
            "gmt_kegg_go": str(gmt_kegg_go) if gmt_kegg_go else None,
            "ppi_edge_list": str(ppi_edge_list) if ppi_edge_list else None,
            "step2_dir": str(step2_dir),
            "step5_dir": str(step5_dir),
            "seed": int(seed),
        },
    }
