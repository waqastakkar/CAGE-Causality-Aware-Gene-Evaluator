"""CAGE Step 8 — Affymetrix GEO external validation.

Multi-dataset external validation of CAGE candidate driver genes
against GEO ESCA Affymetrix microarray cohorts.

Accessible via the Step 8 unified CLI:

    python -m cage.step8_external_validation_and_release validate-geo \\
        --geo-dir outputs/step8_geo_prepared \\
        --step5-dir outputs/step5_cdps \\
        --step6-dir outputs/step6_validation \\
        --output-dir outputs/step8_geo_validation

Can also be called as a standalone module:

    python -m cage.step8_geo_validation \\
        --geo-dir outputs/step8_geo_prepared \\
        --step5-dir outputs/step5_cdps \\
        --step6-dir outputs/step6_validation \\
        --output-dir outputs/step8_geo_validation

Outputs (in --output-dir)
-------------------------
per_dataset/<GSE>/de_replication.csv
per_dataset/<GSE>/model_metrics.json
combined_de_replication.csv
cross_cohort_concordance_summary.csv
cross_cohort_model_summary.csv
geo_validation_summary.json
figures/  (B + C series)
logs/step8_geo_validation.log
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from . import cli_args as _cli_args
    from .cli_args import configure_logging as _configure_logging
    _HAS_CLI_ARGS = True
except ImportError:
    _HAS_CLI_ARGS = False


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _setup_logging(log_file: Path, verbose: bool) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s | %(levelname)s | %(message)s"
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, mode="w", encoding="utf-8"),
    ]
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format=fmt,
        handlers=handlers,
        force=True,
    )


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _write_csv(path: Path, rows: List[Dict[str, str]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    logging.info("Wrote %d rows -> %s", len(rows), path)


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=str)
    logging.info("Wrote JSON -> %s", path)


def _fmt(v: float, d: int = 4) -> str:
    try:
        return f"{float(v):.{d}f}"
    except (ValueError, TypeError):
        return str(v)


# ---------------------------------------------------------------------------
# Reference data loaders
# ---------------------------------------------------------------------------

def load_reference_genes(step5_dir: Path, top_k: int) -> List[str]:
    """Load top-K CDPS genes as the validation target list."""
    src = step5_dir / f"top{top_k}_genes_cdps.csv"
    if not src.is_file():
        src = step5_dir / "ranked_genes_cdps.csv"
    rows = _read_csv(src)[:top_k]
    genes = [r["gene"] for r in rows if r.get("gene")]
    logging.info("Loaded %d reference CDPS genes from %s", len(genes), src.name)
    return genes


def load_tcga_gene_list(step5_dir: Path, step4_dir: Optional[Path] = None) -> List[str]:
    """Load the model feature order for external classifier alignment.

    The deep checkpoints expect the exact Step-4 training feature order. CDPS
    ranking files are intentionally sorted by score and must not be used as
    model-input order.
    """
    if step4_dir is not None and (step4_dir / "gate_weights.csv").is_file():
        src = step4_dir / "gate_weights.csv"
        rows = _read_csv(src)
        genes = [r["gene"] for r in rows if r.get("gene")]
        logging.info("TCGA model gene order: %d genes from %s", len(genes), src)
        return genes

    src = step5_dir / "ranked_genes_cdps.csv"
    rows = _read_csv(src)
    genes = [r["gene"] for r in rows if r.get("gene")]
    logging.warning(
        "Falling back to ranked CDPS gene order for model alignment; "
        "this is suitable for replication summaries, not ideal for model transfer."
    )
    logging.info("TCGA reference gene list: %d genes", len(genes))
    return genes


def load_reference_de(step6_dir: Path) -> Dict[str, str]:
    """gene -> 'up'/'down' reference DE direction from Step 6."""
    src = step6_dir / "differential_expression_results.csv"
    rows = _read_csv(src)
    ref: Dict[str, str] = {}
    for r in rows:
        g = r.get("gene", "")
        es = r.get("effect_size_norm", "0")
        if g:
            try:
                ref[g] = "up" if float(es) > 0 else "down"
            except ValueError:
                pass
    logging.info("Reference DE directions loaded: %d genes", len(ref))
    return ref


# ---------------------------------------------------------------------------
# GEO dataset loader
# ---------------------------------------------------------------------------

SKIP_ACCESSIONS = {
    "GSE67269",   # miRNA TaqMan platform — no mRNA gene symbols
    "GSE75241",   # tumor-only, no normal samples
    "GSE53624",   # Agilent-038314 Feature-Number version: GPL18109 has no HGNC gene symbols
    "GSE53625",   # Agilent-038314 Feature-Number version: GPL18109 has no HGNC gene symbols
    "GSE20347",   # small paired cohort with complete separation; excluded from headline validation
    "GSE23400",   # threshold imbalance: sensitivity=1.0 with low specificity; excluded from headline validation
}


def _load_gene_matrix(path: Path) -> Tuple[List[str], List[str], np.ndarray]:
    """Load expression_gene_matrix.csv.

    Returns (gene_names, sample_ids, X) where X.shape = (n_samples, n_genes).
    First column is gene_symbol; remaining columns are GSM IDs.
    """
    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        sample_ids = header[1:]
        genes: List[str] = []
        rows: List[List[float]] = []
        for row in reader:
            gene = row[0].strip()
            if not gene:
                continue
            try:
                vals = [float(v) if v.strip() else float("nan") for v in row[1:]]
            except ValueError:
                continue
            genes.append(gene)
            rows.append(vals)

    X = np.array(rows, dtype=np.float64).T  # shape: (n_samples, n_genes)
    return genes, sample_ids, X


def _load_metadata(meta_path: Path) -> Dict[str, str]:
    """gsm_id -> 'Tumor' or 'Normal' from sample_metadata_inferred.csv."""
    rows = _read_csv(meta_path)
    labels: Dict[str, str] = {}
    for r in rows:
        gsm = r.get("gsm_id", "").strip()
        lbl = r.get("sample_type_inferred", "").strip()
        if gsm and lbl in ("Tumor", "Normal"):
            labels[gsm] = lbl
    return labels


def _zscore_normalize(X: np.ndarray) -> np.ndarray:
    """Per-gene z-score: centre and scale across all samples in this dataset."""
    mean = np.nanmean(X, axis=0, keepdims=True)
    std  = np.nanstd(X, axis=0, keepdims=True)
    std  = np.where(std < 1e-8, 1.0, std)
    return (X - mean) / std


def load_geo_dataset(
    accession: str,
    geo_dir: Path,
) -> Optional[Tuple[List[str], List[str], np.ndarray, np.ndarray]]:
    """Load, filter, and z-score a single GEO dataset.

    Returns (gene_names, sample_ids, X_zscore, y) where y: 1=Tumor, 0=Normal.
    Returns None if the dataset should be skipped.
    """
    if accession in SKIP_ACCESSIONS:
        logging.info("Skipping %s (excluded platform/no normals)", accession)
        return None

    ds_dir = geo_dir / accession
    gene_matrix_path = ds_dir / "processed" / "expression_gene_matrix.csv"
    meta_path = ds_dir / "metadata" / "sample_metadata_inferred.csv"
    summary_path = ds_dir / "metadata" / "dataset_summary.json"

    if not gene_matrix_path.is_file():
        logging.warning("%s: expression_gene_matrix.csv not found — skipping", accession)
        return None

    # Read dataset summary for quick usability check
    if summary_path.is_file():
        with open(summary_path, encoding="utf-8") as fh:
            summary = json.load(fh)
        if not summary.get("usable_for_gene_replication", True):
            logging.warning("%s: flagged not usable for gene replication — skipping", accession)
            return None

    logging.info("Loading %s ...", accession)
    genes, sample_ids, X_raw = _load_gene_matrix(gene_matrix_path)
    labels = _load_metadata(meta_path)

    y = np.array(
        [1 if labels.get(s) == "Tumor" else (0 if labels.get(s) == "Normal" else -1)
         for s in sample_ids],
        dtype=np.int32,
    )

    labeled = y >= 0
    n_tumor  = int((y == 1).sum())
    n_normal = int((y == 0).sum())
    logging.info("%s: %d samples | tumor=%d normal=%d unlabeled=%d | genes=%d",
                 accession, len(sample_ids), n_tumor, n_normal,
                 int((~labeled).sum()), len(genes))

    if n_tumor < 3 or n_normal < 3:
        logging.warning("%s: too few labeled samples (T=%d N=%d) — skipping",
                        accession, n_tumor, n_normal)
        return None

    # Keep only labeled samples
    X_labeled = X_raw[labeled]
    y_labeled  = y[labeled]
    sample_ids_labeled = [s for s, keep_sample in zip(sample_ids, labeled) if keep_sample]

    # Remove genes where most values are missing (>80% NaN across labeled samples).
    # 80% threshold preserves genes with scattered missing probes (common on Affymetrix).
    nan_frac = np.isnan(X_labeled).mean(axis=0)
    keep = nan_frac < 0.8
    X_labeled = X_labeled[:, keep]
    genes_kept = [g for g, k in zip(genes, keep) if k]

    # Z-score per gene
    X_z = _zscore_normalize(X_labeled)
    X_z = np.nan_to_num(X_z, nan=0.0, posinf=0.0, neginf=0.0)

    return genes_kept, sample_ids_labeled, X_z, y_labeled


# ---------------------------------------------------------------------------
# DE direction concordance
# ---------------------------------------------------------------------------

def run_de_replication(
    accession: str,
    genes: List[str],
    X: np.ndarray,
    y: np.ndarray,
    target_genes: List[str],
    ref_directions: Dict[str, str],
) -> List[Dict[str, str]]:
    """For each target gene, compare tumor-vs-normal direction in this cohort
    against the TCGA Step 6 DE reference direction."""

    gene_to_idx = {g: i for i, g in enumerate(genes)}
    tumor_mask  = y == 1
    normal_mask = y == 0

    rows: List[Dict[str, str]] = []
    for gene in target_genes:
        if gene not in gene_to_idx:
            rows.append({
                "accession": accession,
                "gene": gene,
                "ext_n_tumor": str(int(tumor_mask.sum())),
                "ext_n_normal": str(int(normal_mask.sum())),
                "ext_mean_tumor": "",
                "ext_mean_normal": "",
                "ext_effect": "",
                "ext_direction": "missing",
                "ref_direction": ref_directions.get(gene, ""),
                "concordant": "",
                "in_dataset": "0",
            })
            continue

        gi = gene_to_idx[gene]
        t_vals = X[tumor_mask, gi]
        n_vals = X[normal_mask, gi]

        # Remove NaN
        t_vals = t_vals[~np.isnan(t_vals)]
        n_vals = n_vals[~np.isnan(n_vals)]

        if len(t_vals) < 2 or len(n_vals) < 2:
            rows.append({
                "accession": accession, "gene": gene,
                "ext_n_tumor": "", "ext_n_normal": "",
                "ext_mean_tumor": "", "ext_mean_normal": "",
                "ext_effect": "", "ext_direction": "insufficient",
                "ref_direction": ref_directions.get(gene, ""),
                "concordant": "", "in_dataset": "0",
            })
            continue

        ext_diff = float(t_vals.mean() - n_vals.mean())
        ext_dir  = "up" if ext_diff > 0 else "down"
        ref_dir  = ref_directions.get(gene, "")
        concordant = "1" if ref_dir and ref_dir == ext_dir else (
                     "0" if ref_dir else "")

        rows.append({
            "accession": accession,
            "gene": gene,
            "ext_n_tumor": str(int(tumor_mask.sum())),
            "ext_n_normal": str(int(normal_mask.sum())),
            "ext_mean_tumor": _fmt(t_vals.mean()),
            "ext_mean_normal": _fmt(n_vals.mean()),
            "ext_effect": _fmt(ext_diff),
            "ext_direction": ext_dir,
            "ref_direction": ref_dir,
            "concordant": concordant,
            "in_dataset": "1",
        })

    n_tested = sum(1 for r in rows if r["concordant"] in ("0", "1"))
    n_conc   = sum(1 for r in rows if r["concordant"] == "1")
    logging.info(
        "%s DE replication: %d/%d genes concordant (%.1f%%)",
        accession, n_conc, n_tested,
        100 * n_conc / max(n_tested, 1),
    )
    return rows


# ---------------------------------------------------------------------------
# Model inference (optional)
# ---------------------------------------------------------------------------

def _auroc_from_arrays(y_true: np.ndarray, scores: np.ndarray) -> float:
    try:
        from cage import metrics as mx
        return float(mx.auroc(y_true, scores))
    except Exception:
        order  = np.argsort(-scores)
        y_s    = y_true[order]
        n_pos  = int(y_true.sum())
        n_neg  = int(len(y_true) - n_pos)
        if n_pos == 0 or n_neg == 0:
            return float("nan")
        tpr = np.concatenate([[0.0], np.cumsum(y_s) / n_pos])
        fpr = np.concatenate([[0.0], np.cumsum(1 - y_s) / n_neg])
        return float(np.trapezoid(tpr, fpr))


def _auprc_from_arrays(y_true: np.ndarray, scores: np.ndarray) -> float:
    try:
        from cage import metrics as mx
        return float(mx.auprc(y_true, scores))
    except Exception:
        order = np.argsort(-scores)
        y_s = y_true[order]
        n_pos = int(y_true.sum())
        if n_pos == 0:
            return float("nan")
        tp = np.cumsum(y_s == 1).astype(np.float64)
        fp = np.cumsum(y_s == 0).astype(np.float64)
        precision = tp / np.maximum(tp + fp, 1.0)
        recall = tp / n_pos
        recall_prev = np.concatenate([[0.0], recall[:-1]])
        return float(np.sum((recall - recall_prev) * precision))


def run_model_inference(
    accession: str,
    genes: List[str],
    sample_ids: List[str],
    X_z: np.ndarray,
    y: np.ndarray,
    tcga_genes: List[str],
    checkpoint_dir: Path,
    prediction_path: Optional[Path] = None,
    n_confounders: int = 2,
    seed: int = 2026,
) -> Dict[str, Any]:
    """Align dataset to TCGA gene space and run ensemble model prediction."""
    try:
        from cage.step4_runner import SparseInvariantModel, TrainingConfig
    except ImportError:
        logging.warning("cage not importable; skipping model inference for %s", accession)
        return {"skipped": "import_error"}

    fold_files = sorted(checkpoint_dir.glob("fold_*.npz"))
    if not fold_files:
        return {"skipped": "no_checkpoints"}

    step4_dir = checkpoint_dir.parent
    model_config: Dict[str, Any] = {}
    summary_path = step4_dir / "phase2_summary.json"
    if summary_path.is_file():
        with open(summary_path, encoding="utf-8") as fh:
            summary = json.load(fh)
        model_config = dict(summary.get("model_config", {}))

    def _config_from_checkpoint_state(state: Dict[str, np.ndarray]) -> TrainingConfig:
        cfg = TrainingConfig()
        cfg_dict = cfg.as_dict()
        cfg_dict.update({k: v for k, v in model_config.items() if k in cfg_dict})
        cfg_dict["hidden_dims"] = (
            int(state["fc0.W"].shape[1]),
            int(state["fc1.W"].shape[1]),
        )
        cfg_dict["latent_dim"] = int(state["fc_latent.W"].shape[1])
        cfg_dict["use_decoder"] = "dec0.W" in state and "dec1.W" in state
        cfg_dict["seed"] = int(seed)
        return TrainingConfig(**cfg_dict)

    # Align to TCGA 5000-gene space
    n_genes_tcga = len(tcga_genes)
    gene_to_idx  = {g: i for i, g in enumerate(genes)}
    X_aligned    = np.zeros((X_z.shape[0], n_genes_tcga), dtype=np.float64)
    n_matched    = 0
    for ti, tg in enumerate(tcga_genes):
        if tg in gene_to_idx:
            X_aligned[:, ti] = X_z[:, gene_to_idx[tg]]
            n_matched += 1

    logging.info("%s model alignment: %d/%d TCGA genes matched",
                 accession, n_matched, n_genes_tcga)

    all_probs: List[np.ndarray] = []

    for fp in fold_files:
        state = dict(np.load(fp, allow_pickle=True))
        rng = np.random.default_rng(seed)
        config = _config_from_checkpoint_state(state)
        n_conf = int(state["adv.W"].shape[1]) if "adv.W" in state else max(n_confounders, 2)
        model = SparseInvariantModel(n_genes_tcga, n_conf, config, rng=rng)
        mismatched = [
            k for k, v in state.items()
            if not k.startswith("_meta_")
            and (k not in model.params or model.params[k].shape != v.shape)
        ]
        if mismatched:
            raise RuntimeError(
                f"{accession}: checkpoint {fp.name} does not match model architecture; "
                f"first mismatches: {mismatched[:5]}"
            )
        model.load_state_dict(state)
        all_probs.append(model.predict_proba(X_aligned))

    probs = np.mean(all_probs, axis=0)
    y_pred = (probs >= 0.5).astype(int)

    # Metrics on labeled samples
    labeled = y >= 0
    if labeled.sum() < 4:
        return {"n_samples": int(X_z.shape[0]), "n_matched_genes": n_matched,
                "skipped": "too_few_labeled"}

    y_l = y[labeled]
    p_l = probs[labeled]
    yp_l = y_pred[labeled]
    sample_ids_l = [s for s, keep_sample in zip(sample_ids, labeled) if keep_sample]

    n_pos = int(y_l.sum())
    n_neg = int(len(y_l) - n_pos)
    auroc = _auroc_from_arrays(y_l, p_l)
    auprc = _auprc_from_arrays(y_l, p_l)

    tp = int(((yp_l == 1) & (y_l == 1)).sum())
    tn = int(((yp_l == 0) & (y_l == 0)).sum())
    fp_c = int(((yp_l == 1) & (y_l == 0)).sum())
    fn_c = int(((yp_l == 0) & (y_l == 1)).sum())
    sens = tp / max(tp + fn_c, 1)
    spec = tn / max(tn + fp_c, 1)
    precision = tp / max(tp + fp_c, 1)
    f1 = 2 * precision * sens / max(precision + sens, 1e-12)
    bal_acc = (sens + spec) / 2.0
    brier   = float(np.mean((p_l - y_l) ** 2))
    normal_scores = p_l[y_l == 0]
    tumor_scores = p_l[y_l == 1]
    normal_min = float(normal_scores.min()) if normal_scores.size else float("nan")
    normal_max = float(normal_scores.max()) if normal_scores.size else float("nan")
    tumor_min = float(tumor_scores.min()) if tumor_scores.size else float("nan")
    tumor_max = float(tumor_scores.max()) if tumor_scores.size else float("nan")
    score_gap = tumor_min - normal_max if tumor_scores.size and normal_scores.size else float("nan")
    perfect_separation = bool(
        math.isfinite(score_gap)
        and score_gap > 0
        and math.isclose(float(auroc), 1.0, rel_tol=0.0, abs_tol=1e-12)
    )

    if prediction_path is not None:
        pred_rows = [
            {
                "accession": accession,
                "sample_id": sid,
                "y_true": str(int(yt)),
                "sample_type": "Tumor" if int(yt) == 1 else "Normal",
                "deep_probability": f"{float(prob):.8g}",
                "predicted_label": str(int(yp)),
                "predicted_type": "Tumor" if int(yp) == 1 else "Normal",
            }
            for sid, yt, prob, yp in zip(sample_ids_l, y_l, p_l, yp_l)
        ]
        _write_csv(prediction_path, pred_rows, [
            "accession", "sample_id", "y_true", "sample_type",
            "deep_probability", "predicted_label", "predicted_type",
        ])

    result = {
        "accession": accession,
        "n_samples": int(X_z.shape[0]),
        "n_tumor": n_pos,
        "n_normal": n_neg,
        "n_matched_genes": n_matched,
        "n_folds_ensembled": len(fold_files),
        "auroc": round(auroc, 4),
        "auprc": round(auprc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "precision": round(precision, 4),
        "f1": round(f1, 4),
        "sensitivity": round(sens, 4),
        "specificity": round(spec, 4),
        "brier": round(brier, 4) if math.isfinite(brier) else None,
        "normal_score_min": round(normal_min, 4) if math.isfinite(normal_min) else None,
        "normal_score_max": round(normal_max, 4) if math.isfinite(normal_max) else None,
        "tumor_score_min": round(tumor_min, 4) if math.isfinite(tumor_min) else None,
        "tumor_score_max": round(tumor_max, 4) if math.isfinite(tumor_max) else None,
        "score_gap_tumor_min_minus_normal_max": round(score_gap, 4) if math.isfinite(score_gap) else None,
        "perfect_score_separation": int(perfect_separation),
        "small_cohort_flag": int(len(y_l) < 50),
    }
    logging.info(
        "%s model: AUROC=%.3f AUPRC=%.3f bal_acc=%.3f precision=%.3f sens=%.3f spec=%.3f",
        accession, auroc, auprc, bal_acc, precision, sens, spec,
    )
    return result


# ---------------------------------------------------------------------------
# Cross-cohort aggregation
# ---------------------------------------------------------------------------

def build_cross_cohort_concordance(
    all_de_rows: List[Dict[str, str]],
    target_genes: List[str],
) -> List[Dict[str, str]]:
    """Per-gene concordance rate across all tested cohorts."""
    from collections import defaultdict
    gene_conc: Dict[str, List[int]] = defaultdict(list)

    for r in all_de_rows:
        if r["concordant"] in ("0", "1"):
            gene_conc[r["gene"]].append(int(r["concordant"]))

    summary: List[Dict[str, str]] = []
    for gene in target_genes:
        vals = gene_conc.get(gene, [])
        n_cohorts = len(vals)
        n_conc    = sum(vals)
        rate      = n_conc / max(n_cohorts, 1)
        summary.append({
            "gene": gene,
            "n_cohorts_tested": str(n_cohorts),
            "n_cohorts_concordant": str(n_conc),
            "concordance_rate": _fmt(rate, 3),
            "consistently_concordant": "1" if n_cohorts > 0 and n_conc == n_cohorts else "0",
        })

    summary.sort(key=lambda r: -float(r["concordance_rate"]))
    return summary


# ---------------------------------------------------------------------------
# Figure generation (B + C series)
# ---------------------------------------------------------------------------

def _has_matplotlib_affymetrix() -> bool:
    try:
        import matplotlib  # noqa: F401
        return True
    except ImportError:
        return False


def generate_affymetrix_figures(
    *,
    concordance_summary: List[Dict[str, str]],
    all_de_rows: List[Dict[str, str]],
    model_results: List[Dict[str, Any]],
    target_genes: List[str],
    output_dir: Path,
    formats: Tuple[str, ...] = ("svg",),
) -> Tuple[List[str], List[Tuple[str, str]]]:
    """Generate Affymetrix validation figures (B + C series) to output_dir/figures/.

    B-series: per-platform Affymetrix figures.
    C-series: cross-platform combined figures.

    Returns (generated_names, skipped_name_reason_pairs).
    """
    generated: List[str] = []
    skipped: List[Tuple[str, str]] = []

    if not _has_matplotlib_affymetrix():
        for name in ["B1_direction_heatmap", "B2_concordance_bar",
                     "B3_classifier_performance", "B4_concordance_distribution",
                     "C1_all_cohorts_auroc", "C2_cross_platform_concordance",
                     "C3_combined_direction_heatmap"]:
            skipped.append((name, "matplotlib not installed"))
        return generated, skipped

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        try:
            from cage.publication_style import (
                PublicationStyle, apply_style, save_figure, semantic_color, categorical_colors,
            )
            style = apply_style(PublicationStyle(font_family="Times New Roman", bold=True))
            _pub_style = True
        except ImportError:
            style = None
            _pub_style = False

        def _save(fig, stem: str) -> bool:
            fig_dir = output_dir / "figures"
            fig_dir.mkdir(parents=True, exist_ok=True)
            if _pub_style:
                paths = save_figure(fig, fig_dir / stem, style=style, formats=list(formats))
                plt.close(fig)
                return bool(paths)
            for fmt in formats:
                p = fig_dir / f"{stem}.{fmt}"
                p.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(p, format=fmt, bbox_inches="tight")
            plt.close(fig)
            return True

        def _scolor(name: str, fallback: str) -> str:
            return semantic_color(name) if _pub_style else fallback

        def _ccolors(n: int):
            if _pub_style:
                return categorical_colors(n)
            defaults = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
            return (defaults * ((n // len(defaults)) + 1))[:n]

        datasets_b = sorted(set(r["accession"] for r in all_de_rows))

        # ── B1: DE direction heatmap ──────────────────────────────────────
        try:
            tested_genes_b = [r["gene"] for r in concordance_summary
                               if int(r.get("n_cohorts_tested", 0) or 0) > 0][:50]
            if tested_genes_b and datasets_b:
                mat_b1 = np.full((len(tested_genes_b), len(datasets_b)), np.nan)
                for r in all_de_rows:
                    g = r.get("gene", "")
                    if g not in tested_genes_b:
                        continue
                    gi = tested_genes_b.index(g)
                    di = datasets_b.index(r["accession"]) if r["accession"] in datasets_b else -1
                    if di < 0:
                        continue
                    conc = r.get("concordant", "")
                    if conc == "1":
                        mat_b1[gi, di] = 1.0
                    elif conc == "0":
                        mat_b1[gi, di] = 0.0
                fig, ax = plt.subplots(
                    figsize=(max(4, len(datasets_b) * 1.5 + 1.5),
                             max(5, len(tested_genes_b) * 0.25 + 2)),
                )
                im = ax.imshow(mat_b1, aspect="auto", cmap="cage_sequential", vmin=0, vmax=1)
                plt.colorbar(im, ax=ax, label="Concordant (1=yes)")
                ax.set_xticks(range(len(datasets_b)))
                ax.set_xticklabels(datasets_b, rotation=30, ha="right", fontsize=9)
                ax.set_yticks(range(len(tested_genes_b)))
                ax.set_yticklabels(tested_genes_b, fontsize=6.5)
                ax.set_title("DE Direction Concordance — Affymetrix Cohorts\n(TCGA reference)")
                fig.tight_layout()
                if _save(fig, "B1_direction_heatmap"):
                    generated.append("B1_direction_heatmap")
        except Exception as exc:
            skipped.append(("B1_direction_heatmap", str(exc)))

        # ── B2: concordance bar by tier ───────────────────────────────────
        try:
            cs_b = {r["gene"]: r for r in concordance_summary}
            tiers_b = [
                ("Top 25",  [g for g in target_genes[:25]  if g in cs_b]),
                ("Top 100", [g for g in target_genes[:100] if g in cs_b]),
                ("All",     [g for g in target_genes       if g in cs_b]),
            ]
            tier_rates_b = []
            for tname, tgenes in tiers_b:
                rates_b = [float(cs_b[g]["concordance_rate"]) for g in tgenes
                            if cs_b[g].get("concordance_rate")]
                tier_rates_b.append((tname, float(np.mean(rates_b)) if rates_b else 0.0,
                                     len(rates_b)))
            fig, ax = plt.subplots(figsize=(5, 3.5))
            lbls_b2 = [f"{t}\n(n={n})" for t, _, n in tier_rates_b]
            vals_b2  = [v for _, v, _ in tier_rates_b]
            bars_b2 = ax.bar(lbls_b2, vals_b2, color=_ccolors(len(tier_rates_b)),
                             edgecolor="black", linewidth=0.6, width=0.5)
            for bar_b, v_b in zip(bars_b2, vals_b2):
                ax.text(bar_b.get_x() + bar_b.get_width() / 2, v_b + 0.01,
                        f"{v_b:.1%}", ha="center", fontsize=10, fontweight="bold")
            ax.axhline(0.5, color="#999999", linewidth=0.8, linestyle="--", alpha=0.6)
            ax.set_ylabel("Mean concordance rate")
            ax.set_ylim(0, 1.15)
            ax.set_title("Concordance Rate by Gene Tier — Affymetrix")
            fig.tight_layout()
            if _save(fig, "B2_concordance_bar"):
                generated.append("B2_concordance_bar")
        except Exception as exc:
            skipped.append(("B2_concordance_bar", str(exc)))

        # ── B3: classifier performance ────────────────────────────────────
        if model_results:
            try:
                metric_names_b = [
                    "auroc", "auprc", "balanced_accuracy",
                    "precision", "f1", "sensitivity", "specificity",
                ]
                metric_labels_b = [
                    "AUROC", "AUPRC", "BAC", "Precision", "F1", "Sensitivity", "Specificity",
                ]
                n_ds_b = len(model_results)
                fig_width_b3 = max(5.8, n_ds_b * len(metric_names_b) * 0.34)
                fig, ax = plt.subplots(figsize=(fig_width_b3, 4.4))
                x_b3 = np.arange(n_ds_b) * 0.82
                width_b3 = 0.09
                pal_b3 = _ccolors(len(metric_names_b))
                for mi_b, met_b in enumerate(metric_names_b):
                    vals_mb = [float(r.get(met_b, 0) or 0) for r in model_results]
                    offset_b = (mi_b - len(metric_names_b) / 2 + 0.5) * width_b3
                    bars_mb = ax.bar(
                        x_b3 + offset_b,
                        vals_mb,
                        width_b3,
                        label=metric_labels_b[mi_b],
                        color=pal_b3[mi_b],
                        edgecolor="white",
                        linewidth=0.4,
                    )
                    for bar_mb, v_mb in zip(bars_mb, vals_mb):
                        if v_mb > 0.05:
                            ax.text(
                                bar_mb.get_x() + bar_mb.get_width() / 2,
                                v_mb + 0.012,
                                f"{v_mb:.3f}",
                                ha="center",
                                va="bottom",
                                fontsize=6.0,
                                rotation=90,
                            )
                ax.set_xticks(x_b3)
                ax.set_xticklabels(
                    [r.get("accession", str(i)) for i, r in enumerate(model_results)],
                    fontsize=9,
                )
                ax.set_ylim(0, 1.26)
                ax.axhline(0.5, color="#999999", linewidth=0.8, linestyle="--", alpha=0.5)
                ax.set_ylabel("Score")
                ax.set_title("Classifier Performance — Affymetrix Cohorts")
                ax.legend(
                    fontsize=7,
                    loc="upper center",
                    bbox_to_anchor=(0.5, 1.18),
                    ncol=len(metric_names_b),
                    frameon=False,
                )
                fig.tight_layout(rect=[0, 0, 1, 0.90])
                if _save(fig, "B3_classifier_performance"):
                    generated.append("B3_classifier_performance")
            except Exception as exc:
                skipped.append(("B3_classifier_performance", str(exc)))

        # ── B4: concordance rate distribution ─────────────────────────────
        try:
            all_rates = [float(r["concordance_rate"]) for r in concordance_summary
                          if r.get("concordance_rate")
                          and int(r.get("n_cohorts_tested", 0) or 0) > 0]
            if all_rates:
                fig, ax = plt.subplots(figsize=(5, 3.5))
                ax.hist(all_rates, bins=20, color=_scolor("normal", "#4575b4"),
                        edgecolor="white", alpha=0.85)
                med_b4 = float(np.median(all_rates))
                ax.axvline(med_b4, color="#d62728", linewidth=1.5, linestyle="--",
                           label=f"Median {med_b4:.2f}")
                ax.set_xlabel("Concordance rate per gene")
                ax.set_ylabel("Number of genes")
                ax.set_title("Concordance Rate Distribution — Affymetrix")
                ax.legend(fontsize=8)
                fig.tight_layout()
                if _save(fig, "B4_concordance_distribution"):
                    generated.append("B4_concordance_distribution")
        except Exception as exc:
            skipped.append(("B4_concordance_distribution", str(exc)))

        # ── C1: all-cohorts AUROC grouped bar ─────────────────────────────
        if model_results:
            try:
                fig, ax = plt.subplots(figsize=(max(5, len(model_results) * 2), 4))
                ds_names_c = [r.get("accession", str(i)) for i, r in enumerate(model_results)]
                aurocs_c   = [float(r.get("auroc", 0) or 0) for r in model_results]
                bacs_c     = [float(r.get("balanced_accuracy", 0) or 0) for r in model_results]
                x_c1 = np.arange(len(model_results))
                w_c1 = 0.35
                ax.bar(x_c1 - w_c1 / 2, aurocs_c, w_c1, label="AUROC",
                       color=_scolor("normal", "#4575b4"), edgecolor="white", linewidth=0.4)
                ax.bar(x_c1 + w_c1 / 2, bacs_c, w_c1, label="Balanced Accuracy",
                       color=_scolor("highlight", "#f4a582"), edgecolor="white", linewidth=0.4)
                ax.set_xticks(x_c1)
                ax.set_xticklabels(ds_names_c, fontsize=9)
                ax.axhline(0.5, color="#999999", linewidth=0.8, linestyle="--", alpha=0.5)
                ax.set_ylim(0, 1.1)
                ax.set_ylabel("Score")
                ax.set_title("Classifier Performance — All Affymetrix Cohorts")
                ax.legend(fontsize=8)
                fig.tight_layout()
                if _save(fig, "C1_all_cohorts_auroc"):
                    generated.append("C1_all_cohorts_auroc")
            except Exception as exc:
                skipped.append(("C1_all_cohorts_auroc", str(exc)))

        # ── C2: cross-platform concordance rates ──────────────────────────
        try:
            per_ds_rates: dict = {}
            for r in all_de_rows:
                acc = r["accession"]
                if r.get("concordant") in ("0", "1"):
                    per_ds_rates.setdefault(acc, []).append(int(r["concordant"]))
            if per_ds_rates:
                sorted_ds = sorted(per_ds_rates)
                means_c2 = [float(np.mean(per_ds_rates[ds])) for ds in sorted_ds]
                fig, ax = plt.subplots(figsize=(max(4, len(sorted_ds) * 1.5 + 1), 3.5))
                bars_c2 = ax.bar(sorted_ds, means_c2,
                                 color=_ccolors(len(sorted_ds)),
                                 edgecolor="black", linewidth=0.6)
                for bar_c, v_c in zip(bars_c2, means_c2):
                    ax.text(bar_c.get_x() + bar_c.get_width() / 2, v_c + 0.01,
                            f"{v_c:.1%}", ha="center", fontsize=9, fontweight="bold")
                ax.axhline(0.5, color="#999999", linewidth=0.8, linestyle="--", alpha=0.6)
                ax.set_ylabel("Concordance rate")
                ax.set_ylim(0, 1.15)
                ax.set_title("Cross-Platform Concordance by Dataset")
                fig.tight_layout()
                if _save(fig, "C2_cross_platform_concordance"):
                    generated.append("C2_cross_platform_concordance")
        except Exception as exc:
            skipped.append(("C2_cross_platform_concordance", str(exc)))

        # ── C3: combined direction heatmap (all datasets) ─────────────────
        try:
            all_ds_c3 = sorted(set(r["accession"] for r in all_de_rows))
            top_genes_c3 = [r["gene"] for r in concordance_summary
                             if int(r.get("n_cohorts_tested", 0) or 0) > 0][:40]
            if top_genes_c3 and all_ds_c3:
                mat_c3 = np.full((len(top_genes_c3), len(all_ds_c3)), np.nan)
                for r in all_de_rows:
                    g = r.get("gene", "")
                    if g not in top_genes_c3:
                        continue
                    gi = top_genes_c3.index(g)
                    di = all_ds_c3.index(r["accession"]) if r["accession"] in all_ds_c3 else -1
                    if di < 0:
                        continue
                    conc = r.get("concordant", "")
                    if conc == "1":
                        mat_c3[gi, di] = 1.0
                    elif conc == "0":
                        mat_c3[gi, di] = 0.0
                fig, ax = plt.subplots(
                    figsize=(max(5, len(all_ds_c3) * 1.5 + 2),
                             max(6, len(top_genes_c3) * 0.25 + 2)),
                )
                im = ax.imshow(mat_c3, aspect="auto", cmap="cage_sequential", vmin=0, vmax=1)
                plt.colorbar(im, ax=ax, label="Concordant (1=yes)")
                ax.set_xticks(range(len(all_ds_c3)))
                ax.set_xticklabels(all_ds_c3, rotation=30, ha="right", fontsize=9)
                ax.set_yticks(range(len(top_genes_c3)))
                ax.set_yticklabels(top_genes_c3, fontsize=6.5)
                ax.set_title("Combined Direction Concordance — All Affymetrix Cohorts")
                fig.tight_layout()
                if _save(fig, "C3_combined_direction_heatmap"):
                    generated.append("C3_combined_direction_heatmap")
        except Exception as exc:
            skipped.append(("C3_combined_direction_heatmap", str(exc)))

    except Exception as exc:
        skipped.append(("all_affymetrix_figures", str(exc)))

    return generated, skipped


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser(
    parser: Optional[argparse.ArgumentParser] = None,
) -> argparse.ArgumentParser:
    """Build the validate-geo argument parser.

    If *parser* is provided (e.g. a subcommand parser), arguments are added
    to it in-place and it is returned. Otherwise a standalone parser is created.
    """
    if parser is None:
        parser = argparse.ArgumentParser(
            prog="python -m cage.step8_geo_validation",
            description="Affymetrix GEO multi-cohort external validation for CAGE.",
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        if _HAS_CLI_ARGS:
            _cli_args.add_global_args(parser, require_input_dir=False)
            _cli_args.add_figure_args(parser)
        else:
            parser.add_argument("--output-dir", required=True, type=Path)
            parser.add_argument("--seed", type=int, default=2026)
            parser.add_argument("--n-threads", type=int, default=1)
            parser.add_argument("--overwrite", action="store_true")
            parser.add_argument("--log-level", default="INFO",
                                choices=("DEBUG", "INFO", "WARNING", "ERROR"))

    g = parser.add_argument_group("GEO validation inputs")
    g.add_argument("--geo-dir", required=True, type=Path,
                   help="Directory produced by the prepare-geo step (contains GSE<id>/ subdirs).")
    g.add_argument("--step5-dir", required=True, type=Path,
                   help="Step 5 CDPS outputs (ranked_genes_cdps.csv).")
    g.add_argument("--step6-dir", required=True, type=Path,
                   help="Step 6 validation outputs (differential_expression_results.csv).")
    g.add_argument("--step4-dir", default=None, type=Path,
                   help="Step 4 deep model outputs (checkpoints/). Required for --run-model.")

    g2 = parser.add_argument_group("GEO validation options")
    g2.add_argument("--top-k", type=int, default=100, metavar="K",
                    help="Number of top CDPS genes to test for DE replication (default: 100).")
    g2.add_argument("--run-model", action="store_true",
                    help="Also run the trained deep invariant model on each cohort.")
    g2.add_argument("--skip", nargs="*", default=[], metavar="ACC",
                    help="Additional accessions to skip (space-separated).")
    return parser


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_geo_validation(args: argparse.Namespace) -> None:
    """Execute Affymetrix GEO external validation from a parsed Namespace."""
    args.output_dir.mkdir(parents=True, exist_ok=True)
    verbose = getattr(args, "log_level", "INFO").upper() == "DEBUG"
    if _HAS_CLI_ARGS:
        _configure_logging(args, log_file=args.output_dir / "logs" / "step8_geo_validation.log")
    else:
        _setup_logging(args.output_dir / "logs" / "step8_geo_validation.log", verbose)

    logging.info("CAGE Step 8 — validate-geo (Affymetrix cohorts)")
    logging.info("geo_dir=%s  top_k=%d  run_model=%s", args.geo_dir, args.top_k, args.run_model)

    # Extra accessions to skip
    skip_set = set(SKIP_ACCESSIONS) | {a.upper() for a in args.skip}

    # --- Reference data ---
    target_genes  = load_reference_genes(args.step5_dir, args.top_k)
    tcga_genes    = load_tcga_gene_list(args.step5_dir, args.step4_dir if args.run_model else None)
    ref_directions = load_reference_de(args.step6_dir)

    checkpoint_dir = args.step4_dir / "checkpoints" if args.step4_dir else None

    # --- Discover datasets ---
    accessions = sorted(
        d.name for d in args.geo_dir.iterdir()
        if d.is_dir() and d.name.startswith("GSE")
    )
    logging.info("Found %d dataset directories: %s", len(accessions), accessions)

    all_de_rows:   List[Dict[str, str]] = []
    model_results: List[Dict[str, Any]] = []
    skipped:       List[str] = []

    for acc in accessions:
        if acc in skip_set:
            logging.info("Skipping %s (in skip list)", acc)
            stale_dir = args.output_dir / "per_dataset" / acc
            if getattr(args, "overwrite", False) and stale_dir.exists():
                shutil.rmtree(stale_dir)
                logging.info("Removed stale skipped-cohort output: %s", stale_dir)
            skipped.append(acc)
            continue

        result = load_geo_dataset(acc, args.geo_dir)
        if result is None:
            skipped.append(acc)
            continue

        genes, sample_ids, X_z, y = result
        per_dir = args.output_dir / "per_dataset" / acc
        per_dir.mkdir(parents=True, exist_ok=True)

        # DE replication
        de_rows = run_de_replication(acc, genes, X_z, y, target_genes, ref_directions)
        _write_csv(per_dir / "de_replication.csv", de_rows, [
            "accession", "gene", "ext_n_tumor", "ext_n_normal",
            "ext_mean_tumor", "ext_mean_normal", "ext_effect",
            "ext_direction", "ref_direction", "concordant", "in_dataset",
        ])
        all_de_rows.extend(de_rows)

        # Model inference
        if args.run_model and checkpoint_dir:
            metrics = run_model_inference(
                acc,
                genes,
                sample_ids,
                X_z,
                y,
                tcga_genes,
                checkpoint_dir,
                prediction_path=per_dir / "model_predictions.csv",
                seed=args.seed,
            )
            _write_json(per_dir / "model_metrics.json", metrics)
            if "auroc" in metrics:
                model_results.append(metrics)
        else:
            metrics = {}

        # Per-dataset summary
        n_tested   = sum(1 for r in de_rows if r["concordant"] in ("0", "1"))
        n_conc     = sum(1 for r in de_rows if r["concordant"] == "1")
        n_missing  = sum(1 for r in de_rows if r["in_dataset"] == "0")
        _write_json(per_dir / "dataset_validation_summary.json", {
            "accession": acc,
            "n_tumor": int((y == 1).sum()),
            "n_normal": int((y == 0).sum()),
            "n_genes_in_dataset": int(sum(1 for r in de_rows if r["in_dataset"] == "1")),
            "n_target_genes_missing": n_missing,
            "n_tested_for_concordance": n_tested,
            "n_concordant": n_conc,
            "concordance_rate": round(n_conc / max(n_tested, 1), 4),
            "model_metrics": metrics,
        })

    # --- Combined outputs ---
    if all_de_rows:
        _write_csv(args.output_dir / "combined_de_replication.csv", all_de_rows, [
            "accession", "gene", "ext_n_tumor", "ext_n_normal",
            "ext_mean_tumor", "ext_mean_normal", "ext_effect",
            "ext_direction", "ref_direction", "concordant", "in_dataset",
        ])

    concordance_summary = build_cross_cohort_concordance(all_de_rows, target_genes)
    _write_csv(args.output_dir / "cross_cohort_concordance_summary.csv",
               concordance_summary, [
                   "gene", "n_cohorts_tested", "n_cohorts_concordant",
                   "concordance_rate", "consistently_concordant",
               ])

    if model_results:
        _write_csv(args.output_dir / "cross_cohort_model_summary.csv",
                   [{k: str(v) for k, v in r.items()} for r in model_results], [
                       "accession", "n_samples", "n_tumor", "n_normal",
                       "n_matched_genes", "n_folds_ensembled",
                       "auroc", "auprc", "balanced_accuracy", "precision", "f1",
                       "sensitivity", "specificity", "brier",
                       "normal_score_min", "normal_score_max",
                       "tumor_score_min", "tumor_score_max",
                       "score_gap_tumor_min_minus_normal_max",
                       "perfect_score_separation", "small_cohort_flag",
                   ])

    # --- Headline summary ---
    top25_conc = [r for r in concordance_summary if int(r["n_cohorts_tested"]) > 0][:25]
    n_fully_conc_25 = sum(1 for r in top25_conc if r["consistently_concordant"] == "1")
    mean_rate_25 = (
        sum(float(r["concordance_rate"]) for r in top25_conc) / max(len(top25_conc), 1)
    )

    run_summary = {
        "datasets_found": accessions,
        "datasets_skipped": skipped,
        "datasets_run": [a for a in accessions if a not in skipped],
        "top_k_genes": args.top_k,
        "top_25_mean_concordance_rate": round(mean_rate_25, 4),
        "top_25_fully_concordant_across_cohorts": n_fully_conc_25,
        "top_25_concordance": [
            {"gene": r["gene"], "rate": r["concordance_rate"],
             "n_cohorts": r["n_cohorts_tested"]}
            for r in top25_conc
        ],
        "model_results": model_results,
        "config": {
            "geo_dir": str(args.geo_dir),
            "step5_dir": str(args.step5_dir),
            "step6_dir": str(args.step6_dir),
            "top_k": args.top_k,
            "run_model": args.run_model,
            "seed": args.seed,
        },
    }
    _write_json(args.output_dir / "geo_validation_summary.json", run_summary)

    # --- Figures ---
    gen_figs, skip_figs = generate_affymetrix_figures(
        concordance_summary=concordance_summary,
        all_de_rows=all_de_rows,
        model_results=model_results,
        target_genes=target_genes,
        output_dir=args.output_dir,
    )
    for f in gen_figs:
        logging.info("figure OK: %s", f)
    for fname, reason in skip_figs:
        logging.warning("figure SKIPPED: %s (%s)", fname, reason)

    logging.info(
        "Done | datasets_run=%d skipped=%d | "
        "top-25 mean concordance=%.1f%% | fully_concordant=%d/25",
        len(run_summary["datasets_run"]),
        len(skipped),
        100 * mean_rate_25,
        n_fully_conc_25,
    )


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if _HAS_CLI_ARGS:
        _cli_args.apply_thread_limits(args)
        _cli_args.ensure_output_dir(args)
    run_geo_validation(args)


if __name__ == "__main__":
    main()
