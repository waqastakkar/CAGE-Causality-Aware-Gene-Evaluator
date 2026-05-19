"""CAGE Step 8 runner: external validation & release bundle assembly.

Two main responsibilities:
1. **External cohort validation** (optional): load an independent ESCA cohort,
   run the trained model for predictions, compute replication metrics, and
   check DE direction concordance for top CDPS genes.
2. **Release bundle assembly** (always): collect artifacts from Steps 2-7
   into a clean release directory tree with QC checks and claim boundaries.

Pure Python + numpy.
"""

from __future__ import annotations

import csv
import datetime
import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger("cage.step8_runner")


# ---------------------------------------------------------------------------
# CSV / JSON helpers (reused from step7_runner pattern)
# ---------------------------------------------------------------------------

def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        logger.warning("CSV not found: %s", path)
        return []
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _write_csv(path: Path, rows: List[Dict[str, str]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    logger.info("Wrote %d rows -> %s", len(rows), path)


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        logger.warning("JSON not found: %s", path)
        return {}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=str)
    logger.info("Wrote JSON -> %s", path)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    logger.info("Wrote text -> %s", path)


def _fmt_f(v: float, d: int = 4) -> str:
    try:
        return f"{float(v):.{d}f}"
    except (ValueError, TypeError):
        return str(v)


# ---------------------------------------------------------------------------
# External cohort loading
# ---------------------------------------------------------------------------

def _load_external_cohort(
    normalized_path: Path,
    metadata_path: Path,
    label_column: str,
    reference_genes: List[str],
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """Load external cohort, aligning genes to reference_genes.

    Returns (X, y, sample_barcodes, aligned_gene_names).
    Missing genes are zero-filled; extra genes are dropped.
    """
    # Read normalized matrix: first column = gene, rest = samples
    with open(normalized_path, encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        sample_ids = header[1:]
        gene_data: Dict[str, List[float]] = {}
        for row in reader:
            gene = row[0]
            vals = [float(v) if v else 0.0 for v in row[1:]]
            gene_data[gene] = vals

    n_samples = len(sample_ids)
    n_genes = len(reference_genes)
    X = np.zeros((n_samples, n_genes), dtype=np.float64)
    matched = 0
    for gi, gene in enumerate(reference_genes):
        if gene in gene_data:
            X[:, gi] = gene_data[gene]
            matched += 1

    logger.info("External cohort: %d samples, %d/%d reference genes matched",
                n_samples, matched, n_genes)

    # Read metadata for labels
    meta_rows = _read_csv(metadata_path)
    barcode_to_label: Dict[str, int] = {}
    for r in meta_rows:
        bc = r.get("sample_barcode", r.get("barcode", ""))
        lbl_raw = r.get(label_column, "").strip().lower()
        if lbl_raw in ("tumor", "1", "primary tumor", "01"):
            barcode_to_label[bc] = 1
        elif lbl_raw in ("normal", "0", "solid tissue normal", "11"):
            barcode_to_label[bc] = 0

    y = np.full(n_samples, -1, dtype=np.int32)
    for i, sid in enumerate(sample_ids):
        if sid in barcode_to_label:
            y[i] = barcode_to_label[sid]
        else:
            # Try matching by prefix
            for bc, lbl in barcode_to_label.items():
                if sid.startswith(bc) or bc.startswith(sid):
                    y[i] = lbl
                    break

    labeled = int((y >= 0).sum())
    logger.info("External labels: %d/%d labeled (tumor=%d, normal=%d)",
                labeled, n_samples, int((y == 1).sum()), int((y == 0).sum()))

    return X, y, sample_ids, reference_genes


# ---------------------------------------------------------------------------
# External model inference
# ---------------------------------------------------------------------------

def run_external_predictions(
    X_ext: np.ndarray,
    y_ext: np.ndarray,
    sample_ids: List[str],
    checkpoint_dir: Path,
    n_features: int,
    n_confounders: int = 2,
    seed: int = 2026,
) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    """Run ensemble prediction on external cohort using all fold checkpoints.

    Returns (prediction_rows, summary_metrics).
    """
    from .step4_runner import SparseInvariantModel, TrainingConfig
    from . import deep_model_utils as dm

    rng = np.random.default_rng(seed)
    config = TrainingConfig(seed=seed)

    # Collect fold predictions
    fold_files = sorted(checkpoint_dir.glob("fold_*.npz"))
    if not fold_files:
        logger.warning("No fold checkpoints found in %s", checkpoint_dir)
        return [], {}

    all_probs = []
    for fp in fold_files:
        state = dict(np.load(fp, allow_pickle=True))
        # Infer dimensions from checkpoint
        model = SparseInvariantModel(n_features, n_confounders, config, rng=rng)
        model.load_state_dict(state)
        probs = model.predict_proba(X_ext)
        all_probs.append(probs)
        logger.info("Checkpoint %s -> probs shape %s, mean=%.4f",
                     fp.name, probs.shape, probs.mean())

    # Ensemble average
    ensemble_probs = np.mean(all_probs, axis=0)

    # Per-sample predictions
    pred_rows: List[Dict[str, str]] = []
    for i, sid in enumerate(sample_ids):
        pred_rows.append({
            "sample_barcode": sid,
            "true_label": str(y_ext[i]) if y_ext[i] >= 0 else "",
            "predicted_prob_tumor": _fmt_f(ensemble_probs[i], 6),
            "predicted_class": "1" if ensemble_probs[i] >= 0.5 else "0",
        })

    # Summary metrics (only on labeled samples)
    labeled_mask = y_ext >= 0
    summary: Dict[str, Any] = {
        "n_samples": int(X_ext.shape[0]),
        "n_labeled": int(labeled_mask.sum()),
        "n_folds_ensembled": len(fold_files),
    }

    if labeled_mask.sum() >= 2:
        y_true = y_ext[labeled_mask]
        p = ensemble_probs[labeled_mask]
        y_pred = (p >= 0.5).astype(int)

        # AUROC via trapezoidal
        order = np.argsort(-p)
        y_s = y_true[order]
        n_pos = int(y_true.sum())
        n_neg = int(len(y_true) - n_pos)
        if n_pos > 0 and n_neg > 0:
            tpr_pts = np.cumsum(y_s) / n_pos
            fpr_pts = np.cumsum(1 - y_s) / n_neg
            tpr_pts = np.concatenate([[0.0], tpr_pts])
            fpr_pts = np.concatenate([[0.0], fpr_pts])
            auroc = float(np.trapz(tpr_pts, fpr_pts))
        else:
            auroc = float("nan")

        # Balanced accuracy
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        fp_count = int(((y_pred == 1) & (y_true == 0)).sum())
        sens = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp_count, 1)
        bal_acc = (sens + spec) / 2.0

        # Brier
        brier = float(np.mean((p - y_true) ** 2))

        summary.update({
            "auroc": round(auroc, 6),
            "balanced_accuracy": round(bal_acc, 6),
            "sensitivity": round(sens, 6),
            "specificity": round(spec, 6),
            "brier": round(brier, 6),
            "n_tumor": n_pos,
            "n_normal": n_neg,
        })

    return pred_rows, summary


# ---------------------------------------------------------------------------
# Top gene replication (DE direction concordance)
# ---------------------------------------------------------------------------

def run_top_gene_replication(
    X_ext: np.ndarray,
    y_ext: np.ndarray,
    gene_names: List[str],
    top_genes: List[str],
    reference_de_path: Optional[Path],
) -> List[Dict[str, str]]:
    """Check whether top CDPS genes show same DE direction in external cohort.

    Uses simple mean-difference (tumor - normal) as effect direction.
    """
    labeled_mask = y_ext >= 0
    if labeled_mask.sum() < 4:
        logger.warning("Too few labeled external samples for DE replication")
        return []

    y = y_ext[labeled_mask]
    X = X_ext[labeled_mask]
    tumor_mask = y == 1
    normal_mask = y == 0

    if tumor_mask.sum() < 2 or normal_mask.sum() < 2:
        logger.warning("Need >=2 tumor and >=2 normal for DE replication")
        return []

    # Reference DE directions
    ref_directions: Dict[str, str] = {}
    if reference_de_path and reference_de_path.is_file():
        for r in _read_csv(reference_de_path):
            gene = r.get("gene", "")
            es = r.get("effect_size_norm", "0")
            try:
                ref_directions[gene] = "up" if float(es) > 0 else "down"
            except ValueError:
                pass

    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    rows: List[Dict[str, str]] = []
    for gene in top_genes:
        if gene not in gene_to_idx:
            continue
        gi = gene_to_idx[gene]
        t_vals = X[tumor_mask, gi]
        n_vals = X[normal_mask, gi]
        ext_diff = float(t_vals.mean() - n_vals.mean())
        ext_dir = "up" if ext_diff > 0 else "down"
        ref_dir = ref_directions.get(gene, "")
        concordant = "1" if ref_dir and ref_dir == ext_dir else ("0" if ref_dir else "")

        rows.append({
            "gene": gene,
            "ext_mean_tumor": _fmt_f(t_vals.mean()),
            "ext_mean_normal": _fmt_f(n_vals.mean()),
            "ext_effect_diff": _fmt_f(ext_diff),
            "ext_direction": ext_dir,
            "ref_direction": ref_dir,
            "concordant": concordant,
        })

    n_concordant = sum(1 for r in rows if r["concordant"] == "1")
    n_tested = sum(1 for r in rows if r["concordant"] in ("0", "1"))
    logger.info("DE direction concordance: %d/%d genes concordant", n_concordant, n_tested)
    return rows


# ---------------------------------------------------------------------------
# Release bundle assembly
# ---------------------------------------------------------------------------

def _copy_dir(src: Path, dst: Path, label: str) -> int:
    """Recursively copy src -> dst. Returns number of files copied."""
    if not src.is_dir():
        logger.warning("Source dir not found for %s: %s", label, src)
        return 0
    dst.mkdir(parents=True, exist_ok=True)
    count = 0
    for item in src.rglob("*"):
        if item.is_file():
            rel = item.relative_to(src)
            target = dst / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)
            count += 1
    logger.info("Copied %d files for %s: %s -> %s", count, label, src, dst)
    return count


def _copy_file(src: Path, dst: Path, label: str) -> bool:
    """Copy a single file. Returns True on success."""
    if not src.is_file():
        logger.warning("Source file not found for %s: %s", label, src)
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def assemble_release_bundle(
    bundle_dir: Path,
    step2_dir: Optional[Path],
    step3_dir: Optional[Path],
    step4_dir: Optional[Path],
    step5_dir: Optional[Path],
    step6_dir: Optional[Path],
    step7_dir: Optional[Path],
    step8_output_dir: Optional[Path],
    *,
    step6b_top25_dir: Optional[Path] = None,
    step6b_survival_dir: Optional[Path] = None,
    step8_geo_dir: Optional[Path] = None,
    step8_agilent_dir: Optional[Path] = None,
    copy_final_figures: bool = True,
    copy_key_tables: bool = True,
    build_supplement: bool = True,
) -> Dict[str, Any]:
    """Assemble the release bundle directory tree.

    Structure::

        release_bundle/
            final_figures/      (from step7)
            final_tables/       (from step7)
            manuscript/         (from step7)
            supplementary/      (from step7)
            configs/            (phase summary JSONs from all steps)
            manifests/          (from step7)
            external/           (from step8, if available)
    """
    bundle_dir.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, Any] = {"dirs_copied": {}, "files_copied": 0}

    # -- final_figures from step7 --
    if copy_final_figures and step7_dir:
        n = _copy_dir(step7_dir / "final_figures", bundle_dir / "final_figures", "final_figures")
        manifest["dirs_copied"]["final_figures"] = n
        manifest["files_copied"] += n

    # -- final_tables from step7 --
    if copy_key_tables and step7_dir:
        n = _copy_dir(step7_dir / "final_tables", bundle_dir / "final_tables", "final_tables")
        manifest["dirs_copied"]["final_tables"] = n
        manifest["files_copied"] += n

    # -- manuscript from step7 --
    if step7_dir:
        n = _copy_dir(step7_dir / "manuscript", bundle_dir / "manuscript", "manuscript")
        manifest["dirs_copied"]["manuscript"] = n
        manifest["files_copied"] += n

    # -- supplementary from step7 --
    if build_supplement and step7_dir:
        n = _copy_dir(step7_dir / "supplementary", bundle_dir / "supplementary", "supplementary")
        manifest["dirs_copied"]["supplementary"] = n
        manifest["files_copied"] += n

    # -- manifests from step7 --
    if step7_dir:
        n = _copy_dir(step7_dir / "manifests", bundle_dir / "manifests", "manifests")
        manifest["dirs_copied"]["manifests"] = n
        manifest["files_copied"] += n

    # -- configs: phase summary JSONs --
    configs_dir = bundle_dir / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    config_sources = [
        (step2_dir, "phase1_summary.json"),
        (step3_dir, "phase3_summary.json"),
        (step4_dir, "phase2_summary.json"),
        (step5_dir, "phase3_summary.json"),
        (step6_dir, "phase4_summary.json"),
    ]
    # Rename to avoid collisions
    config_targets = [
        "step2_phase1_summary.json",
        "step3_baselines_summary.json",
        "step4_phase2_summary.json",
        "step5_phase3_summary.json",
        "step6_phase4_summary.json",
    ]
    configs_count = 0
    for (src_dir, fname), tgt_name in zip(config_sources, config_targets):
        if src_dir and (src_dir / fname).is_file():
            shutil.copy2(src_dir / fname, configs_dir / tgt_name)
            configs_count += 1
    if step7_dir and (step7_dir / "phase5_summary.json").is_file():
        shutil.copy2(step7_dir / "phase5_summary.json", configs_dir / "step7_phase5_summary.json")
        configs_count += 1
    manifest["dirs_copied"]["configs"] = configs_count
    manifest["files_copied"] += configs_count
    logger.info("Copied %d config JSONs to configs/", configs_count)

    # -- step6b top-25 prioritization: tables, reports, figures --
    if step6b_top25_dir:
        n = _copy_dir(step6b_top25_dir / "tables", bundle_dir / "step6b_top25" / "tables", "step6b_top25_tables")
        n += _copy_dir(step6b_top25_dir / "reports", bundle_dir / "step6b_top25" / "reports", "step6b_top25_reports")
        n += _copy_dir(step6b_top25_dir / "figures", bundle_dir / "step6b_top25" / "figures", "step6b_top25_figures")
        manifest["dirs_copied"]["step6b_top25"] = n
        manifest["files_copied"] += n

    # -- step6b survival/external: tables, reports, figures --
    if step6b_survival_dir:
        n = _copy_dir(step6b_survival_dir / "tables", bundle_dir / "step6b_survival" / "tables", "step6b_survival_tables")
        n += _copy_dir(step6b_survival_dir / "reports", bundle_dir / "step6b_survival" / "reports", "step6b_survival_reports")
        n += _copy_dir(step6b_survival_dir / "figures", bundle_dir / "step6b_survival" / "figures", "step6b_survival_figures")
        manifest["dirs_copied"]["step6b_survival"] = n
        manifest["files_copied"] += n

    # -- step8 GEO validation results --
    if step8_geo_dir:
        n = _copy_dir(step8_geo_dir / "figures", bundle_dir / "step8_geo_validation" / "figures", "step8_geo_figures")
        n += _copy_dir(step8_geo_dir / "per_dataset", bundle_dir / "step8_geo_validation" / "per_dataset", "step8_geo_per_dataset")
        for fname in ("combined_de_replication.csv", "cross_cohort_concordance_summary.csv",
                      "cross_cohort_model_summary.csv", "geo_validation_summary.json"):
            src = step8_geo_dir / fname
            if src.is_file():
                dst = bundle_dir / "step8_geo_validation" / fname
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                n += 1
        manifest["dirs_copied"]["step8_geo_validation"] = n
        manifest["files_copied"] += n

    # -- step8 Agilent validation results --
    if step8_agilent_dir:
        n = _copy_dir(step8_agilent_dir / "figures", bundle_dir / "step8_agilent_validation" / "figures", "step8_agilent_figures")
        n += _copy_dir(step8_agilent_dir / "per_dataset", bundle_dir / "step8_agilent_validation" / "per_dataset", "step8_agilent_per_dataset")
        for fname in ("combined_de_replication.csv", "concordance_summary.csv",
                      "model_summary.csv", "validation_summary.json"):
            src = step8_agilent_dir / fname
            if src.is_file():
                dst = bundle_dir / "step8_agilent_validation" / fname
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                n += 1
        manifest["dirs_copied"]["step8_agilent_validation"] = n
        manifest["files_copied"] += n

    # -- external cohort validation results from step8 release run --
    if step8_output_dir:
        ext_files = [
            "external_predictions.csv",
            "external_summary_metrics.json",
            "external_top_gene_replication.csv",
        ]
        ext_dir = bundle_dir / "external"
        ext_dir.mkdir(parents=True, exist_ok=True)
        ext_count = 0
        for fname in ext_files:
            src = step8_output_dir / fname
            if src.is_file():
                shutil.copy2(src, ext_dir / fname)
                ext_count += 1
        manifest["dirs_copied"]["external"] = ext_count
        manifest["files_copied"] += ext_count

    return manifest


# ---------------------------------------------------------------------------
# QC checks
# ---------------------------------------------------------------------------

def run_qc_checks(bundle_dir: Path) -> List[Dict[str, str]]:
    """Run QC checks on the assembled release bundle."""
    checks: List[Dict[str, str]] = []

    # Check required directories
    for dname in ["final_tables", "manuscript", "configs", "manifests"]:
        d = bundle_dir / dname
        exists = d.is_dir() and any(d.iterdir())
        checks.append({
            "check": f"directory_{dname}_populated",
            "status": "PASS" if exists else "WARN",
            "detail": str(d),
        })

    # Check key tables
    key_tables = [
        "final_tables/table1_cohort_summary.csv",
        "final_tables/table2_baseline_vs_deep_performance.csv",
        "final_tables/table3_top_ranked_genes.csv",
        "final_tables/table4_validation_augmented_genes.csv",
    ]
    for tbl in key_tables:
        p = bundle_dir / tbl
        checks.append({
            "check": f"table_{Path(tbl).stem}",
            "status": "PASS" if p.is_file() else "MISSING",
            "detail": str(p),
        })

    # Check step6b top-25 prioritization tables
    step6b_tables = [
        "step6b_top25/tables/top25_final_priority_ranking.csv",
        "step6b_top25/tables/top25_integrated_evidence.csv",
        "step6b_top25/tables/top25_manuscript_table.csv",
    ]
    for tbl in step6b_tables:
        p = bundle_dir / tbl
        checks.append({
            "check": f"step6b_{Path(tbl).stem}",
            "status": "PASS" if p.is_file() else "WARN",
            "detail": str(p),
        })

    # Check step6b survival tables
    survival_tables = [
        "step6b_survival/tables/top25_survival_summary.csv",
        "step6b_survival/tables/top25_external_validation.csv",
        "step6b_survival/tables/top25_full_validation_summary.csv",
    ]
    for tbl in survival_tables:
        p = bundle_dir / tbl
        checks.append({
            "check": f"step6b_survival_{Path(tbl).stem}",
            "status": "PASS" if p.is_file() else "WARN",
            "detail": str(p),
        })

    # Check step8 external validation tables
    ext_val_files = [
        "step8_geo_validation/combined_de_replication.csv",
        "step8_agilent_validation/combined_de_replication.csv",
        "step8_agilent_validation/concordance_summary.csv",
    ]
    for tbl in ext_val_files:
        p = bundle_dir / tbl
        checks.append({
            "check": f"ext_val_{Path(tbl).stem}",
            "status": "PASS" if p.is_file() else "WARN",
            "detail": str(p),
        })

    # Check manuscript files
    for md in ["manuscript_report.md", "manuscript_methods.md", "manuscript_results_summary.md"]:
        p = bundle_dir / "manuscript" / md
        checks.append({
            "check": f"manuscript_{Path(md).stem}",
            "status": "PASS" if p.is_file() else "MISSING",
            "detail": str(p),
        })

    # Check manifests
    for mf in ["study_summary.json", "output_manifest.csv", "table_manifest.csv"]:
        p = bundle_dir / "manifests" / mf
        checks.append({
            "check": f"manifest_{Path(mf).stem}",
            "status": "PASS" if p.is_file() else "MISSING",
            "detail": str(p),
        })

    # Check SVG figures (final_figures + step6b panels)
    fig_dir = bundle_dir / "final_figures"
    n_svg = len(list(fig_dir.rglob("*.svg"))) if fig_dir.is_dir() else 0
    checks.append({
        "check": "final_svg_figures_present",
        "status": "PASS" if n_svg > 0 else "WARN",
        "detail": f"{n_svg} SVG files in final_figures/",
    })
    km_panel = bundle_dir / "step6b_survival" / "figures" / "km" / "combined_KM_all25.svg"
    checks.append({
        "check": "step6b_km_panel_present",
        "status": "PASS" if km_panel.is_file() else "WARN",
        "detail": str(km_panel),
    })
    tcga_panel = bundle_dir / "step6b_survival" / "figures" / "boxplots" / "combined_TCGA_boxplots_all25.svg"
    checks.append({
        "check": "step6b_tcga_boxplot_panel_present",
        "status": "PASS" if tcga_panel.is_file() else "WARN",
        "detail": str(tcga_panel),
    })

    n_pass = sum(1 for c in checks if c["status"] == "PASS")
    n_warn = sum(1 for c in checks if c["status"] == "WARN")
    n_miss = sum(1 for c in checks if c["status"] == "MISSING")
    logger.info("QC checks: %d PASS, %d WARN, %d MISSING", n_pass, n_warn, n_miss)

    return checks


# ---------------------------------------------------------------------------
# Claim boundary document
# ---------------------------------------------------------------------------

CLAIM_BOUNDARY_TEXT = """\
# CAGE Claim Boundaries

## Status of Findings

All genes identified in this study are described as **candidate driver genes**
or **computationally prioritized genes**. The rankings are derived from a
multi-component scoring framework (CDPS) integrating deep invariant modeling,
attribution analysis, stability, environment invariance, and counterfactual
perturbation, followed by biological and statistical validation.

## What This Study Claims

- The pipeline identifies genes whose expression patterns are most predictive
  of tumor vs. normal status in TCGA ESCA, robust across defined environments
  (histology, sex, smoking, country, stage).
- The validation layer provides convergent computational evidence (differential
  expression, clinical association, subgroup consistency) supporting the top
  candidates.

## What This Study Does NOT Claim

- **No causal claims**: Computational prioritization does not establish
  causality. Functional validation (e.g., knockdown, CRISPR) is required.
- **No clinical-validation claims**: These genes have not been tested in
  clinical settings or independent prospective cohorts.
- **No therapeutic claims**: No drug targets or biomarkers are proposed.

## External Validation

{external_status}

## Reproducibility

All code, configurations, random seeds, and intermediate outputs are provided
in the release bundle. The pipeline uses deterministic seeds and
patient-grouped cross-validation to prevent data leakage.

---
*Generated by CAGE pipeline on {date}*
"""


# ---------------------------------------------------------------------------
# Reviewer bundle
# ---------------------------------------------------------------------------

def build_reviewer_bundle(
    reviewer_dir: Path,
    bundle_dir: Path,
    output_dir: Path,
    qc_checks: List[Dict[str, str]],
) -> int:
    """Build a reviewer-friendly subset with key tables, drafts, and QC."""
    reviewer_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    # Copy key tables
    for tbl in ["table3_top_ranked_genes.csv", "table4_validation_augmented_genes.csv"]:
        src = bundle_dir / "final_tables" / tbl
        if src.is_file():
            shutil.copy2(src, reviewer_dir / tbl)
            count += 1

    # Copy manuscript report
    src = bundle_dir / "manuscript" / "manuscript_report.md"
    if src.is_file():
        shutil.copy2(src, reviewer_dir / "manuscript_report.md")
        count += 1

    # Copy study summary
    src = bundle_dir / "manifests" / "study_summary.json"
    if src.is_file():
        shutil.copy2(src, reviewer_dir / "study_summary.json")
        count += 1

    # Write QC report
    _write_csv(reviewer_dir / "qc_checks.csv", qc_checks, ["check", "status", "detail"])
    count += 1

    # Copy claim boundaries
    claim_src = output_dir / "claim_boundaries.md"
    if claim_src.is_file():
        shutil.copy2(claim_src, reviewer_dir / "claim_boundaries.md")
        count += 1

    logger.info("Reviewer bundle: %d files -> %s", count, reviewer_dir)
    return count


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_step8(
    step2_dir: Optional[Path],
    step4_dir: Optional[Path],
    step5_dir: Optional[Path],
    step6_dir: Optional[Path],
    step7_dir: Optional[Path],
    output_dir: Path,
    *,
    step6b_top25_dir: Optional[Path] = None,
    step6b_survival_dir: Optional[Path] = None,
    step8_geo_dir: Optional[Path] = None,
    step8_agilent_dir: Optional[Path] = None,
    external_counts: Optional[Path] = None,
    external_normalized: Optional[Path] = None,
    external_metadata: Optional[Path] = None,
    external_label_column: str = "sample_type",
    release_bundle_dir: Optional[Path] = None,
    copy_final_figures: bool = True,
    copy_key_tables: bool = True,
    build_supplement: bool = True,
    do_build_reviewer_bundle: bool = False,
    style: Any = None,
    seed: int = 2026,
) -> Dict[str, Any]:
    """Run the full Step 8 pipeline.

    Returns a summary dict for phase6_summary.json.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle_dir = release_bundle_dir or (output_dir / "release_bundle")

    has_external = all(
        p is not None and p.is_file()
        for p in [external_normalized, external_metadata]
    )

    summary: Dict[str, Any] = {
        "external_validation_run": has_external,
        "release_bundle_dir": str(bundle_dir),
        "config": {
            "seed": seed,
            "copy_final_figures": copy_final_figures,
            "copy_key_tables": copy_key_tables,
            "build_supplement": build_supplement,
            "build_reviewer_bundle": do_build_reviewer_bundle,
        },
    }

    # ---------------------------------------------------------------
    # 1. External cohort validation (optional)
    # ---------------------------------------------------------------
    external_status = "No external cohort was provided. External validation was not performed."

    if has_external:
        logger.info("Step 8: running external cohort validation ...")

        # Load reference gene list from step5
        gene_names: List[str] = []
        if step5_dir and (step5_dir / "ranked_genes_cdps.csv").is_file():
            cdps_rows = _read_csv(step5_dir / "ranked_genes_cdps.csv")
            gene_names = [r["gene"] for r in cdps_rows]
        elif step2_dir and (step2_dir / "normalized_primary_matrix.csv").is_file():
            with open(step2_dir / "normalized_primary_matrix.csv", encoding="utf-8") as fh:
                reader = csv.reader(fh)
                next(reader)  # skip header
                gene_names = [row[0] for row in reader]

        if not gene_names:
            logger.error("Cannot determine reference gene list for external validation")
        else:
            # Load external cohort
            X_ext, y_ext, ext_samples, _ = _load_external_cohort(
                external_normalized, external_metadata,
                external_label_column, gene_names,
            )

            # Model predictions
            checkpoint_dir = step4_dir / "checkpoints" if step4_dir else None
            if checkpoint_dir and checkpoint_dir.is_dir():
                pred_rows, pred_summary = run_external_predictions(
                    X_ext, y_ext, ext_samples, checkpoint_dir,
                    n_features=len(gene_names), seed=seed,
                )
                if pred_rows:
                    _write_csv(
                        output_dir / "external_predictions.csv", pred_rows,
                        ["sample_barcode", "true_label", "predicted_prob_tumor", "predicted_class"],
                    )
                _write_json(output_dir / "external_summary_metrics.json", pred_summary)
                summary["external_metrics"] = pred_summary

                auroc = pred_summary.get("auroc", "N/A")
                external_status = (
                    f"External validation was performed on an independent cohort "
                    f"({pred_summary.get('n_samples', '?')} samples). "
                    f"Ensemble AUROC = {auroc}."
                )
            else:
                logger.warning("No checkpoints found — skipping model inference")

            # Top-gene replication
            top_genes: List[str] = []
            if step5_dir and (step5_dir / "top100_genes_cdps.csv").is_file():
                top_genes = [r["gene"] for r in _read_csv(step5_dir / "top100_genes_cdps.csv")]
            elif step5_dir and (step5_dir / "top25_genes_cdps.csv").is_file():
                top_genes = [r["gene"] for r in _read_csv(step5_dir / "top25_genes_cdps.csv")]

            if top_genes:
                ref_de = step6_dir / "differential_expression_results.csv" if step6_dir else None
                replication_rows = run_top_gene_replication(
                    X_ext, y_ext, gene_names, top_genes, ref_de,
                )
                if replication_rows:
                    _write_csv(
                        output_dir / "external_top_gene_replication.csv", replication_rows,
                        ["gene", "ext_mean_tumor", "ext_mean_normal", "ext_effect_diff",
                         "ext_direction", "ref_direction", "concordant"],
                    )
                    n_conc = sum(1 for r in replication_rows if r["concordant"] == "1")
                    n_tested = sum(1 for r in replication_rows if r["concordant"] in ("0", "1"))
                    summary["replication"] = {
                        "n_genes_tested": n_tested,
                        "n_concordant": n_conc,
                        "concordance_rate": round(n_conc / max(n_tested, 1), 4),
                    }
    else:
        logger.info("Step 8: no external cohort provided — skipping external validation")

    # ---------------------------------------------------------------
    # 2. Release bundle assembly
    # ---------------------------------------------------------------
    logger.info("Step 8: assembling release bundle -> %s", bundle_dir)

    # Find step3_dir (baselines) — may not be passed as arg, infer from step7
    step3_dir_inferred = None
    if step7_dir:
        # Try the standard location
        candidate = step7_dir.parent / "step3_baselines"
        if candidate.is_dir():
            step3_dir_inferred = candidate

    bundle_manifest = assemble_release_bundle(
        bundle_dir=bundle_dir,
        step2_dir=step2_dir,
        step3_dir=step3_dir_inferred,
        step4_dir=step4_dir,
        step5_dir=step5_dir,
        step6_dir=step6_dir,
        step7_dir=step7_dir,
        step8_output_dir=output_dir if has_external else None,
        step6b_top25_dir=step6b_top25_dir,
        step6b_survival_dir=step6b_survival_dir,
        step8_geo_dir=step8_geo_dir,
        step8_agilent_dir=step8_agilent_dir,
        copy_final_figures=copy_final_figures,
        copy_key_tables=copy_key_tables,
        build_supplement=build_supplement,
    )
    summary["bundle_manifest"] = bundle_manifest

    # ---------------------------------------------------------------
    # 3. Claim boundary document
    # ---------------------------------------------------------------
    claim_text = CLAIM_BOUNDARY_TEXT.format(
        external_status=external_status,
        date=datetime.date.today().isoformat(),
    )
    _write_text(output_dir / "claim_boundaries.md", claim_text)
    # Also copy into bundle
    _write_text(bundle_dir / "CLAIM_BOUNDARIES.md", claim_text)

    # ---------------------------------------------------------------
    # 4. QC checks
    # ---------------------------------------------------------------
    logger.info("Step 8: running QC checks ...")
    qc_checks = run_qc_checks(bundle_dir)
    _write_csv(output_dir / "qc_checks.csv", qc_checks, ["check", "status", "detail"])
    summary["qc"] = {
        "n_pass": sum(1 for c in qc_checks if c["status"] == "PASS"),
        "n_warn": sum(1 for c in qc_checks if c["status"] == "WARN"),
        "n_missing": sum(1 for c in qc_checks if c["status"] == "MISSING"),
    }

    # ---------------------------------------------------------------
    # 5. Reviewer bundle (optional)
    # ---------------------------------------------------------------
    if do_build_reviewer_bundle:
        reviewer_dir = output_dir / "reviewer_bundle"
        n_reviewer = build_reviewer_bundle(reviewer_dir, bundle_dir, output_dir, qc_checks)
        summary["reviewer_bundle_files"] = n_reviewer

    # ---------------------------------------------------------------
    # 6. Phase summary
    # ---------------------------------------------------------------
    _write_json(output_dir / "phase6_summary.json", summary)

    logger.info(
        "Step 8 complete | external=%s bundle_files=%d qc=(%dP/%dW/%dM) reviewer=%s",
        has_external,
        bundle_manifest.get("files_copied", 0),
        summary["qc"]["n_pass"],
        summary["qc"]["n_warn"],
        summary["qc"]["n_missing"],
        do_build_reviewer_bundle,
    )

    return summary
