"""CAGE Step 7 runner: manuscript packaging logic.

Collects artifacts from Steps 2-6, assembles:
  - Main tables (1-4) and supplementary tables (S1-S6)
  - Manuscript Markdown drafts (report, methods, results summary)
  - Reproducibility manifests (study_summary.json, output/figure/table manifests)
  - Optionally copies final figures and tables into release-ready directories

Pure Python + numpy; matplotlib used only for optional figure assembly.
"""

from __future__ import annotations

import csv
import datetime
import json
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger("cage.step7_runner")


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def _read_csv(path: Path) -> List[Dict[str, str]]:
    """Read a CSV into a list of row dicts.  Returns [] if file missing."""
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


# ---------------------------------------------------------------------------
# Phase artifact discovery
# ---------------------------------------------------------------------------

@dataclass
class PhaseArtifacts:
    """Resolved paths to all prior-phase artifacts."""
    # Phase I (step2)
    master_samples: Optional[Path] = None
    normalized_matrix: Optional[Path] = None
    counts_matrix: Optional[Path] = None
    grouped_folds: Optional[Path] = None
    phase1_summary: Optional[Path] = None
    # Phase II (step4)
    deep_oof: Optional[Path] = None
    deep_train: Optional[Path] = None
    gate_weights: Optional[Path] = None
    latent_embeddings: Optional[Path] = None
    deep_summary_metrics: Optional[Path] = None
    deep_per_fold: Optional[Path] = None
    deep_training_history: Optional[Path] = None
    phase2_summary: Optional[Path] = None
    # Baselines (step3)
    baseline_summary_metrics: Optional[Path] = None
    baseline_per_fold: Optional[Path] = None
    baseline_oof: Optional[Path] = None
    baseline_train: Optional[Path] = None
    baseline_feature_importance: Optional[Path] = None
    phase3_baselines_summary: Optional[Path] = None
    # Phase III (step5)
    ranked_genes_cdps: Optional[Path] = None
    top25_genes_cdps: Optional[Path] = None
    top100_genes_cdps: Optional[Path] = None
    gene_attribution: Optional[Path] = None
    gene_stability: Optional[Path] = None
    gene_invariance: Optional[Path] = None
    gene_perturbation: Optional[Path] = None
    phase3_summary: Optional[Path] = None
    # Phase IV (step6)
    de_results: Optional[Path] = None
    de_top_hits: Optional[Path] = None
    cdps_de_support: Optional[Path] = None
    clinical_results: Optional[Path] = None
    survival_results: Optional[Path] = None
    subgroup_results: Optional[Path] = None
    final_validated: Optional[Path] = None
    phase4_summary: Optional[Path] = None
    enrichment_summary: Optional[Path] = None
    network_gene_support: Optional[Path] = None
    network_edges: Optional[Path] = None
    # Phase IV-b (step6b top-25 prioritization)
    top25_final_priority_ranking: Optional[Path] = None
    top25_integrated_evidence: Optional[Path] = None
    top25_manuscript_table: Optional[Path] = None
    top25_tier_summary: Optional[Path] = None
    # Phase IV-b (step6b survival/external)
    top25_survival_summary: Optional[Path] = None
    top25_external_validation: Optional[Path] = None
    top25_full_validation_summary: Optional[Path] = None
    # Figure directories from completed phases. These are copied verbatim into
    # the manuscript package so Step 7 does not silently omit usable figures.
    figure_dirs: Dict[str, Path] = field(default_factory=dict)

    missing: List[str] = field(default_factory=list)


def discover_artifacts(
    step2_dir: Optional[Path],
    step3_dir: Optional[Path],
    step4_dir: Optional[Path],
    step5_dir: Optional[Path],
    step6_dir: Optional[Path],
    step8_dir: Optional[Path] = None,
    extra_figure_dirs: Optional[Dict[str, Path]] = None,
    step6b_top25_dir: Optional[Path] = None,
    step6b_survival_dir: Optional[Path] = None,
) -> PhaseArtifacts:
    """Resolve all expected artifact paths and flag missing ones."""
    art = PhaseArtifacts()

    def _set(attr: str, base: Optional[Path], fname: str) -> None:
        if base is None:
            art.missing.append(f"{attr} (no dir)")
            return
        p = base / fname
        if p.is_file():
            setattr(art, attr, p)
        else:
            art.missing.append(f"{attr}: {p}")

    def _add_figure_dir(name: str, base: Optional[Path]) -> None:
        if base is None:
            return
        fig_dir = base / "figures"
        if fig_dir.is_dir():
            art.figure_dirs[name] = fig_dir

    # Phase I
    _set("master_samples", step2_dir, "master_samples_primary.csv")
    _set("normalized_matrix", step2_dir, "normalized_primary_matrix.csv")
    _set("counts_matrix", step2_dir, "counts_primary_matrix.csv")
    _set("grouped_folds", step2_dir, "grouped_outer_folds.csv")
    _set("phase1_summary", step2_dir, "phase1_summary.json")

    # Baselines
    _set("baseline_summary_metrics", step3_dir, "baseline_summary_metrics.csv")
    _set("baseline_per_fold", step3_dir, "baseline_per_fold_metrics.csv")
    _set("baseline_oof", step3_dir, "baseline_oof_predictions.csv")
    _set("baseline_train", step3_dir, "baseline_train_predictions.csv")
    _set("baseline_feature_importance", step3_dir, "baseline_feature_importance.csv")
    _set("phase3_baselines_summary", step3_dir, "phase3_summary.json")

    # Phase II (deep model)
    _set("deep_oof", step4_dir, "deep_oof_predictions.csv")
    _set("deep_train", step4_dir, "deep_train_predictions.csv")
    _set("gate_weights", step4_dir, "gate_weights.csv")
    _set("latent_embeddings", step4_dir, "latent_embeddings.csv")
    _set("deep_summary_metrics", step4_dir, "deep_summary_metrics.csv")
    _set("deep_per_fold", step4_dir, "deep_per_fold_metrics.csv")
    _set("deep_training_history", step4_dir, "deep_training_history.csv")
    _set("phase2_summary", step4_dir, "phase2_summary.json")

    # Phase III (CDPS)
    _set("ranked_genes_cdps", step5_dir, "ranked_genes_cdps.csv")
    _set("top25_genes_cdps", step5_dir, "top25_genes_cdps.csv")
    _set("top100_genes_cdps", step5_dir, "top100_genes_cdps.csv")
    _set("gene_attribution", step5_dir, "gene_attribution_scores.csv")
    _set("gene_stability", step5_dir, "gene_stability_scores.csv")
    _set("gene_invariance", step5_dir, "gene_invariance_scores.csv")
    _set("gene_perturbation", step5_dir, "gene_perturbation_scores.csv")
    _set("phase3_summary", step5_dir, "phase3_summary.json")

    # Phase IV (validation)
    _set("de_results", step6_dir, "differential_expression_results.csv")
    _set("de_top_hits", step6_dir, "differential_expression_top_hits.csv")
    _set("cdps_de_support", step6_dir, "top_cdps_de_support.csv")
    _set("clinical_results", step6_dir, "clinical_association_results.csv")
    _set("survival_results", step6_dir, "survival_gene_summary.csv")
    _set("subgroup_results", step6_dir, "subgroup_sensitivity_summary.csv")
    _set("final_validated", step6_dir, "final_validated_gene_ranking.csv")
    _set("phase4_summary", step6_dir, "phase4_summary.json")

    # Step 6 extras (enrichment, network)
    _set("enrichment_summary", step6_dir, "enrichment_summary_top_pathways.csv")
    _set("network_gene_support", step6_dir, "network_gene_support.csv")
    _set("network_edges", step6_dir, "network_edges_top_genes.csv")

    # Phase IV-b: step6b top-25 prioritization
    _set("top25_final_priority_ranking", step6b_top25_dir, "tables/top25_final_priority_ranking.csv")
    _set("top25_integrated_evidence", step6b_top25_dir, "tables/top25_integrated_evidence.csv")
    _set("top25_manuscript_table", step6b_top25_dir, "tables/top25_manuscript_table.csv")
    _set("top25_tier_summary", step6b_top25_dir, "tables/top25_tier_summary.csv")

    # Phase IV-b: step6b survival/external
    _set("top25_survival_summary", step6b_survival_dir, "tables/top25_survival_summary.csv")
    _set("top25_external_validation", step6b_survival_dir, "tables/top25_external_validation.csv")
    _set("top25_full_validation_summary", step6b_survival_dir, "tables/top25_full_validation_summary.csv")

    _add_figure_dir("step2_cohort", step2_dir)
    _add_figure_dir("step3_baselines", step3_dir)
    _add_figure_dir("step4_deep_model", step4_dir)
    _add_figure_dir("step5_cdps", step5_dir)
    _add_figure_dir("step6_validation", step6_dir)
    _add_figure_dir("step8_external_validation", step8_dir)
    _add_figure_dir("step6b_top25", step6b_top25_dir)
    # step6b_survival has nested subdirs (km/, boxplots/, external/) — register all
    if step6b_survival_dir:
        fig_root = step6b_survival_dir / "figures"
        if fig_root.is_dir():
            art.figure_dirs["step6b_survival"] = fig_root
        for subname in ("km", "boxplots", "external"):
            sub = fig_root / subname
            if sub.is_dir():
                art.figure_dirs[f"step6b_survival_{subname}"] = sub
    for name, fig_dir in (extra_figure_dirs or {}).items():
        if fig_dir.is_dir():
            art.figure_dirs[name] = fig_dir

    if art.missing:
        logger.warning("Missing %d artifacts: %s", len(art.missing),
                        "; ".join(art.missing[:10]))
    else:
        logger.info("All expected artifacts found.")
    return art


# ---------------------------------------------------------------------------
# Table 1: Cohort summary
# ---------------------------------------------------------------------------

def build_table1_cohort_summary(art: PhaseArtifacts) -> List[Dict[str, str]]:
    """Build Table 1 from master_samples_primary.csv."""
    if art.master_samples is None:
        return []
    rows = _read_csv(art.master_samples)
    if not rows:
        return []

    n_total = len(rows)
    n_tumor = sum(1 for r in rows if r.get("label") == "Tumor")
    n_normal = n_total - n_tumor

    # Unique patients
    patients = set(r.get("patient_barcode", "") for r in rows)
    n_patients = len(patients - {""})

    # Environment / subgroup distributions
    env_cols = ["sex", "histology", "smoking_status", "stage_paper", "country"]
    env_dist: Dict[str, Dict[str, int]] = {}
    for col in env_cols:
        counts: Dict[str, int] = {}
        for r in rows:
            val = r.get(col, "").strip() or "Unknown"
            counts[val] = counts.get(val, 0) + 1
        env_dist[col] = counts

    table_rows: List[Dict[str, str]] = [
        {"category": "Total samples", "value": str(n_total), "detail": ""},
        {"category": "Tumor samples", "value": str(n_tumor), "detail": ""},
        {"category": "Normal samples", "value": str(n_normal), "detail": ""},
        {"category": "Unique patients", "value": str(n_patients), "detail": ""},
    ]
    for col, dist in env_dist.items():
        sorted_items = sorted(dist.items(), key=lambda x: -x[1])
        detail = "; ".join(f"{k}={v}" for k, v in sorted_items)
        table_rows.append({
            "category": f"Subgroup: {col}",
            "value": str(len(dist)),
            "detail": detail,
        })

    logger.info("Table 1: %d rows (n=%d samples, %d patients)",
                len(table_rows), n_total, n_patients)
    return table_rows


TABLE1_FIELDS = ["category", "value", "detail"]


# ---------------------------------------------------------------------------
# Table 2: Baseline vs Deep performance
# ---------------------------------------------------------------------------

_PERF_COLS = [
    "model", "aggregation", "auroc", "auroc_ci_lower", "auroc_ci_upper",
    "auprc", "auprc_ci_lower", "auprc_ci_upper",
    "balanced_accuracy", "f1", "brier", "log_loss",
    "sensitivity", "specificity",
]


def build_table2_performance(art: PhaseArtifacts) -> List[Dict[str, str]]:
    """Build Table 2: baseline vs deep model performance comparison."""
    out: List[Dict[str, str]] = []
    for src in [art.baseline_summary_metrics, art.deep_summary_metrics]:
        if src is None:
            continue
        for r in _read_csv(src):
            if r.get("aggregation") == "overall_oof":
                out.append({k: r.get(k, "") for k in _PERF_COLS})
    logger.info("Table 2: %d model rows", len(out))
    return out


# ---------------------------------------------------------------------------
# Table 3: Top ranked genes (CDPS components)
# ---------------------------------------------------------------------------

_TABLE3_COLS = [
    "rank", "gene", "cdps",
    "attribution_norm", "gate_norm", "stability_norm",
    "invariance_norm", "perturbation_norm",
]


def build_table3_top_genes(art: PhaseArtifacts, top_k: int = 25) -> List[Dict[str, str]]:
    """Build Table 3: top CDPS-ranked genes with component decomposition."""
    src = art.top25_genes_cdps if top_k <= 25 else art.top100_genes_cdps
    if src is None:
        src = art.ranked_genes_cdps
    if src is None:
        return []
    rows = _read_csv(src)[:top_k]
    out = [{k: r.get(k, "") for k in _TABLE3_COLS} for r in rows]
    logger.info("Table 3: top %d genes", len(out))
    return out


# ---------------------------------------------------------------------------
# Table 4: Validation-augmented genes
# ---------------------------------------------------------------------------

_TABLE4_COLS = [
    "rank_final", "gene", "final_score", "cdps_rank", "cdps",
    "validation_score", "de_norm", "enrichment_norm", "network_norm",
    "clinical_norm", "subgroup_norm", "external_norm",
]


def build_table4_validation(art: PhaseArtifacts, top_k: int = 25) -> List[Dict[str, str]]:
    """Build Table 4: final validated gene ranking with all evidence layers."""
    if art.final_validated is None:
        return []
    rows = _read_csv(art.final_validated)[:top_k]
    out = [{k: r.get(k, "") for k in _TABLE4_COLS} for r in rows]
    logger.info("Table 4: top %d validated genes", len(out))
    return out


# ---------------------------------------------------------------------------
# Supplementary tables S1-S6
# ---------------------------------------------------------------------------

def build_supplementary_tables(art: PhaseArtifacts) -> Dict[str, Tuple[List[Dict[str, str]], Sequence[str]]]:
    """Build supplementary tables S1-S6. Returns {name: (rows, fieldnames)}."""
    tables: Dict[str, Tuple[List[Dict[str, str]], Sequence[str]]] = {}

    # S1: Full ranked genes
    if art.ranked_genes_cdps:
        rows = _read_csv(art.ranked_genes_cdps)
        if rows:
            tables["supplementary_table_s1_full_ranked_genes"] = (rows, list(rows[0].keys()))

    # S2: DE results
    if art.de_results:
        rows = _read_csv(art.de_results)
        if rows:
            tables["supplementary_table_s2_de_results"] = (rows, list(rows[0].keys()))

    # S3: Enrichment results (may not exist if GMT files not provided)
    # Skip gracefully — no enrichment files to collect

    # S4: Clinical associations
    if art.clinical_results:
        rows = _read_csv(art.clinical_results)
        if rows:
            tables["supplementary_table_s4_clinical_associations"] = (rows, list(rows[0].keys()))

    # S5: Model hyperparameters (from phase2_summary config)
    if art.phase2_summary:
        summary = _read_json(art.phase2_summary)
        config = summary.get("config", {})
        hp_rows = [{"parameter": k, "value": str(v)} for k, v in sorted(config.items())]
        if hp_rows:
            tables["supplementary_table_s5_model_hyperparameters"] = (hp_rows, ["parameter", "value"])

    # S6: Fold-level metrics
    if art.deep_per_fold:
        rows = _read_csv(art.deep_per_fold)
        if rows:
            tables["supplementary_table_s6_fold_level_metrics"] = (rows, list(rows[0].keys()))

    # S3: Enrichment summary (now filled — step6 enrichment_summary_top_pathways.csv)
    if art.enrichment_summary:
        rows = _read_csv(art.enrichment_summary)
        if rows:
            tables["supplementary_table_s3_enrichment_results"] = (rows, list(rows[0].keys()))

    # S7: Top-25 final priority ranking (step6b)
    if art.top25_final_priority_ranking:
        rows = _read_csv(art.top25_final_priority_ranking)
        if rows:
            tables["supplementary_table_s7_top25_priority_ranking"] = (rows, list(rows[0].keys()))

    # S8: Survival summary per gene (step6b)
    if art.top25_survival_summary:
        rows = _read_csv(art.top25_survival_summary)
        if rows:
            tables["supplementary_table_s8_survival_summary"] = (rows, list(rows[0].keys()))

    # S9: External GEO validation concordance (step6b)
    if art.top25_external_validation:
        rows = _read_csv(art.top25_external_validation)
        if rows:
            tables["supplementary_table_s9_external_validation"] = (rows, list(rows[0].keys()))

    # S10: Network/PPI gene support (step6)
    if art.network_gene_support:
        rows = _read_csv(art.network_gene_support)
        if rows:
            tables["supplementary_table_s10_network_gene_support"] = (rows, list(rows[0].keys()))

    logger.info("Supplementary tables built: %s", list(tables.keys()))
    return tables


# ---------------------------------------------------------------------------
# Manuscript Markdown drafts
# ---------------------------------------------------------------------------

def _fmt_float(val: str, decimals: int = 4) -> str:
    try:
        return f"{float(val):.{decimals}f}"
    except (ValueError, TypeError):
        return val


def generate_manuscript_report(
    art: PhaseArtifacts,
    table1: List[Dict[str, str]],
    table2: List[Dict[str, str]],
    table3: List[Dict[str, str]],
    table4: List[Dict[str, str]],
) -> str:
    """Generate manuscript_report.md: overview and key results."""
    lines: List[str] = []
    lines.append("# CAGE: Candidate-driver Invariant Prioritization for TCGA ESCA")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append("This study applies a Candidate-driver Invariant Prioritization framework "
                 "to TCGA esophageal carcinoma (ESCA) data, combining deep invariant modeling "
                 "with multi-layered biological validation to identify computationally "
                 "prioritized candidate driver genes.")
    lines.append("")

    # Cohort summary
    lines.append("## Cohort Summary")
    lines.append("")
    if table1:
        lines.append("| Category | Value | Detail |")
        lines.append("|----------|-------|--------|")
        for r in table1:
            lines.append(f"| {r['category']} | {r['value']} | {r['detail']} |")
        lines.append("")

    # Model performance
    lines.append("## Model Performance (Table 2)")
    lines.append("")
    if table2:
        lines.append("| Model | AUROC | AUROC 95% CI | AUPRC | Bal. Acc. | F1 |")
        lines.append("|-------|-------|-------------|-------|-----------|-----|")
        for r in table2:
            ci = f"[{_fmt_float(r.get('auroc_ci_lower',''), 3)}, {_fmt_float(r.get('auroc_ci_upper',''), 3)}]"
            lines.append(
                f"| {r.get('model','')} | {_fmt_float(r.get('auroc',''), 4)} | {ci} "
                f"| {_fmt_float(r.get('auprc',''), 4)} "
                f"| {_fmt_float(r.get('balanced_accuracy',''), 4)} "
                f"| {_fmt_float(r.get('f1',''), 4)} |"
            )
        lines.append("")

    # Top genes
    lines.append("## Top Candidate Driver Genes (Table 3)")
    lines.append("")
    lines.append("*Note: Genes listed below are computationally prioritized candidates. "
                 "No causal or clinical-validation claims are made.*")
    lines.append("")
    if table3:
        lines.append("| Rank | Gene | CDPS | Attribution | Gate | Stability | Invariance | Perturbation |")
        lines.append("|------|------|------|-------------|------|-----------|------------|--------------|")
        for r in table3[:25]:
            lines.append(
                f"| {r.get('rank','')} | {r.get('gene','')} "
                f"| {_fmt_float(r.get('cdps',''), 4)} "
                f"| {_fmt_float(r.get('attribution_norm',''), 3)} "
                f"| {_fmt_float(r.get('gate_norm',''), 3)} "
                f"| {_fmt_float(r.get('stability_norm',''), 3)} "
                f"| {_fmt_float(r.get('invariance_norm',''), 3)} "
                f"| {_fmt_float(r.get('perturbation_norm',''), 3)} |"
            )
        lines.append("")

    # Validated ranking
    lines.append("## Validated Gene Ranking (Table 4)")
    lines.append("")
    if table4:
        lines.append("| Rank | Gene | Final Score | CDPS | Val. Score | DE | Clinical | Subgroup |")
        lines.append("|------|------|-------------|------|------------|-----|----------|----------|")
        for r in table4[:25]:
            lines.append(
                f"| {r.get('rank_final','')} | {r.get('gene','')} "
                f"| {_fmt_float(r.get('final_score',''), 4)} "
                f"| {_fmt_float(r.get('cdps',''), 4)} "
                f"| {_fmt_float(r.get('validation_score',''), 4)} "
                f"| {_fmt_float(r.get('de_norm',''), 3)} "
                f"| {_fmt_float(r.get('clinical_norm',''), 3)} "
                f"| {_fmt_float(r.get('subgroup_norm',''), 3)} |"
            )
        lines.append("")

    # Claim boundary
    lines.append("## Claim Boundaries")
    lines.append("")
    lines.append("All genes identified in this study are described as **candidate driver genes** "
                 "or **computationally prioritized genes**. No claims of causality or clinical "
                 "utility are made. External validation on independent cohorts is required to "
                 "confirm biological relevance and translational potential.")
    lines.append("")

    return "\n".join(lines)


def generate_manuscript_methods(art: PhaseArtifacts) -> str:
    """Generate manuscript_methods.md: structured methods section."""
    p1 = _read_json(art.phase1_summary) if art.phase1_summary else {}
    p2 = _read_json(art.phase2_summary) if art.phase2_summary else {}
    p3 = _read_json(art.phase3_summary) if art.phase3_summary else {}
    p4 = _read_json(art.phase4_summary) if art.phase4_summary else {}

    cohort = p1.get("cohort", {})
    cv = p1.get("cross_validation", {})
    p2_config = p2.get("config", {})
    p2_cohort = p2.get("cohort", {})
    p3_config = p3.get("config", {})
    p4_config = p4.get("config", {})

    lines: List[str] = []
    lines.append("# Methods")
    lines.append("")

    # Data sources
    lines.append("## Data Sources and Cohort Curation")
    lines.append("")
    lines.append(f"Gene expression data (RNA-seq STAR counts) and clinical metadata "
                 f"were obtained from The Cancer Genome Atlas (TCGA) esophageal "
                 f"carcinoma (ESCA) project. The curated cohort comprises "
                 f"**{cohort.get('n_total_samples', 'N/A')}** samples "
                 f"(**{cohort.get('n_tumor', 'N/A')}** tumor, "
                 f"**{cohort.get('n_normal', 'N/A')}** normal) from "
                 f"**{cohort.get('n_patients', 'N/A')}** patients.")
    lines.append("")

    # Preprocessing
    cfg1 = p1.get("config", {})
    lines.append("## Preprocessing")
    lines.append("")
    lines.append(f"Variance-stabilizing transformation (VST) was applied to raw counts, "
                 f"followed by gene filtering (minimum count >= {cfg1.get('min_count', 'N/A')} "
                 f"in >= {cfg1.get('min_samples_per_gene', 'N/A')} samples, near-zero variance "
                 f"threshold = {cfg1.get('near_zero_variance_threshold', 'N/A')}). "
                 f"The resulting {p2_cohort.get('n_genes', 'N/A')}-gene matrix was z-score "
                 f"normalized across samples.")
    lines.append("")

    # Environments
    envs = cfg1.get("environments", [])
    if envs:
        lines.append(f"Five environment variables were defined for invariant modeling: "
                     f"{', '.join(envs)}.")
        lines.append("")

    # Cross-validation
    lines.append("## Cross-Validation Strategy")
    lines.append("")
    lines.append(f"Patient-grouped nested cross-validation was employed with "
                 f"**{cv.get('n_outer_folds', 'N/A')}** outer folds and "
                 f"**{cv.get('n_inner_folds', 'N/A')}** inner folds "
                 f"(seed = {cv.get('seed', 'N/A')}). Patient grouping ensures "
                 f"no data leakage from paired tumor-normal samples of the same patient.")
    lines.append("")

    # Deep invariant model
    lines.append("## Deep Sparse Invariant Model")
    lines.append("")
    lines.append("A gated sparse neural network with environment-invariant regularization "
                 "was trained to classify tumor vs. normal samples. The model incorporates "
                 "learned gate weights for implicit feature selection and a domain-invariance "
                 "penalty to ensure stable predictions across environment strata.")
    lines.append("")

    # CDPS
    lines.append("## Causality-Aware Priority Score (CDPS)")
    lines.append("")
    weights = p3_config.get("weights", {})
    if weights:
        w_str = ", ".join(f"{k}={v}" for k, v in sorted(weights.items()))
        lines.append(f"The CDPS integrates five normalized components: {w_str}. "
                     f"Attribution was computed via {p3_config.get('attribution_method', 'integrated gradients')} "
                     f"with {p3_config.get('ig_steps', 32)} steps. "
                     f"Stability was assessed using the top {p3_config.get('stability_top_frac', 0.05):.0%} "
                     f"of genes across folds. Normalization: {p3_config.get('normalization', 'minmax')}.")
    lines.append("")

    # Biological validation
    lines.append("## Biological and Statistical Validation")
    lines.append("")
    de_method = p4_config.get("de_method", "welch")
    fdr_t = p4_config.get("fdr_threshold", 0.05)
    lfc_t = p4_config.get("lfc_threshold", 1.0)
    lines.append(f"Differential expression was assessed using a {de_method} test "
                 f"(tumor vs. normal) with Benjamini-Hochberg FDR correction "
                 f"(threshold = {fdr_t}). Effect size filtering required "
                 f"|effect_size_norm| >= {lfc_t}.")
    lines.append("")

    val_weights = p4_config.get("weights", {})
    if val_weights:
        w_str = ", ".join(f"{k}={v}" for k, v in sorted(val_weights.items()))
        lines.append(f"The integrated validation score combines: {w_str}. "
                     f"The final ranking uses 0.5 * CDPS_normalized + 0.5 * validation_score.")
    lines.append("")

    if p4_config.get("run_survival"):
        lines.append("Survival analysis was performed using a two-sample log-rank test "
                     "(median-split) on the tumor-only cohort, with BH FDR correction.")
        lines.append("")

    if p4_config.get("run_subgroup_sensitivity"):
        lines.append("Subgroup robustness was evaluated by checking consistency of tumor-vs-normal "
                     "effect direction within each environment stratum.")
        lines.append("")

    # Reproducibility
    lines.append("## Reproducibility")
    lines.append("")
    lines.append(f"All analyses used seed = {cv.get('seed', 2026)}. "
                 "The pipeline is implemented in pure Python (numpy) without external "
                 "statistical libraries. All statistical tests (Welch t-test, Mann-Whitney U, "
                 "hypergeometric enrichment, log-rank) are implemented from first principles "
                 "using only standard-library math functions and numpy.")
    lines.append("")

    return "\n".join(lines)


def generate_manuscript_results_summary(
    table2: List[Dict[str, str]],
    table3: List[Dict[str, str]],
    table4: List[Dict[str, str]],
    art: PhaseArtifacts,
) -> str:
    """Generate manuscript_results_summary.md with table/figure references."""
    p4 = _read_json(art.phase4_summary) if art.phase4_summary else {}
    de_qc = p4.get("de_qc", {})

    lines: List[str] = []
    lines.append("# Results Summary")
    lines.append("")

    # Model comparison
    lines.append("## Model Performance")
    lines.append("")
    if len(table2) >= 2:
        bl = table2[0]
        dp = table2[1]
        lines.append(f"The deep invariant model achieved AUROC = {_fmt_float(dp.get('auroc',''), 3)} "
                     f"(95% CI [{_fmt_float(dp.get('auroc_ci_lower',''), 3)}, "
                     f"{_fmt_float(dp.get('auroc_ci_upper',''), 3)}]) "
                     f"vs. elastic-net baseline AUROC = {_fmt_float(bl.get('auroc',''), 3)} "
                     f"(**Table 2**, **Figure 4**). "
                     f"Balanced accuracy improved from {_fmt_float(bl.get('balanced_accuracy',''), 3)} "
                     f"to {_fmt_float(dp.get('balanced_accuracy',''), 3)}.")
    lines.append("")

    # CDPS ranking
    lines.append("## Gene Ranking")
    lines.append("")
    if table3:
        top5 = ", ".join(r.get("gene", "") for r in table3[:5])
        lines.append(f"The top 5 CDPS-ranked candidate genes are: {top5} (**Table 3**, **Figure 5**).")
    lines.append("")

    # DE overlap
    lines.append("## Differential Expression")
    lines.append("")
    n_sig = de_qc.get("n_sig_fdr", 0)
    n_pass = de_qc.get("n_sig_fdr_and_effect", 0)
    n_tested = de_qc.get("n_genes_tested", 0)
    if n_tested:
        lines.append(f"Of {n_tested} genes tested, {n_sig} were significant at FDR < 0.05, "
                     f"and {n_pass} also passed the effect-size threshold (**Figure 6a**).")
    lines.append("")

    # Final validated
    lines.append("## Final Validated Ranking")
    lines.append("")
    if table4:
        top5v = ", ".join(r.get("gene", "") for r in table4[:5])
        lines.append(f"After integrating CDPS with biological validation, the top 5 genes are: "
                     f"{top5v} (**Table 4**, **Figure 6**).")
    lines.append("")

    lines.append("*See Methods for full details on all statistical tests and scoring weights.*")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Manifests
# ---------------------------------------------------------------------------

def build_study_summary(
    art: PhaseArtifacts,
    table2: List[Dict[str, str]],
    table4: List[Dict[str, str]],
    seed: int,
) -> Dict[str, Any]:
    """Build study_summary.json for reproducibility."""
    p1 = _read_json(art.phase1_summary) if art.phase1_summary else {}
    p4 = _read_json(art.phase4_summary) if art.phase4_summary else {}

    top_genes = [r.get("gene", "") for r in table4[:25]]
    best_model = {}
    for r in table2:
        if r.get("model") == "deep":
            best_model = {k: r.get(k, "") for k in ["auroc", "auprc", "balanced_accuracy", "f1"]}

    return {
        "study_title": "CAGE: Candidate-driver Invariant Prioritization for TCGA ESCA",
        "generated_at": datetime.datetime.now().isoformat(),
        "seed": seed,
        "cohort": p1.get("cohort", {}),
        "cross_validation": p1.get("cross_validation", {}),
        "best_model_performance": best_model,
        "de_summary": p4.get("de_qc", {}),
        "top_25_validated_genes": top_genes,
        "validation_weights": p4.get("config", {}).get("weights", {}),
        "claim_boundary": (
            "All genes are computationally prioritized candidates. "
            "No causality or clinical-validation claims are made."
        ),
        "missing_artifacts": art.missing,
    }


def build_file_manifest(
    output_dir: Path,
    kind: str,
    glob_pattern: str,
    description_map: Optional[Dict[str, str]] = None,
) -> List[Dict[str, str]]:
    """Scan output_dir for files matching glob_pattern and build a manifest."""
    desc_map = description_map or {}
    manifest: List[Dict[str, str]] = []
    for p in sorted(output_dir.rglob(glob_pattern)):
        rel = str(p.relative_to(output_dir))
        manifest.append({
            "file": rel,
            "kind": kind,
            "size_bytes": str(p.stat().st_size),
            "description": desc_map.get(p.name, ""),
        })
    return manifest


def copy_existing_step_figures(
    art: PhaseArtifacts,
    output_dir: Path,
) -> Tuple[List[str], List[Tuple[str, str]]]:
    """Copy all per-step rendered figures into final_figures/all_step_figures/.

    Step 7 should be a manuscript package, not a bottleneck that hides figures
    already generated by earlier phases. The sidecar style JSON files are copied
    with the SVG/PDF/PNG files so typography/palette provenance stays attached.
    """
    copied: List[str] = []
    skipped: List[Tuple[str, str]] = []
    out_root = output_dir / "final_figures" / "all_step_figures"
    patterns = ("*.svg", "*.pdf", "*.png", "*.style.json")

    for label, src_dir in sorted(art.figure_dirs.items()):
        if not src_dir.is_dir():
            skipped.append((label, f"figure directory not found: {src_dir}"))
            continue
        dst_dir = out_root / label
        dst_dir.mkdir(parents=True, exist_ok=True)
        n = 0
        for pattern in patterns:
            for src in sorted(src_dir.rglob(pattern)):
                dst = dst_dir / src.name
                shutil.copy2(src, dst)
                copied.append(str(dst.relative_to(output_dir)))
                n += 1
        if n == 0:
            skipped.append((label, "no figure files found"))

    if copied:
        rows = [{"file": p, "source": p.split("/")[2] if "/" in p else ""} for p in copied]
        _write_csv(
            output_dir / "manifests" / "source_figure_manifest.csv",
            rows,
            ["file", "source"],
        )
    return copied, skipped


def copy_curated_manuscript_figures(
    art: PhaseArtifacts,
    output_dir: Path,
) -> Tuple[List[str], List[Tuple[str, str]]]:
    """Copy high-value source figures into final_figures/ with manuscript names."""
    copied: List[str] = []
    skipped: List[Tuple[str, str]] = []
    fig_dir = output_dir / "final_figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    selections = [
        ("step2_cohort", "D1_cohort_composition", "figure1_cohort_composition"),
        ("step2_cohort", "D2_clinical_demographics", "figure2_clinical_demographics"),
        ("step3_baselines", "fig_baseline_roc", "figure3_baseline_roc"),
        ("step3_baselines", "fig_baseline_pr", "figure3_baseline_precision_recall"),
        ("step4_deep_model", "fig_deep_roc_pr", "figure4_deep_roc_pr"),
        ("step5_cdps", "fig_G2_top25_components", "figure5_cdps_top25_components"),
        ("step5_cdps", "fig_G3_score_scatter", "figure5_cdps_score_scatter"),
        ("step6_validation", "fig_de_volcano", "figure6_de_volcano"),
        ("step6_validation", "fig_cdps_vs_de", "figure6_cdps_vs_de"),
        ("step6_validation", "fig_final_rank_components", "figure6_final_rank_components"),
        ("step6_validation", "fig_H3_pathway_enrichment", "figure6_pathway_enrichment"),
        ("step6_validation", "fig_H2_clinical_assoc_heatmap", "figure6_clinical_association_heatmap"),
        ("step6_validation", "fig_H4_survival_summary", "figure6_survival_summary"),
        ("step6_validation", "fig_H5_subgroup_sensitivity", "figure6_subgroup_sensitivity"),
        ("step6_validation", "fig_H6_final_ranking_waterfall", "figure6_final_ranking_waterfall"),
        ("step6_validation", "fig_H7_gene_pathway_bubble", "figure6_gene_pathway_bubble"),
        # Step 6b: top-25 final prioritization
        ("step6b_top25", "top25_evidence_component_heatmap", "figure7a_top25_evidence_heatmap"),
        ("step6b_top25", "top25_final_score_waterfall", "figure7b_top25_final_score_waterfall"),
        ("step6b_top25", "top25_de_vs_cdps_scatter", "figure7c_top25_de_vs_cdps_scatter"),
        ("step6b_top25", "top25_tiered_ranking_barplot", "figure7d_top25_tiered_ranking"),
        ("step6b_top25", "top25_external_concordance_heatmap", "figure7e_top25_external_concordance"),
        ("step6b_top25", "top25_clinical_survival_summary", "figure7f_top25_clinical_survival"),
        ("step6b_top25", "top25_network_pathway_support", "figure7g_top25_network_pathway"),
        ("step6b_top25", "top25_missing_evidence_map", "figure7h_top25_missing_evidence"),
        # Step 6b: survival/external validation combined panels
        ("step6b_survival_km", "combined_KM_all25", "figure8a_km_all25_panel"),
        ("step6b_survival_boxplots", "combined_TCGA_boxplots_all25", "figure8b_tcga_boxplots_all25"),
        ("step6b_survival", "top25_external_concordance_heatmap", "figure8c_geo_external_concordance"),
        ("step6b_survival", "top25_survival_summary_plot", "figure8d_survival_summary"),
        # Step 6b: per-dataset GEO combined panels (supplementary)
        ("step6b_survival_external", "GSE38129_all25_boxplots", "figureS1_geo_GSE38129_all25"),
        ("step6b_survival_external", "GSE161533_all25_boxplots", "figureS2_geo_GSE161533_all25"),
        ("step6b_survival_external", "GSE53624_all25_boxplots", "figureS3_geo_GSE53624_all25"),
        ("step6b_survival_external", "GSE53625_all25_boxplots", "figureS4_geo_GSE53625_all25"),
        # Step 8: Affymetrix GEO and Agilent external validation
        ("step8_geo_validation", "C1_all_cohorts_auroc", "figure9a_affymetrix_classifier_performance"),
        ("step8_geo_validation", "C2_cross_platform_concordance", "figure9b_affymetrix_concordance"),
        ("step8_geo_validation", "C3_combined_direction_heatmap", "figure9c_affymetrix_direction_heatmap"),
        ("step8_agilent_validation", "A4_classifier_performance", "figure9d_agilent_classifier_performance"),
        ("step8_agilent_validation", "A2_concordance_bar", "figure9e_agilent_concordance"),
        ("step8_agilent_validation", "A1_direction_heatmap", "figure9f_agilent_direction_heatmap"),
    ]

    for source_label, stem, manuscript_stem in selections:
        src_dir = art.figure_dirs.get(source_label)
        if src_dir is None:
            skipped.append((manuscript_stem, f"source directory missing: {source_label}"))
            continue
        found = False
        for suffix in (".svg", ".pdf", ".png", ".style.json"):
            src = src_dir / f"{stem}{suffix}"
            if not src.is_file():
                continue
            dst = fig_dir / f"{manuscript_stem}{suffix}"
            shutil.copy2(src, dst)
            copied.append(str(dst.relative_to(output_dir)))
            found = True
        if not found:
            skipped.append((manuscript_stem, f"source figure not found: {src_dir / stem}"))

    return copied, skipped


# ---------------------------------------------------------------------------
# Figure assembly (optional — requires matplotlib)
# ---------------------------------------------------------------------------

def _has_matplotlib() -> bool:
    try:
        import matplotlib
        return True
    except ImportError:
        return False


def _prediction_pairs_from_artifacts(
    art: PhaseArtifacts,
) -> Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    """Collect train/OOF prediction arrays for model-performance ROC plots."""
    from . import metrics as mx  # noqa: F401 - imported here to keep optional plotting isolated

    curves: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {}

    if art.baseline_oof:
        rows = _read_csv(art.baseline_oof)
        if rows:
            prob_cols = [c for c in rows[0] if c.endswith("_prob")]
            y = np.array([int(r["y_true"]) for r in rows], dtype=np.int64)
            for col in prob_cols:
                vals = []
                yy = []
                for y_i, r in zip(y, rows):
                    v = r.get(col, "")
                    if v == "":
                        continue
                    yy.append(int(y_i))
                    vals.append(float(v))
                if vals:
                    model = col[:-5]
                    curves.setdefault(model, {})["OOF"] = (
                        np.asarray(yy, dtype=np.int64),
                        np.asarray(vals, dtype=np.float64),
                    )

    if art.baseline_train:
        rows = _read_csv(art.baseline_train)
        by_model: Dict[str, Tuple[List[int], List[float]]] = {}
        for r in rows:
            v = r.get("train_prob", "")
            if v == "":
                continue
            model = r.get("model", "")
            yy, pp = by_model.setdefault(model, ([], []))
            yy.append(int(r["y_true"]))
            pp.append(float(v))
        for model, (yy, pp) in by_model.items():
            if pp:
                curves.setdefault(model, {})["Train"] = (
                    np.asarray(yy, dtype=np.int64),
                    np.asarray(pp, dtype=np.float64),
                )

    if art.deep_oof:
        rows = _read_csv(art.deep_oof)
        yy: List[int] = []
        pp: List[float] = []
        for r in rows:
            v = r.get("deep_prob", "")
            if v == "":
                continue
            yy.append(int(r["y_true"]))
            pp.append(float(v))
        if pp:
            curves.setdefault("deep", {})["OOF"] = (
                np.asarray(yy, dtype=np.int64),
                np.asarray(pp, dtype=np.float64),
            )

    if art.deep_train:
        rows = _read_csv(art.deep_train)
        yy = []
        pp = []
        for r in rows:
            v = r.get("deep_train_prob", "")
            if v == "":
                continue
            yy.append(int(r["y_true"]))
            pp.append(float(v))
        if pp:
            curves.setdefault("deep", {})["Train"] = (
                np.asarray(yy, dtype=np.int64),
                np.asarray(pp, dtype=np.float64),
            )

    return curves


def generate_figures(
    art: PhaseArtifacts,
    output_dir: Path,
    style: Any = None,
) -> Tuple[List[str], List[Tuple[str, str]]]:
    """Generate publication figures. Returns (generated, skipped) lists."""
    generated: List[str] = []
    skipped: List[Tuple[str, str]] = []

    if not _has_matplotlib():
        fig_names = [
            "figure1_study_overview", "figure2_cohort_characteristics",
            "figure3_latent_space", "figure4_model_performance",
            "figure5_gene_ranking", "figure6_biological_validation",
            "figure7_perturbation", "figure8_external_validation",
        ]
        for name in fig_names:
            skipped.append((name, "matplotlib not installed"))
        return generated, skipped

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .publication_style import (
        apply_style, save_figure, cage_palette,
        semantic_color, categorical_colors,
    )

    if style is not None:
        apply_style(style)

    fig_dir = output_dir / "final_figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    palette = cage_palette("semantic")

    # --- Figure 2: Cohort characteristics ---
    try:
        if art.master_samples:
            rows = _read_csv(art.master_samples)
            n_tumor = sum(1 for r in rows if r.get("label") == "Tumor")
            n_normal = len(rows) - n_tumor
            fig, ax = plt.subplots(figsize=(3.5, 2.625))
            bars = ax.bar(["Tumor", "Normal"], [n_tumor, n_normal],
                          color=[palette["tumor"], palette["normal"]])
            for bar, val in zip(bars, [n_tumor, n_normal]):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                        str(val), ha="center", va="bottom", fontweight="bold")
            ax.set_ylabel("Number of Samples")
            ax.set_title("Cohort Composition")
            save_figure(fig, fig_dir / "figure2_cohort_characteristics", style=style)
            generated.append("figure2_cohort_characteristics")
        else:
            skipped.append(("figure2_cohort_characteristics", "master_samples not found"))
    except Exception as e:
        skipped.append(("figure2_cohort_characteristics", str(e)))

    # --- Figure 4: Model performance comparison ---
    try:
        from . import metrics as mx

        curves = _prediction_pairs_from_artifacts(art)
        if curves:
            model_order = [m for m in ("elasticnet", "logistic", "rf", "deep") if m in curves]
            model_order += [m for m in sorted(curves) if m not in model_order]
            colors = categorical_colors(max(len(model_order), 4))
            color_by_model = {m: colors[i] for i, m in enumerate(model_order)}

            fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
            ax_roc, ax_bar = axes
            auc_rows: List[Tuple[str, str, float]] = []

            for model in model_order:
                label_model = "Deep" if model == "deep" else model.capitalize()
                pair = curves.get(model, {}).get("OOF")
                if pair is None:
                    continue
                y_arr, p_arr = pair
                auc = mx.auroc(y_arr, p_arr)
                auc_rows.append((model, "OOF", float(auc)))
                fpr, tpr, _ = mx.roc_curve(y_arr, p_arr)
                ax_roc.plot(
                    fpr, tpr,
                    linewidth=1.5,
                    color=color_by_model[model],
                    label=f"{label_model} ({auc:.3f})",
                )

            ax_roc.plot([0, 1], [0, 1], linestyle=":", color="#888888", linewidth=0.9)
            ax_roc.set_xlabel("False Positive Rate")
            ax_roc.set_ylabel("True Positive Rate")
            ax_roc.set_xlim(-0.02, 1.02)
            ax_roc.set_ylim(-0.02, 1.02)
            ax_roc.set_title("Patient-Grouped OOF ROC")
            ax_roc.legend(loc="lower right", fontsize=6.2, frameon=True)

            x = np.arange(len(model_order), dtype=np.float64)
            oof_vals = []
            for model in model_order:
                oof_vals.append(next((v for m, s, v in auc_rows if m == model and s == "OOF"), np.nan))
            ax_bar.bar(x, oof_vals, 0.55, color=[color_by_model[m] for m in model_order],
                       edgecolor="black", linewidth=0.5)
            for xi, v in zip(x, oof_vals):
                if np.isfinite(v):
                    ax_bar.text(xi, min(1.04, v + 0.015), f"{v:.3f}",
                                ha="center", va="bottom", fontsize=6.5, rotation=90)
            ax_bar.set_xticks(x)
            ax_bar.set_xticklabels(["Deep" if m == "deep" else m.capitalize() for m in model_order],
                                   rotation=25, ha="right", fontsize=8)
            ax_bar.set_ylabel("AUROC")
            ax_bar.set_ylim(0.0, 1.08)
            ax_bar.axhline(0.5, color="#888888", linewidth=0.8, linestyle=":")
            ax_bar.set_title("OOF AUROC Summary")
            fig.suptitle("Model ROC Performance", fontsize=11, fontweight="bold")
            fig.tight_layout(rect=[0, 0, 1, 0.94])
            save_figure(fig, fig_dir / "figure4_model_performance", style=style)
            generated.append("figure4_model_performance")
        elif art.baseline_summary_metrics and art.deep_summary_metrics:
            bl_rows = _read_csv(art.baseline_summary_metrics)
            dp_rows = _read_csv(art.deep_summary_metrics)
            bl_oof = next((r for r in bl_rows if r.get("aggregation") == "overall_oof"), None)
            dp_oof = next((r for r in dp_rows if r.get("aggregation") == "overall_oof"), None)
            if bl_oof and dp_oof:
                metrics = ["auroc", "auprc", "balanced_accuracy", "f1"]
                bl_vals = [float(bl_oof.get(m, 0)) for m in metrics]
                dp_vals = [float(dp_oof.get(m, 0)) for m in metrics]
                x = np.arange(len(metrics))
                w = 0.35
                fig, ax = plt.subplots(figsize=(4.5, 3.0))
                ax.bar(x - w / 2, bl_vals, w, label="Elastic-Net", color=palette["normal"])
                ax.bar(x + w / 2, dp_vals, w, label="Deep Invariant", color=palette["tumor"])
                ax.set_xticks(x)
                ax.set_xticklabels(["AUROC", "AUPRC", "Bal. Acc.", "F1"])
                ax.set_ylim(0, 1.1)
                ax.set_ylabel("Score")
                ax.set_title("Model Performance Comparison")
                ax.legend()
                save_figure(fig, fig_dir / "figure4_model_performance", style=style)
                generated.append("figure4_model_performance")
            else:
                skipped.append(("figure4_model_performance", "no overall_oof rows"))
        else:
            skipped.append(("figure4_model_performance", "metrics not found"))
    except Exception as e:
        skipped.append(("figure4_model_performance", str(e)))

    # --- Figure 5: Top gene CDPS component decomposition ---
    try:
        if art.top25_genes_cdps:
            rows = _read_csv(art.top25_genes_cdps)[:25]
            genes = [r["gene"] for r in rows]
            components = ["attribution_norm", "gate_norm", "stability_norm",
                          "invariance_norm", "perturbation_norm"]
            comp_labels = ["Attribution", "Gate", "Stability", "Invariance", "Perturbation"]
            colors = categorical_colors(5)
            y = np.arange(len(genes))
            fig, ax = plt.subplots(figsize=(5.6, 6.8))
            left = np.zeros(len(genes))
            for ci, comp in enumerate(components):
                vals = np.array([float(r.get(comp, 0)) for r in rows])
                ax.barh(y, vals, left=left, label=comp_labels[ci], color=colors[ci], height=0.7)
                left += vals
            ax.set_yticks(y)
            ax.set_yticklabels(genes)
            ax.invert_yaxis()
            ax.set_xlabel("Normalized Component Score")
            ax.set_title("CDPS Component Decomposition (Top 25)")
            ax.legend(loc="lower right", fontsize=7)
            save_figure(fig, fig_dir / "figure5_gene_ranking", style=style)
            generated.append("figure5_gene_ranking")
        else:
            skipped.append(("figure5_gene_ranking", "top25 not found"))
    except Exception as e:
        skipped.append(("figure5_gene_ranking", str(e)))

    # --- Figure 6: Validation summary (DE volcano-style scatter) ---
    try:
        if art.de_results:
            de_rows = _read_csv(art.de_results)
            effects = []
            neg_log_fdr = []
            colors_arr = []
            for r in de_rows:
                try:
                    es = float(r.get("effect_size_norm", 0))
                    fdr = float(r.get("fdr_bh", r.get("fdr", 1)))
                    fdr = max(fdr, 1e-300)
                    effects.append(es)
                    neg_log_fdr.append(-np.log10(fdr))
                    sig = int(r.get("sig_fdr_and_effect", 0))
                    if sig and es > 0:
                        colors_arr.append(palette["up"])
                    elif sig and es < 0:
                        colors_arr.append(palette["down"])
                    else:
                        colors_arr.append(palette["nonsig"])
                except (ValueError, TypeError):
                    continue
            if effects:
                fig, ax = plt.subplots(figsize=(4.5, 3.5))
                ax.scatter(effects, neg_log_fdr, c=colors_arr, s=8, alpha=0.7, edgecolors="none")
                ax.axhline(-np.log10(0.05), ls="--", color="gray", lw=0.8)
                ax.set_xlabel("Effect Size (z-score units)")
                ax.set_ylabel("-log10(FDR)")
                ax.set_title("Differential Expression: Tumor vs Normal")
                save_figure(fig, fig_dir / "figure6_de_volcano", style=style)
                generated.append("figure6_de_volcano")
            else:
                skipped.append(("figure6_de_volcano", "no valid DE rows"))
        else:
            skipped.append(("figure6_de_volcano", "DE results not found"))
    except Exception as e:
        skipped.append(("figure6_de_volcano", str(e)))

    return generated, skipped


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_step7_packaging(
    step2_dir: Optional[Path],
    step3_dir: Optional[Path],
    step4_dir: Optional[Path],
    step5_dir: Optional[Path],
    step6_dir: Optional[Path],
    output_dir: Path,
    *,
    step6b_top25_dir: Optional[Path] = None,
    step6b_survival_dir: Optional[Path] = None,
    step8_dir: Optional[Path] = None,
    export_markdown: bool = True,
    export_html: bool = False,
    export_docx_ready: bool = False,
    export_latex_ready: bool = False,
    copy_final_figures: bool = True,
    copy_key_tables: bool = True,
    build_supplement: bool = True,
    build_reviewer_bundle: bool = False,
    style: Any = None,
    seed: int = 2026,
) -> Dict[str, Any]:
    """Run the full Step 7 manuscript packaging pipeline.

    Returns a summary dict suitable for phase5_summary.json.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Discover artifacts ---
    logger.info("Step 7: discovering artifacts ...")
    inferred_figure_dirs: Dict[str, Path] = {}
    for label, dirname in [
        ("step8_geo_validation", "step8_geo_validation"),
        ("step8_agilent_validation", "step8_agilent_validation"),
        ("pipeline_figures", "figures/pipeline"),
        ("validation_figures", "figures"),
    ]:
        candidate = output_dir.parent / dirname / "figures"
        if candidate.is_dir():
            inferred_figure_dirs[label] = candidate
        elif dirname.endswith("/pipeline"):
            candidate = output_dir.parent / dirname
            if candidate.is_dir():
                inferred_figure_dirs[label] = candidate

    art = discover_artifacts(
        step2_dir,
        step3_dir,
        step4_dir,
        step5_dir,
        step6_dir,
        step8_dir=step8_dir,
        extra_figure_dirs=inferred_figure_dirs,
        step6b_top25_dir=step6b_top25_dir,
        step6b_survival_dir=step6b_survival_dir,
    )

    # --- 2. Build main tables ---
    logger.info("Step 7: building main tables ...")
    tables_dir = output_dir / "final_tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    table1 = build_table1_cohort_summary(art)
    if table1:
        _write_csv(tables_dir / "table1_cohort_summary.csv", table1, TABLE1_FIELDS)

    table2 = build_table2_performance(art)
    if table2:
        _write_csv(tables_dir / "table2_baseline_vs_deep_performance.csv", table2, _PERF_COLS)

    table3 = build_table3_top_genes(art, top_k=25)
    if table3:
        _write_csv(tables_dir / "table3_top_ranked_genes.csv", table3, _TABLE3_COLS)

    table4 = build_table4_validation(art, top_k=25)
    if table4:
        _write_csv(tables_dir / "table4_validation_augmented_genes.csv", table4, _TABLE4_COLS)

    # --- 3. Build supplementary tables ---
    if build_supplement:
        logger.info("Step 7: building supplementary tables ...")
        supp_dir = output_dir / "supplementary"
        supp_dir.mkdir(parents=True, exist_ok=True)
        supp_tables = build_supplementary_tables(art)
        for name, (rows, fnames) in supp_tables.items():
            _write_csv(supp_dir / f"{name}.csv", rows, fnames)

    # --- 4. Manuscript drafts ---
    if export_markdown:
        logger.info("Step 7: generating manuscript Markdown ...")
        ms_dir = output_dir / "manuscript"
        ms_dir.mkdir(parents=True, exist_ok=True)

        report_md = generate_manuscript_report(art, table1, table2, table3, table4)
        _write_text(ms_dir / "manuscript_report.md", report_md)

        methods_md = generate_manuscript_methods(art)
        _write_text(ms_dir / "manuscript_methods.md", methods_md)

        results_md = generate_manuscript_results_summary(table2, table3, table4, art)
        _write_text(ms_dir / "manuscript_results_summary.md", results_md)

    # --- 5. Figures ---
    logger.info("Step 7: generating figures ...")
    fig_generated, fig_skipped = generate_figures(art, output_dir, style=style)
    curated_figures: List[str] = []
    copied_step_figures: List[str] = []
    if copy_final_figures:
        logger.info("Step 7: copying curated and source figures ...")
        curated_figures, curated_skipped = copy_curated_manuscript_figures(art, output_dir)
        source_figures, source_skipped = copy_existing_step_figures(art, output_dir)
        copied_step_figures = source_figures
        fig_generated.extend(curated_figures)
        fig_skipped.extend(curated_skipped)
        fig_skipped.extend(source_skipped)

    # --- 6. Manifests ---
    logger.info("Step 7: building manifests ...")
    manifest_dir = output_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    study_summary = build_study_summary(art, table2, table4, seed)
    _write_json(manifest_dir / "study_summary.json", study_summary)

    # Output manifest (all files in output_dir)
    output_manifest = build_file_manifest(output_dir, "output", "*")
    if output_manifest:
        _write_csv(manifest_dir / "output_manifest.csv", output_manifest,
                   ["file", "kind", "size_bytes", "description"])

    # Figure manifest
    figure_manifest = build_file_manifest(output_dir, "figure", "*.svg")
    figure_manifest.extend(build_file_manifest(output_dir, "figure", "*.pdf"))
    figure_manifest.extend(build_file_manifest(output_dir, "figure", "*.png"))
    if figure_manifest:
        _write_csv(manifest_dir / "figure_manifest.csv", figure_manifest,
                   ["file", "kind", "size_bytes", "description"])

    # Table manifest
    table_desc = {
        "table1_cohort_summary.csv": "Cohort composition and subgroup distributions",
        "table2_baseline_vs_deep_performance.csv": "Model performance comparison",
        "table3_top_ranked_genes.csv": "Top 25 CDPS-ranked candidate driver genes",
        "table4_validation_augmented_genes.csv": "Top 25 genes with integrated validation scores",
    }
    table_manifest = build_file_manifest(output_dir, "table", "*.csv", table_desc)
    if table_manifest:
        _write_csv(manifest_dir / "table_manifest.csv", table_manifest,
                   ["file", "kind", "size_bytes", "description"])

    # --- 7. Phase summary ---
    phase_summary = {
        "tables_written": {
            "table1": len(table1),
            "table2": len(table2),
            "table3": len(table3),
            "table4": len(table4),
        },
        "supplementary_tables": list(build_supplementary_tables(art).keys()) if build_supplement else [],
        "manuscript_drafts": ["manuscript_report.md", "manuscript_methods.md",
                              "manuscript_results_summary.md"] if export_markdown else [],
        "figures": {
            "generated": fig_generated,
            "curated_copies": curated_figures,
            "source_copies": copied_step_figures,
            "skipped": [{"name": n, "reason": r} for n, r in fig_skipped],
        },
        "manifests": ["study_summary.json", "output_manifest.csv",
                      "figure_manifest.csv", "table_manifest.csv"],
        "missing_artifacts": art.missing,
        "config": {
            "seed": seed,
            "export_markdown": export_markdown,
            "export_html": export_html,
            "copy_final_figures": copy_final_figures,
            "copy_key_tables": copy_key_tables,
            "build_supplement": build_supplement,
            "step8_dir": str(step8_dir) if step8_dir else "",
        },
    }
    _write_json(output_dir / "phase5_summary.json", phase_summary)

    n_tables = sum(1 for v in phase_summary["tables_written"].values() if v > 0)
    logger.info(
        "Step 7 complete | tables=%d supplementary=%d drafts=%d "
        "figures_ok=%d figures_skip=%d manifests=%d missing_artifacts=%d",
        n_tables,
        len(phase_summary["supplementary_tables"]),
        len(phase_summary["manuscript_drafts"]),
        len(fig_generated),
        len(fig_skipped),
        len(phase_summary["manifests"]),
        len(art.missing),
    )

    return phase_summary
