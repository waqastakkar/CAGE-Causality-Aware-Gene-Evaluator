"""CAGE Step 6b: Top-25 CDPS gene final evidence-based prioritization.

This module ingests outputs from Steps 5, 6, and 8 and produces an integrated
evidence table for the top-25 computationally prioritized CDPS genes.

Evidence sources (all optional; graceful degradation when absent):
  * CDPS model scores (Step 5)
  * Differential expression and pathway/network support (Step 6)
  * External GEO / Agilent replication (Step 8)

Scoring logic:
  * Each evidence component is min-max normalized to [0, 1].
  * Missing evidence for a gene is flagged; weights are re-normalized per gene
    over available components so the final score sums to 1.0.
  * Tier 1 / 2 / 3 classification by configurable score thresholds.

Outputs:
  tables/   — integrated evidence CSV, ranked table, tier summary, missing
              evidence report, manuscript-ready table.
  figures/  — waterfall, component heatmap, tiered barplot, concordance
              heatmap, DE vs CDPS scatter, clinical/survival summary, network
              pathway support, missing evidence map.
  reports/  — prioritization summary, methods text, results text (Markdown).
  manifests/— input file manifest, output manifest, figure manifest,
              scoring config, reproducibility JSON.
  logs/     — step log.

Scientific language: all genes described as "computationally prioritized
candidates" that "require experimental confirmation". Never imply causation.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .cli_args import (
    add_figure_args,
    build_step_parser,
    configure_logging,
    style_from_args,
)
from .publication_style import (
    PublicationStyle,
    apply_style,
    categorical_colors,
    log_figure_status,
    cage_palette,
    save_figure,
    semantic_color,
)

logger = logging.getLogger("cage.step6b")

__version__ = "1.0.0"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_TOP_K = 25
DEFAULT_SEED = 2026

_WEIGHT_KEYS = (
    "w_cdps",
    "w_external",
    "w_de",
    "w_clinical_survival",
    "w_pathway_network",
    "w_subgroup",
)

_COMPONENT_LABELS = {
    "cdps": "CDPS / Model",
    "external": "External Replication",
    "de": "Differential Expression",
    "clinical_survival": "Clinical / Survival",
    "pathway_network": "Pathway / Network",
    "subgroup": "Subgroup Robustness",
}


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _safe_read_csv(path: Optional[Path]) -> Optional[List[Dict[str, str]]]:
    """Read a CSV file into a list of row dicts. Returns None if file missing."""
    if path is None or not path.is_file():
        logger.warning("Missing input (skipping): %s", path)
        return None
    rows: List[Dict[str, str]] = []
    with open(path, "r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(dict(row))
    logger.info("Loaded %d rows from %s", len(rows), path.name)
    return rows


def _safe_read_json(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    """Read a JSON file. Returns None if file missing."""
    if path is None or not path.is_file():
        logger.warning("Missing input (skipping): %s", path)
        return None
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    logger.info("Loaded JSON from %s", path.name)
    return data


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    """Write a list of row dicts to CSV with a fixed fieldname order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote %d rows to %s", len(rows), path.name)


def _write_manifest(
    path: Path,
    entries: List[Dict[str, Any]],
    fieldnames: Optional[Sequence[str]] = None,
) -> None:
    if not entries:
        return
    keys = list(fieldnames) if fieldnames is not None else list(entries[0].keys())
    _write_csv(path, entries, keys)


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------


def _to_float(val: Any, default: float = float("nan")) -> float:
    """Convert a string or numeric value to float; return default on failure."""
    if val is None or val == "" or val == "NA" or val == "NaN":
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _minmax_norm(values: List[float]) -> List[float]:
    """Min-max normalize a list of floats to [0, 1]; NaN inputs propagate."""
    arr = np.array(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if len(finite) == 0:
        return [float("nan")] * len(values)
    lo, hi = float(finite.min()), float(finite.max())
    if math.isclose(lo, hi, rel_tol=1e-12, abs_tol=1e-14):
        return [0.5 if np.isfinite(v) else float("nan") for v in arr]
    return [
        float((v - lo) / (hi - lo)) if np.isfinite(v) else float("nan")
        for v in arr
    ]


def _rank_norm(values: List[float], ascending: bool = False) -> List[float]:
    """Rank-based normalization; missing values get NaN.

    ascending=False means the largest value → rank 1 → score 1.0.
    """
    arr = np.array(values, dtype=float)
    finite_idx = [i for i, v in enumerate(arr) if np.isfinite(v)]
    if not finite_idx:
        return [float("nan")] * len(values)
    vals_fin = arr[finite_idx]
    order = np.argsort(vals_fin)
    if not ascending:
        order = order[::-1]
    n = len(finite_idx)
    rank_norm_scores = {finite_idx[i]: float(rank) / (n - 1) if n > 1 else 1.0
                        for rank, i in enumerate(order)}
    return [rank_norm_scores.get(i, float("nan")) for i in range(len(values))]


def _merge_by_gene(
    base: Dict[str, Dict[str, Any]],
    rows: Optional[List[Dict[str, str]]],
    gene_col: str,
    fields: Dict[str, str],
) -> None:
    """Merge selected fields from rows into base dict keyed by gene name.

    Parameters
    ----------
    base:
        Mutable dict { gene_name -> {col: val, ...} }.
    rows:
        CSV rows loaded with _safe_read_csv; None is silently skipped.
    gene_col:
        Name of the column containing the gene symbol in rows.
    fields:
        Mapping { target_col_name -> source_col_name_in_rows }.
    """
    if rows is None:
        return
    for row in rows:
        gene = row.get(gene_col, "").strip()
        if not gene or gene not in base:
            continue
        for target, source in fields.items():
            val = row.get(source, "")
            if target not in base[gene] or base[gene][target] == "":
                base[gene][target] = val


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _score_evidence(
    genes: List[str],
    gene_data: Dict[str, Dict[str, Any]],
    weights: Dict[str, float],
) -> Tuple[List[float], List[str], List[str]]:
    """Compute per-gene weighted final priority scores.

    Returns
    -------
    scores : list[float]
    flags  : list[str]  — pipe-separated missing component flags per gene
    available_components : list[str]  — comma-separated available components
    """
    component_raw: Dict[str, List[float]] = {k: [] for k in weights}

    for gene in genes:
        d = gene_data[gene]
        component_raw["cdps"].append(_to_float(d.get("cdps_score")))
        component_raw["external"].append(_to_float(d.get("external_concordance_rate")))
        de_dir = str(d.get("DE_direction", "")).lower()
        de_fdr = _to_float(d.get("DE_FDR"))
        log2fc = _to_float(d.get("log2FC"))
        if not (math.isnan(log2fc) or math.isnan(de_fdr)):
            component_raw["de"].append(abs(log2fc))
        else:
            component_raw["de"].append(float("nan"))

        clin = _to_float(d.get("clinical_support_score"))
        surv = _to_float(d.get("survival_p_value"))
        if not math.isnan(clin):
            clin_surv = clin
        elif not math.isnan(surv):
            clin_surv = 1.0 - min(1.0, surv)
        else:
            clin_surv = float("nan")
        component_raw["clinical_survival"].append(clin_surv)

        pw = _to_float(d.get("pathway_support_score"))
        net = _to_float(d.get("network_centrality"))
        if not math.isnan(pw) and not math.isnan(net):
            pn = (pw + net) / 2.0
        elif not math.isnan(pw):
            pn = pw
        elif not math.isnan(net):
            pn = net
        else:
            pn = float("nan")
        component_raw["pathway_network"].append(pn)

        component_raw["subgroup"].append(_to_float(d.get("subgroup_robustness_score")))

    normed: Dict[str, List[float]] = {}
    for comp, vals in component_raw.items():
        if comp in ("external",):
            normed[comp] = _minmax_norm(vals)
        elif comp == "clinical_survival":
            normed[comp] = _minmax_norm(vals)
        else:
            normed[comp] = _minmax_norm(vals)

    scores: List[float] = []
    flags_list: List[str] = []
    avail_list: List[str] = []

    for i, gene in enumerate(genes):
        avail_w: Dict[str, float] = {}
        missing: List[str] = []
        for comp, w in weights.items():
            v = normed[comp][i]
            if math.isnan(v):
                missing.append(comp)
            else:
                avail_w[comp] = w

        total_w = sum(avail_w.values())
        if total_w == 0:
            scores.append(float("nan"))
        else:
            score = sum(normed[comp][i] * (w / total_w) for comp, w in avail_w.items())
            scores.append(float(score))

        flags_list.append("|".join(missing) if missing else "none")
        avail_list.append(",".join(avail_w.keys()) if avail_w else "none")

    return scores, flags_list, avail_list


def _assign_tier(
    score: float,
    tier1_threshold: float,
    tier2_threshold: float,
    n_available_components: int,
) -> str:
    """Assign evidence tier based on score and multi-source support."""
    if math.isnan(score):
        return "Tier 3"
    if score >= tier1_threshold and n_available_components >= 3:
        return "Tier 1"
    if score >= tier2_threshold:
        return "Tier 2"
    return "Tier 3"


# ---------------------------------------------------------------------------
# Data loading from all steps
# ---------------------------------------------------------------------------


def _load_gene_universe(
    step5_dir: Path,
    step6_dir: Path,
    top_k: int,
    rng: np.random.Generator,
) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
    """Determine the top-K gene list and build a seed evidence dict."""
    # Priority: step5 top25 list > step5 ranked list > step6 final ranking
    genes: Optional[List[str]] = None

    top25_path = step5_dir / "top25_genes_cdps.csv"
    ranked_path = step5_dir / "ranked_genes_cdps.csv"
    step6_ranked = step6_dir / "final_validated_gene_ranking.csv"

    for candidate_path, gene_col in [
        (top25_path, "gene"),
        (ranked_path, "gene"),
        (step6_ranked, "gene"),
    ]:
        rows = _safe_read_csv(candidate_path)
        if rows:
            genes = [r[gene_col].strip() for r in rows if r.get(gene_col, "").strip()]
            genes = [g for g in genes if g]
            if genes:
                logger.info("Gene universe from %s: %d genes", candidate_path.name, len(genes))
                break

    if not genes:
        raise RuntimeError(
            "Could not find any gene list in step5_dir or step6_dir. "
            "Ensure ranked_genes_cdps.csv or similar file exists."
        )

    genes = genes[:top_k]
    logger.info("Using top %d genes: %s …", len(genes), ", ".join(genes[:5]))

    gene_data: Dict[str, Dict[str, Any]] = {
        g: {
            "gene": g,
            "original_cdps_rank": str(i + 1),
            "cdps_score": "",
            "attribution_score": "",
            "gate_score": "",
            "stability_score": "",
            "invariance_score": "",
            "perturbation_score": "",
            "log2FC": "",
            "DE_pvalue": "",
            "DE_FDR": "",
            "DE_direction": "",
            "pathway_support_score": "",
            "pathway_names": "",
            "network_degree": "",
            "network_centrality": "",
            "clinical_support_score": "",
            "strongest_clinical_variable": "",
            "clinical_p_value": "",
            "survival_p_value": "",
            "hazard_direction": "",
            "subgroup_robustness_score": "",
            "external_concordance_rate": "",
            "n_external_cohorts_tested": "",
            "n_external_cohorts_concordant": "",
        }
        for i, g in enumerate(genes)
    }
    return genes, gene_data


def _load_step5_evidence(
    genes: List[str],
    gene_data: Dict[str, Dict[str, Any]],
    step5_dir: Path,
) -> None:
    # ranked_genes_cdps.csv / top25_genes_cdps.csv use "cdps" not "cdps_score"
    for fname in ("top25_genes_cdps.csv", "ranked_genes_cdps.csv"):
        ranked = _safe_read_csv(step5_dir / fname)
        _merge_by_gene(gene_data, ranked, "gene", {
            "cdps_score": "cdps",
            "attribution_score": "attribution_score",
            "gate_score": "gate_score",
            "stability_score": "stability_score",
            "invariance_score": "invariance_score",
            "perturbation_score": "perturbation_score",
        })

    attr = _safe_read_csv(step5_dir / "gene_attribution_scores.csv")
    _merge_by_gene(gene_data, attr, "gene", {"attribution_score": "attribution_score"})

    stab = _safe_read_csv(step5_dir / "gene_stability_scores.csv")
    _merge_by_gene(gene_data, stab, "gene", {"stability_score": "stability_score"})

    inv = _safe_read_csv(step5_dir / "gene_invariance_scores.csv")
    _merge_by_gene(gene_data, inv, "gene", {"invariance_score": "invariance_score"})

    gate = _safe_read_csv(step5_dir / "gene_perturbation_scores.csv")
    _merge_by_gene(gene_data, gate, "gene", {
        "perturbation_score": "perturbation_score",
        "gate_score": "gate_score",
    })


def _download_string_network(
    genes: List[str],
    *,
    species: int = 9606,
    score_threshold: int = 400,
    save_path: Optional[Path] = None,
) -> List[Tuple[str, str, float]]:
    """Download PPI interactions from the STRING REST API (pure stdlib).

    Queries https://string-db.org/api for interactions among the supplied
    gene list plus their first-degree neighbors. Returns a list of
    (gene1, gene2, combined_score) tuples (score in 0–1 range).

    Parameters
    ----------
    genes : list of gene symbols (HGNC)
    species : NCBI taxonomy ID — 9606 for Homo sapiens
    score_threshold : combined score threshold (0–1000 scale used by STRING;
        400 = medium confidence, 700 = high, 900 = very high)
    save_path : if given, write the edge list as TSV (gene1\\tgene2\\tscore)
        so it can be passed to Step 6 via --ppi-edge-list
    """
    import urllib.request
    import urllib.parse

    identifiers = "\r".join(genes)
    params = urllib.parse.urlencode({
        "identifiers": identifiers,
        "species": str(species),
        "required_score": str(score_threshold),
        "caller_identity": "cage_pipeline",
    })
    url = "https://string-db.org/api/tsv/network?" + params

    logger.info(
        "Querying STRING API for %d genes (species=%d, score>=%d) ...",
        len(genes), species, score_threshold,
    )
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "CAGE/1.0"})
        with urllib.request.urlopen(req, timeout=90) as resp:
            content = resp.read().decode("utf-8")
    except Exception as exc:
        logger.warning("STRING API request failed: %s — network metrics will be empty.", exc)
        return []

    lines = content.strip().split("\n")
    if len(lines) < 2:
        logger.warning("STRING API returned no edges for the supplied genes.")
        return []

    header = lines[0].split("\t")
    try:
        ia = header.index("preferredName_A")
        ib = header.index("preferredName_B")
        isc = header.index("score")
    except ValueError:
        logger.warning("STRING API returned unexpected columns: %s", header)
        return []

    edges: List[Tuple[str, str, float]] = []
    for line in lines[1:]:
        parts = line.split("\t")
        if len(parts) <= max(ia, ib, isc):
            continue
        g1, g2 = parts[ia].strip(), parts[ib].strip()
        score = _to_float(parts[isc])
        if g1 and g2 and g1 != g2 and not math.isnan(score):
            edges.append((g1, g2, score))

    logger.info("STRING API: %d edges returned.", len(edges))

    if save_path is not None and edges:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(str(save_path), "w", encoding="utf-8") as fh:
            fh.write("gene1\tgene2\tscore\n")
            for g1, g2, s in edges:
                fh.write(f"{g1}\t{g2}\t{s:.6f}\n")
        logger.info("STRING edge list saved to %s", save_path)

    return edges


def _compute_network_metrics(
    genes: List[str],
    edges: List[Tuple[str, str, float]],
    gene_data: Dict[str, Dict[str, Any]],
) -> None:
    """Compute degree and normalised centrality from STRING edges and write into gene_data."""
    gene_set = set(genes)
    degree: Dict[str, int] = {g: 0 for g in genes}

    for g1, g2, _score in edges:
        if g1 in gene_set:
            degree[g1] += 1
        if g2 in gene_set:
            degree[g2] += 1

    max_deg = max(degree.values()) if degree and max(degree.values()) > 0 else 1
    populated = sum(1 for d in degree.values() if d > 0)
    logger.info(
        "Network metrics: %d / %d top genes have ≥1 STRING interaction.",
        populated, len(genes),
    )
    for g in genes:
        gene_data[g]["network_degree"] = str(degree[g])
        gene_data[g]["network_centrality"] = f"{degree[g] / max_deg:.6f}"


def _load_gmt_membership(
    genes: List[str],
    gene_data: Dict[str, Dict[str, Any]],
    gmt_paths: List[Optional[Path]],
) -> None:
    """Populate pathway_support_score and pathway_names from GMT files.

    Reads one or more GMT files (tab-delimited: pathway_name, description,
    gene1, gene2, ...) and for each top-25 gene records how many pathways
    it belongs to and lists up to 5 pathway names. This provides per-gene
    membership coverage even for genes that are not in significantly enriched
    pathways as a set.
    """
    gene_set: set = set(genes)
    gene_pw: Dict[str, List[str]] = {g: [] for g in genes}

    for gmt_path in gmt_paths:
        if gmt_path is None or not gmt_path.is_file():
            continue
        logger.info("Reading GMT membership from %s ...", gmt_path.name)
        n_pathways_read = 0
        with open(str(gmt_path), "r", encoding="utf-8") as fh:
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 3:
                    continue
                pw_name = parts[0].strip()
                # parts[1] is description (URL or text) — skip
                pw_genes = set(p.strip() for p in parts[2:] if p.strip())
                for g in gene_set.intersection(pw_genes):
                    gene_pw[g].append(pw_name)
                n_pathways_read += 1
        logger.info("  -> %d pathways scanned from %s", n_pathways_read, gmt_path.name)

    populated = 0
    for g in genes:
        if not gene_pw[g]:
            continue
        pws = gene_pw[g]
        # Overwrite existing (enrichment-derived) data only if we have more pathways
        _raw = _to_float(gene_data[g].get("pathway_support_score", "0"))
        existing = int(_raw) if not math.isnan(_raw) else 0
        if len(pws) >= existing:
            gene_data[g]["pathway_support_score"] = str(len(pws))
            gene_data[g]["pathway_names"] = "|".join(pws[:5])
        populated += 1

    logger.info(
        "GMT membership: %d / %d top genes have pathway support.", populated, len(genes)
    )


def _load_step6_evidence(
    genes: List[str],
    gene_data: Dict[str, Dict[str, Any]],
    step6_dir: Path,
) -> None:
    # differential_expression_results.csv uses effect_size_norm, p_value, fdr_bh
    de = _safe_read_csv(step6_dir / "differential_expression_results.csv")
    if de:
        for row in de:
            gene = row.get("gene", "").strip()
            if gene not in gene_data:
                continue
            eff = _to_float(row.get("effect_size_norm"))
            gene_data[gene]["log2FC"] = str(eff) if not math.isnan(eff) else ""
            gene_data[gene]["DE_pvalue"] = row.get("p_value", "")
            gene_data[gene]["DE_FDR"] = row.get("fdr_bh", "")
            gene_data[gene]["DE_direction"] = "up" if eff > 0 else "down" if eff < 0 else ""

    # top_cdps_de_support.csv — same schema, prefer over full DE table
    de_top = _safe_read_csv(step6_dir / "top_cdps_de_support.csv")
    if de_top:
        for row in de_top:
            gene = row.get("gene", "").strip()
            if gene not in gene_data:
                continue
            eff = _to_float(row.get("effect_size_norm"))
            if not math.isnan(eff):
                gene_data[gene]["log2FC"] = str(eff)
                gene_data[gene]["DE_direction"] = "up" if eff > 0 else "down"
            fdr = row.get("fdr_bh", "")
            if fdr:
                gene_data[gene]["DE_FDR"] = fdr

    # Build per-gene pathway data from enrichment_results_kegg_go.csv first
    # (has overlap_genes column = semicolon-separated gene list per pathway)
    gene_pw: Dict[str, List[str]] = {}
    gene_pw_score: Dict[str, float] = {}
    for enrich_fname in (
        "enrichment_results_kegg_go.csv",
        "enrichment_results_hallmark.csv",
        "enrichment_results_reactome.csv",
    ):
        enrich = _safe_read_csv(step6_dir / enrich_fname)
        if not enrich:
            continue
        for row in enrich:
            pw_name = row.get("pathway", row.get("term", ""))
            # overlap_genes is semicolon-separated in step6 output
            gnames_raw = row.get("overlap_genes", row.get("genes", row.get("gene_list", "")))
            sep = ";" if ";" in gnames_raw else ","
            for g in (x.strip() for x in gnames_raw.split(sep)):
                if not g or g not in gene_data:
                    continue
                gene_pw.setdefault(g, [])
                # strip source prefix (e.g. "kegg_go:KEGG_MEDICUS_REFERENCE_...")
                clean = pw_name.split(":", 1)[-1] if ":" in pw_name else pw_name
                gene_pw[g].append(clean)
                gene_pw_score[g] = gene_pw_score.get(g, 0.0) + 1.0

    if gene_pw:
        for g, pws in gene_pw.items():
            gene_data[g]["pathway_names"] = "|".join(pws[:5])
            gene_data[g]["pathway_support_score"] = str(int(gene_pw_score[g]))
        logger.info("Pathway support populated for %d genes from enrichment files.", len(gene_pw))

    # top_gene_pathway_membership.csv: gene, n_pathways, pathways
    # Use this to fill any genes missed above (and get pathway names)
    memb = _safe_read_csv(step6_dir / "top_gene_pathway_membership.csv")
    if memb:
        for row in memb:
            gene = row.get("gene", "").strip()
            if gene not in gene_data:
                continue
            # Only overwrite if not already populated from enrichment results
            if not gene_data[gene].get("pathway_support_score"):
                n_pw = row.get("n_pathways", "")
                gene_data[gene]["pathway_support_score"] = n_pw
            if not gene_data[gene].get("pathway_names"):
                raw_pws = row.get("pathways", "")
                # format: "source:PATHWAY_NAME|source:PATHWAY2|..."
                cleaned = [
                    p.split(":", 1)[-1] if ":" in p else p
                    for p in raw_pws.split("|") if p.strip()
                ]
                gene_data[gene]["pathway_names"] = "|".join(cleaned[:5])

    net = _safe_read_csv(step6_dir / "network_gene_support.csv")
    _merge_by_gene(gene_data, net, "gene", {
        "network_degree": "degree",
        "network_centrality": "centrality",
    })

    # clinical_association_results.csv — uses min_p_value, best_env
    clin = _safe_read_csv(step6_dir / "clinical_association_results.csv")
    if clin:
        for row in clin:
            gene = row.get("gene", "").strip()
            if gene not in gene_data:
                continue
            pval = _to_float(row.get("min_p_value", row.get("min_p_fdr_bh", "")))
            score = (1.0 - min(1.0, pval)) if not math.isnan(pval) else float("nan")
            gene_data[gene]["clinical_support_score"] = str(score) if not math.isnan(score) else ""
            gene_data[gene]["strongest_clinical_variable"] = row.get("best_env", "")
            gene_data[gene]["clinical_p_value"] = str(pval) if not math.isnan(pval) else ""

    # survival_gene_summary.csv — uses p_value, hazard_direction_high_vs_low
    surv = _safe_read_csv(step6_dir / "survival_gene_summary.csv")
    if surv:
        for row in surv:
            gene = row.get("gene", "").strip()
            if gene not in gene_data:
                continue
            gene_data[gene]["survival_p_value"] = row.get("p_value", "")
            hdir = _to_float(row.get("hazard_direction_high_vs_low", ""))
            gene_data[gene]["hazard_direction"] = "high" if hdir > 0 else "low" if hdir < 0 else ""

    # subgroup_sensitivity_summary.csv — uses agreement_fraction
    sub = _safe_read_csv(step6_dir / "subgroup_sensitivity_summary.csv")
    if sub:
        for row in sub:
            gene = row.get("gene", "").strip()
            if gene not in gene_data:
                continue
            gene_data[gene]["subgroup_robustness_score"] = row.get(
                "agreement_fraction", row.get("robustness_score", "")
            )


def _load_external_evidence(
    genes: List[str],
    gene_data: Dict[str, Dict[str, Any]],
    external_dir: Optional[Path],
    agilent_dir: Optional[Path],
) -> None:
    for ext_dir in (external_dir, agilent_dir):
        if ext_dir is None:
            continue
        for fname, gene_col, fields in [
            ("external_top_gene_replication.csv", "gene", {
                "external_concordance_rate": "concordance_rate",
                "n_external_cohorts_tested": "n_cohorts_tested",
                "n_external_cohorts_concordant": "n_cohorts_concordant",
            }),
            ("cross_cohort_concordance_summary.csv", "gene", {
                "external_concordance_rate": "concordance_rate",
            }),
            ("combined_de_replication.csv", "gene", {
                "external_concordance_rate": "replication_rate",
                "n_external_cohorts_tested": "n_cohorts",
            }),
            ("concordance_summary.csv", "gene", {
                "external_concordance_rate": "concordance_rate",
            }),
        ]:
            rows = _safe_read_csv(ext_dir / fname)
            _merge_by_gene(gene_data, rows, gene_col, fields)

        # Build per-dataset concordance matrix from the long-format combined file.
        # combined_de_replication.csv has one row per (gene, accession/dataset).
        for fname in ("combined_de_replication.csv", "top25_external_validation.csv"):
            rows = _safe_read_csv(ext_dir / fname)
            if not rows:
                continue
            dataset_col = "accession" if "accession" in rows[0] else "dataset"
            for row in rows:
                gene = row.get("gene", "").strip()
                if gene not in gene_data:
                    continue
                ds = row.get(dataset_col, "").strip()
                if not ds:
                    continue
                conc_raw = str(row.get("concordant", "")).strip()
                if conc_raw == "1":
                    status = "concordant"
                elif conc_raw == "0":
                    status = "discordant"
                else:
                    status = "absent"
                pdc = gene_data[gene].setdefault("per_dataset_concordance", {})
                if ds not in pdc:
                    pdc[ds] = status


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------


def _save_figure(
    fig: Any,
    fig_dir: Path,
    stem: str,
    style: PublicationStyle,
    manifest: List[Dict[str, Any]],
    caption: str = "",
) -> None:
    paths = save_figure(
        fig,
        fig_dir / stem,
        style=style,
        formats=style.default_formats,
        metadata={"caption": caption, "step": "6b"},
    )
    for p in paths:
        manifest.append({"figure": p.name, "path": str(p), "caption": caption})


def _fig_waterfall(
    genes: List[str],
    scores: List[float],
    tiers: List[str],
    style: PublicationStyle,
    fig_dir: Path,
    manifest: List[Dict[str, Any]],
) -> str:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return "matplotlib not available"

    palette = cage_palette("semantic")
    tier_colors = {
        "Tier 1": palette["tumor"],
        "Tier 2": palette["enriched"],
        "Tier 3": palette["nonsig"],
    }

    pairs = sorted(
        [(g, s, t) for g, s, t in zip(genes, scores, tiers) if not math.isnan(s)],
        key=lambda x: x[1],
        reverse=True,
    )
    if not pairs:
        return "no data"
    g_sorted, s_sorted, t_sorted = zip(*pairs)

    fig, ax = plt.subplots(figsize=style.figsize("double", aspect=0.5))
    colors = [tier_colors.get(t, "#B0B0B0") for t in t_sorted]
    ax.barh(range(len(g_sorted)), s_sorted, color=colors, edgecolor="none", height=0.7)
    ax.set_yticks(range(len(g_sorted)))
    ax.set_yticklabels(g_sorted, fontsize=style.tick_label_font_size)
    ax.set_xlabel("Integrated Priority Score", fontsize=style.axis_label_font_size)
    ax.set_title("Top-25 CDPS Genes — Integrated Priority Score", fontsize=style.title_font_size)
    from matplotlib.patches import Patch
    legend_elems = [Patch(facecolor=c, label=t) for t, c in tier_colors.items()]
    ax.legend(handles=legend_elems, loc="lower right", fontsize=style.legend_font_size)
    fig.tight_layout()

    _save_figure(fig, fig_dir, "top25_final_score_waterfall", style, manifest,
                 caption="Horizontal waterfall plot of integrated priority scores for the top-25 CDPS candidate genes, coloured by evidence tier.")
    return "ok"


def _fig_component_heatmap(
    genes: List[str],
    gene_data: Dict[str, Dict[str, Any]],
    weights: Dict[str, float],
    style: PublicationStyle,
    fig_dir: Path,
    manifest: List[Dict[str, Any]],
) -> str:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return "matplotlib not available"

    components = list(_COMPONENT_LABELS.keys())
    comp_raws: Dict[str, List[float]] = {c: [] for c in components}
    for gene in genes:
        d = gene_data[gene]
        comp_raws["cdps"].append(_to_float(d.get("cdps_score")))
        comp_raws["external"].append(_to_float(d.get("external_concordance_rate")))
        comp_raws["de"].append(abs(_to_float(d.get("log2FC"))))
        clin = _to_float(d.get("clinical_support_score"))
        surv = _to_float(d.get("survival_p_value"))
        clin_surv = clin if not math.isnan(clin) else (1.0 - min(1.0, surv) if not math.isnan(surv) else float("nan"))
        comp_raws["clinical_survival"].append(clin_surv)
        pw = _to_float(d.get("pathway_support_score"))
        net = _to_float(d.get("network_centrality"))
        comp_raws["pathway_network"].append((pw + net) / 2 if not (math.isnan(pw) or math.isnan(net)) else (pw if not math.isnan(pw) else net))
        comp_raws["subgroup"].append(_to_float(d.get("subgroup_robustness_score")))

    normed: Dict[str, List[float]] = {c: _minmax_norm(vals) for c, vals in comp_raws.items()}
    mat = np.full((len(genes), len(components)), np.nan)
    for j, comp in enumerate(components):
        for i, v in enumerate(normed[comp]):
            mat[i, j] = v

    fig, ax = plt.subplots(figsize=style.figsize("double", aspect=0.7))
    masked = np.ma.masked_invalid(mat)
    im = ax.imshow(masked.T, aspect="auto", cmap="cage_sequential", vmin=0, vmax=1)
    ax.set_xticks(range(len(genes)))
    ax.set_xticklabels(genes, rotation=60, ha="right", fontsize=style.tick_label_font_size)
    ax.set_yticks(range(len(components)))
    ax.set_yticklabels([_COMPONENT_LABELS[c] for c in components], fontsize=style.tick_label_font_size)
    ax.set_title("Evidence Component Heatmap (Normalized 0–1)", fontsize=style.title_font_size)
    plt.colorbar(im, ax=ax, shrink=0.6, label="Normalized score")
    fig.tight_layout()

    _save_figure(fig, fig_dir, "top25_evidence_component_heatmap", style, manifest,
                 caption="Heatmap of min-max normalized evidence component scores across the top-25 CDPS candidate genes. Grey cells indicate missing data.")
    return "ok"


def _fig_tiered_barplot(
    genes: List[str],
    scores: List[float],
    tiers: List[str],
    style: PublicationStyle,
    fig_dir: Path,
    manifest: List[Dict[str, Any]],
) -> str:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return "matplotlib not available"

    palette = cage_palette("semantic")
    tier_colors = {"Tier 1": palette["tumor"], "Tier 2": palette["enriched"], "Tier 3": palette["nonsig"]}
    colors = [tier_colors.get(t, "#B0B0B0") for t in tiers]

    fig, ax = plt.subplots(figsize=style.figsize("double", aspect=0.4))
    xs = range(len(genes))
    s_plot = [s if not math.isnan(s) else 0.0 for s in scores]
    ax.bar(xs, s_plot, color=colors, edgecolor="none", width=0.7)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(genes, rotation=60, ha="right", fontsize=style.tick_label_font_size)
    ax.set_ylabel("Integrated Priority Score", fontsize=style.axis_label_font_size)
    ax.set_title("Tiered Ranking of Top-25 CDPS Candidates", fontsize=style.title_font_size)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=c, label=t) for t, c in tier_colors.items()],
              fontsize=style.legend_font_size)
    fig.tight_layout()

    _save_figure(fig, fig_dir, "top25_tiered_ranking_barplot", style, manifest,
                 caption="Bar plot of integrated priority scores for the top-25 CDPS candidate genes, coloured by evidence tier (Tier 1 / 2 / 3).")
    return "ok"


def _fig_external_concordance(
    genes: List[str],
    gene_data: Dict[str, Dict[str, Any]],
    style: PublicationStyle,
    fig_dir: Path,
    manifest: List[Dict[str, Any]],
) -> str:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        from matplotlib.patches import Patch
    except ImportError:
        return "matplotlib not available"

    # Gather all datasets seen across genes
    all_datasets: List[str] = []
    for g in genes:
        for ds in gene_data[g].get("per_dataset_concordance", {}):
            if ds not in all_datasets:
                all_datasets.append(ds)

    # Fall back to aggregate bar chart if per-dataset data is absent
    if not all_datasets:
        rates = [_to_float(gene_data[g].get("external_concordance_rate")) for g in genes]
        if all(math.isnan(r) for r in rates):
            return "no external concordance data"
        fig, ax = plt.subplots(figsize=style.figsize("double", aspect=0.35))
        colors = [
            semantic_color("tumor") if (not math.isnan(r) and r >= 0.7)
            else semantic_color("normal") if (not math.isnan(r) and r >= 0.4)
            else semantic_color("nonsig")
            for r in rates
        ]
        ax.bar(range(len(genes)), [r if not math.isnan(r) else 0.0 for r in rates],
               color=colors, edgecolor="none", width=0.7)
        ax.set_xticks(range(len(genes)))
        ax.set_xticklabels(genes, rotation=60, ha="right", fontsize=style.tick_label_font_size)
        ax.set_ylabel("External Concordance Rate", fontsize=style.axis_label_font_size)
        ax.set_ylim(0, 1.05)
        ax.set_title("External Replication Concordance — Top-25 CDPS Genes",
                     fontsize=style.title_font_size)
        fig.tight_layout()
        _save_figure(fig, fig_dir, "top25_external_concordance_heatmap", style, manifest,
                     caption="External replication concordance rates for the top-25 CDPS candidate genes.")
        return "ok"

    # Proper gene × dataset concordance heatmap
    _COLOR_CONCORD = "#00A087"   # green  — palette sequential stop
    _COLOR_DISCORD = "#E64B35"   # red-orange — palette diverging high
    _COLOR_ABSENT  = "#D0D0D0"   # light grey

    cell_val = {"concordant": 1, "discordant": 0, "absent": 2}
    numeric = [
        [cell_val.get(gene_data[g].get("per_dataset_concordance", {}).get(ds, "absent"), 2)
         for ds in all_datasets]
        for g in genes
    ]

    cmap = ListedColormap([_COLOR_DISCORD, _COLOR_CONCORD, _COLOR_ABSENT])
    fig_h = max(5, len(genes) * 0.38 + 1.8)
    fig_w = max(5, len(all_datasets) * 1.6 + 2.2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(numeric, cmap=cmap, vmin=0, vmax=2, aspect="auto")

    mark = {"concordant": "+", "discordant": "−", "absent": "·"}
    for gi, g in enumerate(genes):
        for di, ds in enumerate(all_datasets):
            val = gene_data[g].get("per_dataset_concordance", {}).get(ds, "absent")
            col = "white" if val != "absent" else "#777777"
            ax.text(di, gi, mark.get(val, "·"), ha="center", va="center",
                    fontsize=style.tick_label_font_size, color=col)

    ax.set_xticks(range(len(all_datasets)))
    ax.set_xticklabels(all_datasets, rotation=30, ha="right",
                       fontsize=style.tick_label_font_size)
    ax.set_yticks(range(len(genes)))
    ax.set_yticklabels(genes, fontsize=style.tick_label_font_size)
    ax.set_title("External Replication Concordance — Top-25 CDPS Genes",
                 fontsize=style.title_font_size, fontweight="bold")

    legend_elements = [
        Patch(facecolor=_COLOR_CONCORD, label="Concordant (+)"),
        Patch(facecolor=_COLOR_DISCORD, label="Discordant (−)"),
        Patch(facecolor=_COLOR_ABSENT,  label="Not measured (·)"),
    ]
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.01, 1),
              loc="upper left", fontsize=style.legend_font_size, frameon=True)
    fig.tight_layout()

    _save_figure(fig, fig_dir, "top25_external_concordance_heatmap", style, manifest,
                 caption="Gene × dataset concordance heatmap for the top-25 CDPS candidate genes "
                         "across GEO / Agilent validation cohorts. Green = concordant, "
                         "Red = discordant, Grey = not measured.")
    return "ok"


def _fig_de_vs_cdps(
    genes: List[str],
    gene_data: Dict[str, Dict[str, Any]],
    style: PublicationStyle,
    fig_dir: Path,
    manifest: List[Dict[str, Any]],
) -> str:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return "matplotlib not available"

    xs = [_to_float(gene_data[g].get("cdps_score")) for g in genes]
    ys = [_to_float(gene_data[g].get("log2FC")) for g in genes]

    pairs = [(x, y, g) for x, y, g in zip(xs, ys, genes)
             if not (math.isnan(x) or math.isnan(y))]
    if len(pairs) < 2:
        return "insufficient data for DE vs CDPS scatter"

    xv, yv, gv = zip(*pairs)
    dirs = [str(gene_data[g].get("DE_direction", "")).lower() for g in gv]
    colors = [semantic_color("up") if d == "up" else semantic_color("down") if d == "down" else semantic_color("nonsig") for d in dirs]

    fig, ax = plt.subplots(figsize=style.figsize("single", aspect=1.0))
    ax.scatter(xv, yv, c=colors, s=40, edgecolors="none", alpha=0.85)
    for x_, y_, g_ in zip(xv, yv, gv):
        ax.annotate(g_, (x_, y_), fontsize=max(5, style.annotation_font_size),
                    ha="left", va="bottom", xytext=(2, 2), textcoords="offset points")
    ax.axhline(0, lw=0.8, ls="--", color="#888888")
    ax.set_xlabel("CDPS Score (Step 5)", fontsize=style.axis_label_font_size)
    ax.set_ylabel("Log₂ Fold Change (tumor vs normal)", fontsize=style.axis_label_font_size)
    ax.set_title("CDPS Score vs Differential Expression", fontsize=style.title_font_size)
    fig.tight_layout()

    _save_figure(fig, fig_dir, "top25_de_vs_cdps_scatter", style, manifest,
                 caption="Scatter plot of CDPS model score versus log2 fold change (tumor vs normal) for the top-25 CDPS candidate genes.")
    return "ok"


def _fig_clinical_survival(
    genes: List[str],
    gene_data: Dict[str, Dict[str, Any]],
    style: PublicationStyle,
    fig_dir: Path,
    manifest: List[Dict[str, Any]],
) -> str:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return "matplotlib not available"

    clin_scores = [_to_float(gene_data[g].get("clinical_support_score")) for g in genes]
    surv_pvals = [_to_float(gene_data[g].get("survival_p_value")) for g in genes]

    has_clin = any(not math.isnan(c) for c in clin_scores)
    has_surv = any(not math.isnan(p) for p in surv_pvals)
    if not has_clin and not has_surv:
        return "no clinical/survival data"

    fig, axes = plt.subplots(1, 2, figsize=style.figsize("double", aspect=0.45))

    if has_clin:
        cs = [c if not math.isnan(c) else 0.0 for c in clin_scores]
        axes[0].barh(range(len(genes)), cs, color=semantic_color("validated"), edgecolor="none", height=0.7)
        axes[0].set_yticks(range(len(genes)))
        axes[0].set_yticklabels(genes, fontsize=style.tick_label_font_size)
        axes[0].set_xlabel("Clinical Support Score", fontsize=style.axis_label_font_size)
        axes[0].set_title("Clinical Association", fontsize=style.title_font_size)
    else:
        axes[0].text(0.5, 0.5, "No data", ha="center", va="center",
                     transform=axes[0].transAxes, fontsize=style.annotation_font_size)
        axes[0].axis("off")

    if has_surv:
        neg_log_p = [(-math.log10(p) if not math.isnan(p) and p > 0 else 0.0) for p in surv_pvals]
        dirs = [str(gene_data[g].get("hazard_direction", "")).lower() for g in genes]
        colors = [semantic_color("tumor") if d == "high" else semantic_color("normal") if d == "low" else semantic_color("nonsig") for d in dirs]
        axes[1].barh(range(len(genes)), neg_log_p, color=colors, edgecolor="none", height=0.7)
        axes[1].set_yticks(range(len(genes)))
        axes[1].set_yticklabels(genes, fontsize=style.tick_label_font_size)
        axes[1].axvline(-math.log10(0.05), lw=0.8, ls="--", color="#888888")
        axes[1].set_xlabel("−log₁₀(log-rank p-value)", fontsize=style.axis_label_font_size)
        axes[1].set_title("Survival Association", fontsize=style.title_font_size)
    else:
        axes[1].text(0.5, 0.5, "No data", ha="center", va="center",
                     transform=axes[1].transAxes, fontsize=style.annotation_font_size)
        axes[1].axis("off")

    fig.tight_layout()
    _save_figure(fig, fig_dir, "top25_clinical_survival_summary", style, manifest,
                 caption="Clinical association scores (left) and survival log-rank significance (right) for the top-25 CDPS candidate genes.")
    return "ok"


def _fig_network_pathway(
    genes: List[str],
    gene_data: Dict[str, Dict[str, Any]],
    style: PublicationStyle,
    fig_dir: Path,
    manifest: List[Dict[str, Any]],
) -> str:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return "matplotlib not available"

    pw_scores = [_to_float(gene_data[g].get("pathway_support_score")) for g in genes]
    net_deg = [_to_float(gene_data[g].get("network_degree")) for g in genes]

    has_pw = any(not math.isnan(p) for p in pw_scores)
    has_net = any(not math.isnan(n) for n in net_deg)
    if not has_pw and not has_net:
        return "no pathway/network data"

    fig, axes = plt.subplots(1, 2, figsize=style.figsize("double", aspect=0.45))

    if has_pw:
        pw = [p if not math.isnan(p) else 0.0 for p in pw_scores]
        axes[0].barh(range(len(genes)), pw, color=semantic_color("enriched"), edgecolor="none", height=0.7)
        axes[0].set_yticks(range(len(genes)))
        axes[0].set_yticklabels(genes, fontsize=style.tick_label_font_size)
        axes[0].set_xlabel("Pathway Support Score", fontsize=style.axis_label_font_size)
        axes[0].set_title("Pathway Enrichment", fontsize=style.title_font_size)
    else:
        axes[0].text(0.5, 0.5, "No data", ha="center", va="center",
                     transform=axes[0].transAxes, fontsize=style.annotation_font_size)
        axes[0].axis("off")

    if has_net:
        nd = [n if not math.isnan(n) else 0.0 for n in net_deg]
        axes[1].barh(range(len(genes)), nd, color=semantic_color("subgroup_a"), edgecolor="none", height=0.7)
        axes[1].set_yticks(range(len(genes)))
        axes[1].set_yticklabels(genes, fontsize=style.tick_label_font_size)
        axes[1].set_xlabel("Network Degree", fontsize=style.axis_label_font_size)
        axes[1].set_title("PPI Network Support", fontsize=style.title_font_size)
    else:
        axes[1].text(0.5, 0.5, "No data", ha="center", va="center",
                     transform=axes[1].transAxes, fontsize=style.annotation_font_size)
        axes[1].axis("off")

    fig.tight_layout()
    _save_figure(fig, fig_dir, "top25_network_pathway_support", style, manifest,
                 caption="Pathway enrichment support scores (left) and PPI network degree (right) for the top-25 CDPS candidate genes.")
    return "ok"


def _fig_missing_evidence(
    genes: List[str],
    flags: List[str],
    style: PublicationStyle,
    fig_dir: Path,
    manifest: List[Dict[str, Any]],
) -> str:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
    except ImportError:
        return "matplotlib not available"

    components = list(_COMPONENT_LABELS.keys())
    mat = np.zeros((len(genes), len(components)), dtype=float)
    for i, flag_str in enumerate(flags):
        missing = set(flag_str.split("|")) if flag_str != "none" else set()
        for j, comp in enumerate(components):
            mat[i, j] = 1.0 if comp in missing else 0.0

    # Binary colormap: present=#00A087 (green), missing=#E64B35 (red-orange) — house palette
    _cmap_missing = ListedColormap(["#00A087", "#E64B35"])
    fig, ax = plt.subplots(figsize=style.figsize("double", aspect=0.55))
    ax.imshow(mat.T, aspect="auto", cmap=_cmap_missing, vmin=0, vmax=1)
    ax.set_xticks(range(len(genes)))
    ax.set_xticklabels(genes, rotation=60, ha="right", fontsize=style.tick_label_font_size)
    ax.set_yticks(range(len(components)))
    ax.set_yticklabels([_COMPONENT_LABELS[c] for c in components], fontsize=style.tick_label_font_size)
    ax.set_title("Missing Evidence Map (Red = Missing)", fontsize=style.title_font_size)
    fig.tight_layout()

    _save_figure(fig, fig_dir, "top25_missing_evidence_map", style, manifest,
                 caption="Evidence availability map for the top-25 CDPS candidate genes. Red cells indicate missing data for the corresponding evidence component.")
    return "ok"


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def _write_reports(
    genes: List[str],
    gene_data: Dict[str, Dict[str, Any]],
    scores: List[float],
    tiers: List[str],
    weights: Dict[str, float],
    tier1_thr: float,
    tier2_thr: float,
    report_dir: Path,
    run_ts: str,
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)

    tier1 = [g for g, t in zip(genes, tiers) if t == "Tier 1"]
    tier2 = [g for g, t in zip(genes, tiers) if t == "Tier 2"]
    tier3 = [g for g, t in zip(genes, tiers) if t == "Tier 3"]
    valid_scores = [s for s in scores if not math.isnan(s)]
    mean_score = sum(valid_scores) / len(valid_scores) if valid_scores else float("nan")

    summary = f"""# CAGE Step 6b — Top-25 CDPS Final Prioritization Summary

Generated: {run_ts}

## Overview

This report summarizes the integrated evidence-based prioritization of the top-25
computationally prioritized CDPS (Candidate Diagnostic/Prognostic Signature) genes.
All genes are **computational candidates** and require experimental confirmation before
any mechanistic or clinical conclusions can be drawn.

## Tier Distribution

| Tier | Count | Genes |
|------|-------|-------|
| Tier 1 (score ≥ {tier1_thr:.2f}, multi-source) | {len(tier1)} | {", ".join(tier1) or "None"} |
| Tier 2 (score ≥ {tier2_thr:.2f}) | {len(tier2)} | {", ".join(tier2) or "None"} |
| Tier 3 (score < {tier2_thr:.2f}) | {len(tier3)} | {", ".join(tier3) or "None"} |

## Score Summary

- Mean priority score: {mean_score:.3f}
- Total genes evaluated: {len(genes)}
- Genes with complete evidence: {sum(1 for g, f in zip(genes, [gene_data[g].get("_flags", "none") for g in genes]) if f == "none")}

## Evidence Weights Applied

| Component | Weight |
|-----------|--------|
"""
    for comp, w in weights.items():
        summary += f"| {_COMPONENT_LABELS.get(comp, comp)} | {w:.2f} |\n"

    summary += """
## Interpretation

Tier 1 genes show convergent support across at least three independent evidence
streams and represent the highest-priority candidates for further investigation.
Tier 2 genes are computationally supported but rely on fewer evidence types.
Tier 3 genes have limited multi-stream support and should be interpreted cautiously.

**All findings are computational predictions. None of these genes have been
experimentally validated as causal drivers of esophageal squamous cell carcinoma.**
"""
    (report_dir / "top25_prioritization_summary.md").write_text(summary, encoding="utf-8")

    methods = f"""# Methods: Top-25 CDPS Gene Prioritization (Step 6b)

## Evidence Integration

The top-{len(genes)} computationally prioritized CDPS genes were subjected to
integrated evidence scoring using six evidence components:

1. **CDPS/Model score** (weight {weights.get('cdps', 0):.2f}): Composite model score
   from Step 5, incorporating attribution, gate, stability, invariance, and
   perturbation sub-scores.
2. **External replication** (weight {weights.get('external', 0):.2f}): Concordance rate
   across independent GEO Affymetrix and Agilent microarray validation cohorts.
3. **Differential expression** (weight {weights.get('de', 0):.2f}): Absolute log₂
   fold change (tumor vs. adjacent normal) from Welch t-test with
   Benjamini–Hochberg FDR correction.
4. **Clinical/survival association** (weight {weights.get('clinical_survival', 0):.2f}):
   Clinico-pathologic association score and log-rank survival p-value.
5. **Pathway/network support** (weight {weights.get('pathway_network', 0):.2f}):
   Pathway enrichment membership count and PPI network degree/centrality.
6. **Subgroup robustness** (weight {weights.get('subgroup', 0):.2f}): Direction
   consistency of tumor-vs-normal effect across environment strata.

Each component was min-max normalized to [0, 1] across the gene set. For genes
with missing evidence in one or more components, the available component weights
were re-normalized to sum to 1.0 so the final score remains on a [0, 1] scale.

## Tier Classification

- **Tier 1**: integrated priority score ≥ {tier1_thr:.2f} AND evidence available
  from ≥ 3 independent components.
- **Tier 2**: integrated priority score ≥ {tier2_thr:.2f}.
- **Tier 3**: integrated priority score < {tier2_thr:.2f} or score unavailable.

## Statistical Language

All genes are described as *computationally prioritized candidates*. No gene is
described as experimentally confirmed or causally implicated in ESCA without
independent wet-lab validation.
"""
    (report_dir / "top25_methods_text.md").write_text(methods, encoding="utf-8")

    ranked_pairs = sorted(zip(genes, scores, tiers), key=lambda x: (math.isnan(x[1]), -x[1] if not math.isnan(x[1]) else 0))
    results = f"""# Results: Top-25 CDPS Gene Prioritization

## Summary

Integrated evidence-based prioritization identified {len(tier1)} Tier-1,
{len(tier2)} Tier-2, and {len(tier3)} Tier-3 candidate genes among the top-25
computationally prioritized CDPS genes.

## Ranked Gene List

| Rank | Gene | Priority Score | Tier |
|------|------|---------------|------|
"""
    for rank, (g, s, t) in enumerate(ranked_pairs, 1):
        s_str = f"{s:.3f}" if not math.isnan(s) else "N/A"
        results += f"| {rank} | {g} | {s_str} | {t} |\n"

    results += """
## Notes

- Scores and tiers reflect computational evidence only.
- Missing evidence components were excluded per-gene with proportional weight redistribution.
- All candidate genes require experimental confirmation.
"""
    (report_dir / "top25_results_text.md").write_text(results, encoding="utf-8")
    logger.info("Wrote reports to %s", report_dir)


# ---------------------------------------------------------------------------
# Manifest and reproducibility
# ---------------------------------------------------------------------------


def _write_manifests(
    manifest_dir: Path,
    input_entries: List[Dict[str, Any]],
    output_entries: List[Dict[str, Any]],
    figure_manifest: List[Dict[str, Any]],
    weights: Dict[str, float],
    tier1_thr: float,
    tier2_thr: float,
    top_k: int,
    seed: int,
    run_ts: str,
    script_hash: str,
) -> None:
    manifest_dir.mkdir(parents=True, exist_ok=True)

    _write_manifest(manifest_dir / "input_file_manifest.csv", input_entries,
                    ["label", "path", "exists", "n_rows"])
    _write_manifest(manifest_dir / "output_manifest.csv", output_entries,
                    ["output", "path", "type"])
    _write_manifest(manifest_dir / "figure_manifest.csv", figure_manifest,
                    ["figure", "path", "caption"])

    scoring_cfg = {
        "weights": weights,
        "tier1_threshold": tier1_thr,
        "tier2_threshold": tier2_thr,
        "top_k": top_k,
        "normalization": "minmax",
        "missing_handling": "per_gene_weight_renormalization",
    }
    (manifest_dir / "scoring_config.json").write_text(
        json.dumps(scoring_cfg, indent=2), encoding="utf-8"
    )

    repro = {
        "timestamp": run_ts,
        "seed": seed,
        "script_sha256": script_hash,
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "platform": sys.platform,
    }
    (manifest_dir / "reproducibility.json").write_text(
        json.dumps(repro, indent=2), encoding="utf-8"
    )
    logger.info("Wrote manifests to %s", manifest_dir)


# ---------------------------------------------------------------------------
# CLI parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = build_step_parser(
        prog="python -m cage.step6b_top25_final_prioritization",
        step_title="Step 6b — Top-25 CDPS Gene Final Evidence Prioritization",
        step_description=(
            "Integrates CDPS model scores, differential expression, external "
            "replication, clinical/survival associations, pathway/network support, "
            "and subgroup robustness into a single ranked priority score for the "
            "top-25 computationally prioritized candidate genes."
        ),
        inputs_doc=(
            "Step 5 CDPS outputs (--step5-dir), Step 6 validation outputs "
            "(--step6-dir), Step 8 external GEO outputs (--external-dir), "
            "Step 8 Agilent outputs (--agilent-dir). All inputs are optional; "
            "missing files degrade gracefully."
        ),
        outputs_doc=(
            "tables/  — integrated evidence, ranked priority, tier summary, "
            "missing evidence report, manuscript table.\n"
            "figures/ — 8 publication-grade SVG figures.\n"
            "reports/ — prioritization summary, methods text, results text (MD).\n"
            "manifests/ — input/output/figure manifests, scoring config, "
            "reproducibility JSON.\n"
            "logs/   — step6b log."
        ),
        example=(
            "python -m cage.step6b_top25_final_prioritization \\\n"
            "  --step5-dir outputs/step5_cdps \\\n"
            "  --step6-dir outputs/step6_validation \\\n"
            "  --external-dir outputs/step8_geo_validation \\\n"
            "  --agilent-dir outputs/step8_agilent_validation \\\n"
            "  --output-dir outputs/step6b_top25_prioritization \\\n"
            "  --top-k 25 --seed 2026 \\\n"
            "  --figure-format svg --extra-figure-format pdf \\\n"
            "  --font-family 'DejaVu Sans' --font-size 10 --palette cage \\\n"
            "  --w-cdps 0.30 --w-external 0.20 --w-de 0.15 \\\n"
            "  --w-clinical-survival 0.15 --w-pathway-network 0.10 --w-subgroup 0.10 \\\n"
            "  --tier1-threshold 0.75 --tier2-threshold 0.50"
        ),
        require_input_dir=False,
    )

    inp = parser.add_argument_group("Input directories")
    inp.add_argument("--step5-dir", type=Path, default=Path("outputs/step5_cdps"),
                     metavar="DIR", help="Step 5 CDPS output directory.")
    inp.add_argument("--step6-dir", type=Path, default=Path("outputs/step6_validation"),
                     metavar="DIR", help="Step 6 biological validation output directory.")
    inp.add_argument("--external-dir", type=Path, default=None,
                     metavar="DIR", help="Step 8 GEO external validation output directory (optional).")
    inp.add_argument("--agilent-dir", type=Path, default=None,
                     metavar="DIR", help="Step 8 Agilent validation output directory (optional).")

    gmt_grp = parser.add_argument_group("GMT pathway files (optional, for per-gene membership lookup)")
    gmt_grp.add_argument("--kegg-gmt", type=Path, default=None, metavar="FILE",
                         help="KEGG Medicus GMT file for per-gene pathway membership "
                              "(e.g. data/c2.cp.kegg_medicus.v2026.1.Hs.symbols.gmt).")
    gmt_grp.add_argument("--hallmark-gmt", type=Path, default=None, metavar="FILE",
                         help="MSigDB Hallmark GMT file for per-gene pathway membership "
                              "(e.g. data/h.all.v2026.1.Hs.symbols.gmt).")
    gmt_grp.add_argument("--reactome-gmt", type=Path, default=None, metavar="FILE",
                         help="Reactome GMT file for per-gene pathway membership.")

    net_grp = parser.add_argument_group("STRING PPI network (auto-downloaded, no file needed)")
    net_grp.add_argument("--download-string-ppi", action="store_true",
                         help="Download PPI interactions for the top-K genes from STRING REST API "
                              "(https://string-db.org). Requires internet access. "
                              "Saves edge list to <output-dir>/manifests/string_ppi_edges.tsv "
                              "which can also be reused with Step 6 --ppi-edge-list.")
    net_grp.add_argument("--string-score-threshold", type=int, default=400, metavar="INT",
                         help="STRING combined score threshold 0–1000 "
                              "(400=medium, 700=high, 900=very high; default: 400).")
    net_grp.add_argument("--string-species", type=int, default=9606, metavar="INT",
                         help="NCBI taxonomy ID for STRING query (default: 9606 = Homo sapiens).")

    scoring = parser.add_argument_group("Gene selection and scoring")
    scoring.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, metavar="INT",
                         help=f"Number of top CDPS genes to prioritize (default: {DEFAULT_TOP_K}).")
    scoring.add_argument("--w-cdps", type=float, default=0.30, metavar="W",
                         help="Weight for CDPS/model score component (default: 0.30).")
    scoring.add_argument("--w-external", type=float, default=0.20, metavar="W",
                         help="Weight for external replication component (default: 0.20).")
    scoring.add_argument("--w-de", type=float, default=0.15, metavar="W",
                         help="Weight for differential expression component (default: 0.15).")
    scoring.add_argument("--w-clinical-survival", type=float, default=0.15, metavar="W",
                         help="Weight for clinical/survival component (default: 0.15).")
    scoring.add_argument("--w-pathway-network", type=float, default=0.10, metavar="W",
                         help="Weight for pathway/network component (default: 0.10).")
    scoring.add_argument("--w-subgroup", type=float, default=0.10, metavar="W",
                         help="Weight for subgroup robustness component (default: 0.10).")
    scoring.add_argument("--tier1-threshold", type=float, default=0.75, metavar="FLOAT",
                         help="Score threshold for Tier 1 classification (default: 0.75).")
    scoring.add_argument("--tier2-threshold", type=float, default=0.50, metavar="FLOAT",
                         help="Score threshold for Tier 2 classification (default: 0.50).")

    # output-dir is already added by build_step_parser / add_global_args
    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    run_ts = datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"
    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(args, log_file=log_dir / "step6b_top25_final_prioritization.log")

    logger.info("=" * 70)
    logger.info("CAGE Step 6b — Top-25 Final Prioritization")
    logger.info("Run timestamp: %s", run_ts)
    logger.info("Output directory: %s", out_dir)
    logger.info("Step 5 directory: %s", args.step5_dir)
    logger.info("Step 6 directory: %s", args.step6_dir)
    logger.info("External dir: %s", args.external_dir)
    logger.info("Agilent dir: %s", args.agilent_dir)
    logger.info("Top-K: %d  Seed: %d", args.top_k, args.seed)
    logger.info("=" * 70)

    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)

    weights: Dict[str, float] = {
        "cdps": args.w_cdps,
        "external": args.w_external,
        "de": args.w_de,
        "clinical_survival": args.w_clinical_survival,
        "pathway_network": args.w_pathway_network,
        "subgroup": args.w_subgroup,
    }
    w_sum = sum(weights.values())
    if not math.isclose(w_sum, 1.0, abs_tol=1e-6):
        logger.warning("Weights sum to %.4f (not 1.0); will be re-normalized per-gene as usual.", w_sum)

    logger.info("Evidence weights: %s", weights)

    # ---- load gene universe -------------------------------------------------
    genes, gene_data = _load_gene_universe(args.step5_dir, args.step6_dir, args.top_k, rng)

    # ---- load evidence -------------------------------------------------------
    _load_step5_evidence(genes, gene_data, args.step5_dir)
    _load_step6_evidence(genes, gene_data, args.step6_dir)
    _load_external_evidence(genes, gene_data, args.external_dir, args.agilent_dir)

    # GMT membership lookup — runs after step6 enrichment so it can augment
    # genes that have no enrichment hit but are individually in pathways.
    gmt_paths: List[Optional[Path]] = [
        getattr(args, "kegg_gmt", None),
        getattr(args, "hallmark_gmt", None),
        getattr(args, "reactome_gmt", None),
    ]
    if any(p is not None for p in gmt_paths):
        _load_gmt_membership(genes, gene_data, gmt_paths)
    else:
        logger.info(
            "No GMT files supplied (--kegg-gmt / --hallmark-gmt / --reactome-gmt). "
            "Pathway membership uses Step 6 enrichment outputs only."
        )

    # STRING PPI network download
    if getattr(args, "download_string_ppi", False):
        string_save = out_dir / "manifests" / "string_ppi_edges.tsv"
        string_edges = _download_string_network(
            genes,
            species=getattr(args, "string_species", 9606),
            score_threshold=getattr(args, "string_score_threshold", 400),
            save_path=string_save,
        )
        if string_edges:
            _compute_network_metrics(genes, string_edges, gene_data)
            logger.info(
                "STRING edge list saved to %s — reuse with Step 6: "
                "--run-network --ppi-edge-list %s",
                string_save, string_save,
            )
    else:
        logger.info(
            "No STRING download requested. Pass --download-string-ppi to fetch "
            "PPI interactions online. Network column will be empty."
        )

    # ---- scoring -------------------------------------------------------------
    scores, flags, avail_comps = _score_evidence(genes, gene_data, weights)

    # count available components per gene
    n_avail = [
        len([c for c in comp.split(",") if c and c != "none"])
        for comp in avail_comps
    ]

    tiers = [
        _assign_tier(s, args.tier1_threshold, args.tier2_threshold, na)
        for s, na in zip(scores, n_avail)
    ]

    # attach per-gene computed fields
    for i, gene in enumerate(genes):
        gene_data[gene]["final_priority_score"] = (
            f"{scores[i]:.6f}" if not math.isnan(scores[i]) else ""
        )
        gene_data[gene]["evidence_tier"] = tiers[i]
        gene_data[gene]["missing_evidence_flags"] = flags[i]
        gene_data[gene]["_flags"] = flags[i]

    # ---- tables --------------------------------------------------------------
    tbl_dir = out_dir / "tables"
    tbl_dir.mkdir(parents=True, exist_ok=True)

    evidence_cols = [
        "gene", "original_cdps_rank", "cdps_score", "attribution_score",
        "gate_score", "stability_score", "invariance_score", "perturbation_score",
        "log2FC", "DE_pvalue", "DE_FDR", "DE_direction",
        "pathway_support_score", "pathway_names", "network_degree", "network_centrality",
        "clinical_support_score", "strongest_clinical_variable", "clinical_p_value",
        "survival_p_value", "hazard_direction", "subgroup_robustness_score",
        "external_concordance_rate", "n_external_cohorts_tested", "n_external_cohorts_concordant",
        "final_priority_score", "evidence_tier", "missing_evidence_flags",
    ]
    all_rows = [gene_data[g] for g in genes]
    _write_csv(tbl_dir / "top25_integrated_evidence.csv", all_rows, evidence_cols)

    ranked_pairs = sorted(
        zip(genes, scores, tiers),
        key=lambda x: (math.isnan(x[1]), -x[1] if not math.isnan(x[1]) else 0),
    )
    ranked_rows = []
    for rank, (g, s, t) in enumerate(ranked_pairs, 1):
        row = dict(gene_data[g])
        row["priority_rank"] = str(rank)
        ranked_rows.append(row)
    ranked_cols = ["priority_rank"] + evidence_cols
    _write_csv(tbl_dir / "top25_final_priority_ranking.csv", ranked_rows, ranked_cols)

    tier_summary_rows = []
    for tier_label in ("Tier 1", "Tier 2", "Tier 3"):
        tier_genes = [g for g, t in zip(genes, tiers) if t == tier_label]
        tier_scores = [s for g, s, t in zip(genes, scores, tiers) if t == tier_label]
        mean_s = (sum(tier_scores) / len(tier_scores)) if tier_scores else float("nan")
        tier_summary_rows.append({
            "tier": tier_label,
            "n_genes": len(tier_genes),
            "genes": "|".join(tier_genes),
            "mean_score": f"{mean_s:.4f}" if not math.isnan(mean_s) else "",
            "score_threshold": f"{args.tier1_threshold}" if tier_label == "Tier 1" else f"{args.tier2_threshold}" if tier_label == "Tier 2" else f"<{args.tier2_threshold}",
        })
    _write_csv(tbl_dir / "top25_tier_summary.csv", tier_summary_rows,
               ["tier", "n_genes", "genes", "mean_score", "score_threshold"])

    missing_rows = [
        {"gene": g, "missing_components": flags[i], "n_missing": len([c for c in flags[i].split("|") if c and c != "none"])}
        for i, g in enumerate(genes)
    ]
    _write_csv(tbl_dir / "top25_missing_evidence_report.csv", missing_rows,
               ["gene", "missing_components", "n_missing"])

    ms_cols = ["gene", "evidence_tier", "final_priority_score", "log2FC", "DE_FDR",
               "external_concordance_rate", "survival_p_value", "pathway_names",
               "missing_evidence_flags"]
    ms_rows = [gene_data[g] for g in [r[0] for r in ranked_pairs]]
    for row in ms_rows:
        row["final_priority_score"] = row.get("final_priority_score", "")
    _write_csv(tbl_dir / "top25_manuscript_table.csv", ms_rows, ms_cols)

    # ---- figures -------------------------------------------------------------
    style = style_from_args(args)
    try:
        apply_style(style)
    except ImportError as _mpl_err:
        logger.warning("matplotlib not available — figures will be skipped: %s", _mpl_err)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    figure_manifest: List[Dict[str, Any]] = []
    fig_status: List[str] = []
    fig_skipped: List[Tuple[str, str]] = []

    for name, fn, kwargs in [
        ("waterfall", _fig_waterfall, dict(genes=genes, scores=scores, tiers=tiers)),
        ("component_heatmap", _fig_component_heatmap, dict(genes=genes, gene_data=gene_data, weights=weights)),
        ("tiered_barplot", _fig_tiered_barplot, dict(genes=genes, scores=scores, tiers=tiers)),
        ("external_concordance", _fig_external_concordance, dict(genes=genes, gene_data=gene_data)),
        ("de_vs_cdps", _fig_de_vs_cdps, dict(genes=genes, gene_data=gene_data)),
        ("clinical_survival", _fig_clinical_survival, dict(genes=genes, gene_data=gene_data)),
        ("network_pathway", _fig_network_pathway, dict(genes=genes, gene_data=gene_data)),
        ("missing_evidence", _fig_missing_evidence, dict(genes=genes, flags=flags)),
    ]:
        try:
            result = fn(style=style, fig_dir=fig_dir, manifest=figure_manifest, **kwargs)
            if result == "ok":
                fig_status.append(name)
            else:
                fig_skipped.append((name, result))
        except Exception as exc:
            logger.exception("Figure %s failed: %s", name, exc)
            fig_skipped.append((name, str(exc)))

    log_figure_status(fig_status, fig_skipped, logger_name="cage.step6b")

    # ---- reports -------------------------------------------------------------
    _write_reports(genes, gene_data, scores, tiers, weights,
                   args.tier1_threshold, args.tier2_threshold,
                   out_dir / "reports", run_ts)

    # ---- manifests -----------------------------------------------------------
    script_path = Path(__file__)
    script_hash = hashlib.sha256(script_path.read_bytes()).hexdigest()

    input_entries: List[Dict[str, Any]] = []
    for label, path in [
        ("step5_dir", args.step5_dir),
        ("step6_dir", args.step6_dir),
        ("external_dir", args.external_dir),
        ("agilent_dir", args.agilent_dir),
    ]:
        p = Path(path) if path else None
        input_entries.append({
            "label": label,
            "path": str(p) if p else "",
            "exists": str(p.exists()) if p else "False",
            "n_rows": "",
        })

    output_entries: List[Dict[str, Any]] = []
    for p in sorted(out_dir.rglob("*")):
        if p.is_file() and "manifest" not in p.parts[-2:]:
            output_entries.append({
                "output": p.name,
                "path": str(p),
                "type": p.suffix.lstrip("."),
            })

    _write_manifests(
        out_dir / "manifests",
        input_entries, output_entries, figure_manifest,
        weights, args.tier1_threshold, args.tier2_threshold,
        args.top_k, args.seed, run_ts, script_hash,
    )

    # ---- final log -----------------------------------------------------------
    tier1_genes = [g for g, t in zip(genes, tiers) if t == "Tier 1"]
    tier2_genes = [g for g, t in zip(genes, tiers) if t == "Tier 2"]
    logger.info("=" * 70)
    logger.info("Step 6b complete.")
    logger.info("Tier 1 (%d genes): %s", len(tier1_genes), ", ".join(tier1_genes) or "None")
    logger.info("Tier 2 (%d genes): %s", len(tier2_genes), ", ".join(tier2_genes) or "None")
    logger.info("Figures: %d generated, %d skipped.", len(fig_status), len(fig_skipped))
    logger.info("Outputs: %s", out_dir)
    logger.info("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
