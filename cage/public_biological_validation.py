"""Public biological validation for CAGE-prioritized ESCA genes.

This downstream module validates final candidate genes against public
resources without changing the core CAGE/CDPS training workflow.

Run directly with:

    python -m cage.public_biological_validation --genes FOXS1 ESM1 KIF2C
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import re
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError
from scipy import stats

from .publication_style import PublicationStyle, apply_style, save_figure

LOGGER = logging.getLogger("cage.public_validation")

DEFAULT_GENES = ("FOXS1", "ESM1", "KIF2C")
DEFAULT_MARKERS: Mapping[str, str] = {
    "PDCD1": "checkpoint",
    "CD274": "checkpoint",
    "CTLA4": "checkpoint",
    "LAG3": "checkpoint",
    "TIGIT": "checkpoint",
    "HAVCR2": "checkpoint",
    "PDCD1LG2": "checkpoint",
    "IDO1": "checkpoint",
    "CXCL9": "immune chemokine",
    "CXCL10": "immune chemokine",
    "VEGFA": "angiogenesis",
    "PECAM1": "endothelial",
    "VWF": "endothelial",
    "COL1A1": "stromal",
    "ACTA2": "stromal",
    "EPCAM": "epithelial",
    "MKI67": "proliferation",
    "TOP2A": "proliferation",
    "CD8A": "T cell",
    "CD8B": "T cell",
    "CD4": "T cell",
    "MS4A1": "B cell",
    "CD68": "myeloid",
    "CD163": "myeloid",
    "ITGAM": "myeloid",
    "FCGR3A": "myeloid/NK",
    "NKG7": "NK/cytotoxic",
    "GNLY": "NK/cytotoxic",
    "FOXP3": "Treg",
}

AXIS_MARKERS: Mapping[str, tuple[str, ...]] = {
    "FOXS1": ("COL1A1", "ACTA2", "VWF", "PECAM1"),
    "ESM1": ("PECAM1", "VWF", "VEGFA", "CD274"),
    "KIF2C": ("MKI67", "TOP2A"),
}

GENE_AXIS: Mapping[str, str] = {
    "FOXS1": "possible invasive/stromal remodeling axis",
    "ESM1": "vascular/endothelial and tumor microenvironment axis",
    "KIF2C": "mitotic spindle/cell-cycle proliferation axis",
}

VALIDATION_FIGURE_FORMATS = ("svg", "png")


def validation_style(*, formats: Sequence[str] = VALIDATION_FIGURE_FORMATS) -> PublicationStyle:
    """Publication style for manuscript-facing public validation figures."""
    return PublicationStyle(
        font_family="Times New Roman",
        font_fallbacks=("Times New Roman", "Times", "STIXGeneral", "DejaVu Serif", "serif"),
        bold=True,
        base_font_size=10,
        axis_label_font_size=10,
        tick_label_font_size=9,
        legend_font_size=9,
        title_font_size=12,
        colorbar_label_font_size=9,
        default_formats=tuple(formats),
    )

HPA_KEYWORDS = (
    "gene",
    "ensembl",
    "evidence",
    "protein",
    "rna",
    "tissue",
    "cancer",
    "pathology",
    "tcga",
    "prognostic",
    "subcellular",
    "location",
    "reliability",
    "esophagus",
    "oesophagus",
    "esophageal",
)


@dataclass
class ValidationOutputs:
    hpa_summary: Path | None = None
    cbio_summary: Path | None = None
    immune_summary: Path | None = None
    singlecell_summary: Path | None = None
    integrated_summary: Path | None = None
    interpretation: Path | None = None


def normalize_gene(gene: str) -> str:
    return str(gene).strip().upper()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def setup_logging(output_dir: Path, log_level: str = "INFO") -> None:
    ensure_dir(output_dir)
    log_path = output_dir / "public_biological_validation.log"
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    level = getattr(logging, log_level.upper(), logging.INFO)
    root.setLevel(level)
    fmt = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stream = logging.StreamHandler(stream=sys.stderr)
    stream.setFormatter(fmt)
    root.addHandler(stream)
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)


def resolve_input_path(path: Path | None, fallback_names: Sequence[str]) -> Path | None:
    """Resolve a user path, allowing case-insensitive and nearby fallbacks."""
    if path and path.exists():
        return path

    candidates: list[Path] = []
    if path is not None:
        candidates.append(path)
        parent = path.parent if str(path.parent) != "." else Path.cwd()
        candidates.extend(parent / name for name in fallback_names)
        if parent.exists():
            target = path.name.lower()
            for child in parent.iterdir():
                if child.name.lower() == target:
                    return child

    for base in (Path("data/HPA"), Path("Data/HPA"), Path("data"), Path("Data")):
        candidates.extend(base / name for name in fallback_names)

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return candidate
    return None


def _open_hpa_table(path: Path) -> pd.DataFrame:
    suffixes = [s.lower() for s in path.suffixes]
    if ".zip" in suffixes:
        with zipfile.ZipFile(path) as zf:
            members = [m for m in zf.namelist() if not m.endswith("/")]
            tsv_members = [m for m in members if m.lower().endswith((".tsv", ".txt", ".csv"))]
            if not tsv_members:
                raise ValueError(f"No TSV/TXT/CSV member found inside {path}")
            member = tsv_members[0]
            sep = "," if member.lower().endswith(".csv") else "\t"
            with zf.open(member) as handle:
                return pd.read_csv(handle, sep=sep, dtype=str, low_memory=False)
    if path.suffix.lower() == ".json":
        return pd.read_json(path, dtype=str)
    sep = "," if path.suffix.lower() == ".csv" else "\t"
    return pd.read_csv(path, sep=sep, dtype=str, low_memory=False)


def _detect_gene_column(columns: Sequence[str]) -> str:
    normalized = {c.strip().lower(): c for c in columns}
    for name in ("gene", "gene name", "gene_name", "symbol", "hgnc symbol", "hugo gene symbol"):
        if name in normalized:
            return normalized[name]
    for col in columns:
        low = col.strip().lower()
        if low == "name" or ("gene" in low and "synonym" not in low and "description" not in low):
            return col
    raise ValueError("Could not detect a gene symbol column in the HPA file.")


def _hpa_column_category(column: str) -> str:
    low = column.lower()
    labels = []
    for key in ("esophagus", "oesophagus", "esophageal", "cancer", "pathology", "tcga"):
        if key in low:
            labels.append(key)
    for key in ("protein", "rna", "evidence", "reliability", "subcellular", "location"):
        if key in low:
            labels.append(key)
    return ";".join(labels) if labels else "other"


def _has_hpa_value(row: pd.Series, columns: Sequence[str]) -> float:
    for col in columns:
        if col not in row.index:
            continue
        val = str(row[col]).strip()
        if val and val.lower() not in {"nan", "na", "none", ""}:
            return 1.0
    return 0.0


def _plot_hpa_evidence(hpa: pd.DataFrame, genes: Sequence[str], out: Path) -> None:
    if hpa.empty:
        return
    gene_cols = [c for c in hpa.columns if c.lower() in {"gene", "gene name", "symbol"} or c.lower().startswith("gene")]
    gene_col = gene_cols[0] if gene_cols else hpa.columns[0]
    hpa = hpa.copy()
    hpa["_gene_upper"] = hpa[gene_col].map(normalize_gene)

    evidence_groups: Mapping[str, tuple[str, ...]] = {
        "Protein evidence": ("Evidence", "HPA evidence", "UniProt evidence", "NeXtProt evidence"),
        "RNA tissue": ("RNA tissue specificity", "RNA tissue distribution", "RNA tissue specific nTPM"),
        "RNA cancer": ("RNA cancer specificity", "RNA cancer distribution", "RNA cancer specific pTPM"),
        "Subcellular": ("Subcellular location", "Subcellular main location", "Subcellular additional location"),
        "Reliability": ("Reliability (IH)", "Reliability (IF)", "Antibody"),
        "Cancer prognostic": tuple(c for c in hpa.columns if "cancer prognostics" in c.lower()),
    }

    rows: list[dict[str, Any]] = []
    for gene in genes:
        gene_u = normalize_gene(gene)
        sub = hpa[hpa["_gene_upper"].eq(gene_u)]
        row = sub.iloc[0] if not sub.empty else pd.Series(dtype=object)
        record = {"gene": gene_u}
        for label, cols in evidence_groups.items():
            record[label] = _has_hpa_value(row, cols)
        rows.append(record)

    plot_data = pd.DataFrame(rows)
    plot_data.to_csv(out / "hpa_candidate_evidence_plot_data.csv", index=False)
    matrix = plot_data.set_index("gene")
    if matrix.empty:
        return

    style = apply_style(validation_style(formats=("svg", "png")))
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    im = ax.imshow(matrix.values.astype(float), cmap="cage_sequential", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns, rotation=35, ha="right")
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_yticklabels(matrix.index)
    ax.set_title("HPA Public Evidence Availability For Candidate Genes", fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Evidence present", fontweight="bold")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, "Yes" if matrix.iloc[i, j] > 0 else "No", ha="center", va="center", fontsize=7)
    save_figure(
        fig,
        out / "figure_hpa_candidate_evidence_heatmap",
        style=style,
        formats=("svg", "png"),
        metadata={"source_data": "hpa_candidate_evidence_plot_data.csv"},
    )


def load_hpa(
    hpa_file: Path,
    genes: Sequence[str],
    output_dir: Path,
    logger: logging.Logger = LOGGER,
) -> pd.DataFrame:
    """Load HPA data, extract candidate rows, and write validation outputs."""
    out = ensure_dir(output_dir / "hpa")
    log_lines: list[str] = []
    resolved = resolve_input_path(
        hpa_file,
        ("proteinatlas.tsv.zip", "proteinatlas.tsv", "proteinatlas.txt", "proteinatlas.csv", "proteinatlas.json"),
    )
    if resolved is None:
        msg = f"HPA input not found: {hpa_file}"
        logger.warning(msg)
        write_text(out / "hpa_missing_fields_report.txt", msg + "\n")
        pd.DataFrame({"column": [], "category": [], "selected_for_summary": []}).to_csv(
            out / "hpa_available_columns.csv", index=False
        )
        empty = pd.DataFrame({"Gene": list(genes), "hpa_status": "input file missing"})
        empty.to_csv(out / "hpa_candidate_gene_summary.csv", index=False)
        return empty

    if resolved != hpa_file:
        log_lines.append(f"Requested HPA file {hpa_file} was not found; using {resolved}.")
        logger.warning(log_lines[-1])

    df = _open_hpa_table(resolved)
    df.columns = [str(c).strip().strip('"') for c in df.columns]
    gene_col = _detect_gene_column(df.columns)
    gene_set = {normalize_gene(g) for g in genes}
    candidates = df[df[gene_col].map(normalize_gene).isin(gene_set)].copy()
    found = set(candidates[gene_col].map(normalize_gene))
    for missing in sorted(gene_set - found):
        log_lines.append(f"Gene {missing} was not found in HPA file {resolved}.")

    available_cols = pd.DataFrame(
        {
            "column": df.columns,
            "category": [_hpa_column_category(c) for c in df.columns],
            "selected_for_summary": [
                any(key in c.lower() for key in HPA_KEYWORDS) for c in df.columns
            ],
        }
    )
    available_cols.to_csv(out / "hpa_available_columns.csv", index=False)

    required_groups = {
        "gene_name": ("gene", "symbol"),
        "ensembl_id": ("ensembl",),
        "protein_evidence": ("protein", "evidence"),
        "rna_tissue_evidence": ("rna", "tissue"),
        "cancer_pathology_evidence": ("cancer", "pathology", "tcga", "prognostic"),
        "subcellular_location": ("subcellular", "location"),
        "reliability": ("reliability",),
        "esophagus_related": ("esophagus", "oesophagus", "esophageal"),
    }
    missing_groups: list[str] = []
    for group, keys in required_groups.items():
        if not any(any(key in c.lower() for key in keys) for c in df.columns):
            missing_groups.append(group)

    selected_cols = [c for c in df.columns if any(key in c.lower() for key in HPA_KEYWORDS)]
    ordered_cols = []
    for col in [gene_col, "Ensembl", "Evidence", "HPA evidence", "RNA tissue specificity",
                "RNA tissue distribution", "RNA cancer specificity", "RNA cancer distribution",
                "Subcellular location", "Subcellular main location"]:
        if col in candidates.columns and col not in ordered_cols:
            ordered_cols.append(col)
    for col in selected_cols:
        if col not in ordered_cols:
            ordered_cols.append(col)
    for col in candidates.columns:
        if col not in ordered_cols:
            ordered_cols.append(col)
    candidates = candidates.reindex(columns=ordered_cols)

    if candidates.empty:
        candidates = pd.DataFrame({gene_col: list(genes), "hpa_status": "gene not found"})
    candidates.to_csv(out / "hpa_candidate_gene_summary.csv", index=False)
    _plot_hpa_evidence(candidates, genes, out)

    if missing_groups:
        log_lines.append(
            "Missing expected HPA evidence groups: " + ", ".join(missing_groups) + ". "
            "All available columns for candidate genes were still saved."
        )
    else:
        log_lines.append("All broad HPA evidence groups were detected.")
    log_lines.append(f"HPA source used: {resolved}")
    log_lines.append(f"Candidate rows written: {len(candidates)}")
    write_text(out / "hpa_missing_fields_report.txt", "\n".join(log_lines) + "\n")
    return candidates


def _requests_session():
    import requests

    session = requests.Session()
    session.headers.update(
        {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "CAGE-public-biological-validation/0.1",
        }
    )
    return session


def _api_request(
    session: Any,
    method: str,
    url: str,
    *,
    json_body: Mapping[str, Any] | None = None,
    timeout: int = 60,
) -> tuple[Any | None, str | None]:
    try:
        response = session.request(method, url, json=json_body, timeout=timeout)
        response.raise_for_status()
        if not response.text.strip():
            return [], None
        return response.json(), None
    except Exception as exc:  # pragma: no cover - network dependent
        return None, f"{method} {url} failed: {exc}"


def _records_to_frame(records: Any) -> pd.DataFrame:
    if records is None:
        return pd.DataFrame()
    if isinstance(records, dict):
        if "data" in records and isinstance(records["data"], list):
            return pd.DataFrame(records["data"])
        return pd.DataFrame([records])
    if isinstance(records, list):
        return pd.DataFrame(records)
    return pd.DataFrame()


def _choose_profile(profiles: pd.DataFrame, kind: str) -> str | None:
    if profiles.empty:
        return None
    df = profiles.copy()
    def col(name: str) -> pd.Series:
        if name in df.columns:
            return df[name].astype(str)
        return pd.Series([""] * len(df), index=df.index, dtype=str)

    hay = (
        col("molecularProfileId")
        + " "
        + col("name")
        + " "
        + col("molecularAlterationType")
        + " "
        + col("datatype")
    ).str.lower()

    if kind == "mutation":
        mask = hay.str.contains("mutation", na=False)
    elif kind == "cna":
        mask = hay.str.contains("gistic|copy|cna|discrete", regex=True, na=False)
    elif kind == "mrna":
        z_mask = hay.str.contains("mrna", na=False) & hay.str.contains("z-score|zscore|zscores", regex=True, na=False)
        if z_mask.any():
            return str(df.loc[z_mask, "molecularProfileId"].iloc[0])
        mask = hay.str.contains("mrna|rna_seq|expression", regex=True, na=False)
    else:
        return None
    if mask.any():
        return str(df.loc[mask, "molecularProfileId"].iloc[0])
    return None


def _filter_by_entrez(frame: pd.DataFrame, entrez: int | None) -> pd.DataFrame:
    if frame.empty or entrez is None or "entrezGeneId" not in frame.columns:
        return pd.DataFrame()
    ids = pd.to_numeric(frame["entrezGeneId"], errors="coerce")
    return frame[ids.eq(entrez)]


def _numeric_column(frame: pd.DataFrame, columns: str | Sequence[str]) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=float, index=frame.index)
    names = [columns] if isinstance(columns, str) else list(columns)
    for column in names:
        if column in frame.columns:
            return pd.to_numeric(frame[column], errors="coerce")
    return pd.Series(np.nan, index=frame.index, dtype=float)


def _plot_cbioportal_summary(summary: pd.DataFrame, out: Path) -> None:
    if summary.empty or "gene" not in summary.columns:
        return
    metrics = [
        ("mutation_frequency", "Mutation"),
        ("cna_frequency", "CNA"),
        ("mrna_upregulated_frequency", "mRNA z >= 2"),
        ("any_alteration_frequency", "Any alteration"),
    ]
    available = [(col, label) for col, label in metrics if col in summary.columns]
    if not available:
        return
    plot_data = summary[["gene"] + [col for col, _ in available]].copy()
    plot_data = plot_data.rename(columns={col: label for col, label in available})
    plot_data.to_csv(out / "cbioportal_alteration_frequency_plot_data.csv", index=False)

    style = apply_style(validation_style(formats=("svg", "png")))
    import matplotlib.pyplot as plt

    genes = plot_data["gene"].astype(str).tolist()
    grouped_cols = [label for _, label in available]
    values = plot_data[grouped_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    colors = ["#E64B35", "#4DBBD5", "#00A087", "#3C5488"]

    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    x = np.arange(len(genes))
    width = 0.18 if len(grouped_cols) > 1 else 0.45
    offsets = (np.arange(len(grouped_cols)) - (len(grouped_cols) - 1) / 2) * width
    for idx, col in enumerate(grouped_cols):
        bars = ax.bar(x + offsets[idx], values[col].values * 100.0, width=width, label=col, color=colors[idx % len(colors)])
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.4,
                f"{height:.1f}",
                ha="center",
                va="bottom",
                fontsize=7,
                fontweight="bold",
            )
    ax.set_xticks(x)
    ax.set_xticklabels(genes)
    ax.set_ylabel("Frequency (%)", fontweight="bold")
    ax.set_title("cBioPortal TCGA-ESCA Genomic And mRNA Support", fontweight="bold")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.set_ylim(0, max(10.0, float(values.max().max() * 100.0) + 5.0))
    save_figure(
        fig,
        out / "figure_cbioportal_candidate_gene_alteration_frequency",
        style=style,
        formats=("svg", "png"),
        metadata={"source_data": "cbioportal_alteration_frequency_plot_data.csv"},
    )

    heat = values.copy()
    heat.index = genes
    fig, ax = plt.subplots(figsize=(7.2, 2.7))
    im = ax.imshow(heat.values * 100.0, cmap="cage_sequential", aspect="auto")
    ax.set_xticks(range(heat.shape[1]))
    ax.set_xticklabels(heat.columns, rotation=25, ha="right")
    ax.set_yticks(range(heat.shape[0]))
    ax.set_yticklabels(heat.index)
    ax.set_title("cBioPortal Candidate Gene Evidence Heatmap", fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Frequency (%)", fontweight="bold")
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            ax.text(j, i, f"{heat.iloc[i, j] * 100.0:.1f}", ha="center", va="center", fontsize=7, fontweight="bold")
    save_figure(
        fig,
        out / "figure_cbioportal_candidate_gene_evidence_heatmap",
        style=style,
        formats=("svg", "png"),
        metadata={"source_data": "cbioportal_alteration_frequency_plot_data.csv"},
    )


def _choose_sample_list(sample_lists: pd.DataFrame, study_id: str) -> str | None:
    if sample_lists.empty:
        return f"{study_id}_all"
    ids = sample_lists.get("sampleListId", pd.Series(dtype=str)).astype(str)
    categories = sample_lists.get("category", pd.Series([""] * len(sample_lists))).astype(str).str.lower()
    for pattern in (f"{study_id}_all", "all_cases_in_study", "_all"):
        mask = ids.str.lower().eq(pattern.lower()) | categories.str.contains(pattern, regex=False, na=False)
        if mask.any():
            return ids[mask].iloc[0]
    return ids.iloc[0]


def _gene_entrez_ids(session: Any, base_url: str, genes: Sequence[str], log: list[str]) -> dict[str, int]:
    url = f"{base_url}/genes/fetch?geneIdType=HUGO_GENE_SYMBOL"
    payload = list(genes)
    data, err = _api_request(session, "POST", url, json_body=payload)
    if err:
        log.append(err)
        return {}
    frame = _records_to_frame(data)
    mapping: dict[str, int] = {}
    for _, row in frame.iterrows():
        symbol = normalize_gene(row.get("hugoGeneSymbol", row.get("symbol", "")))
        entrez = row.get("entrezGeneId")
        if symbol and pd.notna(entrez):
            mapping[symbol] = int(entrez)
    missing = [g for g in genes if normalize_gene(g) not in mapping]
    if missing:
        log.append("Could not resolve Entrez IDs for: " + ", ".join(missing))
    return mapping


def _fetch_profile_data(
    session: Any,
    base_url: str,
    profile_id: str | None,
    endpoint_suffix: str,
    sample_list_id: str | None,
    entrez_ids: Sequence[int],
    log: list[str],
) -> pd.DataFrame:
    if not profile_id or not sample_list_id or not entrez_ids:
        log.append(f"Skipped {endpoint_suffix}: missing profile, sample list, or Entrez IDs.")
        return pd.DataFrame()
    url = f"{base_url}/molecular-profiles/{profile_id}/{endpoint_suffix}/fetch?projection=DETAILED"
    data, err = _api_request(
        session,
        "POST",
        url,
        json_body={"entrezGeneIds": list(entrez_ids), "sampleListId": sample_list_id},
    )
    if err:
        log.append(err)
        return pd.DataFrame()
    return _records_to_frame(data)


def query_cbioportal(
    study_id: str,
    genes: Sequence[str],
    output_dir: Path,
    base_url: str = "https://www.cbioportal.org/api",
    logger: logging.Logger = LOGGER,
) -> pd.DataFrame:
    """Query cBioPortal for mutation, CNA, mRNA, clinical, and summary data."""
    out = ensure_dir(output_dir / "cbioportal")
    log: list[str] = [f"cBioPortal base URL: {base_url}", f"Study ID: {study_id}"]
    try:
        session = _requests_session()
    except Exception as exc:
        msg = f"Could not import/use requests for cBioPortal API: {exc}"
        logger.warning(msg)
        log.append(msg)
        write_text(out / "cbioportal_api_log.txt", "\n".join(log) + "\n")
        empty = pd.DataFrame({"gene": list(genes), "cbioportal_status": "requests unavailable"})
        empty.to_csv(out / "cbioportal_candidate_gene_summary.csv", index=False)
        for name in ("molecular_profiles", "mutations", "cna", "mrna", "clinical"):
            pd.DataFrame().to_csv(out / f"cbioportal_{name}.csv", index=False)
        return empty

    profiles_data, err = _api_request(session, "GET", f"{base_url}/studies/{study_id}/molecular-profiles")
    if err:
        log.append(err)
        profiles = pd.DataFrame()
    else:
        profiles = _records_to_frame(profiles_data)
    profiles.to_csv(out / "cbioportal_molecular_profiles.csv", index=False)

    sample_lists_data, err = _api_request(session, "GET", f"{base_url}/studies/{study_id}/sample-lists")
    if err:
        log.append(err)
        sample_lists = pd.DataFrame()
    else:
        sample_lists = _records_to_frame(sample_lists_data)
    sample_list_id = _choose_sample_list(sample_lists, study_id)
    log.append(f"Selected sample list: {sample_list_id}")

    entrez_map = _gene_entrez_ids(session, base_url, genes, log)
    entrez_ids = [entrez_map[g] for g in sorted(entrez_map)]
    symbol_by_entrez = {v: k for k, v in entrez_map.items()}

    mutation_profile = _choose_profile(profiles, "mutation")
    cna_profile = _choose_profile(profiles, "cna")
    mrna_profile = _choose_profile(profiles, "mrna")
    log.extend(
        [
            f"Selected mutation profile: {mutation_profile}",
            f"Selected CNA profile: {cna_profile}",
            f"Selected mRNA profile: {mrna_profile}",
        ]
    )

    mutations = _fetch_profile_data(
        session, base_url, mutation_profile, "mutations", sample_list_id, entrez_ids, log
    )
    cna = _fetch_profile_data(
        session, base_url, cna_profile, "discrete-copy-number", sample_list_id, entrez_ids, log
    )
    mrna = _fetch_profile_data(
        session, base_url, mrna_profile, "molecular-data", sample_list_id, entrez_ids, log
    )

    mutations.to_csv(out / "cbioportal_mutations.csv", index=False)
    cna.to_csv(out / "cbioportal_cna.csv", index=False)
    mrna.to_csv(out / "cbioportal_mrna.csv", index=False)

    clinical_sample, err_s = _api_request(
        session, "GET", f"{base_url}/studies/{study_id}/clinical-data?clinicalDataType=SAMPLE&projection=DETAILED"
    )
    if err_s:
        log.append(err_s)
        clinical = pd.DataFrame()
    else:
        clinical_raw = _records_to_frame(clinical_sample)
        if not clinical_raw.empty and {"sampleId", "clinicalAttributeId", "value"}.issubset(clinical_raw.columns):
            clinical = clinical_raw.pivot_table(
                index="sampleId", columns="clinicalAttributeId", values="value", aggfunc="first"
            ).reset_index()
        else:
            clinical = clinical_raw
    clinical.to_csv(out / "cbioportal_clinical.csv", index=False)

    sample_ids: set[str] = set()
    for frame in (mutations, cna, mrna):
        if "sampleId" in frame.columns:
            sample_ids.update(frame["sampleId"].dropna().astype(str))
    total_samples = max(len(sample_ids), 1)
    if clinical is not None and not clinical.empty:
        first_id = "sampleId" if "sampleId" in clinical.columns else clinical.columns[0]
        total_samples = max(total_samples, clinical[first_id].dropna().astype(str).nunique())

    summary_rows: list[dict[str, Any]] = []
    for gene in genes:
        gene_u = normalize_gene(gene)
        entrez = entrez_map.get(gene_u)
        mut_g = _filter_by_entrez(mutations, entrez)
        cna_g = _filter_by_entrez(cna, entrez)
        mrna_g = _filter_by_entrez(mrna, entrez)

        mut_samples = set(mut_g.get("sampleId", pd.Series(dtype=str)).dropna().astype(str))
        cna_values = _numeric_column(cna_g, ("value", "alteration"))
        cna_alt = cna_g.loc[cna_values.fillna(0).ne(0)] if not cna_g.empty else pd.DataFrame()
        cna_samples = set(cna_alt.get("sampleId", pd.Series(dtype=str)).dropna().astype(str))
        mrna_values = _numeric_column(mrna_g, ("value", "alteration"))
        is_zscore = bool(mrna_profile and re.search(r"z[-_ ]?score|zscore|zscores", mrna_profile, re.I))
        mrna_up = mrna_g.loc[mrna_values.ge(2.0)] if is_zscore and not mrna_g.empty else pd.DataFrame()
        mrna_samples = set(mrna_up.get("sampleId", pd.Series(dtype=str)).dropna().astype(str))
        altered_samples = mut_samples | cna_samples | mrna_samples
        summary_rows.append(
            {
                "gene": gene_u,
                "entrez_gene_id": entrez,
                "total_samples_estimated": total_samples,
                "mutation_count": int(len(mut_g)),
                "mutation_sample_count": len(mut_samples),
                "mutation_frequency": len(mut_samples) / total_samples,
                "cna_altered_sample_count": len(cna_samples),
                "cna_frequency": len(cna_samples) / total_samples,
                "mrna_profile": mrna_profile,
                "mrna_upregulated_sample_count": len(mrna_samples),
                "mrna_upregulated_frequency": len(mrna_samples) / total_samples if is_zscore else np.nan,
                "mrna_upregulation_rule": "z-score >= 2" if is_zscore else "not computed; selected profile is not a z-score profile",
                "any_altered_sample_count": len(altered_samples),
                "any_alteration_frequency": len(altered_samples) / total_samples,
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out / "cbioportal_candidate_gene_summary.csv", index=False)
    _plot_cbioportal_summary(summary, out)
    write_text(out / "cbioportal_api_log.txt", "\n".join(log) + "\n")
    return summary


def _score_expression_candidate(path: Path) -> int:
    text = str(path).lower()
    score = 0
    for key, weight in (
        ("esca_vst_normalized_matrix", 80),
        ("tcga_esca", 40),
        ("vst_normalized", 35),
        ("normalized_primary_matrix", 30),
        ("normalized_expression", 25),
        ("expression_matrix", 20),
        ("step2_cohort", 15),
    ):
        if key in text:
            score += weight
    if "geo" in text or "gse" in text:
        score -= 30
    if path.suffix.lower() in {".csv", ".tsv", ".txt"}:
        score += 5
    return score


def find_expression_matrix(search_dirs: Sequence[Path], logger: logging.Logger = LOGGER) -> Path | None:
    names = ("vst_normalized", "expression_matrix", "normalized_expression", "TCGA_ESCA", "normalized_primary_matrix")
    candidates: list[Path] = []
    for base in search_dirs:
        if not base.exists():
            continue
        for root, _, files in os.walk(base):
            root_path = Path(root)
            for file in files:
                path = root_path / file
                low = file.lower()
                if path.suffix.lower() in {".csv", ".tsv", ".txt"} and any(n.lower() in low for n in names):
                    candidates.append(path)
    if not candidates:
        return None
    candidates = sorted(candidates, key=lambda p: (_score_expression_candidate(p), -len(str(p))), reverse=True)
    logger.info("Selected expression matrix: %s", candidates[0])
    return candidates[0]


def _csv_sep(path: Path) -> str:
    return "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","


def load_expression_subset(path: Path, genes: Sequence[str], logger: logging.Logger = LOGGER) -> pd.DataFrame:
    """Return expression as genes x samples for only requested genes."""
    sep = _csv_sep(path)
    header = pd.read_csv(path, sep=sep, nrows=0)
    cols = list(header.columns)
    wanted = {normalize_gene(g) for g in genes}
    upper_cols = {normalize_gene(c): c for c in cols}
    present_as_columns = [upper_cols[g] for g in wanted if g in upper_cols]
    first_col = cols[0] if cols else None

    if present_as_columns and first_col is not None:
        usecols = [first_col] + present_as_columns
        df = pd.read_csv(path, sep=sep, usecols=usecols)
        df = df.set_index(first_col).T
        df.index = [normalize_gene(i) for i in df.index]
        logger.info("Loaded expression subset from sample-by-gene matrix: %s", path)
        return df.apply(pd.to_numeric, errors="coerce")

    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, sep=sep, index_col=0, chunksize=1000):
        chunk.index = chunk.index.map(normalize_gene)
        hit = chunk.loc[chunk.index.intersection(wanted)]
        if not hit.empty:
            chunks.append(hit)
    if not chunks:
        logger.warning("No requested genes were found in expression matrix %s", path)
        return pd.DataFrame()
    df = pd.concat(chunks)
    df = df[~df.index.duplicated(keep="first")]
    logger.info("Loaded expression subset from gene-by-sample matrix: %s", path)
    return df.apply(pd.to_numeric, errors="coerce")


def _bh(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    mask = np.isfinite(arr)
    out = np.full(arr.shape, np.nan, dtype=float)
    if not mask.any():
        return out
    pvals = arr[mask]
    order = np.argsort(pvals)
    ranked = pvals[order]
    n = float(len(ranked))
    adjusted = ranked * n / (np.arange(len(ranked)) + 1.0)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    restored = np.empty_like(adjusted)
    restored[order] = adjusted
    out[mask] = restored
    return out


def compute_gene_marker_correlations(
    expression_file: Path | None,
    genes: Sequence[str],
    output_dir: Path,
    markers: Mapping[str, str] = DEFAULT_MARKERS,
    logger: logging.Logger = LOGGER,
) -> pd.DataFrame:
    """Compute Pearson/Spearman correlations between candidate and marker genes."""
    out = ensure_dir(output_dir / "immune")
    log: list[str] = []
    if expression_file is None:
        expression_file = find_expression_matrix(
            [Path("data/processed"), Path("Data/processed"), Path("results"), Path("outputs"), Path(".")],
            logger=logger,
        )
    if expression_file is None or not expression_file.exists():
        msg = "No expression matrix found for immune/TME validation."
        log.append(msg)
        write_text(out / "immune_validation_log.txt", "\n".join(log) + "\n")
        empty = pd.DataFrame()
        empty.to_csv(out / "immune_marker_gene_correlations.csv", index=False)
        empty.to_csv(out / "immune_marker_gene_correlations_fdr.csv", index=False)
        pd.DataFrame({"status": [msg]}).to_csv(out / "immune_validation_summary.csv", index=False)
        return empty

    log.append(f"Selected expression matrix: {expression_file}")
    all_genes = list(dict.fromkeys([normalize_gene(g) for g in genes] + [normalize_gene(m) for m in markers]))
    expr = load_expression_subset(expression_file, all_genes, logger=logger)
    present = set(expr.index)
    missing_candidates = [g for g in genes if normalize_gene(g) not in present]
    missing_markers = [m for m in markers if normalize_gene(m) not in present]
    if missing_candidates:
        log.append("Missing candidate genes in expression matrix: " + ", ".join(missing_candidates))
    if missing_markers:
        log.append("Missing marker genes in expression matrix: " + ", ".join(missing_markers))

    rows: list[dict[str, Any]] = []
    for gene in [normalize_gene(g) for g in genes if normalize_gene(g) in present]:
        x = pd.to_numeric(expr.loc[gene], errors="coerce")
        for marker, category in markers.items():
            marker_u = normalize_gene(marker)
            if marker_u not in present or marker_u == gene:
                continue
            y = pd.to_numeric(expr.loc[marker_u], errors="coerce")
            valid = x.notna() & y.notna()
            n = int(valid.sum())
            if n < 3 or x[valid].nunique() < 2 or y[valid].nunique() < 2:
                pearson_r = pearson_p = spearman_r = spearman_p = np.nan
            else:
                pearson_r, pearson_p = stats.pearsonr(x[valid], y[valid])
                spearman_r, spearman_p = stats.spearmanr(x[valid], y[valid])
            rows.append(
                {
                    "candidate_gene": gene,
                    "marker_gene": marker_u,
                    "marker_category": category,
                    "n_samples": n,
                    "pearson_r": pearson_r,
                    "pearson_p": pearson_p,
                    "spearman_r": spearman_r,
                    "spearman_p": spearman_p,
                }
            )

    corr = pd.DataFrame(rows)
    corr.to_csv(out / "immune_marker_gene_correlations.csv", index=False)
    if not corr.empty:
        corr["pearson_fdr"] = _bh(corr["pearson_p"])
        corr["spearman_fdr"] = _bh(corr["spearman_p"])
        corr["significant_spearman_fdr_0_05"] = corr["spearman_fdr"].le(0.05)
    corr.to_csv(out / "immune_marker_gene_correlations_fdr.csv", index=False)

    summary_rows: list[dict[str, Any]] = []
    for gene in [normalize_gene(g) for g in genes]:
        sub = corr[corr["candidate_gene"].eq(gene)] if not corr.empty else pd.DataFrame()
        sig = sub[sub.get("spearman_fdr", pd.Series(dtype=float)).le(0.05)] if not sub.empty else pd.DataFrame()
        top = sub.reindex(sub["spearman_r"].abs().sort_values(ascending=False).index).head(5) if not sub.empty else pd.DataFrame()
        summary_rows.append(
            {
                "candidate_gene": gene,
                "markers_tested": int(len(sub)),
                "significant_marker_correlations_spearman_fdr_0_05": int(len(sig)),
                "top_abs_spearman_markers": "; ".join(
                    f"{r.marker_gene} ({r.spearman_r:.2f})" for r in top.itertuples()
                    if pd.notna(r.spearman_r)
                ),
                "axis_markers_present": "; ".join(m for m in AXIS_MARKERS.get(gene, ()) if m in present),
            }
        )
    pd.DataFrame(summary_rows).to_csv(out / "immune_validation_summary.csv", index=False)
    _plot_correlation_heatmap(corr, genes, markers, out / "figure_candidate_gene_immune_correlation_heatmap")
    log.append("Default marker-gene correlation analysis completed. Optional R deconvolution was not required.")
    write_text(out / "immune_validation_log.txt", "\n".join(log) + "\n")
    return corr


def _plot_correlation_heatmap(
    corr: pd.DataFrame,
    genes: Sequence[str],
    markers: Mapping[str, str],
    base_path: Path,
) -> None:
    if corr.empty:
        return
    matrix = corr.pivot(index="candidate_gene", columns="marker_gene", values="spearman_r")
    row_order = [normalize_gene(g) for g in genes if normalize_gene(g) in matrix.index]
    col_order = [normalize_gene(m) for m in markers if normalize_gene(m) in matrix.columns]
    matrix = matrix.reindex(index=row_order, columns=col_order)
    if matrix.empty:
        return
    style = apply_style(validation_style(formats=("svg", "png")))
    import matplotlib.pyplot as plt

    fig_w = max(7.2, 0.28 * max(1, len(matrix.columns)))
    fig_h = max(2.4, 0.45 * max(1, len(matrix.index)) + 1.4)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(matrix.values.astype(float), cmap="cage_diverging", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    ax.set_title("Candidate Gene Association With Immune/TME Markers", fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Spearman rho", fontweight="bold")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix.iloc[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6, color="#222222")
    save_figure(fig, base_path, style=style, formats=("svg", "png"), metadata={"analysis": "immune_marker_correlations"})


def _read_table_auto(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            members = [m for m in zf.namelist() if m.lower().endswith((".txt", ".tsv", ".csv"))]
            preferred = [m for m in members if "majorlineage" in m.lower()]
            member = preferred[0] if preferred else members[0]
            sep = "," if member.lower().endswith(".csv") else "\t"
            with zf.open(member) as handle:
                return pd.read_csv(handle, sep=sep)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    sep = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    return pd.read_csv(path, sep=sep)


def _find_file_by_keywords(base_dirs: Sequence[Path], keywords: Sequence[str]) -> Path | None:
    hits = _find_files_by_keywords(base_dirs, keywords)
    return hits[0] if hits else None


def _find_files_by_keywords(base_dirs: Sequence[Path], keywords: Sequence[str]) -> list[Path]:
    hits: list[Path] = []
    for base in base_dirs:
        if not base.exists():
            continue
        for root, _, files in os.walk(base):
            for file in files:
                low = file.lower()
                if any(k in low for k in keywords) and Path(file).suffix.lower() in {".csv", ".tsv", ".txt", ".parquet", ".zip"}:
                    hits.append(Path(root) / file)
    return sorted(hits, key=lambda p: len(str(p)))


def _detect_cell_type_column(metadata: pd.DataFrame) -> str | None:
    preferred = (
        "cell_type",
        "celltype",
        "cell type",
        "cell_type_annotation",
        "annotation",
        "major_cell_type",
        "cluster",
    )
    normalized = {str(c).strip().lower(): c for c in metadata.columns}
    for name in preferred:
        if name in normalized:
            return normalized[name]
    for col in metadata.columns:
        low = str(col).lower()
        if "cell" in low and "type" in low:
            return col
    return None


def _cluster_average_long_from_file(path: Path, genes: Sequence[str]) -> pd.DataFrame:
    avg = _read_table_auto(path)
    if avg.empty:
        return pd.DataFrame()
    first = avg.columns[0] if len(avg.columns) else None
    upper_cols = {normalize_gene(c): c for c in avg.columns}
    avg_index_genes = [normalize_gene(i) for i in avg.index]
    gene_targets = [normalize_gene(g) for g in genes]
    if any(g in avg_index_genes for g in gene_targets):
        avg = avg.copy()
        avg.index = avg_index_genes
        hits = avg.loc[avg.index.intersection(gene_targets)]
        return hits.T.reset_index(names="cell_type").melt(
            id_vars="cell_type", var_name="gene", value_name="average_expression"
        )
    if first and any(g in upper_cols for g in gene_targets):
        rows = []
        for _, row in avg.iterrows():
            for gene in gene_targets:
                col = upper_cols.get(gene)
                if col:
                    rows.append({"cell_type": row[first], "gene": gene, "average_expression": row[col]})
        return pd.DataFrame(rows)
    if first:
        avg = avg.set_index(first)
        avg.index = avg.index.map(normalize_gene)
        hits = avg.loc[avg.index.intersection(gene_targets)]
        if not hits.empty:
            return hits.T.reset_index(names="cell_type").melt(
                id_vars="cell_type", var_name="gene", value_name="average_expression"
            )
    return pd.DataFrame()


def _subset_sc_expression(path: Path, genes: Sequence[str]) -> pd.DataFrame:
    sep = _csv_sep(path)
    header = pd.read_csv(path, sep=sep, nrows=0)
    cols = list(header.columns)
    wanted = {normalize_gene(g) for g in genes}
    upper_cols = {normalize_gene(c): c for c in cols}
    first_col = cols[0] if cols else None
    present_as_columns = [upper_cols[g] for g in wanted if g in upper_cols]
    if present_as_columns and first_col is not None:
        df = pd.read_csv(path, sep=sep, usecols=[first_col] + present_as_columns)
        return df.set_index(first_col).rename(columns=normalize_gene).apply(pd.to_numeric, errors="coerce")

    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, sep=sep, index_col=0, chunksize=1000):
        chunk.index = chunk.index.map(normalize_gene)
        hit = chunk.loc[chunk.index.intersection(wanted)]
        if not hit.empty:
            chunks.append(hit)
    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks).T.apply(pd.to_numeric, errors="coerce")


def run_singlecell_summary(
    singlecell_dir: Path,
    genes: Sequence[str],
    output_dir: Path,
    logger: logging.Logger = LOGGER,
) -> pd.DataFrame:
    """Optional TISCH2/processed single-cell validation."""
    out = ensure_dir(output_dir / "singlecell")
    base_dirs = [singlecell_dir, Path("data/external/TISCH2"), Path("Data/external/TISCH2")]
    if not any(base.exists() for base in base_dirs):
        msg = "Single-cell validation skipped because no processed single-cell dataset was found."
        write_text(out / "singlecell_validation_skipped.txt", msg + "\n")
        logger.warning(msg)
        return pd.DataFrame()

    cluster_paths = _find_files_by_keywords(base_dirs, ("cluster_average_expression", "cluster_avg", "average_expression", "expression"))
    cluster_path = None
    matrix_path = _find_file_by_keywords(base_dirs, ("expression_matrix", "matrix"))
    metadata_path = _find_file_by_keywords(base_dirs, ("cell_metadata", "cell_type_annotation", "metadata", "annotation"))

    avg_long = pd.DataFrame()
    pct_long = pd.DataFrame()
    log_lines: list[str] = []

    for candidate_path in cluster_paths:
        candidate_long = _cluster_average_long_from_file(candidate_path, genes)
        log_lines.append(
            f"Checked cluster-average expression file: {candidate_path} "
            f"({candidate_long['gene'].nunique() if not candidate_long.empty else 0} candidate genes found)."
        )
        if not candidate_long.empty:
            cluster_path = candidate_path
            avg_long = candidate_long
            log_lines.append(f"Using cluster-average expression file: {cluster_path}")
            break

    if not avg_long.empty:
        pass
    elif matrix_path is not None and metadata_path is not None:
        log_lines.append(f"Using expression matrix file: {matrix_path}")
        log_lines.append(f"Using cell metadata file: {metadata_path}")
        expr = _subset_sc_expression(matrix_path, genes)
        meta = _read_table_auto(metadata_path)
        cell_type_col = _detect_cell_type_column(meta)
        if expr.empty or cell_type_col is None:
            log_lines.append("Single-cell files were found, but gene expression or cell-type columns could not be detected.")
        else:
            cell_id_col = meta.columns[0]
            meta = meta.set_index(cell_id_col)
            expr.index = expr.index.astype(str)
            meta.index = meta.index.astype(str)
            common = expr.index.intersection(meta.index)
            expr = expr.loc[common]
            meta = meta.loc[common]
            rows_avg: list[dict[str, Any]] = []
            rows_pct: list[dict[str, Any]] = []
            for cell_type, idx in meta.groupby(cell_type_col).groups.items():
                sub = expr.loc[idx]
                for gene in [normalize_gene(g) for g in genes if normalize_gene(g) in sub.columns]:
                    values = pd.to_numeric(sub[gene], errors="coerce")
                    rows_avg.append(
                        {"cell_type": cell_type, "gene": gene, "average_expression": float(values.mean())}
                    )
                    rows_pct.append(
                        {
                            "cell_type": cell_type,
                            "gene": gene,
                            "percent_expressing": float(values.gt(0).mean() * 100.0),
                        }
                    )
            avg_long = pd.DataFrame(rows_avg)
            pct_long = pd.DataFrame(rows_pct)

    if avg_long.empty:
        msg = "Single-cell validation skipped because no processed single-cell dataset was found."
        write_text(out / "singlecell_validation_skipped.txt", msg + "\n" + "\n".join(log_lines) + "\n")
        return pd.DataFrame()

    avg_long["average_expression"] = pd.to_numeric(avg_long["average_expression"], errors="coerce")
    avg_long.to_csv(out / "singlecell_gene_by_celltype.csv", index=False)
    if pct_long.empty:
        pct_long = avg_long.assign(percent_expressing=np.nan)[["cell_type", "gene", "percent_expressing"]]
    pct_long.to_csv(out / "singlecell_percent_expressing.csv", index=False)
    summary = (
        avg_long.sort_values("average_expression", ascending=False)
        .groupby("gene")
        .head(3)
        .groupby("gene")
        .agg(top_cell_types=("cell_type", lambda x: "; ".join(map(str, x))),
             max_average_expression=("average_expression", "max"))
        .reset_index()
    )
    summary.to_csv(out / "singlecell_candidate_gene_summary.csv", index=False)
    _plot_singlecell(avg_long, pct_long, out)
    write_text(out / "singlecell_validation_log.txt", "\n".join(log_lines) + "\n")
    return summary


def _plot_singlecell(avg_long: pd.DataFrame, pct_long: pd.DataFrame, out: Path) -> None:
    if avg_long.empty:
        return
    style = apply_style(validation_style(formats=("svg",)))
    import matplotlib.pyplot as plt

    merged = avg_long.merge(pct_long, on=["cell_type", "gene"], how="left")
    cell_types = list(dict.fromkeys(merged["cell_type"].astype(str)))
    genes = list(dict.fromkeys(merged["gene"].astype(str)))
    x_pos = {g: i for i, g in enumerate(genes)}
    y_pos = {c: i for i, c in enumerate(cell_types)}

    fig, ax = plt.subplots(figsize=(max(4, len(genes) * 0.8), max(3, len(cell_types) * 0.28)))
    sizes = merged["percent_expressing"].fillna(40).clip(lower=5, upper=100)
    sc = ax.scatter(
        merged["gene"].map(x_pos),
        merged["cell_type"].astype(str).map(y_pos),
        s=sizes,
        c=merged["average_expression"],
        cmap="cage_sequential",
        edgecolor="#333333",
        linewidth=0.4,
    )
    ax.set_xticks(range(len(genes)))
    ax.set_xticklabels(genes)
    ax.set_yticks(range(len(cell_types)))
    ax.set_yticklabels(cell_types)
    ax.set_title("Single-Cell Candidate Gene Expression By Cell Type", fontweight="bold")
    cbar = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Average expression", fontweight="bold")
    save_figure(fig, out / "figure_singlecell_dotplot", style=style, formats=("svg",))

    matrix = avg_long.pivot_table(index="cell_type", columns="gene", values="average_expression", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(max(4, len(matrix.columns) * 0.8), max(3, len(matrix.index) * 0.28)))
    im = ax.imshow(matrix.values, aspect="auto", cmap="cage_sequential")
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns)
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    ax.set_title("Single-Cell Average Expression Heatmap", fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    save_figure(fig, out / "figure_singlecell_heatmap", style=style, formats=("svg",))


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()


def _hpa_support_for_gene(hpa: pd.DataFrame, gene: str) -> str:
    if hpa.empty:
        return "not available"
    gene_cols = [c for c in hpa.columns if c.lower() in {"gene", "gene name", "symbol"} or c.lower().startswith("gene")]
    col = gene_cols[0] if gene_cols else hpa.columns[0]
    row = hpa[hpa[col].map(normalize_gene).eq(normalize_gene(gene))]
    if row.empty:
        return "not found"
    values = []
    for key in ("Evidence", "HPA evidence", "RNA tissue specificity", "RNA cancer specificity", "Subcellular location"):
        if key in row.columns:
            val = str(row[key].iloc[0]).strip()
            if val and val.lower() != "nan":
                values.append(f"{key}: {val}")
    return "; ".join(values[:5]) if values else "available HPA row without canonical evidence fields"


def _immune_axis_support(corr: pd.DataFrame, gene: str) -> tuple[float, str]:
    if corr.empty:
        return np.nan, "not available"
    gene_u = normalize_gene(gene)
    markers = AXIS_MARKERS.get(gene_u, ())
    sub = corr[corr["candidate_gene"].eq(gene_u) & corr["marker_gene"].isin(markers)]
    if sub.empty:
        return np.nan, "axis markers not present"
    sig = sub[sub.get("spearman_fdr", pd.Series(dtype=float)).le(0.05)]
    pos = sub[sub["spearman_r"].gt(0)]
    score = float(pos["spearman_r"].mean()) if not pos.empty else float(sub["spearman_r"].mean())
    pieces = [
        f"{r.marker_gene} rho={r.spearman_r:.2f}, FDR={r.spearman_fdr:.3g}"
        for r in sub.sort_values("spearman_r", ascending=False).itertuples()
        if pd.notna(r.spearman_r)
    ]
    prefix = "supported by significant positive correlations" if not sig.empty and sig["spearman_r"].gt(0).any() else "association observed"
    return score, prefix + ": " + "; ".join(pieces)


def build_integrated_summary(
    genes: Sequence[str],
    output_dir: Path,
    logger: logging.Logger = LOGGER,
) -> pd.DataFrame:
    """Build integrated validation table, heatmap, and manuscript paragraph."""
    out = ensure_dir(output_dir / "integrated")
    hpa = _safe_read_csv(output_dir / "hpa" / "hpa_candidate_gene_summary.csv")
    cbio = _safe_read_csv(output_dir / "cbioportal" / "cbioportal_candidate_gene_summary.csv")
    immune = _safe_read_csv(output_dir / "immune" / "immune_marker_gene_correlations_fdr.csv")
    single = _safe_read_csv(output_dir / "singlecell" / "singlecell_candidate_gene_summary.csv")

    rows: list[dict[str, Any]] = []
    for gene in genes:
        gene_u = normalize_gene(gene)
        cb = cbio[cbio.get("gene", pd.Series(dtype=str)).map(normalize_gene).eq(gene_u)] if not cbio.empty else pd.DataFrame()
        sc = single[single.get("gene", pd.Series(dtype=str)).map(normalize_gene).eq(gene_u)] if not single.empty else pd.DataFrame()
        immune_score, immune_text = _immune_axis_support(immune, gene_u)
        hpa_text = _hpa_support_for_gene(hpa, gene_u)
        any_alt = float(cb["any_alteration_frequency"].iloc[0]) if not cb.empty and "any_alteration_frequency" in cb else np.nan
        mrna_up = float(cb["mrna_upregulated_frequency"].iloc[0]) if not cb.empty and "mrna_upregulated_frequency" in cb else np.nan
        single_text = str(sc["top_cell_types"].iloc[0]) if not sc.empty and "top_cell_types" in sc else "not available"
        support_signals = sum(
            [
                hpa_text not in {"not available", "not found"},
                pd.notna(any_alt) and any_alt > 0,
                pd.notna(mrna_up) and mrna_up > 0,
                pd.notna(immune_score) and abs(immune_score) >= 0.25,
                single_text != "not available",
            ]
        )
        level = "moderate public-data support" if support_signals >= 3 else "limited public-data support" if support_signals >= 1 else "insufficient public-data support"
        rows.append(
            {
                "gene": gene_u,
                "proposed_axis": GENE_AXIS.get(gene_u, "candidate biological axis"),
                "hpa_support_summary": hpa_text,
                "cbioportal_any_alteration_frequency": any_alt,
                "cbioportal_mrna_upregulated_frequency": mrna_up,
                "immune_tme_axis_score": immune_score,
                "immune_tme_support_summary": immune_text,
                "singlecell_celltype_localization": single_text,
                "integrated_support_level": level,
                "interpretation_boundary": "Public-data validation only; requires functional validation.",
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "candidate_gene_public_validation_summary.csv", index=False)
    _plot_integrated_heatmap(summary, out / "figure_integrated_public_validation_heatmap")
    _write_interpretation_markdown(summary, out / "candidate_gene_biological_interpretation.md")
    return summary


def _plot_integrated_heatmap(summary: pd.DataFrame, base_path: Path) -> None:
    if summary.empty:
        return
    metrics = pd.DataFrame(
        {
            "gene": summary["gene"],
            "HPA row/evidence": summary["hpa_support_summary"].map(
                lambda x: 0.0 if str(x) in {"not available", "not found"} else 1.0
            ),
            "cBio alteration freq": pd.to_numeric(summary["cbioportal_any_alteration_frequency"], errors="coerce").fillna(0),
            "cBio mRNA up freq": pd.to_numeric(summary["cbioportal_mrna_upregulated_frequency"], errors="coerce").fillna(0),
            "Immune/TME axis rho": pd.to_numeric(summary["immune_tme_axis_score"], errors="coerce").fillna(0),
            "Single-cell available": summary["singlecell_celltype_localization"].map(lambda x: 0.0 if str(x) == "not available" else 1.0),
        }
    ).set_index("gene")
    style = apply_style(validation_style(formats=("svg", "png")))
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    im = ax.imshow(metrics.values, aspect="auto", cmap="cage_sequential")
    ax.set_xticks(range(metrics.shape[1]))
    ax.set_xticklabels(metrics.columns, rotation=35, ha="right")
    ax.set_yticks(range(metrics.shape[0]))
    ax.set_yticklabels(metrics.index)
    ax.set_title("Integrated Public Validation Evidence", fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Evidence metric", fontweight="bold")
    for i in range(metrics.shape[0]):
        for j in range(metrics.shape[1]):
            ax.text(j, i, f"{metrics.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
    save_figure(fig, base_path, style=style, formats=("svg", "png"))


def _write_interpretation_markdown(summary: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Public biological validation of CAGE-prioritized ESCA candidate genes",
        "",
        "This section summarizes public-data validation for CAGE/CDPS-prioritized candidate genes. "
        "These analyses provide public protein-level evidence, genomic alteration analysis, "
        "immune/TME association, and optional single-cell cell-type localization; they do not "
        "constitute wet-lab or functional validation.",
        "",
        "## Integrated Results Paragraph",
        "",
        "**Public biological validation of CAGE-prioritized ESCA candidate genes.** "
        "Public biological validation was performed for FOXS1, ESM1, and KIF2C using Human Protein "
        "Atlas annotations, cBioPortal TCGA-ESCA molecular profiles, TCGA-ESCA immune/TME marker-gene "
        "correlation analysis, and optional processed single-cell data when available. ",
    ]
    gene_sentences = []
    for row in summary.itertuples(index=False):
        hpa = row.hpa_support_summary
        cbio = row.cbioportal_any_alteration_frequency
        immune = row.immune_tme_support_summary
        sc = row.singlecell_celltype_localization
        cbio_text = "cBioPortal alteration support was not available"
        if pd.notna(cbio):
            cbio_text = f"genomic alteration analysis estimated an alteration frequency of {cbio:.3f}"
        sc_text = "single-cell cell-type localization was not available"
        if str(sc) != "not available":
            sc_text = f"single-cell cell-type localization suggested higher expression in {sc}"
        gene_sentences.append(
            f"{row.gene} was evaluated as a {row.proposed_axis}; the observed evidence is {row.integrated_support_level}. "
            f"HPA support: {hpa}. {cbio_text}. Immune/TME association: {immune}. {sc_text}. "
            "Together, these results are consistent with a possible role only where supported by the data and require functional validation."
        )
    lines.append(" ".join(gene_sentences))
    lines.extend(
        [
            "",
            "## Per-Gene Interpretation",
            "",
        ]
    )
    for row in summary.itertuples(index=False):
        lines.extend(
            [
                f"### {row.gene}",
                "",
                f"- Proposed model: {row.proposed_axis}.",
                f"- HPA/public protein-level evidence: {row.hpa_support_summary}.",
                f"- cBioPortal/genomic alteration analysis: any alteration frequency = {row.cbioportal_any_alteration_frequency}.",
                f"- Immune/TME association: {row.immune_tme_support_summary}.",
                f"- Single-cell cell-type localization: {row.singlecell_celltype_localization}.",
                "- Claim boundary: this supports a possible association and requires functional validation.",
                "",
            ]
        )
    write_text(path, "\n".join(lines) + "\n")


def write_manuscript_figure_manifest(output_dir: Path) -> None:
    """Write a compact map from manuscript figures to source plot data."""
    rows = [
        {
            "figure": "hpa/figure_hpa_candidate_evidence_heatmap.svg",
            "source_data": "hpa/hpa_candidate_evidence_plot_data.csv",
            "manuscript_use": "HPA public protein/RNA/cancer evidence availability panel",
        },
        {
            "figure": "cbioportal/figure_cbioportal_candidate_gene_alteration_frequency.svg",
            "source_data": "cbioportal/cbioportal_alteration_frequency_plot_data.csv",
            "manuscript_use": "TCGA-ESCA mutation/CNA/mRNA alteration frequency panel",
        },
        {
            "figure": "cbioportal/figure_cbioportal_candidate_gene_evidence_heatmap.svg",
            "source_data": "cbioportal/cbioportal_alteration_frequency_plot_data.csv",
            "manuscript_use": "Compact genomic alteration support heatmap",
        },
        {
            "figure": "immune/figure_candidate_gene_immune_correlation_heatmap.svg",
            "source_data": "immune/immune_marker_gene_correlations_fdr.csv",
            "manuscript_use": "Immune/TME marker-gene correlation heatmap",
        },
        {
            "figure": "singlecell/figure_singlecell_dotplot.svg",
            "source_data": "singlecell/singlecell_gene_by_celltype.csv",
            "manuscript_use": "TISCH2 cell-type candidate expression dot plot",
        },
        {
            "figure": "singlecell/figure_singlecell_heatmap.svg",
            "source_data": "singlecell/singlecell_gene_by_celltype.csv",
            "manuscript_use": "TISCH2 cell-type candidate expression heatmap",
        },
        {
            "figure": "integrated/figure_integrated_public_validation_heatmap.svg",
            "source_data": "integrated/candidate_gene_public_validation_summary.csv",
            "manuscript_use": "Integrated public biological validation evidence summary",
        },
    ]
    pd.DataFrame(rows).to_csv(output_dir / "manuscript_figure_data_manifest.csv", index=False)


def run_public_biological_validation(args: argparse.Namespace) -> ValidationOutputs:
    genes = [normalize_gene(g) for g in args.genes]
    out = ensure_dir(args.output_dir)
    setup_logging(out, args.log_level)
    LOGGER.info("Starting public biological validation for genes: %s", ", ".join(genes))

    outputs = ValidationOutputs()
    load_hpa(args.hpa, genes, out)
    outputs.hpa_summary = out / "hpa" / "hpa_candidate_gene_summary.csv"

    if not args.skip_cbioportal:
        query_cbioportal(args.cbioportal_study_id, genes, out, base_url=args.cbioportal_base_url)
        outputs.cbio_summary = out / "cbioportal" / "cbioportal_candidate_gene_summary.csv"
    else:
        LOGGER.warning("Skipping cBioPortal validation by user request.")

    expression_file = args.expression_file
    compute_gene_marker_correlations(expression_file, genes, out)
    outputs.immune_summary = out / "immune" / "immune_validation_summary.csv"

    run_singlecell_summary(args.singlecell_dir, genes, out)
    sc_summary = out / "singlecell" / "singlecell_candidate_gene_summary.csv"
    outputs.singlecell_summary = sc_summary if sc_summary.exists() else None

    build_integrated_summary(genes, out)
    outputs.integrated_summary = out / "integrated" / "candidate_gene_public_validation_summary.csv"
    outputs.interpretation = out / "integrated" / "candidate_gene_biological_interpretation.md"
    write_manuscript_figure_manifest(out)
    LOGGER.info("Public biological validation complete: %s", out)
    return outputs


def _load_config_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "YAML config support requires PyYAML. Install it or pass CLI arguments directly."
            ) from exc
        data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")
    validation = data.get("validation", data)
    if not isinstance(validation, dict):
        raise ValueError("Config key 'validation' must contain a mapping.")
    return validation


def _apply_config(args: argparse.Namespace) -> argparse.Namespace:
    config_path = getattr(args, "config", None)
    if config_path is None:
        return args
    cfg = _load_config_file(config_path)
    defaults = build_parser().parse_args([])
    key_map = {
        "genes": "genes",
        "hpa_file": "hpa",
        "cbioportal_study_id": "cbioportal_study_id",
        "singlecell_dir": "singlecell_dir",
        "output_dir": "output_dir",
        "expression_file": "expression_file",
    }
    for cfg_key, arg_key in key_map.items():
        if cfg_key not in cfg:
            continue
        current = getattr(args, arg_key)
        default = getattr(defaults, arg_key)
        if current != default:
            continue
        value = cfg[cfg_key]
        if arg_key == "genes":
            value = list(value)
        elif arg_key in {"hpa", "singlecell_dir", "output_dir", "expression_file"} and value is not None:
            value = Path(value)
        setattr(args, arg_key, value)
    return args


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cage-public-validation",
        description="Run public biological validation for final CAGE/CDPS ESCA candidate genes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--genes", nargs="+", default=list(DEFAULT_GENES), help="Candidate gene symbols.")
    parser.add_argument("--config", type=Path, default=None, help="Optional YAML/JSON config containing a validation section.")
    parser.add_argument("--hpa", type=Path, default=Path("data/HPA/proteinatlas.tsv"), help="HPA proteinatlas TSV/CSV/JSON file, optionally zipped.")
    parser.add_argument("--output", "--output-dir", dest="output_dir", type=Path, default=Path("outputs/step9_public_biological_validation"), help="Validation output directory.")
    parser.add_argument("--cbioportal-study-id", default="esca_tcga", help="cBioPortal study ID.")
    parser.add_argument("--cbioportal-base-url", default="https://www.cbioportal.org/api", help="cBioPortal API base URL.")
    parser.add_argument("--singlecell-dir", type=Path, default=Path("data/TISCH2"), help="Optional processed TISCH2/single-cell directory.")
    parser.add_argument("--expression-file", type=Path, default=None, help="Optional TCGA ESCA expression matrix override.")
    parser.add_argument("--skip-cbioportal", action="store_true", help="Skip network cBioPortal calls.")
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args = _apply_config(args)
    run_public_biological_validation(args)


if __name__ == "__main__":  # pragma: no cover
    main()
