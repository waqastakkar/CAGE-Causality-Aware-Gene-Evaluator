"""CAGE Step 8 — GEO dataset preparation.

Downloads and prepares GEO external-validation datasets for ESCA / ESCC studies.
Accessible via the Step 8 unified CLI:

    python -m cage.step8_external_validation_and_release prepare-geo \\
        --gse GSE38129 GSE161533 GSE53624 GSE53625 \\
        --output-dir outputs/step8_geo_prepared

Can also be called as a standalone module:

    python -m cage.step8_prepare_geo \\
        --gse GSE38129 GSE161533 GSE53624 GSE53625 \\
        --output-dir outputs/step8_geo_prepared

Main features
-------------
1. Downloads one or more GEO Series accessions (GSE...) using GEOparse.
2. Optionally downloads supplementary files.
3. Extracts per-sample metadata into tidy tables.
4. Heuristically infers tumor / normal labels from metadata text.
5. Builds a probe-level expression matrix from GSM tables.
6. Maps probes to gene symbols using GPL annotation where possible.
7. Collapses probe-level data to gene-level expression.
8. Writes per-dataset outputs plus a cross-dataset manifest.

Outputs per accession
---------------------
<output-dir>/
  GSE<id>/
    raw/
    metadata/
    processed/
    logs/
  external_validation_manifest.csv

Dependencies
------------
pip install GEOparse pandas numpy requests
(Run from the miniconda environment that has these packages installed.)
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import re
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# pandas, requests, and GEOparse are imported lazily so that build_parser()
# and the argument-definition path work in any Python environment.  The actual
# processing functions will raise ImportError at call time if unavailable.
try:
    import pandas as pd
    import requests
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False

try:
    import GEOparse
    _HAS_GEOPARSE = True
except ImportError:
    _HAS_GEOPARSE = False

try:
    from . import cli_args as _cli_args
    from .cli_args import configure_logging as _configure_logging
    _HAS_CLI_ARGS = True
except ImportError:
    _HAS_CLI_ARGS = False


# ----------------------------- Configuration -------------------------------- #

TUMOR_PATTERNS = [
    r"\btumou?r\b",
    r"\bcancer\b",
    r"\bcarcinoma\b",
    r"\bescc\b",
    r"\besca\b",
    r"\bprimary tumor\b",
    r"\bmalignan",
    r"\badenocarcinoma\b",
    r"\bsquamous cell carcinoma\b",
]

NORMAL_PATTERNS = [
    r"\bnormal\b",
    r"\badjacent normal\b",
    r"\bnon[- ]tumou?r\b",
    r"\bcontrol\b",
    r"\bhealthy\b",
    r"\bmatched normal\b",
    r"\bpara[- ]cancer\b",
    r"\bnoncancer",
    r"\bbenign\b",
    r"\bnormal esophageal epithel",
]

ESCA_PATTERNS = [
    r"\besophageal\b",
    r"\besophagus\b",
    r"\besca\b",
    r"\bescc\b",
    r"\beac\b",
]

HISTOLOGY_PATTERNS = {
    "ESCC": [
        r"\bescc\b",
        r"\bsquamous\b",
        r"\besophageal squamous\b",
    ],
    "EAC": [
        r"\beac\b",
        r"\badenocarcinoma\b",
        r"\besophageal adenocarcinoma\b",
    ],
}

MISSING_TOKENS = {"", "nan", "none", "na", "n/a", "unknown", "not available", "null"}


# ----------------------------- Data classes --------------------------------- #

@dataclass
class DatasetSummary:
    accession: str
    title: str
    platform_id: Optional[str]
    platform_title: Optional[str]
    n_samples_total: int
    n_tumor: int
    n_normal: int
    n_unknown: int
    histology_summary: str
    paired_hint: str
    probe_matrix_shape: str
    gene_matrix_shape: str
    gene_symbol_mapping_success: bool
    usable_for_classifier: bool
    usable_for_gene_replication: bool
    notes: str


# ----------------------------- Logging -------------------------------------- #

def setup_logger(log_file: Path, verbose: bool = False) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    handlers: List[logging.Handler] = [
        logging.FileHandler(log_file, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
        force=True,
    )


# ----------------------------- Utility helpers ------------------------------ #

def safe_str(x: object) -> str:
    if x is None:
        return ""
    return str(x).strip()


def first_nonempty(values: Iterable[object]) -> str:
    for value in values:
        text = safe_str(value)
        if text and text.lower() not in MISSING_TOKENS:
            return text
    return ""


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", safe_str(text)).strip().lower()


def any_match(patterns: Sequence[str], text: str) -> bool:
    text_n = normalize_text(text)
    return any(re.search(p, text_n, flags=re.IGNORECASE) for p in patterns)


def guess_histology(text: str) -> str:
    for label, patterns in HISTOLOGY_PATTERNS.items():
        if any_match(patterns, text):
            return label
    return "Unknown"


def infer_sample_type(text: str) -> str:
    text_n = normalize_text(text)
    has_tumor = any(re.search(p, text_n, flags=re.IGNORECASE) for p in TUMOR_PATTERNS)
    has_normal = any(re.search(p, text_n, flags=re.IGNORECASE) for p in NORMAL_PATTERNS)

    # Prefer specific "adjacent normal" style labels over generic cancer words
    if has_normal and not has_tumor:
        return "Normal"
    if has_tumor and not has_normal:
        return "Tumor"

    # Resolve ambiguous cases
    if has_tumor and has_normal:
        # If explicit adjacent/matched normal exists, call Normal
        if re.search(r"\badjacent normal\b|\bmatched normal\b|\bnormal tissue\b", text_n):
            return "Normal"
        return "Tumor"

    return "Unknown"


def read_gz_text_lines(path: Path, max_lines: int = 20) -> List[str]:
    lines: List[str] = []
    try:
        with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as fh:
            for i, line in enumerate(fh):
                lines.append(line.rstrip("\n"))
                if i + 1 >= max_lines:
                    break
    except Exception:
        return []
    return lines


def choose_platform_id(gse_obj, sample_metadata: pd.DataFrame) -> Optional[str]:
    """
    Choose the best platform from the dataset.
    Preference:
    1. Most frequent platform in sample metadata.
    2. If only one GPL exists in gse.gpls, use it.
    """
    if "platform_id" in sample_metadata.columns:
        counts = (
            sample_metadata["platform_id"]
            .astype(str)
            .replace({"": np.nan, "nan": np.nan})
            .dropna()
            .value_counts()
        )
        if not counts.empty:
            return str(counts.index[0])

    gpl_ids = list(getattr(gse_obj, "gpls", {}).keys())
    if len(gpl_ids) == 1:
        return gpl_ids[0]

    return None


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


# ----------------------------- GEO parsing ---------------------------------- #

def _geo_soft_url(accession: str) -> str:
    """Build NCBI HTTPS URL for a GSE SOFT.gz file.

    GEO FTP layout: geo/series/<prefix>nnn/<accession>/soft/<accession>_family.soft.gz
    where <prefix> is the accession number with the last three digits replaced by 'nnn'.
    e.g. GSE20347 -> GSE20nnn, GSE161533 -> GSE161nnn
    """
    m = re.match(r"^(GSE)(\d+)$", accession.upper())
    if not m:
        raise ValueError(f"Cannot parse GEO accession: {accession!r}")
    num = m.group(2)
    # The directory prefix drops the last 3 digits
    prefix = num[:-3] + "nnn" if len(num) > 3 else "0nnn"
    return (
        f"https://ftp.ncbi.nlm.nih.gov/geo/series/{m.group(1)}{prefix}"
        f"/{accession}/soft/{accession}_family.soft.gz"
    )


def _server_file_size(url: str, timeout: int = 30) -> Optional[int]:
    """Return Content-Length from a HEAD request, or None if unavailable."""
    try:
        r = requests.head(url, timeout=timeout, allow_redirects=True)
        r.raise_for_status()
        cl = r.headers.get("Content-Length")
        return int(cl) if cl else None
    except Exception:
        return None


def _download_with_retry(
    url: str,
    dest: Path,
    *,
    max_retries: int = 5,
    retry_delay: float = 10.0,
    chunk_size: int = 1 << 20,  # 1 MB
    timeout: int = 120,
) -> Path:
    """Download *url* to *dest* with byte-range resumption and retry.

    Strategy:
    - A `.tmp` file accumulates bytes across attempts.
    - On each retry, a ``Range: bytes=<resumed_at>-`` header requests only
      the missing tail — so a 192 MB file that dies at byte 192,747,510 will
      fetch only the remaining 41,293 bytes on the next attempt rather than
      restarting from zero.
    - Gzip integrity is verified before the atomic rename to *dest*.
    - Returns *dest* on success; raises after *max_retries* exhausted.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest.with_suffix(".tmp")

    # Learn the full size once so we can log progress and detect completion.
    total_size = _server_file_size(url, timeout=30)
    if total_size:
        logging.info("Remote file size: %.1f MB", total_size / 1e6)

    for attempt in range(1, max_retries + 1):
        # How many bytes do we already have from a previous (partial) attempt?
        resumed_at = tmp_path.stat().st_size if tmp_path.is_file() else 0

        if total_size and resumed_at >= total_size:
            # We have all bytes — just verify and rename.
            logging.info("File already fully downloaded (%d bytes); verifying …", resumed_at)
        else:
            headers: Dict[str, str] = {}
            if resumed_at > 0:
                headers["Range"] = f"bytes={resumed_at}-"
                logging.info(
                    "Resuming at byte %d (%.1f MB already on disk), attempt %d/%d: %s",
                    resumed_at, resumed_at / 1e6, attempt, max_retries, url,
                )
            else:
                logging.info(
                    "Downloading (attempt %d/%d): %s", attempt, max_retries, url
                )

            try:
                with requests.get(
                    url, stream=True, timeout=timeout, headers=headers
                ) as resp:
                    # 206 Partial Content = server honoured Range header
                    # 200 OK = server ignored Range (will restart from 0)
                    if resp.status_code == 200 and resumed_at > 0:
                        logging.warning(
                            "Server returned 200 (ignored Range header); "
                            "restarting from byte 0"
                        )
                        resumed_at = 0
                    resp.raise_for_status()

                    mode = "ab" if resumed_at > 0 else "wb"
                    with open(tmp_path, mode) as fh:
                        for chunk in resp.iter_content(chunk_size=chunk_size):
                            if chunk:
                                fh.write(chunk)

                    downloaded_now = tmp_path.stat().st_size
                    logging.info(
                        "Download segment done: %.1f MB on disk",
                        downloaded_now / 1e6,
                    )

            except Exception as exc:
                wait = retry_delay * (2 ** (attempt - 1))
                if attempt < max_retries:
                    on_disk = tmp_path.stat().st_size if tmp_path.is_file() else 0
                    logging.warning(
                        "Attempt %d/%d failed after %.1f MB (%s). "
                        "Resuming in %.0fs …",
                        attempt, max_retries, on_disk / 1e6, exc, wait,
                    )
                    time.sleep(wait)
                    continue
                else:
                    tmp_path.unlink(missing_ok=True)
                    raise RuntimeError(
                        f"Failed to download {url} after {max_retries} attempts. "
                        f"Last error: {exc}"
                    ) from exc

        # Verify gzip integrity of whatever is on disk now
        try:
            with gzip.open(tmp_path, "rb") as gz:
                gz.read(128)
        except Exception as gz_exc:
            # Corrupted — delete and retry from scratch
            logging.warning(
                "Gzip verification failed (%s); discarding partial file and retrying.",
                gz_exc,
            )
            tmp_path.unlink(missing_ok=True)
            wait = retry_delay * (2 ** (attempt - 1))
            if attempt < max_retries:
                time.sleep(wait)
                continue
            else:
                raise RuntimeError(
                    f"Downloaded file is corrupt after {max_retries} attempts: {url}"
                ) from gz_exc

        tmp_path.rename(dest)
        logging.info(
            "Download complete: %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6
        )
        return dest

    raise RuntimeError(  # pragma: no cover
        f"Failed to download {url} after {max_retries} attempts."
    )


def _load_gse_from_file(soft_gz_path: Path, annotate_gpl: bool = False) -> object:
    """Parse a local SOFT.gz file with GEOparse (no network access)."""
    logging.info("Parsing %s with GEOparse ...", soft_gz_path.name)
    gse = GEOparse.get_GEO(filepath=str(soft_gz_path), silent=True)
    return gse


def download_gse(
    accession: str,
    destdir: Path,
    download_supp: bool = False,
    annotate_gpl: bool = False,
    max_retries: int = 5,
    retry_delay: float = 10.0,
) -> object:
    """Download and parse a GEO series.

    Bypasses GEOparse's fragile FTP size-check downloader in favour of a
    streaming HTTPS downloader with exponential-backoff retry.  Only the
    SOFT.gz parsing step is delegated to GEOparse.
    """
    destdir.mkdir(parents=True, exist_ok=True)
    soft_fname = f"{accession}_family.soft.gz"
    soft_dest = destdir / soft_fname

    if soft_dest.is_file():
        # Validate existing file before trusting it
        try:
            with gzip.open(soft_dest, "rb") as gz:
                gz.read(128)
            logging.info("Reusing cached SOFT.gz: %s", soft_dest)
        except Exception:
            logging.warning("Cached SOFT.gz is corrupt; re-downloading: %s", soft_dest)
            soft_dest.unlink(missing_ok=True)

    if not soft_dest.is_file():
        url = _geo_soft_url(accession)
        _download_with_retry(
            url,
            soft_dest,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )

    gse = _load_gse_from_file(soft_dest, annotate_gpl=annotate_gpl)

    if download_supp:
        try:
            logging.info("Downloading supplementary files for %s", accession)
            gse.download_supplementary_files(
                directory=str(destdir / "supplementary"),
                download_sra=False,
                email=None,
            )
        except Exception as exc:
            logging.warning("Supplementary download failed for %s: %s", accession, exc)

    return gse


def extract_sample_metadata(gse_obj, accession: str) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    for gsm_id, gsm in gse_obj.gsms.items():
        meta = getattr(gsm, "metadata", {}) or {}
        title = first_nonempty(meta.get("title", []))
        source_name = first_nonempty(meta.get("source_name_ch1", []))
        description = first_nonempty(meta.get("description", []))
        characteristics = meta.get("characteristics_ch1", []) or []
        characteristics_text = " | ".join([safe_str(x) for x in characteristics if safe_str(x)])

        platform_id = first_nonempty(meta.get("platform_id", []))
        channel_count = first_nonempty(meta.get("channel_count", []))

        combined_text = " | ".join(
            [
                title,
                source_name,
                description,
                characteristics_text,
            ]
        )

        sample_type = infer_sample_type(combined_text)
        histology = guess_histology(combined_text)
        esca_related = any_match(ESCA_PATTERNS, combined_text)

        rows.append(
            {
                "accession": accession,
                "gsm_id": gsm_id,
                "title": title,
                "source_name": source_name,
                "description": description,
                "characteristics_text": characteristics_text,
                "platform_id": platform_id,
                "channel_count": channel_count,
                "combined_text": combined_text,
                "sample_type_inferred": sample_type,
                "histology_inferred": histology,
                "esca_related_text": bool(esca_related),
            }
        )

    df = pd.DataFrame(rows)

    if not df.empty:
        # Optional pairing hint from title if present, e.g. patient IDs
        df["pairing_key_hint"] = df["title"].astype(str).str.extract(
            r"([A-Za-z]*\d{1,4}[A-Za-z]*)", expand=False
        ).fillna("")

    return df


# -------------------------- Expression matrix build -------------------------- #

def extract_expression_table_from_gsm(gsm) -> Optional[pd.DataFrame]:
    """
    Expected common GSM table structure:
    - ID_REF / VALUE
    - ID / VALUE
    - probe / value
    - gene / value

    Returns a 2-column dataframe: feature_id, value
    """
    table = getattr(gsm, "table", None)
    if table is None or not isinstance(table, pd.DataFrame) or table.empty:
        return None

    cols_lower = {c.lower(): c for c in table.columns}

    id_candidates = ["id_ref", "id", "probe", "probeset", "gene", "genesymbol", "symbol"]
    value_candidates = ["value", "signal", "intensity", "expr", "expression"]

    id_col = next((cols_lower[c] for c in id_candidates if c in cols_lower), None)
    val_col = next((cols_lower[c] for c in value_candidates if c in cols_lower), None)

    if id_col is None or val_col is None:
        # fallback: use first two columns if second is numeric-ish
        if table.shape[1] >= 2:
            cand = table.iloc[:, :2].copy()
            cand.columns = ["feature_id", "value"]
            cand["value"] = pd.to_numeric(cand["value"], errors="coerce")
            cand = cand.dropna(subset=["value"])
            if not cand.empty:
                return cand
        return None

    out = table[[id_col, val_col]].copy()
    out.columns = ["feature_id", "value"]
    out["feature_id"] = out["feature_id"].astype(str).str.strip()
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna(subset=["feature_id", "value"])
    out = out[out["feature_id"] != ""]
    return out


def build_probe_matrix(gse_obj, sample_metadata: pd.DataFrame) -> pd.DataFrame:
    sample_tables: List[pd.DataFrame] = []

    for gsm_id, gsm in gse_obj.gsms.items():
        expr = extract_expression_table_from_gsm(gsm)
        if expr is None or expr.empty:
            logging.warning("No usable expression table for sample %s", gsm_id)
            continue
        expr = expr.rename(columns={"value": gsm_id})
        sample_tables.append(expr)

    if not sample_tables:
        raise RuntimeError("No sample-level expression tables could be extracted.")

    merged = sample_tables[0]
    for df in sample_tables[1:]:
        merged = merged.merge(df, on="feature_id", how="outer")

    merged = merged.drop_duplicates(subset=["feature_id"]).reset_index(drop=True)

    # Keep only samples that are present in metadata
    sample_cols = [c for c in merged.columns if c in set(sample_metadata["gsm_id"])]
    merged = merged[["feature_id"] + sample_cols]

    return merged


# -------------------------- Platform annotation ----------------------------- #

def choose_gene_symbol_column(gpl_table: pd.DataFrame) -> Optional[str]:
    cols = list(gpl_table.columns)
    candidates = [
        "Gene Symbol",
        "GENE_SYMBOL",
        "Gene symbol",
        "Symbol",
        "gene_symbol",
        "GENE SYMBOL",
        "ILMN_Gene",
        "Gene",
        "GB_ACC",
        # NOTE: SPOT_ID is intentionally omitted — it is always a probe
        # identifier (e.g. Agilent CB_XXXXXX / control names), never an HGNC
        # gene symbol, so treating it as a gene column produces garbage mappings.
    ]

    for cand in candidates:
        if cand in cols:
            return cand

    # More flexible fallback
    lower_map = {c.lower(): c for c in cols}
    fuzzy_candidates = [
        "gene symbol",
        "gene_symbol",
        "symbol",
        "gene",
        "ilmn_gene",
    ]
    for cand in fuzzy_candidates:
        if cand in lower_map:
            return lower_map[cand]

    return None


def choose_probe_id_column(gpl_table: pd.DataFrame) -> Optional[str]:
    cols = list(gpl_table.columns)

    if "ID" in cols:
        return "ID"

    lower_map = {c.lower(): c for c in cols}
    for cand in ["id", "id_ref", "probe", "probeset", "spot_id"]:
        if cand in lower_map:
            return lower_map[cand]

    return None


def clean_gene_symbol(symbol: str) -> str:
    text = safe_str(symbol)

    if not text:
        return ""

    # Remove common separators / multiple genes
    text = re.split(r"///|//|;|,|\|", text)[0].strip()
    text = re.sub(r"\s+", "", text)

    if text.lower() in MISSING_TOKENS:
        return ""

    return text


def get_platform_annotation(gse_obj, platform_id: str) -> Optional[pd.DataFrame]:
    gpls = getattr(gse_obj, "gpls", {})
    if platform_id not in gpls:
        return None

    gpl = gpls[platform_id]
    gpl_table = getattr(gpl, "table", None)
    if gpl_table is None or not isinstance(gpl_table, pd.DataFrame) or gpl_table.empty:
        return None

    probe_col = choose_probe_id_column(gpl_table)
    gene_col = choose_gene_symbol_column(gpl_table)

    if probe_col is None or gene_col is None:
        return None

    ann = gpl_table[[probe_col, gene_col]].copy()
    ann.columns = ["feature_id", "gene_symbol"]
    ann["feature_id"] = ann["feature_id"].astype(str).str.strip()
    ann["gene_symbol"] = ann["gene_symbol"].astype(str).map(clean_gene_symbol)
    ann = ann[(ann["feature_id"] != "") & (ann["gene_symbol"] != "")]
    ann = ann.drop_duplicates(subset=["feature_id"])
    return ann


def collapse_to_gene_level(
    probe_matrix: pd.DataFrame,
    annotation: Optional[pd.DataFrame],
    method: str = "median",
) -> Tuple[pd.DataFrame, bool]:
    """
    Returns gene-level matrix with first column gene_symbol.
    """
    if annotation is None or annotation.empty:
        # If no annotation exists, check whether feature_id already looks like gene symbols.
        gene_like = probe_matrix["feature_id"].astype(str).str.match(r"^[A-Za-z0-9\-_\.]{2,20}$")
        if gene_like.mean() > 0.8:
            gene_df = probe_matrix.copy()
            gene_df = gene_df.rename(columns={"feature_id": "gene_symbol"})
            return gene_df, False
        return pd.DataFrame(), False

    merged = probe_matrix.merge(annotation, on="feature_id", how="inner")
    if merged.empty:
        return pd.DataFrame(), False

    sample_cols = [c for c in merged.columns if c not in {"feature_id", "gene_symbol"}]

    if method == "mean":
        gene_df = merged.groupby("gene_symbol", as_index=False)[sample_cols].mean(numeric_only=True)
    elif method == "max":
        gene_df = merged.groupby("gene_symbol", as_index=False)[sample_cols].max(numeric_only=True)
    else:
        gene_df = merged.groupby("gene_symbol", as_index=False)[sample_cols].median(numeric_only=True)

    gene_df = gene_df.drop_duplicates(subset=["gene_symbol"]).reset_index(drop=True)
    return gene_df, True


# ---------------------------- Usability scoring ----------------------------- #

def paired_hint_from_metadata(df: pd.DataFrame) -> str:
    if "pairing_key_hint" not in df.columns or df.empty:
        return "unknown"

    tmp = df[["pairing_key_hint", "sample_type_inferred"]].copy()
    tmp = tmp[tmp["pairing_key_hint"].astype(str) != ""]
    if tmp.empty:
        return "unknown"

    counts = (
        tmp.groupby("pairing_key_hint")["sample_type_inferred"]
        .nunique()
        .reset_index(name="n_types")
    )
    if (counts["n_types"] >= 2).any():
        return "possible_paired"
    return "not_obvious"


def summarize_histology(df: pd.DataFrame) -> str:
    if df.empty or "histology_inferred" not in df.columns:
        return ""
    vc = df["histology_inferred"].fillna("Unknown").value_counts()
    return "; ".join([f"{k}:{v}" for k, v in vc.items()])


def decide_usability(
    sample_metadata: pd.DataFrame,
    gene_matrix: pd.DataFrame,
) -> Tuple[bool, bool, str]:
    n_tumor = int((sample_metadata["sample_type_inferred"] == "Tumor").sum())
    n_normal = int((sample_metadata["sample_type_inferred"] == "Normal").sum())
    n_genes = max(gene_matrix.shape[0] - 1, 0) if not gene_matrix.empty else 0

    notes: List[str] = []

    usable_classifier = (
        n_tumor >= 10 and
        n_normal >= 5 and
        n_genes >= 1000
    )
    usable_gene_rep = (
        n_tumor >= 5 and
        n_normal >= 3 and
        n_genes >= 200
    )

    if n_tumor < 10 or n_normal < 5:
        notes.append("small_class_sizes_for_classifier")
    if n_genes < 1000:
        notes.append("limited_gene_overlap_or_mapping")
    if n_normal == 0:
        notes.append("no_clearly_inferred_normal_samples")
    if n_tumor == 0:
        notes.append("no_clearly_inferred_tumor_samples")

    if not notes:
        notes.append("looks_usable")

    return usable_classifier, usable_gene_rep, ";".join(notes)


# ------------------------------- Main worker -------------------------------- #

def process_accession(
    accession: str,
    outdir: Path,
    download_supp: bool,
    collapse_method: str,
    verbose: bool,
    max_retries: int = 5,
    retry_delay: float = 10.0,
) -> DatasetSummary:
    ds_root = ensure_dir(outdir / accession)
    raw_dir = ensure_dir(ds_root / "raw")
    meta_dir = ensure_dir(ds_root / "metadata")
    proc_dir = ensure_dir(ds_root / "processed")
    logs_dir = ensure_dir(ds_root / "logs")

    setup_logger(logs_dir / f"{accession}.log", verbose=verbose)
    logging.info("Processing %s", accession)

    gse = download_gse(
        accession=accession,
        destdir=raw_dir,
        download_supp=download_supp,
        annotate_gpl=False,
        max_retries=max_retries,
        retry_delay=retry_delay,
    )

    # Save basic series info
    series_info = {
        "accession": accession,
        "title": safe_str(getattr(gse, "name", "")),
        "metadata": getattr(gse, "metadata", {}),
        "n_gsms": len(getattr(gse, "gsms", {})),
        "n_gpls": len(getattr(gse, "gpls", {})),
        "gpl_ids": list(getattr(gse, "gpls", {}).keys()),
    }
    with open(meta_dir / "series_info.json", "w", encoding="utf-8") as fh:
        json.dump(series_info, fh, indent=2, ensure_ascii=False)

    sample_metadata = extract_sample_metadata(gse, accession)
    sample_metadata.to_csv(meta_dir / "sample_metadata_inferred.csv", index=False)

    if sample_metadata.empty:
        raise RuntimeError(f"No sample metadata extracted for {accession}")

    probe_matrix = build_probe_matrix(gse, sample_metadata)
    probe_matrix.to_csv(proc_dir / "expression_probe_matrix.csv", index=False)

    platform_id = choose_platform_id(gse, sample_metadata)
    platform_title = None
    annotation = None

    if platform_id and platform_id in getattr(gse, "gpls", {}):
        gpl = gse.gpls[platform_id]
        platform_title = first_nonempty(getattr(gpl, "metadata", {}).get("title", []))
        annotation = get_platform_annotation(gse, platform_id)

        if annotation is not None and not annotation.empty:
            annotation.to_csv(meta_dir / f"{platform_id}_probe_to_gene_symbol.csv", index=False)
        else:
            logging.warning("No usable gene-symbol annotation for %s (%s)", accession, platform_id)
    else:
        logging.warning("Could not determine a dominant platform for %s", accession)

    gene_matrix, mapped = collapse_to_gene_level(
        probe_matrix=probe_matrix,
        annotation=annotation,
        method=collapse_method,
    )

    if not gene_matrix.empty:
        gene_matrix.to_csv(proc_dir / "expression_gene_matrix.csv", index=False)
    else:
        logging.warning("Gene-level matrix could not be produced for %s", accession)

    # Save class summary
    class_counts = (
        sample_metadata["sample_type_inferred"]
        .value_counts(dropna=False)
        .rename_axis("sample_type")
        .reset_index(name="n_samples")
    )
    class_counts.to_csv(meta_dir / "sample_type_counts.csv", index=False)

    histology_counts = (
        sample_metadata["histology_inferred"]
        .value_counts(dropna=False)
        .rename_axis("histology")
        .reset_index(name="n_samples")
    )
    histology_counts.to_csv(meta_dir / "histology_counts.csv", index=False)

    usable_classifier, usable_gene_replication, notes = decide_usability(
        sample_metadata=sample_metadata,
        gene_matrix=gene_matrix,
    )

    summary = DatasetSummary(
        accession=accession,
        title=safe_str(getattr(gse, "name", "")),
        platform_id=platform_id,
        platform_title=platform_title,
        n_samples_total=int(sample_metadata.shape[0]),
        n_tumor=int((sample_metadata["sample_type_inferred"] == "Tumor").sum()),
        n_normal=int((sample_metadata["sample_type_inferred"] == "Normal").sum()),
        n_unknown=int((sample_metadata["sample_type_inferred"] == "Unknown").sum()),
        histology_summary=summarize_histology(sample_metadata),
        paired_hint=paired_hint_from_metadata(sample_metadata),
        probe_matrix_shape=f"{probe_matrix.shape[0]}x{probe_matrix.shape[1]-1}",
        gene_matrix_shape=(
            f"{gene_matrix.shape[0]}x{gene_matrix.shape[1]-1}"
            if not gene_matrix.empty else "0x0"
        ),
        gene_symbol_mapping_success=bool(mapped),
        usable_for_classifier=bool(usable_classifier),
        usable_for_gene_replication=bool(usable_gene_replication),
        notes=notes,
    )

    with open(meta_dir / "dataset_summary.json", "w", encoding="utf-8") as fh:
        json.dump(asdict(summary), fh, indent=2, ensure_ascii=False)

    logging.info("Finished %s", accession)
    logging.info("Summary: %s", asdict(summary))
    return summary


# ---------------------------------- CLI ------------------------------------- #

def build_parser(
    parser: Optional[argparse.ArgumentParser] = None,
) -> argparse.ArgumentParser:
    """Build the prepare-geo argument parser.

    If *parser* is provided (e.g. a subcommand parser from the Step 8
    dispatcher), arguments are added to it in-place and it is returned.
    Otherwise a standalone ArgumentParser is created.
    """
    if parser is None:
        parser = argparse.ArgumentParser(
            prog="python -m cage.step8_prepare_geo",
            description=(
                "Download and prepare GEO datasets for CAGE external validation."
            ),
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        if _HAS_CLI_ARGS:
            _cli_args.add_global_args(parser, require_input_dir=False)
            _cli_args.add_figure_args(parser)
        else:
            parser.add_argument("--output-dir", required=True, type=Path,
                                help="Output directory for all prepared datasets.")
            parser.add_argument("--seed", type=int, default=2026)
            parser.add_argument("--n-threads", type=int, default=1)
            parser.add_argument("--overwrite", action="store_true")
            parser.add_argument("--log-level", default="INFO",
                                choices=("DEBUG", "INFO", "WARNING", "ERROR"))

    g = parser.add_argument_group("GEO download options")
    g.add_argument(
        "--gse",
        nargs="+",
        required=True,
        metavar="ACC",
        help="One or more GEO series accessions, e.g. GSE38129 GSE161533 GSE53624 GSE53625.",
    )
    g.add_argument(
        "--download-supp",
        action="store_true",
        help="Download supplementary files when available.",
    )
    g.add_argument(
        "--collapse-method",
        choices=["median", "mean", "max"],
        default="median",
        metavar="METHOD",
        help="How to collapse multiple probes per gene symbol (default: median).",
    )
    g.add_argument(
        "--continue-on-error",
        action="store_true",
        default=True,
        help="Continue processing remaining accessions if one fails (default: on).",
    )
    g.add_argument(
        "--no-continue-on-error",
        dest="continue_on_error",
        action="store_false",
        help="Abort on first failure.",
    )
    g.add_argument(
        "--max-retries",
        type=int,
        default=5,
        metavar="N",
        help="Max HTTPS download retries per accession (exponential back-off, default: 5).",
    )
    g.add_argument(
        "--retry-delay",
        type=float,
        default=10.0,
        metavar="SECS",
        help="Initial retry delay in seconds (doubles each attempt, default: 10).",
    )
    g.add_argument(
        "--platform-auto",
        action="store_true",
        default=True,
        help="Automatically pick the dominant platform for probe-to-gene mapping (default: on).",
    )
    return parser


def run_prepare_geo(args: argparse.Namespace) -> None:
    """Execute GEO dataset download and preparation from a parsed Namespace."""
    if not _HAS_PANDAS or not _HAS_GEOPARSE:
        raise SystemExit(
            "prepare-geo requires pandas, requests, and GEOparse.\n"
            "Install them with: pip install GEOparse pandas requests\n"
            "Run this step from the miniconda environment."
        )
    outdir = Path(args.output_dir).resolve()
    ensure_dir(outdir)

    verbose = getattr(args, "log_level", "INFO").upper() == "DEBUG"
    if _HAS_CLI_ARGS:
        _configure_logging(args, log_file=outdir / "logs" / "step8_prepare_geo.log")
    else:
        setup_logger(outdir / "logs" / "prepare_geo.log", verbose=verbose)

    logging.info("CAGE Step 8 — prepare-geo")
    logging.info("Accessions: %s", ", ".join(args.gse))

    summaries: List[DatasetSummary] = []
    failures: List[Dict[str, str]] = []

    for accession in args.gse:
        try:
            summary = process_accession(
                accession=accession,
                outdir=outdir,
                download_supp=args.download_supp,
                collapse_method=args.collapse_method,
                verbose=verbose,
                max_retries=args.max_retries,
                retry_delay=args.retry_delay,
            )
            summaries.append(summary)
        except Exception as exc:
            logging.exception("Failed processing %s", accession)
            failures.append({"accession": accession, "error": str(exc)})
            if not args.continue_on_error:
                raise

    if summaries:
        manifest = pd.DataFrame([asdict(x) for x in summaries])
        manifest.to_csv(outdir / "external_validation_manifest.csv", index=False)

    if failures:
        pd.DataFrame(failures).to_csv(outdir / "failed_accessions.csv", index=False)

    logging.info("Done | successful=%d failed=%d", len(summaries), len(failures))


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if _HAS_CLI_ARGS:
        _cli_args.apply_thread_limits(args)
        _cli_args.ensure_output_dir(args)
    run_prepare_geo(args)


if __name__ == "__main__":
    main()