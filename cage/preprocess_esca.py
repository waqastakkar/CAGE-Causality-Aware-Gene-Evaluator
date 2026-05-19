"""CAGE preprocessing helpers (Phase I).

Pure-function utilities used by :mod:`step2_build_cohort` to load,
harmonise, filter, and transform the TCGA ESCA input files into the
artefacts consumed by all downstream phases.

Key design decisions:
* numpy + csv only (no pandas/scipy dependency) so the pipeline runs in
  any Python 3.9+ environment that has numpy.
* Every function that samples or shuffles takes an explicit ``seed``
  or ``rng`` parameter for deterministic behaviour.
* Gene selection and z-scoring are done *globally* in Step 2 to produce
  a single ``normalized_primary_matrix.csv``.  Step 4 re-normalises
  per-fold using training-set statistics for the deep model.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import random
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

logger = logging.getLogger("cage.preprocess")


# =====================================================================
# I / O helpers
# =====================================================================

def load_csv_matrix(
    path: str | Path,
    dtype: type = np.float64,
) -> Tuple[List[str], List[str], np.ndarray]:
    """Load a gene × sample CSV (R-style row-named matrix).

    Returns
    -------
    gene_names : list[str]
        Row identifiers (first column values).
    sample_ids : list[str]
        Column header values (everything after the empty first cell).
    matrix : ndarray, shape (n_genes, n_samples)
    """
    path = Path(path)
    logger.info("Loading expression matrix from %s ...", path.name)
    gene_names: list[str] = []
    data_rows: list[list[float]] = []

    with open(path, "r", newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        sample_ids = header[1:]  # first cell is empty or an index label

        for row in reader:
            gene_names.append(row[0])
            data_rows.append([float(v) if v not in ("", "NA", "NaN", "nan") else 0.0
                              for v in row[1:]])

    matrix = np.array(data_rows, dtype=dtype)
    logger.info(
        "  -> %d genes x %d samples  (%s)",
        matrix.shape[0], matrix.shape[1], path.name,
    )
    return gene_names, sample_ids, matrix


def load_metadata(path: str | Path) -> List[Dict[str, str]]:
    """Load metadata CSV into a list of ordered dicts (one per sample).

    The first (unnamed) column is renamed to ``row_index``.
    """
    path = Path(path)
    logger.info("Loading metadata from %s ...", path.name)
    records: list[dict[str, str]] = []
    with open(path, "r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        # Rename the unnamed first column produced by R's write.csv
        if fieldnames and fieldnames[0] == "":
            fieldnames[0] = "row_index"
            reader.fieldnames = fieldnames
        for row in reader:
            records.append(dict(row))
    logger.info("  -> %d records, %d fields", len(records), len(fieldnames))
    return records


def write_csv_matrix(
    path: str | Path,
    row_names: Sequence[str],
    col_names: Sequence[str],
    matrix: np.ndarray,
    *,
    index_label: str = "",
    fmt: str = "%.6f",
) -> None:
    """Write a matrix to CSV with row and column headers.

    ``matrix`` shape must be ``(len(row_names), len(col_names))``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n_rows, n_cols = matrix.shape
    assert n_rows == len(row_names) and n_cols == len(col_names), (
        f"Shape mismatch: matrix {matrix.shape} vs names ({len(row_names)}, {len(col_names)})"
    )
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([index_label] + list(col_names))
        for i, rname in enumerate(row_names):
            writer.writerow([rname] + [fmt % v for v in matrix[i]])
    logger.info("Wrote matrix %s  (%d x %d)", path.name, n_rows, n_cols)


def write_csv_records(
    path: str | Path,
    records: Sequence[Dict[str, Any]],
    *,
    fieldnames: Sequence[str] | None = None,
) -> None:
    """Write a list of dicts to a CSV file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        logger.warning("write_csv_records: empty record list, writing header-only CSV to %s", path)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            fh.write("")
        return
    fields = list(fieldnames) if fieldnames else list(records[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for rec in records:
            writer.writerow(rec)
    logger.info("Wrote %d records to %s", len(records), path.name)


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=True, default=str)
    logger.info("Wrote JSON to %s", path.name)


# =====================================================================
# Barcode harmonisation
# =====================================================================

def add_harmonized_identifiers(records: List[Dict[str, str]]) -> None:
    """Add ``patient_barcode`` and ``sample_type_code`` to each record in-place.

    TCGA barcode anatomy (example ``TCGA-L5-A43C-01A-11R-A24K-31``):
        Chars  0-11 : patient barcode  (``TCGA-L5-A43C``)
        Chars 13-14 : sample-type code (``01`` = primary tumour, ``11`` = normal)
    """
    for rec in records:
        bc = rec.get("barcode", "")
        rec["patient_barcode"] = bc[:12] if len(bc) >= 12 else bc
        rec["sample_type_code"] = bc[13:15] if len(bc) >= 15 else rec.get("sample_type_id", "")
    logger.info("Harmonised identifiers for %d records", len(records))


# =====================================================================
# Sample filtering
# =====================================================================

def filter_primary_samples(
    records: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    """Keep primary tumour (01) and solid-tissue normal (11) only.

    * Removes recurrent / metastatic / ambiguous specimens.
    * If a patient has >1 tumour aliquot, retains the first barcode
      lexicographically so the choice is deterministic.
    * Adds a human-readable ``label`` field (``Tumor`` or ``Normal``).

    Returns a *new* list (input is not modified).
    """
    kept: list[dict[str, str]] = []
    seen_tumor_patients: set[str] = set()

    # Sort by barcode so the "first" tumour is deterministic
    sorted_recs = sorted(records, key=lambda r: r.get("barcode", ""))

    for rec in sorted_recs:
        stc = rec.get("sample_type_code", rec.get("sample_type_id", ""))
        if stc not in ("01", "11"):
            continue
        pat = rec["patient_barcode"]
        if stc == "01":
            if pat in seen_tumor_patients:
                continue
            seen_tumor_patients.add(pat)
        new_rec = dict(rec)
        new_rec["label"] = "Tumor" if stc == "01" else "Normal"
        new_rec["label_int"] = "1" if stc == "01" else "0"
        kept.append(new_rec)

    n_tumor = sum(1 for r in kept if r["label"] == "Tumor")
    n_normal = sum(1 for r in kept if r["label"] == "Normal")
    logger.info(
        "filter_primary_samples: kept %d tumour + %d normal = %d samples "
        "(%d patients)",
        n_tumor, n_normal, len(kept), len(set(r["patient_barcode"] for r in kept)),
    )
    return kept


# =====================================================================
# Expression matrix processing
# =====================================================================

def align_matrix_to_samples(
    gene_names: List[str],
    sample_ids: List[str],
    matrix: np.ndarray,
    target_barcodes: Sequence[str],
) -> Tuple[List[str], np.ndarray]:
    """Select and reorder columns of a genes×samples matrix to match *target_barcodes*.

    Returns ``(gene_names, sub_matrix)`` with columns ordered as in
    *target_barcodes*.  Raises if any target barcode is missing.
    """
    col_index = {sid: i for i, sid in enumerate(sample_ids)}
    missing = [b for b in target_barcodes if b not in col_index]
    if missing:
        raise ValueError(
            f"{len(missing)} target barcodes not found in matrix columns, "
            f"e.g. {missing[:5]}"
        )
    col_order = [col_index[b] for b in target_barcodes]
    return gene_names, matrix[:, col_order]


def remove_duplicate_genes(
    gene_names: List[str],
    matrix: np.ndarray,
) -> Tuple[List[str], np.ndarray]:
    """Remove duplicate gene names, keeping the row with highest variance.

    Parameters
    ----------
    matrix : ndarray, shape (n_genes, n_samples)
    """
    if len(gene_names) == len(set(gene_names)):
        logger.info("remove_duplicate_genes: no duplicates found")
        return gene_names, matrix

    # Build a map: gene_name -> list of row indices
    name_to_rows: dict[str, list[int]] = {}
    for i, g in enumerate(gene_names):
        name_to_rows.setdefault(g, []).append(i)

    keep_indices: list[int] = []
    n_dups = 0
    for name, rows in name_to_rows.items():
        if len(rows) == 1:
            keep_indices.append(rows[0])
        else:
            n_dups += len(rows) - 1
            variances = [float(np.var(matrix[r, :])) for r in rows]
            best = rows[int(np.argmax(variances))]
            keep_indices.append(best)

    keep_indices.sort()
    new_genes = [gene_names[i] for i in keep_indices]
    new_matrix = matrix[keep_indices, :]
    logger.info(
        "remove_duplicate_genes: dropped %d duplicate rows (%d -> %d genes)",
        n_dups, len(gene_names), len(new_genes),
    )
    return new_genes, new_matrix


def filter_near_zero_variance(
    gene_names: List[str],
    matrix: np.ndarray,
    threshold: float = 0.01,
) -> Tuple[List[str], np.ndarray]:
    """Remove genes whose variance across samples is below *threshold*.

    Parameters
    ----------
    matrix : ndarray, shape (n_genes, n_samples)
    """
    variances = np.var(matrix, axis=1)
    mask = variances >= threshold
    new_genes = [g for g, m in zip(gene_names, mask) if m]
    new_matrix = matrix[mask, :]
    logger.info(
        "filter_near_zero_variance (threshold=%.4f): %d -> %d genes",
        threshold, len(gene_names), len(new_genes),
    )
    return new_genes, new_matrix


def select_top_variable_genes(
    gene_names: List[str],
    matrix: np.ndarray,
    n_top: int,
) -> Tuple[List[str], np.ndarray]:
    """Retain the *n_top* highest-variance genes.

    Parameters
    ----------
    matrix : ndarray, shape (n_genes, n_samples)
    """
    n_genes = matrix.shape[0]
    if n_genes <= n_top:
        logger.info(
            "select_top_variable_genes: n_genes=%d <= n_top=%d, keeping all",
            n_genes, n_top,
        )
        return gene_names, matrix

    variances = np.var(matrix, axis=1)
    top_idx = np.argsort(variances)[-n_top:]
    top_idx.sort()  # preserve original gene order

    new_genes = [gene_names[i] for i in top_idx]
    new_matrix = matrix[top_idx, :]
    logger.info(
        "select_top_variable_genes: %d -> %d genes (min var kept: %.4f)",
        n_genes, len(new_genes), float(variances[top_idx[0]]),
    )
    return new_genes, new_matrix


def zscore_normalize(
    matrix_sxg: np.ndarray,
    epsilon: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score standardise a samples × genes matrix.

    Returns ``(z_matrix, means, stds)`` where means and stds are per-gene
    vectors of shape ``(n_genes,)``.
    """
    means = np.mean(matrix_sxg, axis=0)
    stds = np.std(matrix_sxg, axis=0)
    stds[stds < epsilon] = 1.0  # avoid division by zero
    z_matrix = (matrix_sxg - means) / stds
    logger.info(
        "zscore_normalize: %d samples x %d genes  (mean of means=%.4f)",
        z_matrix.shape[0], z_matrix.shape[1], float(np.mean(means)),
    )
    return z_matrix, means, stds


# =====================================================================
# Raw counts processing
# =====================================================================

def filter_low_expression(
    gene_names: List[str],
    matrix: np.ndarray,
    min_count: int = 10,
    min_samples: int = 10,
) -> Tuple[List[str], np.ndarray]:
    """Remove genes with low expression from a counts matrix.

    Keeps genes where count >= *min_count* in at least *min_samples* samples.

    Parameters
    ----------
    matrix : ndarray, shape (n_genes, n_samples)  — raw integer counts.
    """
    n_pass = np.sum(matrix >= min_count, axis=1)
    mask = n_pass >= min_samples
    new_genes = [g for g, m in zip(gene_names, mask) if m]
    new_matrix = matrix[mask, :]
    logger.info(
        "filter_low_expression (count>=%d in >=%d samples): %d -> %d genes",
        min_count, min_samples, len(gene_names), len(new_genes),
    )
    return new_genes, new_matrix


def compute_library_sizes(matrix: np.ndarray) -> np.ndarray:
    """Return per-sample total counts (column sums) for QC.

    Parameters
    ----------
    matrix : ndarray, shape (n_genes, n_samples)
    """
    return np.sum(matrix, axis=0)


# =====================================================================
# Environment & confounder encoding
# =====================================================================

_SMOKER_LABELS = {
    "current smoker", "current reformed smoker for < or = 15 yrs",
    "current reformed smoker for > 15 yrs",
}
_NON_SMOKER_LABELS = {"lifelong non-smoker"}

_ASIA_COUNTRIES = {"vietnam", "china", "japan", "south_korea", "korea",
                   "taiwan", "thailand", "india", "singapore", "philippines"}

# Stage classification is order-sensitive: check *late* prefixes first
# because every Roman numeral stage starts with "i", so a naive prefix
# match against ("i", "ia", ...) would swallow IIIA/IIIB/IV as "early".
_LATE_STAGE_PREFIXES = ("iv", "iii")  # IV, IVA, IVB, III, IIIA, IIIB, IIIC
_EARLY_STAGE_PREFIXES = ("ii", "i")   # II, IIA, IIB, I, IA, IB


def _safe_lower(v: str | None) -> str:
    """Lower-case a string, treating None / 'NA' / '' as empty."""
    if v is None:
        return ""
    v = v.strip()
    if v.upper() in ("NA", "NOT REPORTED", "UNKNOWN", ""):
        return ""
    return v.lower()


def assign_environment_strata(
    records: List[Dict[str, str]],
    env_names: Sequence[str] = ("smoking", "sex", "histology", "country", "stage"),
) -> None:
    """Add binary environment columns to each record *in place*.

    For each environment the column is ``env_<name>`` with values
    ``"0"`` / ``"1"`` / ``""`` (missing).
    """
    for rec in records:
        for env in env_names:
            rec[f"env_{env}"] = _compute_env(rec, env)

    # Propagate patient-level env values to paired normal samples.
    # Normal samples lack tumor-specific TCGA metadata (histology, stage, country)
    # but share patient_barcode — inherit values from the paired tumor record.
    patient_envs: Dict[str, Dict[str, str]] = {}
    for rec in records:
        pid = rec.get("patient_barcode", "")
        if not pid:
            continue
        for env in env_names:
            col = f"env_{env}"
            v = rec.get(col, "")
            if v:
                patient_envs.setdefault(pid, {})[col] = v

    propagated = 0
    for rec in records:
        pid = rec.get("patient_barcode", "")
        p_envs = patient_envs.get(pid, {})
        for col, v in p_envs.items():
            if not rec.get(col, ""):
                rec[col] = v
                propagated += 1
    if propagated:
        logger.info("  Propagated %d missing env values from paired tumor samples", propagated)

    # Log distribution
    for env in env_names:
        col = f"env_{env}"
        ctr = Counter(r[col] for r in records)
        logger.info("  env %-12s  0=%d  1=%d  missing=%d",
                     env, ctr.get("0", 0), ctr.get("1", 0), ctr.get("", 0))


def _compute_env(rec: Dict[str, str], env: str) -> str:
    """Return ``"0"`` / ``"1"`` / ``""`` for one environment assignment."""
    if env == "smoking":
        v = _safe_lower(rec.get("tobacco_smoking_status", ""))
        if v in _SMOKER_LABELS:
            return "1"
        if v in _NON_SMOKER_LABELS:
            return "0"
        return ""

    if env == "sex":
        v = _safe_lower(rec.get("gender", ""))
        if v == "male":
            return "1"
        if v == "female":
            return "0"
        return ""

    if env == "histology":
        v = _safe_lower(rec.get("paper_Histological.Type", ""))
        if v == "escc":
            return "1"
        if v == "ac":
            return "0"
        return ""

    if env == "country":
        v = _safe_lower(rec.get("paper_Country",
                                rec.get("country_of_residence_at_enrollment", "")))
        if not v:
            return ""
        return "1" if v in _ASIA_COUNTRIES else "0"

    if env == "stage":
        v = _safe_lower(rec.get("paper_Pathologic.stage",
                                rec.get("ajcc_pathologic_stage", "")))
        if not v:
            return ""
        # Normalise: strip "stage " prefix and any leading non-i characters
        # (handles "Stage IIA", "iIIA", etc.)
        v = v.replace("stage ", "").lstrip()
        while v and v[0] not in ("i", "v"):
            v = v[1:]
        # Order matters: match late prefixes first (IV, III) so naive "i"
        # doesn't swallow III / IV as early.
        for p in _LATE_STAGE_PREFIXES:
            if v.startswith(p):
                return "1"
        for p in _EARLY_STAGE_PREFIXES:
            if v.startswith(p):
                return "0"
        return ""

    logger.warning("Unknown environment name: %s", env)
    return ""


def encode_confounders(
    records: List[Dict[str, str]],
    columns: Sequence[str],
    min_levels: int = 2,
) -> Dict[str, Dict[str, int]]:
    """Integer-encode categorical columns in-place.

    For each column, adds ``<column>_enc`` with the integer code.
    Returns a dict mapping ``column -> {level: code}``.
    Columns with fewer than *min_levels* non-missing values are skipped.
    """
    mappings: dict[str, dict[str, int]] = {}
    for col in columns:
        vals = [_safe_lower(r.get(col, "")) for r in records]
        levels = sorted(set(v for v in vals if v))
        if len(levels) < min_levels:
            logger.info("encode_confounders: skipping '%s' (%d levels)", col, len(levels))
            for r in records:
                r[f"{col}_enc"] = ""
            continue
        level_map = {lv: i for i, lv in enumerate(levels)}
        mappings[col] = level_map
        for r, v in zip(records, vals):
            r[f"{col}_enc"] = str(level_map[v]) if v in level_map else ""
        logger.info("encode_confounders: '%s' -> %d levels", col, len(levels))

    return mappings


# =====================================================================
# Master sample table
# =====================================================================

# Columns to extract into the clean master table (source -> target rename)
_MASTER_COLUMNS: list[tuple[str, str]] = [
    ("barcode", "sample_barcode"),
    ("patient_barcode", "patient_barcode"),
    ("sample_type_code", "sample_type_code"),
    ("label", "label"),
    ("label_int", "label_int"),
    ("age_at_index", "age"),
    ("gender", "sex"),
    ("race", "race"),
    ("paper_Country", "country"),
    ("tobacco_smoking_status", "smoking_status"),
    ("alcohol_history", "alcohol_history"),
    ("paper_Histological.Type", "histology"),
    ("ajcc_pathologic_stage", "stage_ajcc"),
    ("paper_Pathologic.stage", "stage_paper"),
    ("vital_status", "vital_status"),
    ("days_to_death", "days_to_death"),
    ("days_to_last_follow_up", "days_to_last_follow_up"),
    ("residual_disease", "residual_disease"),
]


def build_master_sample_table(
    records: List[Dict[str, str]],
    env_names: Sequence[str] = ("smoking", "sex", "histology", "country", "stage"),
) -> List[Dict[str, str]]:
    """Construct the lean master sample table from enriched metadata records.

    Picks essential clinical/demographic fields plus environment strata.
    """
    master: list[dict[str, str]] = []
    for rec in records:
        row: dict[str, str] = {}
        for src, tgt in _MASTER_COLUMNS:
            row[tgt] = rec.get(src, "")
        for env in env_names:
            row[f"env_{env}"] = rec.get(f"env_{env}", "")
        master.append(row)
    logger.info("build_master_sample_table: %d rows, %d columns",
                len(master), len(master[0]) if master else 0)
    return master


# =====================================================================
# Cross-validation fold assignment
# =====================================================================

def build_patient_grouped_folds(
    records: List[Dict[str, str]],
    n_outer: int = 5,
    n_inner: int = 3,
    seed: int = 2026,
) -> List[Dict[str, str]]:
    """Assign patient-grouped, stratified outer (and inner) CV folds.

    * Patients that contribute *Normal* samples are spread across folds
      first (round-robin after shuffle) so every fold has normals.
    * Remaining tumour-only patients are then shuffled and distributed
      to balance fold sizes.
    * Inner folds are assigned *within each outer-fold training set*
      using the same grouped logic (for step 4 hyperparameter tuning).

    Returns a list of dicts with keys:
        ``sample_barcode``, ``patient_barcode``, ``outer_fold``
    """
    rng = random.Random(seed)

    # Collect patients and whether they have normals
    patient_has_normal: dict[str, bool] = {}
    patient_samples: dict[str, list[str]] = {}
    for rec in records:
        pat = rec["patient_barcode"]
        bc = rec.get("barcode", rec.get("sample_barcode", ""))
        patient_samples.setdefault(pat, []).append(bc)
        if rec.get("label") == "Normal":
            patient_has_normal[pat] = True
        elif pat not in patient_has_normal:
            patient_has_normal[pat] = False

    normal_patients = sorted(p for p, has in patient_has_normal.items() if has)
    tumor_only_patients = sorted(p for p, has in patient_has_normal.items() if not has)

    rng.shuffle(normal_patients)
    rng.shuffle(tumor_only_patients)

    # Round-robin normal patients across folds, then fill with tumour-only
    fold_assignment: dict[str, int] = {}
    for i, pat in enumerate(normal_patients):
        fold_assignment[pat] = i % n_outer
    for i, pat in enumerate(tumor_only_patients):
        # Pick the fold with fewest patients so far
        fold_sizes = Counter(fold_assignment.values())
        for k in range(n_outer):
            fold_sizes.setdefault(k, 0)
        target_fold = min(range(n_outer), key=lambda f: fold_sizes[f])
        fold_assignment[pat] = target_fold

    # Inner folds (within each outer training set)
    inner_assignments: dict[str, dict[int, int]] = {}  # pat -> {outer_fold -> inner_fold}
    for outer in range(n_outer):
        # Training patients for this outer fold
        train_pats_norm = [p for p in normal_patients if fold_assignment[p] != outer]
        train_pats_tumor = [p for p in tumor_only_patients if fold_assignment[p] != outer]
        rng_inner = random.Random(seed + outer + 1)
        rng_inner.shuffle(train_pats_norm)
        rng_inner.shuffle(train_pats_tumor)
        # Same logic: spread normals first, then tumour-only
        inner_map: dict[str, int] = {}
        for i, pat in enumerate(train_pats_norm):
            inner_map[pat] = i % n_inner
        for i, pat in enumerate(train_pats_tumor):
            inner_sizes = Counter(inner_map.values())
            for k in range(n_inner):
                inner_sizes.setdefault(k, 0)
            target = min(range(n_inner), key=lambda f: inner_sizes[f])
            inner_map[pat] = target
        for pat, ifold in inner_map.items():
            inner_assignments.setdefault(pat, {})[outer] = ifold

    # Build output records
    fold_records: list[dict[str, str]] = []
    for rec in records:
        pat = rec["patient_barcode"]
        bc = rec.get("barcode", rec.get("sample_barcode", ""))
        outer = fold_assignment[pat]
        row: dict[str, str] = {
            "sample_barcode": bc,
            "patient_barcode": pat,
            "outer_fold": str(outer),
        }
        for o in range(n_outer):
            key = f"inner_fold_outer{o}"
            inner_val = inner_assignments.get(pat, {}).get(o)
            row[key] = str(inner_val) if inner_val is not None else ""
        fold_records.append(row)

    # Log fold distribution
    for f in range(n_outer):
        pats_in_fold = [p for p, af in fold_assignment.items() if af == f]
        n_t = sum(1 for r in records
                  if r["patient_barcode"] in set(pats_in_fold) and r.get("label") == "Tumor")
        n_n = sum(1 for r in records
                  if r["patient_barcode"] in set(pats_in_fold) and r.get("label") == "Normal")
        logger.info("  outer fold %d: %d patients  (%d T, %d N)",
                     f, len(pats_in_fold), n_t, n_n)

    return fold_records


# =====================================================================
# Phase 1 summary
# =====================================================================

def build_phase1_summary(
    *,
    n_patients: int,
    n_tumor: int,
    n_normal: int,
    n_genes_vst_raw: int,
    n_genes_after_dedup: int,
    n_genes_after_var_filter: int,
    n_genes_selected: int,
    n_genes_counts_raw: int,
    n_genes_counts_filtered: int,
    n_outer_folds: int,
    n_inner_folds: int,
    seed: int,
    env_distributions: Dict[str, Dict[str, int]],
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Assemble the phase-1 reproducibility summary."""
    summary: dict[str, Any] = {
        "phase": "I",
        "step": "step2_build_cohort",
        "cohort": {
            "n_patients": n_patients,
            "n_tumor": n_tumor,
            "n_normal": n_normal,
            "n_total_samples": n_tumor + n_normal,
        },
        "gene_filtering": {
            "vst_raw": n_genes_vst_raw,
            "after_dedup": n_genes_after_dedup,
            "after_variance_filter": n_genes_after_var_filter,
            "selected_top_variable": n_genes_selected,
            "counts_raw": n_genes_counts_raw,
            "counts_after_low_expression_filter": n_genes_counts_filtered,
        },
        "cross_validation": {
            "n_outer_folds": n_outer_folds,
            "n_inner_folds": n_inner_folds,
            "seed": seed,
        },
        "environment_distributions": env_distributions,
    }
    if extra:
        summary.update(extra)
    return summary


# =====================================================================
# Optional figures (graceful degradation when matplotlib unavailable)
# =====================================================================

def _has_matplotlib() -> bool:
    try:
        import matplotlib  # noqa: F401
        return True
    except ImportError:
        return False


def generate_cohort_figures(
    master_records: List[Dict[str, str]],
    output_dir: str | Path,
    style: Any = None,
    formats: Sequence[str] = ("svg",),
) -> Tuple[List[str], List[Tuple[str, str]]]:
    """Generate Phase-I QC figures if matplotlib is available.

    Produces:
      D1_cohort_composition  — tumor/normal bar + histology donut
      D2_clinical_demographics — 2×3 grid (age, sex, country, smoking, stage, vital)
      fig_environment_strata — binary env columns bar chart

    Returns ``(generated_names, skipped_name_reason_pairs)``.
    """
    generated: list[str] = []
    skipped: list[tuple[str, str]] = []

    fig_names = ["D1_cohort_composition", "D2_clinical_demographics", "fig_environment_strata"]
    if not _has_matplotlib():
        for n in fig_names:
            skipped.append((n, "matplotlib not installed"))
        return generated, skipped

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from collections import defaultdict as _dd
        from .publication_style import (
            apply_style, save_figure, semantic_color, categorical_colors,
        )
        if style is not None:
            apply_style(style)

        fig_dir = Path(output_dir) / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)

        tumors  = [r for r in master_records if r.get("label", "").lower() == "tumor"]
        normals = [r for r in master_records if r.get("label", "").lower() == "normal"]

        # ── D1: cohort composition ─────────────────────────────────────────
        try:
            hist_counts: dict[str, int] = _dd(int)
            for r in tumors:
                h = r.get("histology") or "Unknown"
                hist_counts[h.strip() or "Unknown"] += 1

            fig, axes = plt.subplots(1, 2, figsize=(9, 4))

            # left: tumor/normal bar
            ax = axes[0]
            n_t, n_n = len(tumors), len(normals)
            bars = ax.bar(
                ["Tumor", "Normal"], [n_t, n_n],
                color=[semantic_color("tumor"), semantic_color("normal")],
                edgecolor="black", linewidth=0.8, width=0.5,
            )
            for bar, val in zip(bars, [n_t, n_n]):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        str(val), ha="center", va="bottom", fontsize=10, fontweight="bold")
            ax.set_ylabel("Samples")
            ax.set_title("Sample Type Distribution")
            ax.set_ylim(0, max(n_t, n_n) * 1.15)

            # right: histology donut (tumor only)
            ax2 = axes[1]
            h_labels = sorted(hist_counts, key=lambda k: -hist_counts[k])
            h_vals   = [hist_counts[k] for k in h_labels]
            colors_h = categorical_colors(max(len(h_labels), 2))[:len(h_labels)]
            if h_vals:
                wedges, _, autotexts = ax2.pie(
                    h_vals, labels=h_labels, colors=colors_h,
                    autopct=lambda p: f"{p:.0f}%" if p > 3 else "",
                    wedgeprops=dict(width=0.45, edgecolor="white", linewidth=1.2),
                    startangle=90, textprops=dict(fontsize=8),
                )
                for at in autotexts:
                    at.set_fontsize(8)
                ax2.text(0, 0, "Tumor\nhistology", ha="center", va="center",
                         fontsize=9, fontweight="bold")
            ax2.set_title("Histology (Tumor Samples)")

            fig.suptitle(
                f"CAGE Cohort — n = {len(master_records)} samples",
                fontsize=12, fontweight="bold",
            )
            fig.tight_layout()
            paths = save_figure(fig, fig_dir / "D1_cohort_composition",
                                style=style, formats=formats)
            if paths:
                generated.append("D1_cohort_composition")
        except Exception as exc:
            skipped.append(("D1_cohort_composition", str(exc)))

        # ── D2: clinical demographics 2×3 grid ────────────────────────────
        try:
            def _counts(field: str, rows: list) -> dict:
                c: dict[str, int] = _dd(int)
                for r in rows:
                    v = (r.get(field) or "NA").strip() or "NA"
                    c[v] += 1
                return dict(c)

            age_vals = []
            for r in tumors:
                try:
                    age_vals.append(float(r.get("age") or ""))
                except (ValueError, TypeError):
                    pass

            sex_c     = _counts("sex", tumors)
            country_c = _counts("country", tumors)
            stage_c   = _counts("stage_paper", tumors)
            vital_c   = _counts("vital_status", tumors)
            smk       = {"Never/Unknown": 0, "Smoker": 0}
            for r in tumors:
                smk["Smoker" if r.get("env_smoking") == "1" else "Never/Unknown"] += 1

            pal = categorical_colors(6)
            fig, axes = plt.subplots(2, 3, figsize=(13, 8))

            # (0,0) Age histogram
            ax = axes[0, 0]
            if age_vals:
                ax.hist(age_vals, bins=15, color=pal[0], edgecolor="white", alpha=0.85)
                med = float(np.median(age_vals))
                ax.axvline(med, color="#d62728", linewidth=1.5, linestyle="--",
                           label=f"Median {med:.0f}")
                ax.legend(fontsize=8)
            ax.set_xlabel("Age at diagnosis")
            ax.set_ylabel("Count")
            ax.set_title("Age Distribution (Tumor)")

            # (0,1) Sex
            ax = axes[0, 1]
            sk = sorted(sex_c, key=lambda x: -sex_c[x])
            ax.bar(sk, [sex_c[k] for k in sk], color=pal[1], edgecolor="white", width=0.5)
            for i, k in enumerate(sk):
                ax.text(i, sex_c[k] + 0.3, str(sex_c[k]), ha="center", fontsize=9)
            ax.set_title("Sex (Tumor)")
            ax.set_ylabel("Count")

            # (0,2) Country
            ax = axes[0, 2]
            ck = sorted(country_c, key=lambda x: -country_c[x])[:12]
            ax.barh(ck, [country_c[k] for k in ck], color=pal[2], edgecolor="white")
            for i, k in enumerate(ck):
                ax.text(country_c[k] + 0.1, i, str(country_c[k]), va="center", fontsize=8)
            ax.set_title("Country (Tumor)")
            ax.set_xlabel("Count")

            # (1,0) Smoking
            ax = axes[1, 0]
            ax.bar(list(smk), list(smk.values()), color=[pal[3], "#d62728"],
                   edgecolor="white", width=0.5)
            for i, (k, v) in enumerate(smk.items()):
                ax.text(i, v + 0.3, str(v), ha="center", fontsize=9)
            ax.set_title("Smoking Status (Tumor)")
            ax.set_ylabel("Count")

            # (1,1) Stage
            ax = axes[1, 1]
            order = ["I", "IA", "IB", "II", "IIA", "IIB", "III", "IIIA", "IIIB", "IIIC", "IV", "NA"]
            all_keys = [k for k in order if k in stage_c] + \
                       sorted(k for k in stage_c if k not in order)
            vals = [stage_c.get(k, 0) for k in all_keys]
            ax.bar(all_keys, vals, color=pal[4], edgecolor="white")
            ax.set_xticklabels(all_keys, rotation=35, ha="right", fontsize=8)
            ax.set_title("AJCC Stage (Tumor)")
            ax.set_ylabel("Count")

            # (1,2) Vital status
            ax = axes[1, 2]
            vk = sorted(vital_c, key=lambda x: -vital_c[x])
            ax.bar(vk, [vital_c[k] for k in vk], color=pal[5], edgecolor="white", width=0.5)
            for i, k in enumerate(vk):
                ax.text(i, vital_c[k] + 0.3, str(vital_c[k]), ha="center", fontsize=9)
            ax.set_title("Vital Status (Tumor)")
            ax.set_ylabel("Count")

            fig.suptitle("Clinical Demographics — Tumor Samples", fontsize=12, fontweight="bold")
            fig.tight_layout()
            paths = save_figure(fig, fig_dir / "D2_clinical_demographics",
                                style=style, formats=formats)
            if paths:
                generated.append("D2_clinical_demographics")
        except Exception as exc:
            skipped.append(("D2_clinical_demographics", str(exc)))

        # ── Environment strata QC ──────────────────────────────────────────
        try:
            if master_records:
                env_cols = [c for c in master_records[0] if c.startswith("env_")]
                n_envs = len(env_cols)
                if n_envs > 0:
                    fig, axes = plt.subplots(1, n_envs, figsize=(2.2 * n_envs, 2.8))
                    if n_envs == 1:
                        axes = [axes]
                    palette = categorical_colors(3)
                    for ax, col in zip(axes, env_cols):
                        ctr = Counter(r[col] for r in master_records)
                        env_labels = {"0": "Group 0", "1": "Group 1", "": "Missing"}
                        cats = ["0", "1", ""]
                        vals = [ctr.get(c, 0) for c in cats]
                        ax.bar([env_labels[c] for c in cats], vals,
                               color=[palette[0], palette[1], "#CCCCCC"],
                               edgecolor="black", linewidth=0.6)
                        ax.set_title(col.replace("env_", "").capitalize(), fontsize=9)
                        ax.tick_params(axis="x", rotation=30, labelsize=7)
                    fig.suptitle("Environment Strata", fontsize=11, fontweight="bold")
                    fig.tight_layout(rect=[0, 0, 1, 0.92])
                    paths = save_figure(fig, fig_dir / "fig_environment_strata",
                                        style=style, formats=formats)
                    if paths:
                        generated.append("fig_environment_strata")
        except Exception as exc:
            skipped.append(("fig_environment_strata", str(exc)))

    except Exception as exc:
        skipped.append(("all_figures", str(exc)))

    return generated, skipped


def generate_gene_filtering_figure(
    filter_counts: Dict[str, Any],
    output_dir: str | Path,
    style: Any = None,
    formats: Sequence[str] = ("svg",),
) -> Tuple[bool, str]:
    """Render a horizontal-funnel figure of gene counts at each filtering stage.

    Parameters
    ----------
    filter_counts:
        Dict with keys: vst_raw, after_dedup, after_var_filter, selected,
        counts_raw, counts_filtered, n_samples.
    output_dir:
        Directory under which ``figures/`` is created.
    style, formats:
        Publication style and output formats (passed through to save_figure).

    Returns ``(success: bool, reason_if_skipped: str)``.
    """
    if not _has_matplotlib():
        return False, "matplotlib not installed"

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from .publication_style import apply_style, save_figure

        stages = [
            ("VST raw", filter_counts.get("vst_raw", 0)),
            ("After dedup", filter_counts.get("after_dedup", 0)),
            ("After variance\nfilter", filter_counts.get("after_var_filter", 0)),
            ("Selected for\nmodelling", filter_counts.get("selected", 0)),
        ]
        labels = [s[0] for s in stages]
        values = [s[1] for s in stages]
        max_val = max(values) if values else 1

        # Gradient blue → teal to show progressive reduction
        bar_colors = ["#4575b4", "#74add1", "#abd9e9", "#2ca25f"]

        fig, ax = plt.subplots(figsize=(8, 4))
        if style is not None:
            apply_style(fig, style)

        bars = ax.barh(
            labels[::-1], values[::-1],
            color=bar_colors[::-1],
            edgecolor="black", linewidth=0.7,
            height=0.55,
        )
        for bar, val in zip(bars, values[::-1]):
            pct = 100 * val / max_val
            ax.text(
                val + max_val * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:,}  ({pct:.0f}%)",
                va="center", ha="left", fontsize=9,
            )

        ax.set_xlabel("Number of genes")
        ax.set_xlim(0, max_val * 1.25)
        ax.set_title(
            f"Gene Filtering Funnel — CAGE Step 2\n"
            f"({filter_counts.get('n_samples', '?')} samples)",
            fontweight="bold",
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()

        fig_dir = Path(output_dir) / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        saved = save_figure(
            fig, fig_dir / "fig_gene_filtering_funnel",
            style=style, formats=list(formats),
        )
        return bool(saved), ""
    except Exception as exc:
        return False, str(exc)
