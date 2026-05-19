"""CAGE Step 8 — Agilent GEO external validation.

Targeted external validation of CAGE candidate driver genes using the
two large Agilent lncRNA+mRNA ESCC cohorts (GSE53624, GSE53625).

GPL18109 was submitted to GEO without HGNC gene symbols. This module resolves
gene symbols by exact 60-mer probe-sequence matching against NCBI RefSeq mRNAs.

Accessible via the Step 8 unified CLI:

    python -m cage.step8_external_validation_and_release validate-agilent \\
        --geo-dir outputs/step8_geo_prepared \\
        --step5-dir outputs/step5_cdps \\
        --step6-dir outputs/step6_validation \\
        --output-dir outputs/step8_agilent_validation

Can also be called as a standalone module:

    python -m cage.step8_agilent_validation \\
        --geo-dir outputs/step8_geo_prepared \\
        --step5-dir outputs/step5_cdps \\
        --step6-dir outputs/step6_validation \\
        --output-dir outputs/step8_agilent_validation

Outputs (in --output-dir)
-------------------------
annotation/probe_gene_map.csv, annotation_summary.json, refseq_cache/
per_dataset/{GSE}/de_replication.csv, model_metrics.json, dataset_summary.json
combined_de_replication.csv, concordance_summary.csv, model_summary.csv
analysis_report.txt, validation_summary.json
figures/  (A series)
logs/step8_agilent_validation.log
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import logging
import math
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
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
    handlers: list = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(log_file), mode="w", encoding="utf-8"),
    ]
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format=fmt,
        handlers=handlers,
        force=True,
    )


# ---------------------------------------------------------------------------
# CSV / JSON helpers
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
    logging.info("Wrote %d rows -> %s", len(rows), path.name)


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def _default(v: Any) -> Any:
        if isinstance(v, float) and not math.isfinite(v):
            return None
        return str(v)

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=_default)
    logging.info("Wrote JSON -> %s", path.name)


def _fmt(v: float, d: int = 4) -> str:
    try:
        return f"{float(v):.{d}f}"
    except (ValueError, TypeError):
        return str(v)


# ---------------------------------------------------------------------------
# Step 1: Extract probe sequences from SOFT.gz
# ---------------------------------------------------------------------------

def _revcomp(seq: str) -> str:
    comp = str.maketrans("ACGTacgt", "TGCAtgca")
    return seq.translate(comp)[::-1]


def load_probe_sequences(soft_gz: Path) -> Dict[str, str]:
    """Parse GPL18109 table from a GSE SOFT.gz -> {feature_id: sequence}."""
    result: Dict[str, str] = {}
    in_platform = False
    in_table = False
    col_id = col_seq = None

    logging.info("Extracting probe sequences from %s ...", soft_gz.name)
    with gzip.open(str(soft_gz), "rt", encoding="latin-1", errors="replace") as fh:
        for line in fh:
            line = line.rstrip()
            if "^PLATFORM" in line:
                in_platform = True
            if not in_platform:
                continue
            if line == "!platform_table_begin":
                in_table = True
                continue
            if line == "!platform_table_end":
                break
            if not in_table:
                continue

            parts = line.split("\t")
            if col_id is None:
                # header row
                hdr = [p.strip().upper() for p in parts]
                col_id  = next((i for i, h in enumerate(hdr) if h == "ID"),       None)
                col_seq = next((i for i, h in enumerate(hdr) if h == "SEQUENCE"), None)
                continue

            if col_id is None or col_seq is None:
                continue
            if len(parts) <= max(col_id, col_seq):
                continue

            fid = parts[col_id].strip()
            seq = parts[col_seq].strip().upper()
            if fid and seq:
                result[fid] = seq

    logging.info("Loaded %d probe sequences", len(result))
    return result


# ---------------------------------------------------------------------------
# Step 2: Fetch RefSeq mRNA sequences from NCBI
# ---------------------------------------------------------------------------

_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def _ncbi_get(url: str, max_retries: int = 3, pause: float = 0.34) -> bytes:
    for attempt in range(max_retries):
        try:
            time.sleep(pause)
            with urllib.request.urlopen(url, timeout=30) as resp:
                return resp.read()
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            logging.debug("NCBI retry %d/%d (%s): %s", attempt + 1, max_retries, url[:80], exc)
            time.sleep(2 ** attempt * pause)
    raise RuntimeError(f"NCBI request failed after {max_retries} retries: {url[:80]}")


def _search_nm_accessions(gene: str) -> List[str]:
    """Return NM_ accession IDs for a gene (Homo sapiens RefSeq mRNA)."""
    term = urllib.parse.quote(
        f"{gene}[Gene Name] AND Homo sapiens[Organism] AND mRNA[FILT] AND NM_[ACCN]"
    )
    url = f"{_EUTILS}/esearch.fcgi?db=nucleotide&term={term}&retmax=10&retmode=xml"
    try:
        xml_bytes = _ncbi_get(url)
        tree = ET.fromstring(xml_bytes)
        ids = [e.text for e in tree.findall(".//Id") if e.text]
        return ids
    except Exception as exc:
        logging.debug("esearch failed for %s: %s", gene, exc)
        return []


def _search_nr_accessions(gene: str) -> List[str]:
    """Return NR_ (non-coding RefSeq RNA) accession IDs for a gene.

    Used as a fallback for lncRNAs (NR_ prefix) and pseudogenes that have no
    NM_ mRNA but do have annotated transcripts in RefSeq.
    """
    term = urllib.parse.quote(
        f"{gene}[Gene Name] AND Homo sapiens[Organism] AND NR_[ACCN]"
    )
    url = f"{_EUTILS}/esearch.fcgi?db=nucleotide&term={term}&retmax=10&retmode=xml"
    try:
        xml_bytes = _ncbi_get(url)
        tree = ET.fromstring(xml_bytes)
        ids = [e.text for e in tree.findall(".//Id") if e.text]
        return ids
    except Exception as exc:
        logging.debug("NR_ esearch failed for %s: %s", gene, exc)
        return []


def _fetch_fasta_sequence(ncbi_id: str) -> str:
    """Fetch FASTA sequence for an NCBI nucleotide ID."""
    url = (f"{_EUTILS}/efetch.fcgi?db=nuccore&id={ncbi_id}"
           f"&rettype=fasta&retmode=text")
    try:
        raw = _ncbi_get(url).decode("utf-8", errors="replace")
        lines = raw.splitlines()
        seq_lines = [l for l in lines if l and not l.startswith(">")]
        return "".join(seq_lines).upper()
    except Exception as exc:
        logging.debug("efetch failed for %s: %s", ncbi_id, exc)
        return ""


def fetch_refseq_sequence(gene: str, cache_dir: Path) -> str:
    """Fetch (and cache) RefSeq sequence for *gene*.

    Search order:
      1. NM_ (RefSeq mRNA) — protein-coding genes
      2. Relaxed mRNA filter — catches some coding genes under alternate names
      3. NR_ (RefSeq non-coding RNA) — lncRNAs, pseudogene transcripts
    Returns empty string if no sequence is found (e.g. miRNAs too short for
    60-mer probes, or genes with no deposited RefSeq transcript).
    """
    cache_file = cache_dir / f"{gene}.fasta.txt"
    if cache_file.is_file() and cache_file.stat().st_size > 0:
        return cache_file.read_text(encoding="utf-8").strip()

    logging.debug("Fetching RefSeq sequence for %s ...", gene)
    ncbi_ids = _search_nm_accessions(gene)

    if not ncbi_ids:
        # Fallback 1: relax mRNA filter (catches genes where the name search is
        # slightly ambiguous but still returns the right mRNA)
        term = urllib.parse.quote(
            f"{gene}[Gene Name] AND Homo sapiens[Organism] AND mRNA[FILT]"
        )
        url = f"{_EUTILS}/esearch.fcgi?db=nucleotide&term={term}&retmax=5&retmode=xml"
        try:
            xml_bytes = _ncbi_get(url)
            tree = ET.fromstring(xml_bytes)
            ncbi_ids = [e.text for e in tree.findall(".//Id") if e.text]
        except Exception:
            pass

    if not ncbi_ids:
        # Fallback 2: non-coding RefSeq (NR_) — lncRNAs and pseudogene transcripts
        # that have probe representation on lncRNA+mRNA Agilent arrays.
        ncbi_ids = _search_nr_accessions(gene)
        if ncbi_ids:
            logging.info("%s: no NM_ found; using NR_ (non-coding RefSeq) accessions", gene)

    # Concatenate all isoform sequences (separated by a run of N so we
    # don't accidentally match a probe spanning the join)
    all_seq = ""
    for nid in ncbi_ids[:5]:
        seq = _fetch_fasta_sequence(nid)
        if seq:
            all_seq += seq + "N" * 10

    if all_seq:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(all_seq, encoding="utf-8")

    return all_seq


# ---------------------------------------------------------------------------
# Step 3: Map probes to genes via 60-mer exact matching
# ---------------------------------------------------------------------------

def build_probe_gene_map(
    probe_seqs: Dict[str, str],
    target_genes: List[str],
    cache_dir: Path,
) -> Tuple[Dict[str, str], Dict[str, List[str]], Dict[str, str]]:
    """
    Returns:
      probe_to_gene  : {feature_id -> gene_symbol}
      gene_to_probes : {gene_symbol -> [feature_ids]}
      gene_status    : {gene_symbol -> 'found' | 'no_probes' | 'no_refseq'}
    """
    probe_to_gene:  Dict[str, str]        = {}
    gene_to_probes: Dict[str, List[str]]  = {}
    gene_status:    Dict[str, str]        = {}

    n_genes = len(target_genes)
    for gi, gene in enumerate(target_genes, 1):
        logging.info("[%d/%d] Mapping probes for %s ...", gi, n_genes, gene)
        mrna = fetch_refseq_sequence(gene, cache_dir)
        if not mrna:
            logging.warning("No RefSeq sequence for %s — skipping", gene)
            gene_status[gene] = "no_refseq"
            continue

        matched: List[str] = []
        mrna_rc = _revcomp(mrna)
        for fid, probe in probe_seqs.items():
            if len(probe) < 20:
                continue
            if probe in mrna or probe in mrna_rc:
                matched.append(fid)
                # Only assign if not already claimed by a higher-priority gene
                if fid not in probe_to_gene:
                    probe_to_gene[fid] = gene

        if matched:
            gene_to_probes[gene] = matched
            gene_status[gene] = "found"
            logging.info("  %s: %d matching probes", gene, len(matched))
        else:
            gene_status[gene] = "no_probes"
            logging.debug("  %s: 0 matching probes", gene)

    found_count = sum(1 for s in gene_status.values() if s == "found")
    logging.info(
        "Probe mapping complete: %d/%d target genes have probes",
        found_count, n_genes,
    )
    return probe_to_gene, gene_to_probes, gene_status


# ---------------------------------------------------------------------------
# Step 4: Load probe expression matrix
# ---------------------------------------------------------------------------

def load_probe_matrix(path: Path) -> Tuple[List[str], List[str], np.ndarray]:
    """Load expression_probe_matrix.csv.

    Returns (feature_ids, sample_ids, X) where X.shape = (n_samples, n_probes).
    First column header is 'feature_id'; rest are GSM IDs.
    """
    logging.info("Loading probe matrix from %s ...", path.name)
    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        sample_ids = header[1:]
        feature_ids: List[str] = []
        rows: List[List[float]] = []
        for row in reader:
            fid = row[0].strip()
            if not fid:
                continue
            try:
                vals = [float(v) if v.strip() else float("nan") for v in row[1:]]
            except ValueError:
                continue
            feature_ids.append(fid)
            rows.append(vals)

    X = np.array(rows, dtype=np.float64).T  # (n_samples, n_probes)
    logging.info(
        "Probe matrix: %d samples x %d probes", X.shape[0], X.shape[1]
    )
    return feature_ids, sample_ids, X


# ---------------------------------------------------------------------------
# Step 5: Collapse probes to gene-level matrix
# ---------------------------------------------------------------------------

def collapse_to_gene_matrix(
    feature_ids: List[str],
    sample_ids: List[str],
    X: np.ndarray,
    gene_to_probes: Dict[str, List[str]],
    method: str = "median",
) -> Tuple[List[str], List[str], np.ndarray]:
    """Aggregate matched probes to gene-level expression.

    Returns (gene_names, sample_ids, X_gene) where X_gene.shape=(n_samples, n_genes).
    """
    fid_to_col = {fid: i for i, fid in enumerate(feature_ids)}
    gene_names: List[str] = []
    gene_rows:  List[np.ndarray] = []

    for gene, probes in gene_to_probes.items():
        cols = [fid_to_col[p] for p in probes if p in fid_to_col]
        if not cols:
            continue
        sub = X[:, cols]  # (n_samples, n_matched_probes)
        if method == "mean":
            agg = np.nanmean(sub, axis=1)
        elif method == "max":
            agg = np.nanmax(sub, axis=1)
        else:
            agg = np.nanmedian(sub, axis=1)
        gene_names.append(gene)
        gene_rows.append(agg)

    if not gene_rows:
        return [], sample_ids, np.empty((len(sample_ids), 0))

    X_gene = np.stack(gene_rows, axis=1)  # (n_samples, n_genes)
    logging.info("Gene matrix: %d samples x %d genes", X_gene.shape[0], X_gene.shape[1])
    return gene_names, sample_ids, X_gene


# ---------------------------------------------------------------------------
# Step 6: Metadata loader
# ---------------------------------------------------------------------------

def load_metadata(meta_path: Path) -> Dict[str, str]:
    """gsm_id -> 'Tumor' | 'Normal'."""
    rows = _read_csv(meta_path)
    labels: Dict[str, str] = {}
    for r in rows:
        gsm = r.get("gsm_id", "").strip()
        lbl = r.get("sample_type_inferred", "").strip()
        if gsm and lbl in ("Tumor", "Normal"):
            labels[gsm] = lbl
    return labels


# ---------------------------------------------------------------------------
# Step 7: Z-score normalisation
# ---------------------------------------------------------------------------

def zscore(X: np.ndarray) -> np.ndarray:
    mean = np.nanmean(X, axis=0, keepdims=True)
    std  = np.nanstd(X, axis=0, keepdims=True)
    std  = np.where(std < 1e-8, 1.0, std)
    return (X - mean) / std


# ---------------------------------------------------------------------------
# Step 8: Reference data loaders
# ---------------------------------------------------------------------------

def load_reference_genes(step5_dir: Path, top_k: int) -> List[str]:
    src = step5_dir / f"top{top_k}_genes_cdps.csv"
    if not src.is_file():
        src = step5_dir / "ranked_genes_cdps.csv"
    rows = _read_csv(src)[:top_k]
    genes = [r["gene"] for r in rows if r.get("gene")]
    logging.info("Target CDPS genes: %d (from %s)", len(genes), src.name)
    return genes


def load_tcga_gene_list(step5_dir: Path, step4_dir: Optional[Path] = None) -> List[str]:
    """Load the model feature order for external classifier alignment.

    The deep checkpoints expect the exact Step-4 training feature order. CDPS
    ranking files are sorted by score and must not be used as model-input order.
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
    return genes


def load_reference_de(step6_dir: Path) -> Dict[str, str]:
    rows = _read_csv(step6_dir / "differential_expression_results.csv")
    ref: Dict[str, str] = {}
    for r in rows:
        g  = r.get("gene", "")
        es = r.get("effect_size_norm", "0")
        if g:
            try:
                ref[g] = "up" if float(es) > 0 else "down"
            except ValueError:
                pass
    logging.info("Reference DE directions: %d genes", len(ref))
    return ref


# ---------------------------------------------------------------------------
# Step 9: DE direction concordance
# ---------------------------------------------------------------------------

def run_de_replication(
    accession: str,
    gene_names: List[str],
    X: np.ndarray,
    y: np.ndarray,
    target_genes: List[str],
    ref_directions: Dict[str, str],
    gene_status: Dict[str, str],
) -> List[Dict[str, str]]:
    gene_to_col = {g: i for i, g in enumerate(gene_names)}
    tumor_mask  = y == 1
    normal_mask = y == 0
    rows: List[Dict[str, str]] = []

    for gene in target_genes:
        reason = gene_status.get(gene, "unknown")
        if gene not in gene_to_col:
            rows.append({
                "accession": accession, "gene": gene,
                "ext_n_tumor": str(int(tumor_mask.sum())),
                "ext_n_normal": str(int(normal_mask.sum())),
                "ext_mean_tumor": "", "ext_mean_normal": "",
                "ext_effect": "",
                "ext_direction": f"missing:{reason}",
                "ref_direction": ref_directions.get(gene, ""),
                "concordant": "", "in_dataset": "0",
                "n_probes_matched": "0",
            })
            continue

        gi = gene_to_col[gene]
        t_vals = X[tumor_mask, gi]
        n_vals = X[normal_mask, gi]
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
                "n_probes_matched": str(len(gene_to_col)),
            })
            continue

        diff = float(t_vals.mean() - n_vals.mean())
        ext_dir = "up" if diff > 0 else "down"
        ref_dir = ref_directions.get(gene, "")
        concordant = "1" if ref_dir and ref_dir == ext_dir else (
                     "0" if ref_dir else "")

        rows.append({
            "accession": accession, "gene": gene,
            "ext_n_tumor": str(int(tumor_mask.sum())),
            "ext_n_normal": str(int(normal_mask.sum())),
            "ext_mean_tumor": _fmt(t_vals.mean()),
            "ext_mean_normal": _fmt(n_vals.mean()),
            "ext_effect": _fmt(diff),
            "ext_direction": ext_dir,
            "ref_direction": ref_dir,
            "concordant": concordant,
            "in_dataset": "1",
            "n_probes_matched": "1",
        })

    n_tested = sum(1 for r in rows if r["concordant"] in ("0", "1"))
    n_conc   = sum(1 for r in rows if r["concordant"] == "1")
    logging.info(
        "%s DE replication: %d/%d genes concordant (%.1f%%)",
        accession, n_conc, n_tested, 100 * n_conc / max(n_tested, 1),
    )
    return rows


# ---------------------------------------------------------------------------
# Step 10: Model inference (optional)
# ---------------------------------------------------------------------------

def _auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    try:
        from cage import metrics as mx
        return float(mx.auroc(y_true, scores))
    except Exception:
        order = np.argsort(-scores)
        ys = y_true[order]
        n_pos = int(y_true.sum())
        n_neg = len(y_true) - n_pos
        if n_pos == 0 or n_neg == 0:
            return float("nan")
        tpr = np.concatenate([[0.0], np.cumsum(ys) / n_pos])
        fpr = np.concatenate([[0.0], np.cumsum(1 - ys) / n_neg])
        return float(np.trapezoid(tpr, fpr))


def _auprc(y_true: np.ndarray, scores: np.ndarray) -> float:
    try:
        from cage import metrics as mx
        return float(mx.auprc(y_true, scores))
    except Exception:
        order = np.argsort(-scores)
        ys = y_true[order]
        n_pos = int(y_true.sum())
        if n_pos == 0:
            return float("nan")
        tp = np.cumsum(ys == 1).astype(np.float64)
        fp = np.cumsum(ys == 0).astype(np.float64)
        precision = tp / np.maximum(tp + fp, 1.0)
        recall = tp / n_pos
        recall_prev = np.concatenate([[0.0], recall[:-1]])
        return float(np.sum((recall - recall_prev) * precision))


def run_model_inference(
    accession: str,
    gene_names: List[str],
    X_z: np.ndarray,
    y: np.ndarray,
    tcga_genes: List[str],
    checkpoint_dir: Path,
    seed: int = 2026,
) -> Dict[str, Any]:
    try:
        from cage.step4_runner import SparseInvariantModel, TrainingConfig
    except ImportError:
        return {"skipped": "cage_import_error"}

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

    n_tcga = len(tcga_genes)
    g2i = {g: i for i, g in enumerate(gene_names)}
    X_aligned = np.zeros((X_z.shape[0], n_tcga), dtype=np.float64)
    n_matched = 0
    for ti, tg in enumerate(tcga_genes):
        if tg in g2i:
            X_aligned[:, ti] = X_z[:, g2i[tg]]
            n_matched += 1

    logging.info("%s model: %d/%d TCGA genes matched", accession, n_matched, n_tcga)

    all_probs: List[np.ndarray] = []
    for fp in fold_files:
        state = dict(np.load(str(fp), allow_pickle=True))
        rng = np.random.default_rng(seed)
        config = _config_from_checkpoint_state(state)
        n_conf = int(state["adv.W"].shape[1]) if "adv.W" in state else 2
        model = SparseInvariantModel(n_tcga, n_conf, config, rng=rng)
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
    labeled = y >= 0

    if labeled.sum() < 4:
        return {"n_matched_genes": n_matched, "skipped": "too_few_labeled"}

    y_l, p_l, yp_l = y[labeled], probs[labeled], y_pred[labeled]
    n_pos, n_neg = int(y_l.sum()), int(len(y_l) - y_l.sum())
    auroc = _auroc(y_l, p_l)
    auprc = _auprc(y_l, p_l)
    tp = int(((yp_l == 1) & (y_l == 1)).sum())
    tn = int(((yp_l == 0) & (y_l == 0)).sum())
    fp_c = int(((yp_l == 1) & (y_l == 0)).sum())
    fn_c = int(((yp_l == 0) & (y_l == 1)).sum())
    sens = tp / max(tp + fn_c, 1)
    spec = tn / max(tn + fp_c, 1)
    precision = tp / max(tp + fp_c, 1)
    f1 = 2 * precision * sens / max(precision + sens, 1e-12)
    brier_raw = float(np.mean((p_l - y_l) ** 2))

    result: Dict[str, Any] = {
        "accession": accession,
        "n_samples": int(X_z.shape[0]),
        "n_tumor": n_pos,
        "n_normal": n_neg,
        "n_matched_genes": n_matched,
        "n_folds_ensembled": len(fold_files),
        "auroc": round(auroc, 4),
        "auprc": round(auprc, 4),
        "balanced_accuracy": round((sens + spec) / 2, 4),
        "precision": round(precision, 4),
        "f1": round(f1, 4),
        "sensitivity": round(sens, 4),
        "specificity": round(spec, 4),
        "brier": round(brier_raw, 4) if math.isfinite(brier_raw) else None,
    }
    logging.info(
        "%s model: AUROC=%.3f AUPRC=%.3f precision=%.3f sens=%.3f spec=%.3f",
        accession, auroc, auprc, precision, sens, spec,
    )
    return result


# ---------------------------------------------------------------------------
# Cross-cohort aggregation
# ---------------------------------------------------------------------------

def build_concordance_summary(
    all_de_rows: List[Dict[str, str]],
    target_genes: List[str],
) -> List[Dict[str, str]]:
    from collections import defaultdict
    gene_conc: Dict[str, List[int]] = defaultdict(list)
    for r in all_de_rows:
        if r["concordant"] in ("0", "1"):
            gene_conc[r["gene"]].append(int(r["concordant"]))

    summary: List[Dict[str, str]] = []
    for gene in target_genes:
        vals = gene_conc.get(gene, [])
        n_cohorts = len(vals)
        n_conc = sum(vals)
        rate = n_conc / max(n_cohorts, 1)
        summary.append({
            "gene": gene,
            "n_cohorts_tested": str(n_cohorts),
            "n_cohorts_concordant": str(n_conc),
            "concordance_rate": _fmt(rate, 3),
            "consistently_concordant": "1" if n_cohorts > 0 and n_conc == n_cohorts else "0",
        })
    summary.sort(key=lambda r: (-float(r["concordance_rate"]), -int(r["n_cohorts_tested"])))
    return summary


# ---------------------------------------------------------------------------
# Analysis report
# ---------------------------------------------------------------------------

def write_analysis_report(
    out_path: Path,
    datasets_run: List[str],
    gene_status: Dict[str, str],
    all_de_rows: List[Dict[str, str]],
    concordance_summary: List[Dict[str, str]],
    model_results: List[Dict[str, Any]],
    target_genes: List[str],
) -> None:
    n_found   = sum(1 for s in gene_status.values() if s == "found")
    n_noprobe = sum(1 for s in gene_status.values() if s == "no_probes")
    n_noseq   = sum(1 for s in gene_status.values() if s == "no_refseq")

    fully_conc4 = [r for r in concordance_summary
                   if int(r["n_cohorts_tested"]) == 2 and r["consistently_concordant"] == "1"]
    fully_conc3 = [r for r in concordance_summary
                   if int(r["n_cohorts_tested"]) == 2 and r["concordance_rate"] == "1.000"]

    with open(out_path, "w", encoding="utf-8") as fh:
        def p(s: str = "") -> None:
            fh.write(s + "\n")

        p("=" * 70)
        p("CAGE — Agilent lncRNA+mRNA Cohort External Validation Report")
        p("=" * 70)
        p()

        p("Cohorts")
        p("-" * 40)
        for ds in datasets_run:
            rows_ds = [r for r in all_de_rows if r["accession"] == ds]
            n_t = rows_ds[0]["ext_n_tumor"] if rows_ds else "?"
            n_n = rows_ds[0]["ext_n_normal"] if rows_ds else "?"
            n_conc_ds = sum(1 for r in rows_ds if r["concordant"] == "1")
            n_tested_ds = sum(1 for r in rows_ds if r["concordant"] in ("0","1"))
            pct = 100*n_conc_ds/max(n_tested_ds,1)
            p(f"  {ds}: {n_t} tumor / {n_n} normal  |  "
              f"DE concordance {n_conc_ds}/{n_tested_ds} ({pct:.1f}%)")
        p()

        p("Probe Annotation (via NCBI RefSeq 60-mer matching)")
        p("-" * 40)
        p(f"  Target genes tested      : {len(target_genes)}")
        p(f"  Genes with probe matches : {n_found}")
        p(f"  Genes without probes     : {n_noprobe}")
        p(f"  Genes without RefSeq seq : {n_noseq}")
        p()

        p("DE Direction Concordance Summary (both cohorts)")
        p("-" * 40)
        both_concordant = [r for r in concordance_summary
                           if int(r["n_cohorts_tested"]) == 2
                           and r["consistently_concordant"] == "1"]
        one_concordant  = [r for r in concordance_summary
                           if int(r["n_cohorts_tested"]) == 2
                           and r["concordance_rate"] == "0.500"]
        discordant      = [r for r in concordance_summary
                           if int(r["n_cohorts_tested"]) == 2
                           and r["concordance_rate"] == "0.000"]

        p(f"  Concordant in both cohorts   : {len(both_concordant)}")
        if both_concordant:
            p("    " + ", ".join(r["gene"] for r in both_concordant[:20]))
        p()
        p(f"  Concordant in 1/2 cohorts    : {len(one_concordant)}")
        p(f"  Discordant in both cohorts   : {len(discordant)}")
        p()

        # Genes only found in 1 cohort
        one_tested = [r for r in concordance_summary if int(r["n_cohorts_tested"]) == 1]
        p(f"  Genes tested in only 1 cohort: {len(one_tested)}")
        p()

        p("Top concordant genes (found in both cohorts, 100% concordance)")
        p("-" * 40)
        for r in both_concordant[:30]:
            p(f"  {r['gene']:<20}  concordance_rate={r['concordance_rate']}  "
              f"n_cohorts={r['n_cohorts_tested']}")
        p()

        if model_results:
            p("Classifier Performance (TCGA RNA-seq -> Agilent Microarray)")
            p("-" * 40)
            p(f"  Note: Model trained on TCGA RNA-seq (VST); tested on microarray")
            p(f"  log-intensity after per-gene z-score. Domain shift is expected.")
            p()
            for m in model_results:
                if "auroc" in m:
                    p(f"  {m['accession']}: AUROC={m['auroc']:.3f}  "
                      f"AUPRC={m.get('auprc', float('nan')):.3f}  "
                      f"Precision={m.get('precision', float('nan')):.3f}  "
                      f"Sens={m['sensitivity']:.3f}  Spec={m['specificity']:.3f}  "
                      f"n_genes_matched={m['n_matched_genes']}")
            p()

        p("Genes not found on Agilent array (no probe match via 60-mer)")
        p("-" * 40)
        no_probe = [g for g, s in gene_status.items() if s == "no_probes"]
        no_seq   = [g for g, s in gene_status.items() if s == "no_refseq"]
        if no_probe:
            p("  No probe match (RefSeq fetched but 0 probes hit):")
            p("  " + ", ".join(no_probe))
        if no_seq:
            p("  No RefSeq sequence retrieved (likely novel/lncRNA without NM_):")
            p("  " + ", ".join(no_seq))
        p()

        p("=" * 70)
        p("END OF REPORT")
        p("=" * 70)

    logging.info("Analysis report -> %s", out_path.name)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

AGILENT_DATASETS = ["GSE53624", "GSE53625"]


def process_dataset(
    accession: str,
    geo_dir: Path,
    probe_seqs: Dict[str, str],
    gene_to_probes: Dict[str, List[str]],
    gene_status: Dict[str, str],
    target_genes: List[str],
    ref_directions: Dict[str, str],
    tcga_genes: List[str],
    checkpoint_dir: Optional[Path],
    run_model: bool,
    seed: int,
    out_dir: Path,
) -> Tuple[Optional[List[Dict[str, str]]], Optional[Dict[str, Any]]]:
    """Process one Agilent GSE dataset.  Returns (de_rows, model_metrics)."""
    ds_dir = geo_dir / accession
    probe_mat_path = ds_dir / "processed" / "expression_probe_matrix.csv"
    meta_path      = ds_dir / "metadata" / "sample_metadata_inferred.csv"

    if not probe_mat_path.is_file():
        logging.warning("%s: expression_probe_matrix.csv not found — skipping", accession)
        return None, None

    # Load probe matrix
    feature_ids, sample_ids, X_probe = load_probe_matrix(probe_mat_path)

    # Load metadata and label vector
    labels = load_metadata(meta_path)
    y = np.array(
        [1 if labels.get(s) == "Tumor" else (0 if labels.get(s) == "Normal" else -1)
         for s in sample_ids],
        dtype=np.int32,
    )
    labeled = y >= 0
    n_tumor  = int((y == 1).sum())
    n_normal = int((y == 0).sum())
    logging.info("%s: %d samples | tumor=%d normal=%d", accession, len(sample_ids), n_tumor, n_normal)

    if n_tumor < 3 or n_normal < 3:
        logging.warning("%s: too few samples — skipping", accession)
        return None, None

    # Keep labeled samples only
    X_lab = X_probe[labeled]
    y_lab = y[labeled]

    # Collapse probes to gene matrix
    gene_names, sample_ids_lab, X_gene = collapse_to_gene_matrix(
        feature_ids, [s for s, lbl in zip(sample_ids, labeled) if lbl],
        X_lab, gene_to_probes,
    )

    if X_gene.size == 0:
        logging.warning("%s: gene matrix empty after probe collapse — skipping", accession)
        return None, None

    # Write gene-level matrix so downstream modules (e.g. step6b survival boxplots)
    # can load it via the standard expression_gene_matrix.csv path.
    gene_mat_path = geo_dir / accession / "processed" / "expression_gene_matrix.csv"
    try:
        gene_mat_path.parent.mkdir(parents=True, exist_ok=True)
        with open(gene_mat_path, "w", newline="", encoding="utf-8") as _fh:
            _w = csv.writer(_fh)
            _w.writerow(["gene_symbol"] + list(sample_ids_lab))
            for _gi, _gene in enumerate(gene_names):
                _w.writerow([_gene] + [f"{v:.6f}" for v in X_gene[:, _gi]])
        logging.info("%s: wrote gene matrix (%d genes × %d samples)",
                     accession, len(gene_names), len(sample_ids_lab))
    except Exception as _exc:
        logging.warning("%s: could not write expression_gene_matrix.csv: %s", accession, _exc)

    # Z-score per gene
    X_z = zscore(X_gene)
    X_z = np.nan_to_num(X_z, nan=0.0, posinf=0.0, neginf=0.0)

    # Per-dataset output dir
    per_dir = out_dir / "per_dataset" / accession
    per_dir.mkdir(parents=True, exist_ok=True)

    # DE replication
    de_rows = run_de_replication(
        accession, gene_names, X_z, y_lab, target_genes, ref_directions, gene_status,
    )
    _write_csv(per_dir / "de_replication.csv", de_rows, [
        "accession", "gene", "ext_n_tumor", "ext_n_normal",
        "ext_mean_tumor", "ext_mean_normal", "ext_effect",
        "ext_direction", "ref_direction", "concordant", "in_dataset",
        "n_probes_matched",
    ])

    # Model inference
    model_metrics: Dict[str, Any] = {}
    if run_model and checkpoint_dir and checkpoint_dir.is_dir():
        model_metrics = run_model_inference(
            accession, gene_names, X_z, y_lab, tcga_genes, checkpoint_dir, seed,
        )
        _write_json(per_dir / "model_metrics.json", model_metrics)

    # Dataset summary
    n_tested = sum(1 for r in de_rows if r["concordant"] in ("0", "1"))
    n_conc   = sum(1 for r in de_rows if r["concordant"] == "1")
    n_missing = sum(1 for r in de_rows if r["in_dataset"] == "0")
    _write_json(per_dir / "dataset_summary.json", {
        "accession": accession,
        "n_tumor": n_tumor, "n_normal": n_normal,
        "n_genes_resolved": len(gene_names),
        "n_target_missing": n_missing,
        "n_tested_concordance": n_tested,
        "n_concordant": n_conc,
        "concordance_rate": round(n_conc / max(n_tested, 1), 4),
        "model_metrics": model_metrics,
    })
    return de_rows, model_metrics if "auroc" in model_metrics else None


# ---------------------------------------------------------------------------
# Figure generation (A-series)
# ---------------------------------------------------------------------------

def _has_matplotlib_agilent() -> bool:
    try:
        import matplotlib  # noqa: F401
        return True
    except ImportError:
        return False


def generate_agilent_figures(
    *,
    conc_summary: List[Dict[str, str]],
    all_de_rows: List[Dict[str, str]],
    gene_status: Dict[str, str],
    model_results: List[Dict[str, Any]],
    target_genes: List[str],
    output_dir: Path,
    formats: Tuple[str, ...] = ("svg",),
) -> Tuple[List[str], List[Tuple[str, str]]]:
    """Generate Agilent validation figures (A-series) to output_dir/figures/.

    Returns (generated_names, skipped_name_reason_pairs).
    """
    generated: List[str] = []
    skipped: List[Tuple[str, str]] = []

    if not _has_matplotlib_agilent():
        for name in ["A1_direction_heatmap", "A2_concordance_bar",
                     "A3_probe_annotation_donut", "A4_classifier_performance",
                     "A5_effect_scatter", "A6_probes_per_gene"]:
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
            else:
                for fmt in formats:
                    p = fig_dir / f"{stem}.{fmt}"
                    p.parent.mkdir(parents=True, exist_ok=True)
                    fig.savefig(p, format=fmt, bbox_inches="tight")
                plt.close(fig)
                return True

        def _scolor(name: str, fallback: str) -> str:
            if _pub_style:
                return semantic_color(name)
            return fallback

        def _ccolors(n: int):
            if _pub_style:
                return categorical_colors(n)
            defaults = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
            return (defaults * ((n // len(defaults)) + 1))[:n]

        # ── A1: direction heatmap (top genes × datasets) ─────────────────
        try:
            datasets = sorted(set(r["accession"] for r in all_de_rows))
            tested_genes = [r["gene"] for r in conc_summary
                             if int(r.get("n_cohorts_tested", 0) or 0) > 0][:50]
            if tested_genes and datasets:
                mat_a1 = np.full((len(tested_genes), len(datasets)), np.nan)
                for r in all_de_rows:
                    g = r.get("gene", "")
                    if g not in tested_genes:
                        continue
                    gi = tested_genes.index(g)
                    di = datasets.index(r["accession"]) if r["accession"] in datasets else -1
                    if di < 0:
                        continue
                    conc = r.get("concordant", "")
                    if conc == "1":
                        mat_a1[gi, di] = 1.0
                    elif conc == "0":
                        mat_a1[gi, di] = 0.0
                fig, ax = plt.subplots(
                    figsize=(max(4, len(datasets) * 1.5 + 1.5),
                             max(5, len(tested_genes) * 0.25 + 2)),
                )
                im = ax.imshow(mat_a1, aspect="auto", cmap="cage_sequential", vmin=0, vmax=1)
                plt.colorbar(im, ax=ax, label="Concordant (1=yes)")
                ax.set_xticks(range(len(datasets)))
                ax.set_xticklabels(datasets, rotation=30, ha="right", fontsize=9)
                ax.set_yticks(range(len(tested_genes)))
                ax.set_yticklabels(tested_genes, fontsize=6.5)
                ax.set_title("DE Direction Concordance — Agilent Cohorts\n(TCGA reference)")
                fig.tight_layout()
                if _save(fig, "A1_direction_heatmap"):
                    generated.append("A1_direction_heatmap")
        except Exception as exc:
            skipped.append(("A1_direction_heatmap", str(exc)))

        # ── A2: concordance bar chart by tier ─────────────────────────────
        try:
            cs_dict = {r["gene"]: r for r in conc_summary}
            tiers = [
                ("Top 25",  [g for g in target_genes[:25]  if g in cs_dict]),
                ("Top 100", [g for g in target_genes[:100] if g in cs_dict]),
                ("All",     [g for g in target_genes       if g in cs_dict]),
            ]
            tier_rates = []
            for tier_name, tier_genes in tiers:
                rates = [float(cs_dict[g]["concordance_rate"]) for g in tier_genes
                          if cs_dict[g].get("concordance_rate")]
                tier_rates.append((tier_name, float(np.mean(rates)) if rates else 0.0,
                                   len(rates)))
            fig, ax = plt.subplots(figsize=(5, 3.5))
            labels_a2 = [f"{t}\n(n={n})" for t, _, n in tier_rates]
            vals_a2   = [v for _, v, _ in tier_rates]
            bars_a2 = ax.bar(labels_a2, vals_a2,
                             color=_ccolors(len(tier_rates)),
                             edgecolor="black", linewidth=0.6, width=0.5)
            for bar, v in zip(bars_a2, vals_a2):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                        f"{v:.1%}", ha="center", fontsize=10, fontweight="bold")
            ax.axhline(0.5, color="#999999", linewidth=0.8, linestyle="--", alpha=0.6)
            ax.set_ylabel("Mean concordance rate")
            ax.set_ylim(0, 1.15)
            ax.set_title("Concordance Rate by Gene Tier — Agilent")
            fig.tight_layout()
            if _save(fig, "A2_concordance_bar"):
                generated.append("A2_concordance_bar")
        except Exception as exc:
            skipped.append(("A2_concordance_bar", str(exc)))

        # ── A3: probe annotation donut ────────────────────────────────────
        try:
            n_found_a3   = sum(1 for s in gene_status.values() if s == "found")
            n_noprobe_a3 = sum(1 for s in gene_status.values() if s == "no_probes")
            n_noseq_a3   = sum(1 for s in gene_status.values() if s == "no_refseq")
            sizes_a3  = [n_found_a3, n_noprobe_a3, n_noseq_a3]
            labels_a3 = ["Probe matched", "No probes", "No RefSeq"]
            colors_a3 = [_scolor("enriched", "#2ca02c"), _scolor("highlight", "#ff7f0e"),
                          "#aaaaaa"]
            fig, ax = plt.subplots(figsize=(4.5, 4))
            wedges, _, autotexts = ax.pie(
                sizes_a3, labels=labels_a3, colors=colors_a3,
                autopct=lambda p: f"{p:.0f}%" if p > 3 else "",
                wedgeprops=dict(width=0.45, edgecolor="white", linewidth=1.2),
                startangle=90, textprops=dict(fontsize=9),
            )
            for at in autotexts:
                at.set_fontsize(9)
            ax.text(0, 0, f"n={len(gene_status)}\ngenes", ha="center", va="center",
                    fontsize=10, fontweight="bold")
            ax.set_title("Probe Annotation Status\n(Target Genes)")
            fig.tight_layout()
            if _save(fig, "A3_probe_annotation_donut"):
                generated.append("A3_probe_annotation_donut")
        except Exception as exc:
            skipped.append(("A3_probe_annotation_donut", str(exc)))

        # ── A4: classifier performance ────────────────────────────────────
        if model_results:
            try:
                metric_names = [
                    "auroc", "auprc", "balanced_accuracy",
                    "precision", "f1", "sensitivity", "specificity",
                ]
                metric_labels = [
                    "AUROC", "AUPRC", "BAC", "Precision", "F1", "Sensitivity", "Specificity",
                ]
                n_ds = len(model_results)
                fig, ax = plt.subplots(figsize=(max(5.8, n_ds * 2.3), 4.4))
                x_a4 = np.arange(n_ds) * 0.82
                width_a4 = 0.09
                pal_a4 = _ccolors(len(metric_names))
                for mi, met in enumerate(metric_names):
                    vals_m = [float(r.get(met, 0) or 0) for r in model_results]
                    offset = (mi - len(metric_names) / 2 + 0.5) * width_a4
                    bars_m = ax.bar(x_a4 + offset, vals_m, width_a4, label=metric_labels[mi],
                                    color=pal_a4[mi], edgecolor="white", linewidth=0.4)
                    for bar, v in zip(bars_m, vals_m):
                        if v > 0.05:
                            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.012,
                                    f"{v:.3f}", ha="center", va="bottom",
                                    fontsize=6.0, rotation=90)
                ax.set_xticks(x_a4)
                ds_labels_a4 = [r.get("accession", str(i)) for i, r in enumerate(model_results)]
                ax.set_xticklabels(ds_labels_a4, fontsize=9)
                ax.set_ylim(0, 1.26)
                ax.axhline(0.5, color="#999999", linewidth=0.8, linestyle="--", alpha=0.5)
                ax.set_ylabel("Score")
                ax.set_title("Classifier Performance — Agilent Cohorts")
                ax.legend(
                    fontsize=7,
                    loc="upper center",
                    bbox_to_anchor=(0.5, 1.18),
                    ncol=len(metric_names),
                    frameon=False,
                )
                fig.tight_layout(rect=[0, 0, 1, 0.90])
                if _save(fig, "A4_classifier_performance"):
                    generated.append("A4_classifier_performance")
            except Exception as exc:
                skipped.append(("A4_classifier_performance", str(exc)))

        # ── A5: effect size scatter ───────────────────────────────────────
        try:
            datasets_a5 = sorted(set(r["accession"] for r in all_de_rows))
            pal_a5 = _ccolors(max(len(datasets_a5), 2))
            fig, ax = plt.subplots(figsize=(5, 4.5))
            for di, ds in enumerate(datasets_a5):
                rows_ds = [r for r in all_de_rows if r["accession"] == ds]
                xs_a5 = []
                colors_a5_pts = []
                for r in rows_ds:
                    eff = r.get("ext_effect")
                    try:
                        xs_a5.append(float(eff))
                        colors_a5_pts.append(
                            _scolor("enriched", "#2ca02c") if r.get("concordant") == "1"
                            else _scolor("tumor", "#d62728")
                        )
                    except (TypeError, ValueError):
                        pass
                if xs_a5:
                    jitter = np.random.default_rng(di + 42).random(len(xs_a5)) * 0.4 - 0.2
                    ax.scatter(xs_a5, [di] * len(xs_a5) + jitter,
                               c=colors_a5_pts, s=10, alpha=0.6, edgecolors="none")
            ax.set_yticks(range(len(datasets_a5)))
            ax.set_yticklabels(datasets_a5, fontsize=9)
            ax.axvline(0, color="#666666", linewidth=0.7, linestyle="--", alpha=0.5)
            ax.set_xlabel("GEO effect size (tumor - normal)")
            ax.set_title("GEO Effect Sizes — Agilent Cohorts\n(green=concordant, red=discordant)")
            fig.tight_layout()
            if _save(fig, "A5_effect_scatter"):
                generated.append("A5_effect_scatter")
        except Exception as exc:
            skipped.append(("A5_effect_scatter", str(exc)))

        # ── A6: probes per gene ───────────────────────────────────────────
        try:
            probe_counts = {}
            for r in all_de_rows:
                g = r.get("gene", "")
                n_p = r.get("n_probes_matched", "")
                try:
                    probe_counts[g] = int(n_p)
                except (ValueError, TypeError):
                    pass
            if probe_counts:
                vals_a6 = list(probe_counts.values())
                fig, ax = plt.subplots(figsize=(5, 3.5))
                ax.hist(vals_a6, bins=max(10, len(set(vals_a6))),
                        color=_scolor("normal", "#4575b4"),
                        edgecolor="white", alpha=0.85)
                ax.set_xlabel("Number of matched probes per gene")
                ax.set_ylabel("Number of genes")
                ax.set_title("Probe Coverage — Agilent GPL18109")
                med_a6 = float(np.median(vals_a6))
                ax.axvline(med_a6, color="#d62728", linewidth=1.5, linestyle="--",
                           label=f"Median {med_a6:.0f}")
                ax.legend(fontsize=8)
                fig.tight_layout()
                if _save(fig, "A6_probes_per_gene"):
                    generated.append("A6_probes_per_gene")
        except Exception as exc:
            skipped.append(("A6_probes_per_gene", str(exc)))

    except Exception as exc:
        skipped.append(("all_agilent_figures", str(exc)))

    return generated, skipped


def build_parser(
    parser: Optional[argparse.ArgumentParser] = None,
) -> argparse.ArgumentParser:
    """Build the validate-agilent argument parser.

    If *parser* is provided (e.g. a subcommand parser), arguments are added
    to it in-place and it is returned. Otherwise a standalone parser is created.
    """
    if parser is None:
        parser = argparse.ArgumentParser(
            prog="python -m cage.step8_agilent_validation",
            description=(
                "Agilent GPL18109 lncRNA+mRNA GEO external validation for CAGE."
            ),
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

    g = parser.add_argument_group("Agilent validation inputs")
    g.add_argument("--geo-dir", required=True, type=Path,
                   help="Root GEO directory (contains GSE53624/, GSE53625/).")
    g.add_argument("--step5-dir", required=True, type=Path,
                   help="Step 5 CDPS outputs (ranked_genes_cdps.csv).")
    g.add_argument("--step6-dir", required=True, type=Path,
                   help="Step 6 validation outputs (differential_expression_results.csv).")
    g.add_argument("--step4-dir", default=None, type=Path,
                   help="Step 4 deep model outputs (checkpoints/). Required for --run-model.")

    g2 = parser.add_argument_group("Agilent validation options")
    g2.add_argument("--datasets", nargs="+", default=AGILENT_DATASETS, metavar="ACC",
                    help="Which GEO accessions to process (default: GSE53624 GSE53625).")
    g2.add_argument("--top-k", type=int, default=100, metavar="K",
                    help="Top K CDPS genes to validate (default: 100).")
    g2.add_argument("--run-model", action="store_true",
                    help="Also run the trained deep invariant model on each cohort.")
    g2.add_argument("--collapse", default="median", choices=["median", "mean", "max"],
                    metavar="METHOD",
                    help="Probe-to-gene aggregation method (default: median).")
    return parser


def run_agilent_validation(args: argparse.Namespace) -> None:
    """Execute Agilent GEO external validation from a parsed Namespace."""
    args.output_dir.mkdir(parents=True, exist_ok=True)
    verbose = getattr(args, "log_level", "INFO").upper() == "DEBUG"
    if _HAS_CLI_ARGS:
        _configure_logging(args,
                           log_file=args.output_dir / "logs" / "step8_agilent_validation.log")
    else:
        _setup_logging(args.output_dir / "logs" / "step8_agilent_validation.log", verbose)

    logging.info("CAGE Step 8 — validate-agilent (GPL18109 Agilent cohorts)")
    logging.info("datasets=%s  top_k=%d  run_model=%s",
                 args.datasets, args.top_k, args.run_model)

    # ---- Reference data ---------------------------------------------------
    target_genes   = load_reference_genes(args.step5_dir, args.top_k)
    tcga_genes     = load_tcga_gene_list(
        args.step5_dir,
        args.step4_dir if args.run_model else None,
    )
    ref_directions = load_reference_de(args.step6_dir)
    checkpoint_dir = args.step4_dir / "checkpoints" if args.step4_dir else None

    # ---- Probe sequences from first available SOFT.gz ---------------------
    ann_dir = args.output_dir / "annotation"
    ann_dir.mkdir(parents=True, exist_ok=True)
    probe_seq_cache = ann_dir / "probe_sequences.json"
    probe_seqs: Dict[str, str] = {}

    if probe_seq_cache.is_file():
        logging.info("Loading cached probe sequences ...")
        with open(probe_seq_cache, encoding="utf-8") as fh:
            probe_seqs = json.load(fh)
    else:
        for acc in args.datasets:
            soft_gz = args.geo_dir / acc / "raw" / f"{acc}_family.soft.gz"
            if soft_gz.is_file():
                probe_seqs = load_probe_sequences(soft_gz)
                break
        if not probe_seqs:
            logging.error("No SOFT.gz file found. Cannot extract probe sequences.")
            sys.exit(1)
        with open(probe_seq_cache, "w", encoding="utf-8") as fh:
            json.dump(probe_seqs, fh)
        logging.info("Cached %d probe sequences to %s", len(probe_seqs), probe_seq_cache.name)

    # ---- Build probe -> gene map ------------------------------------------
    map_cache = ann_dir / "probe_gene_map.csv"
    status_cache = ann_dir / "gene_status.json"
    gene_to_probes: Dict[str, List[str]] = {}
    gene_status:    Dict[str, str]       = {}

    if map_cache.is_file() and status_cache.is_file():
        logging.info("Loading cached probe→gene map ...")
        rows = _read_csv(map_cache)
        from collections import defaultdict
        g2p_raw: Dict[str, List[str]] = defaultdict(list)
        for r in rows:
            g2p_raw[r["gene"]].append(r["feature_id"])
        gene_to_probes = dict(g2p_raw)
        with open(status_cache, encoding="utf-8") as fh:
            gene_status = json.load(fh)
    else:
        refseq_cache = ann_dir / "refseq_cache"
        probe_to_gene, gene_to_probes, gene_status = build_probe_gene_map(
            probe_seqs, target_genes, refseq_cache,
        )
        # Write probe-gene map CSV
        map_rows = [
            {"feature_id": fid, "gene": gene}
            for gene, fids in gene_to_probes.items()
            for fid in fids
        ]
        _write_csv(map_cache, map_rows, ["feature_id", "gene"])
        with open(status_cache, "w", encoding="utf-8") as fh:
            json.dump(gene_status, fh, indent=2)

    # Annotation summary
    n_found   = sum(1 for s in gene_status.values() if s == "found")
    n_noprobe = sum(1 for s in gene_status.values() if s == "no_probes")
    n_noseq   = sum(1 for s in gene_status.values() if s == "no_refseq")
    _write_json(ann_dir / "annotation_summary.json", {
        "n_target_genes": len(target_genes),
        "n_genes_with_probes": n_found,
        "n_genes_no_probes": n_noprobe,
        "n_genes_no_refseq": n_noseq,
        "genes_found": [g for g, s in gene_status.items() if s == "found"],
        "genes_no_probes": [g for g, s in gene_status.items() if s == "no_probes"],
        "genes_no_refseq": [g for g, s in gene_status.items() if s == "no_refseq"],
        "probe_counts": {g: len(fids) for g, fids in gene_to_probes.items()},
    })
    logging.info(
        "Annotation: %d/%d genes have probe matches (%d no_probes, %d no_refseq)",
        n_found, len(target_genes), n_noprobe, n_noseq,
    )

    # ---- Process datasets -------------------------------------------------
    all_de_rows:    List[Dict[str, str]] = []
    model_results:  List[Dict[str, Any]] = []
    datasets_run:   List[str]            = []

    for acc in args.datasets:
        de_rows, metrics = process_dataset(
            acc, args.geo_dir,
            probe_seqs, gene_to_probes, gene_status,
            target_genes, ref_directions,
            tcga_genes, checkpoint_dir, args.run_model, args.seed,
            args.output_dir,
        )
        if de_rows is not None:
            all_de_rows.extend(de_rows)
            datasets_run.append(acc)
        if metrics:
            model_results.append(metrics)

    # ---- Combined outputs -------------------------------------------------
    if all_de_rows:
        _write_csv(args.output_dir / "combined_de_replication.csv", all_de_rows, [
            "accession", "gene", "ext_n_tumor", "ext_n_normal",
            "ext_mean_tumor", "ext_mean_normal", "ext_effect",
            "ext_direction", "ref_direction", "concordant", "in_dataset",
            "n_probes_matched",
        ])

    conc_summary = build_concordance_summary(all_de_rows, target_genes)
    _write_csv(args.output_dir / "concordance_summary.csv", conc_summary, [
        "gene", "n_cohorts_tested", "n_cohorts_concordant",
        "concordance_rate", "consistently_concordant",
    ])

    if model_results:
        _write_csv(args.output_dir / "model_summary.csv",
                   [{k: str(v) for k, v in r.items()} for r in model_results], [
                       "accession", "n_samples", "n_tumor", "n_normal",
                       "n_matched_genes", "n_folds_ensembled",
                       "auroc", "auprc", "balanced_accuracy", "precision", "f1",
                       "sensitivity", "specificity", "brier",
                   ])

    # ---- Analysis report --------------------------------------------------
    write_analysis_report(
        args.output_dir / "analysis_report.txt",
        datasets_run, gene_status,
        all_de_rows, conc_summary, model_results, target_genes,
    )

    # ---- Headline JSON summary --------------------------------------------
    top25 = [r for r in conc_summary if int(r["n_cohorts_tested"]) > 0][:25]
    mean_rate = sum(float(r["concordance_rate"]) for r in top25) / max(len(top25), 1)
    n_fully_25 = sum(1 for r in top25 if r["consistently_concordant"] == "1")

    _write_json(args.output_dir / "validation_summary.json", {
        "platform": "GPL18109 Agilent-038314 lncRNA+mRNA",
        "datasets_run": datasets_run,
        "n_target_genes": len(target_genes),
        "n_genes_with_probes": n_found,
        "top_25_mean_concordance_rate": round(mean_rate, 4),
        "top_25_fully_concordant": n_fully_25,
        "concordance_details": [
            {"gene": r["gene"], "rate": r["concordance_rate"],
             "n_cohorts": r["n_cohorts_tested"]}
            for r in top25
        ],
        "model_results": model_results,
        "config": {
            "top_k": args.top_k,
            "collapse": args.collapse,
            "run_model": args.run_model,
            "seed": args.seed,
        },
    })

    # ---- Figures -------------------------------------------------------------
    gen_figs, skip_figs = generate_agilent_figures(
        conc_summary=conc_summary,
        all_de_rows=all_de_rows,
        gene_status=gene_status,
        model_results=model_results,
        target_genes=target_genes,
        output_dir=args.output_dir,
    )
    for f in gen_figs:
        logging.info("figure OK: %s", f)
    for name, reason in skip_figs:
        logging.warning("figure SKIPPED: %s (%s)", name, reason)

    logging.info(
        "Done | datasets=%d | genes_with_probes=%d/%d | "
        "top25 mean concordance=%.1f%% | fully_concordant=%d/25",
        len(datasets_run), n_found, len(target_genes),
        100 * mean_rate, n_fully_25,
    )


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if _HAS_CLI_ARGS:
        _cli_args.apply_thread_limits(args)
        _cli_args.ensure_output_dir(args)
    run_agilent_validation(args)


if __name__ == "__main__":
    main()
