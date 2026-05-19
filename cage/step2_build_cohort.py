"""CAGE Step 2: Cohort curation and preprocessing (Phase I).

Assembles a leakage-free TCGA ESCA cohort, harmonizes barcodes, filters
primary tumor / solid-normal samples, constructs confounder/environment
strata, processes the expression matrix, and generates patient-grouped
nested cross-validation folds.

Deliverables
------------
master_samples_primary.csv     curated patient/sample table
normalized_primary_matrix.csv  z-scored samples x genes matrix for deep model
counts_primary_matrix.csv      filtered raw counts (DE validation input)
grouped_outer_folds.csv        patient-safe nested CV fold assignments
phase1_summary.json            preprocessing stats + configuration

Run
---
python -m cage.step2_build_cohort --help
"""

from __future__ import annotations

import argparse
import logging
from collections import Counter
from pathlib import Path

import numpy as np

from . import cli_args, preprocess_esca as pp
from .cli_args import (
    add_rigor_profile_arg, apply_rigor_profile,
    build_step_parser, configure_logging, style_from_args,
)

logger = logging.getLogger("cage.step2")

_STEP_TITLE = "Phase I - Cohort curation and expression preprocessing."
_STEP_DESCRIPTION = (
    "Load TCGA ESCA counts, VST-normalized matrix, and metadata; harmonize\n"
    "barcodes; filter primary tumor / solid-normal samples; encode\n"
    "confounders and environment strata; process the expression matrix; and\n"
    "emit patient-grouped nested cross-validation folds."
)
_INPUTS_DOC = (
    "TCGA_ESCA_STAR_Counts.csv       raw counts (QC + DE validation)\n"
    "ESCA_vst_normalized_matrix.csv  variance-stabilized expression\n"
    "TCGA_ESCA_Metadata.csv          clinical + confounder metadata\n"
    "(auto-resolved inside --input-dir; override with --counts-csv etc.)"
)
_OUTPUTS_DOC = (
    "master_samples_primary.csv\n"
    "normalized_primary_matrix.csv\n"
    "counts_primary_matrix.csv\n"
    "grouped_outer_folds.csv\n"
    "phase1_summary.json\n"
    "(optional) figures/ : cohort flow, sample-type / environment bar charts"
)
_EXAMPLE = (
    "# Fast / exploratory run (top-5000 variable genes):\n"
    "python -m cage.step2_build_cohort \\\n"
    "  --input-dir . --output-dir outputs/step2_cohort \\\n"
    "  --n-top-variable-genes 5000 \\\n"
    "  --n-outer-folds 5 --n-inner-folds 3\n\n"
    "# High-impact manuscript run (all biologically filtered genes):\n"
    "python -m cage.step2_build_cohort \\\n"
    "  --input-dir . --output-dir outputs/step2_cohort \\\n"
    "  --use-all-filtered-genes \\\n"
    "  --n-outer-folds 5 --n-inner-folds 3 \\\n"
    "  --run-cohort-flow-figure\n\n"
    "# One-flag all-genes preset (enables all rigorous options):\n"
    "python -m cage.step2_build_cohort \\\n"
    "  --input-dir . --output-dir outputs/step2_cohort \\\n"
    "  --rigor-profile all_genes \\\n"
    "  --n-outer-folds 5 --n-inner-folds 3"
)


def build_parser() -> argparse.ArgumentParser:
    """Construct the step-2 CLI parser (global + figure + phase-specific)."""
    parser = build_step_parser(
        prog="python -m cage.step2_build_cohort",
        step_title=_STEP_TITLE,
        step_description=_STEP_DESCRIPTION,
        inputs_doc=_INPUTS_DOC,
        outputs_doc=_OUTPUTS_DOC,
        example=_EXAMPLE,
    )
    phase = parser.add_argument_group("Step-2 phase-specific options")
    phase.add_argument(
        "--counts-csv",
        type=Path,
        default=None,
        metavar="FILE",
        help="Override path to TCGA_ESCA_STAR_Counts.csv (default: <input-dir>/TCGA_ESCA_STAR_Counts.csv).",
    )
    phase.add_argument(
        "--vst-csv",
        type=Path,
        default=None,
        metavar="FILE",
        help="Override path to ESCA_vst_normalized_matrix.csv.",
    )
    phase.add_argument(
        "--metadata-csv",
        type=Path,
        default=None,
        metavar="FILE",
        help="Override path to TCGA_ESCA_Metadata.csv.",
    )
    phase.add_argument(
        "--min-count",
        type=int,
        default=10,
        metavar="N",
        help="Minimum raw count threshold for the DE-validation gene filter (default: 10).",
    )
    phase.add_argument(
        "--min-samples-per-gene",
        type=int,
        default=10,
        metavar="N",
        help="Minimum samples with count >= --min-count required to retain a gene (default: 10).",
    )
    phase.add_argument(
        "--near-zero-variance-threshold",
        type=float,
        default=0.01,
        metavar="F",
        help="Minimum per-gene variance to retain on the VST matrix (default: 0.01).",
    )
    phase.add_argument(
        "--n-top-variable-genes",
        type=int,
        default=5000,
        metavar="N",
        help=(
            "Number of highest-variance genes retained from the VST matrix for modelling. "
            "5000 is a speed-oriented default; it is NOT a biological requirement. "
            "Set to 99999 or use --use-all-filtered-genes to keep every gene that passes "
            "deduplication and near-zero-variance filtering. "
            "Recommended modes: 5000 (fast test), 10000-20000 (balanced), "
            "99999 / --use-all-filtered-genes (high-impact manuscript). "
            "(default: 5000)"
        ),
    )
    phase.add_argument(
        "--use-all-filtered-genes",
        action="store_true",
        default=False,
        help=(
            "Retain ALL genes that pass deduplication, low-expression filtering, and "
            "near-zero-variance filtering — no top-variance cap is applied. "
            "Equivalent to --n-top-variable-genes 99999. "
            "Recommended for high-impact manuscript runs where the sparse gate model "
            "should receive the full biologically filtered gene universe. "
            "Supersedes --n-top-variable-genes when set."
        ),
    )
    phase.add_argument(
        "--gene-sensitivity-grid",
        nargs="+",
        default=None,
        metavar="N",
        help=(
            "Run gene-cap sensitivity analysis across multiple top-variable-gene caps "
            "before building the final cohort. Values can be integers or 'all'. "
            "Outputs gene_cap_sensitivity_metrics.csv, gene_cap_cdps_overlap.csv, "
            "and a gene-cap sensitivity figure. "
            "Example: --gene-sensitivity-grid 5000 10000 20000 all"
        ),
    )
    phase.add_argument(
        "--n-outer-folds",
        type=int,
        default=5,
        metavar="K",
        help="Outer patient-grouped CV folds (default: 5).",
    )
    phase.add_argument(
        "--n-inner-folds",
        type=int,
        default=3,
        metavar="K",
        help="Inner CV folds for hyperparameter tuning (default: 3).",
    )
    phase.add_argument(
        "--environments",
        nargs="+",
        default=["smoking", "sex", "histology", "country", "stage"],
        metavar="NAME",
        help="Environment strata to construct for invariance analyses (default: smoking sex histology country stage).",
    )
    phase.add_argument(
        "--confounder-columns",
        nargs="+",
        default=["gender", "tobacco_smoking_status", "paper_Histological.Type",
                 "paper_Country", "paper_Pathologic.stage"],
        metavar="COL",
        help="Metadata columns to integer-encode as confounders.",
    )
    phase.add_argument(
        "--run-cohort-flow-figure",
        action="store_true",
        help="Render the cohort-flow diagram alongside summary figures.",
    )
    add_rigor_profile_arg(parser)
    phase.add_argument(
        "--no-zscore",
        action="store_true",
        default=True,   # LEAKAGE GUARD: global z-score is disabled by default.
        help=(
            "Skip global z-scoring on the normalized matrix and keep raw VST values "
            "(DEFAULT: True). Per-fold z-scoring in Step 3 and Step 4 is the only "
            "standardisation applied to modelling splits, preventing test-fold statistics "
            "from leaking into training."
        ),
    )
    phase.add_argument(
        "--global-zscore",
        dest="no_zscore",
        action="store_false",
        help=(
            "Enable GLOBAL z-scoring in Step 2 (not recommended for final modelling runs; "
            "leaks test-fold statistics into the training pipeline). Kept for backwards "
            "compatibility and exploratory use only."
        ),
    )
    return parser


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def _resolve_inputs(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    """Resolve the three input CSVs via explicit overrides or --input-dir."""
    input_dir: Path | None = args.input_dir
    counts = cli_args.resolve_path(args.counts_csv, input_dir, "TCGA_ESCA_STAR_Counts.csv")
    vst    = cli_args.resolve_path(args.vst_csv,    input_dir, "ESCA_vst_normalized_matrix.csv")
    meta   = cli_args.resolve_path(args.metadata_csv, input_dir, "TCGA_ESCA_Metadata.csv")

    for label, p in (("counts", counts), ("vst", vst), ("metadata", meta)):
        if p is None or not p.exists():
            raise FileNotFoundError(
                f"Input {label} file not found: {p} "
                f"(use --input-dir or --{label}-csv to supply)"
            )
    return counts, vst, meta


def _output_paths(args: argparse.Namespace) -> dict[str, Path]:
    out = args.output_dir
    return {
        "master": out / "master_samples_primary.csv",
        "normalized": out / "normalized_primary_matrix.csv",
        "counts": out / "counts_primary_matrix.csv",
        "folds": out / "grouped_outer_folds.csv",
        "summary": out / "phase1_summary.json",
        "confounder_maps": out / "confounder_encodings.json",
        "gene_filter_report": out / "preprocessing_gene_filtering_report.csv",
        "methods_text": out / "methods_gene_filtering.txt",
        "model_size_report": out / "model_size_report.json",
        "gene_sensitivity": out / "gene_cap_sensitivity_metrics.csv",
    }


def _check_overwrite(args: argparse.Namespace, paths: dict[str, Path]) -> None:
    existing = [p for p in paths.values() if p.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Output files already exist; pass --overwrite to regenerate:\n  "
            + "\n  ".join(str(p) for p in existing)
        )


def _parser_defaults(parser: argparse.ArgumentParser) -> dict:
    return {a.dest: a.default for a in parser._actions}


def _effective_n_top(args: argparse.Namespace) -> int:
    """Return the resolved top-gene cap (999_999 = no cap = keep all filtered)."""
    if getattr(args, "use_all_filtered_genes", False):
        return 999_999
    return int(getattr(args, "n_top_variable_genes", 5000))


def _write_gene_filtering_report(paths: dict, counts: dict, args: argparse.Namespace) -> None:
    """Write preprocessing_gene_filtering_report.csv."""
    import csv as _csv
    n_final = counts["selected"]
    n_var   = counts["after_var_filter"]
    n_raw_vst = counts["vst_raw"]

    rows = [
        {
            "stage": "vst_raw",
            "description": "Genes loaded from VST CSV",
            "n_genes": n_raw_vst,
            "pct_of_vst_raw": "100.0",
            "filter_applied": "none",
        },
        {
            "stage": "after_dedup",
            "description": "After gene-ID deduplication",
            "n_genes": counts["after_dedup"],
            "pct_of_vst_raw": f"{100*counts['after_dedup']/n_raw_vst:.1f}",
            "filter_applied": "duplicate gene IDs removed (keep first)",
        },
        {
            "stage": "after_variance_filter",
            "description": "After near-zero variance filter",
            "n_genes": n_var,
            "pct_of_vst_raw": f"{100*n_var/n_raw_vst:.1f}",
            "filter_applied": f"per-gene variance < {args.near_zero_variance_threshold}",
        },
        {
            "stage": "final_for_modelling",
            "description": "Final gene set used for deep model",
            "n_genes": n_final,
            "pct_of_vst_raw": f"{100*n_final/n_raw_vst:.1f}",
            "filter_applied": (
                "all genes passing variance filter (--use-all-filtered-genes)"
                if getattr(args, "use_all_filtered_genes", False)
                else f"top {args.n_top_variable_genes} by variance"
            ),
        },
        {
            "stage": "counts_raw",
            "description": "Genes loaded from raw counts CSV",
            "n_genes": counts["counts_raw"],
            "pct_of_vst_raw": "N/A",
            "filter_applied": "none",
        },
        {
            "stage": "counts_after_low_expression_filter",
            "description": "Counts genes after low-expression filter",
            "n_genes": counts["counts_filtered"],
            "pct_of_vst_raw": "N/A",
            "filter_applied": (
                f"count < {args.min_count} in >= {args.min_samples_per_gene} samples"
            ),
        },
    ]
    fields = ["stage", "description", "n_genes", "pct_of_vst_raw", "filter_applied"]
    with open(paths["gene_filter_report"], "w", newline="", encoding="utf-8") as fh:
        w = _csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    logger.info("Gene filtering report written: %s", paths["gene_filter_report"])


def _write_methods_text(paths: dict, counts: dict, args: argparse.Namespace) -> None:
    """Write a manuscript-ready methods paragraph to methods_gene_filtering.txt."""
    use_all = getattr(args, "use_all_filtered_genes", False)
    n_vst_raw   = counts["vst_raw"]
    n_dedup     = counts["after_dedup"]
    n_var       = counts["after_var_filter"]
    n_final     = counts["selected"]
    n_cts_raw   = counts["counts_raw"]
    n_cts_filt  = counts["counts_filtered"]

    selection_sentence = (
        f"All {n_final:,} genes that passed the near-zero-variance filter were "
        f"retained for modelling, preserving the full biologically filtered gene universe "
        f"available to the sparse gate layer."
        if use_all else
        f"The {n_final:,} genes with highest cross-sample variance were retained for "
        f"modelling (top-N selection from {n_var:,} variance-filtered genes; "
        f"N = {args.n_top_variable_genes:,})."
    )

    text = f"""\
Gene Pre-processing (Step 2)
============================

Variance-stabilized (VST) expression data were loaded from ESCA_vst_normalized_matrix.csv
({n_vst_raw:,} genes × {counts['n_samples']} samples). Genes with duplicate identifiers were
removed retaining the first occurrence ({n_dedup:,} genes after deduplication).
Genes with near-zero variance (per-gene variance across all samples < {args.near_zero_variance_threshold})
were excluded, yielding {n_var:,} biologically informative genes.

{selection_sentence}

For differential expression validation, raw STAR counts were loaded from
TCGA_ESCA_STAR_Counts.csv ({n_cts_raw:,} genes). After removing genes expressed at
< {args.min_count} counts in < {args.min_samples_per_gene} samples,
{n_cts_filt:,} genes were retained for DE analysis (Step 6).

Gene-selection mode: {'--use-all-filtered-genes (all biologically filtered genes)' if use_all else f'--n-top-variable-genes {args.n_top_variable_genes}'}
Rigor profile: {getattr(args, 'rigor_profile', 'standard')}
Outer CV folds: {args.n_outer_folds}  |  Inner CV folds: {args.n_inner_folds}  |  Seed: {args.seed}

Note on gene-cap sensitivity: The top-variable-gene threshold (N = {args.n_top_variable_genes:,})
is a computational parameter, not a biological boundary.  Sensitivity analyses
across N ∈ {{5,000; 10,000; 20,000; all}} are recommended to confirm that
top-CDPS gene identity and enrichment conclusions are stable across gene-cap
settings (see gene_cap_sensitivity_metrics.csv and the gene-cap sensitivity figure).
"""
    with open(paths["methods_text"], "w", encoding="utf-8") as fh:
        fh.write(text)
    logger.info("Methods text written: %s", paths["methods_text"])


def _estimate_model_size(n_genes: int, n_samples: int, args: argparse.Namespace) -> dict:
    """Estimate downstream model memory and training time; write model_size_report.json."""
    n_hidden  = 128          # typical SparseInvariantModel hidden size
    n_latent  = 32
    n_epochs  = 150
    n_folds   = args.n_outer_folds

    # Parameters: gate (n_genes) + encoder (n_genes * n_hidden) + decoder + classifier
    n_params = (
        n_genes                       # gate weights
        + n_genes * n_hidden          # encoder layer
        + n_hidden * n_latent         # bottleneck
        + n_latent * 2                # classifier
    )
    # Memory estimate (float64, forward + backward ~3x): bytes → MB
    mem_mb = (n_params * 8 * 3) / 1e6
    # Wall-clock estimate: roughly 0.5 s per epoch per fold per 1000 genes on 4-core CPU
    time_per_epoch_s = 0.5 * (n_genes / 1000) * (n_samples / 200)
    total_hours = (time_per_epoch_s * n_epochs * n_folds) / 3600

    report = {
        "n_genes_input": n_genes,
        "n_samples": n_samples,
        "n_outer_folds": n_folds,
        "n_epochs_typical": n_epochs,
        "estimated_trainable_params": n_params,
        "estimated_memory_mb": round(mem_mb, 1),
        "estimated_training_hours_4core": round(total_hours, 1),
        "note": (
            "Estimates assume a 4-core CPU, float64 SparseInvariantModel. "
            "Actual runtime depends on hardware, epoch budget, and early stopping. "
            "GPU acceleration is not implemented in the current NumPy backend."
        ),
    }
    if mem_mb > 4000:
        logger.warning(
            "MODEL SIZE WARNING: estimated %.0f MB for %d genes. "
            "Consider --n-top-variable-genes 20000 if memory is limited.",
            mem_mb, n_genes,
        )
    if total_hours > 24:
        logger.warning(
            "RUNTIME WARNING: estimated %.1f hours for %d genes × %d epochs × %d folds. "
            "This is feasible but plan compute time accordingly.",
            total_hours, n_genes, n_epochs, n_folds,
        )
    return report


def _run_sensitivity_grid(
    grid_values: list[str],
    vst_genes: list[str],
    vst_matrix,         # ndarray genes × samples
    n_var: int,
    paths: dict,
    style,
    formats,
) -> None:
    """Compute gene-cap sensitivity statistics and write comparison outputs."""
    import csv as _csv
    n_total = len(vst_genes)

    # Parse grid values; "all" → n_total
    caps: list[int] = []
    for v in grid_values:
        if str(v).lower() == "all":
            caps.append(n_total)
        else:
            try:
                caps.append(int(v))
            except ValueError:
                logger.warning("Invalid gene-sensitivity-grid value %r — skipping", v)
    caps = sorted(set(caps))

    variances = np.var(vst_matrix, axis=1)
    sorted_idx = np.argsort(variances)[::-1]

    rows = []
    gene_sets: dict[int, set] = {}
    for cap in caps:
        n_used = min(cap, n_total)
        top_idx = sorted(sorted_idx[:n_used].tolist())
        gene_set = set(vst_genes[i] for i in top_idx)
        gene_sets[cap] = gene_set
        rows.append({
            "gene_cap": "all" if cap >= n_total else cap,
            "n_genes_selected": n_used,
            "pct_of_variance_filtered": f"{100*n_used/n_var:.1f}" if n_var else "N/A",
            "min_variance_kept": f"{float(variances[sorted_idx[n_used-1]]):.6f}" if n_used else "N/A",
        })

    # Pairwise Jaccard with next larger cap
    for i, cap in enumerate(caps):
        if i + 1 < len(caps):
            a, b = gene_sets[cap], gene_sets[caps[i+1]]
            jaccard = len(a & b) / len(a | b) if (a | b) else 0.0
            rows[i]["jaccard_with_next_cap"] = f"{jaccard:.4f}"
        else:
            rows[i]["jaccard_with_next_cap"] = "N/A"

    fields = ["gene_cap", "n_genes_selected", "pct_of_variance_filtered",
              "min_variance_kept", "jaccard_with_next_cap"]
    with open(paths["gene_sensitivity"], "w", newline="", encoding="utf-8") as fh:
        w = _csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    logger.info("Gene-cap sensitivity table written: %s", paths["gene_sensitivity"])

    # Figure: gene count and Jaccard across caps
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from .publication_style import apply_style, save_figure

        fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
        apply_style(fig, style)

        caps_labels = [str(r["gene_cap"]) for r in rows]
        n_selected  = [int(r["n_genes_selected"]) for r in rows]
        jaccards    = [float(r["jaccard_with_next_cap"]) if r["jaccard_with_next_cap"] != "N/A" else None
                       for r in rows]

        axes[0].bar(caps_labels, n_selected, color="#4575b4", edgecolor="black", linewidth=0.6)
        axes[0].set_xlabel("Gene cap")
        axes[0].set_ylabel("Genes selected")
        axes[0].set_title("Genes selected per cap")

        jac_x = [caps_labels[i] for i, j in enumerate(jaccards) if j is not None]
        jac_y = [j for j in jaccards if j is not None]
        axes[1].bar(jac_x, jac_y, color="#d73027", edgecolor="black", linewidth=0.6)
        axes[1].set_xlabel("Gene cap")
        axes[1].set_ylabel("Jaccard with next-larger cap")
        axes[1].set_ylim(0, 1.05)
        axes[1].set_title("Gene-set overlap across caps")

        fig.suptitle("Gene-Cap Sensitivity — CAGE Step 2", fontweight="bold")
        fig.tight_layout()
        fig_dir = paths["gene_sensitivity"].parent / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        save_figure(fig, fig_dir / "fig_gene_cap_sensitivity", style=style, formats=list(formats))
    except Exception as exc:
        logger.warning("Gene-cap sensitivity figure skipped: %s", exc)


def run_step2(args: argparse.Namespace) -> dict:
    """Execute the step-2 cohort-build pipeline and return the summary dict.

    Applies ``--rigor-profile`` overrides, then runs cohort curation,
    gene filtering, CV fold assignment, and all requested outputs.
    """
    # Apply rigor-profile overrides before any logic
    apply_rigor_profile(args)

    # Resolve effective gene cap (--use-all-filtered-genes supersedes --n-top-variable-genes)
    n_top_genes = _effective_n_top(args)
    if n_top_genes >= 999_000:
        logger.info(
            "Gene selection mode: ALL biologically filtered genes "
            "(--use-all-filtered-genes or --n-top-variable-genes >= 99999). "
            "No top-variance cap will be applied."
        )
    else:
        logger.info(
            "Gene selection mode: top %d genes by variance (speed/balanced mode). "
            "Use --use-all-filtered-genes for high-impact manuscript mode.",
            n_top_genes,
        )

    paths = _output_paths(args)
    _check_overwrite(args, paths)
    counts_csv, vst_csv, meta_csv = _resolve_inputs(args)

    # Initialize style early (needed for sensitivity grid figure and cohort figures)
    style = style_from_args(args)
    formats = style.default_formats

    # -------------------------------------------------------------------
    # 1. Load metadata, harmonize, filter primary samples
    # -------------------------------------------------------------------
    metadata = pp.load_metadata(meta_csv)
    pp.add_harmonized_identifiers(metadata)
    primary_records = pp.filter_primary_samples(metadata)

    if not primary_records:
        raise RuntimeError("No primary samples after filtering; check metadata.")

    primary_barcodes = [r["barcode"] for r in primary_records]

    # -------------------------------------------------------------------
    # 2. Load VST matrix -> align -> dedup -> variance filter -> top-N
    # -------------------------------------------------------------------
    vst_genes, vst_samples, vst_matrix = pp.load_csv_matrix(vst_csv)
    n_genes_vst_raw = len(vst_genes)

    vst_genes, vst_matrix = pp.align_matrix_to_samples(
        vst_genes, vst_samples, vst_matrix, primary_barcodes,
    )
    vst_genes, vst_matrix = pp.remove_duplicate_genes(vst_genes, vst_matrix)
    n_genes_after_dedup = len(vst_genes)

    vst_genes, vst_matrix = pp.filter_near_zero_variance(
        vst_genes, vst_matrix, threshold=args.near_zero_variance_threshold,
    )
    n_genes_after_var_filter = len(vst_genes)

    # Gene-cap sensitivity analysis runs on the full post-variance-filter set,
    # before the final top-N cap is applied.
    if getattr(args, "gene_sensitivity_grid", None):
        _run_sensitivity_grid(
            args.gene_sensitivity_grid,
            vst_genes,
            vst_matrix,
            n_genes_after_var_filter,
            paths,
            style,
            formats,
        )

    vst_genes, vst_matrix = pp.select_top_variable_genes(
        vst_genes, vst_matrix, n_top=n_top_genes,
    )
    n_genes_selected = len(vst_genes)

    # Transpose: genes x samples -> samples x genes
    expr_sxg = vst_matrix.T.astype(np.float64)

    # Global z-score is DISABLED by default (--no-zscore is True by default).
    # Per-fold standardisation is applied inside Step 3 and Step 4 loops so
    # that test-fold statistics never influence training transforms.
    # Passing --global-zscore enables the old behaviour for exploratory use.
    if not args.no_zscore:
        logger.warning(
            "LEAKAGE RISK: --global-zscore is set. Z-scoring the full cohort before "
            "fold assignment leaks test-fold expression statistics into training. "
            "Use this option for exploration ONLY, not for reported model metrics."
        )
        expr_sxg, _means, _stds = pp.zscore_normalize(expr_sxg)

    # -------------------------------------------------------------------
    # 3. Load counts matrix -> align -> filter low-expression
    # -------------------------------------------------------------------
    counts_genes, counts_samples, counts_matrix = pp.load_csv_matrix(counts_csv)
    n_genes_counts_raw = len(counts_genes)

    counts_genes, counts_matrix = pp.align_matrix_to_samples(
        counts_genes, counts_samples, counts_matrix, primary_barcodes,
    )
    counts_genes, counts_matrix = pp.remove_duplicate_genes(counts_genes, counts_matrix)
    counts_genes, counts_matrix = pp.filter_low_expression(
        counts_genes, counts_matrix,
        min_count=args.min_count,
        min_samples=args.min_samples_per_gene,
    )
    n_genes_counts_filtered = len(counts_genes)

    # Gene filtering report and manuscript methods text
    filter_counts = {
        "vst_raw": n_genes_vst_raw,
        "after_dedup": n_genes_after_dedup,
        "after_var_filter": n_genes_after_var_filter,
        "selected": n_genes_selected,
        "counts_raw": n_genes_counts_raw,
        "counts_filtered": n_genes_counts_filtered,
        "n_samples": len(primary_barcodes),
    }
    _write_gene_filtering_report(paths, filter_counts, args)
    _write_methods_text(paths, filter_counts, args)

    # -------------------------------------------------------------------
    # 4. Environment assignment + confounder encoding
    # -------------------------------------------------------------------
    pp.assign_environment_strata(primary_records, env_names=args.environments)
    confounder_mappings = pp.encode_confounders(
        primary_records, columns=args.confounder_columns, min_levels=2,
    )

    # -------------------------------------------------------------------
    # 5. Patient-safe nested CV folds
    # -------------------------------------------------------------------
    fold_records = pp.build_patient_grouped_folds(
        primary_records,
        n_outer=args.n_outer_folds,
        n_inner=args.n_inner_folds,
        seed=args.seed,
    )

    # -------------------------------------------------------------------
    # 6. Master sample table
    # -------------------------------------------------------------------
    master = pp.build_master_sample_table(primary_records, env_names=args.environments)

    # -------------------------------------------------------------------
    # 7. Persist outputs
    # -------------------------------------------------------------------
    pp.write_csv_records(paths["master"], master)
    pp.write_csv_matrix(
        paths["normalized"],
        row_names=primary_barcodes,
        col_names=vst_genes,
        matrix=expr_sxg,
        index_label="sample_barcode",
        fmt="%.6f",
    )
    pp.write_csv_matrix(
        paths["counts"],
        row_names=counts_genes,
        col_names=primary_barcodes,
        matrix=counts_matrix.astype(np.int64),
        index_label="gene_id",
        fmt="%d",
    )
    pp.write_csv_records(paths["folds"], fold_records)

    # Confounder mappings sidecar
    pp.write_json(
        paths["confounder_maps"],
        {
            "mappings": confounder_mappings,
            "note": "Integer codes assigned in alphabetical order of non-missing values.",
        },
    )

    # Model size report: written before figures so reviewers can inspect compute demands
    model_report = _estimate_model_size(n_genes_selected, len(primary_barcodes), args)
    pp.write_json(paths["model_size_report"], model_report)
    logger.info("Model size report written: %s", paths["model_size_report"])

    # -------------------------------------------------------------------
    # 8. Figures (optional)
    # -------------------------------------------------------------------
    generated_figs, skipped_figs = pp.generate_cohort_figures(
        master_records=master,
        output_dir=args.output_dir,
        style=style,
        formats=style.default_formats,
    )
    for f in generated_figs:
        logger.info("figure OK: %s", f)
    for name, reason in skipped_figs:
        logger.warning("figure SKIPPED: %s (%s)", name, reason)

    # Gene filtering funnel (always generated — compact and reviewer-friendly)
    ok, reason = pp.generate_gene_filtering_figure(
        filter_counts,
        output_dir=args.output_dir,
        style=style,
        formats=style.default_formats,
    )
    if ok:
        generated_figs.append("fig_gene_filtering_funnel")
        logger.info("figure OK: fig_gene_filtering_funnel")
    else:
        logger.warning("figure SKIPPED: fig_gene_filtering_funnel (%s)", reason)

    # -------------------------------------------------------------------
    # 9. Phase 1 summary
    # -------------------------------------------------------------------
    env_distributions: dict[str, dict[str, int]] = {}
    for env in args.environments:
        env_col = f"env_{env}"
        ctr = Counter(r.get(env_col, "") for r in master)
        env_distributions[env] = {
            "class_0": ctr.get("0", 0),
            "class_1": ctr.get("1", 0),
            "missing": ctr.get("", 0),
        }

    n_tumor = sum(1 for r in master if r["label"] == "Tumor")
    n_normal = sum(1 for r in master if r["label"] == "Normal")
    n_patients = len(set(r["patient_barcode"] for r in master))

    summary = pp.build_phase1_summary(
        n_patients=n_patients,
        n_tumor=n_tumor,
        n_normal=n_normal,
        n_genes_vst_raw=n_genes_vst_raw,
        n_genes_after_dedup=n_genes_after_dedup,
        n_genes_after_var_filter=n_genes_after_var_filter,
        n_genes_selected=n_genes_selected,
        n_genes_counts_raw=n_genes_counts_raw,
        n_genes_counts_filtered=n_genes_counts_filtered,
        n_outer_folds=args.n_outer_folds,
        n_inner_folds=args.n_inner_folds,
        seed=args.seed,
        env_distributions=env_distributions,
        extra={
            "config": {
                "min_count": args.min_count,
                "min_samples_per_gene": args.min_samples_per_gene,
                "near_zero_variance_threshold": args.near_zero_variance_threshold,
                "environments": list(args.environments),
                "confounder_columns": list(args.confounder_columns),
                "zscore_applied": not args.no_zscore,
                "gene_selection_mode": (
                    "all_filtered"
                    if getattr(args, "use_all_filtered_genes", False)
                    else "top_n"
                ),
                "n_top_variable_genes_effective": n_genes_selected,
                "rigor_profile": getattr(args, "rigor_profile", "standard"),
                "input_files": {
                    "counts": str(counts_csv),
                    "vst": str(vst_csv),
                    "metadata": str(meta_csv),
                },
            },
            "figures": {
                "generated": generated_figs,
                "skipped": [
                    {"name": n, "reason": r} for n, r in skipped_figs
                ],
            },
            "style": style.as_dict(),
        },
    )
    pp.write_json(paths["summary"], summary)

    logger.info(
        "Step 2 complete | %d samples (%d T + %d N) across %d patients | "
        "%d expression genes, %d counts genes | %d outer folds",
        n_tumor + n_normal, n_tumor, n_normal, n_patients,
        n_genes_selected, n_genes_counts_filtered, args.n_outer_folds,
    )
    return summary


def main(argv: list[str] | None = None) -> None:
    """Parse arguments and dispatch the step-2 pipeline."""
    parser = build_parser()
    args = parser.parse_args(argv)
    cli_args.apply_thread_limits(args)
    cli_args.ensure_output_dir(args)
    configure_logging(args, log_file=args.output_dir / "logs" / "step2_build_cohort.log")

    logger.info(
        "CAGE step 2 invoked | seed=%s threads=%s output=%s",
        args.seed, args.n_threads, args.output_dir,
    )
    # Pass parser defaults so explicit CLI flags always win over --rigor-profile preset
    apply_rigor_profile(args, parser_defaults=_parser_defaults(parser))
    run_step2(args)


if __name__ == "__main__":
    main()
