"""CAGE Step 2 (supplement): Gene-cap sensitivity analysis.

Run this script post-hoc to confirm that CDPS top-gene identity is stable
across different top-variable-gene caps (e.g. 5000, 10000, 20000, all).

The script re-applies deduplication and near-zero-variance filtering on the
original VST matrix, then sweeps the requested gene caps, computing pairwise
Jaccard overlaps and writing a comparison table and figure.

Run
---
python -m cage.step2_gene_sensitivity --help
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from . import cli_args, preprocess_esca as pp
from .cli_args import (
    add_rigor_profile_arg, apply_rigor_profile,
    build_step_parser, configure_logging, style_from_args,
)
from .step2_build_cohort import (
    _run_sensitivity_grid,
    _output_paths as _step2_output_paths,
)

logger = logging.getLogger("cage.step2_sensitivity")

_STEP_TITLE = "Step 2 supplement – Gene-cap sensitivity analysis."
_STEP_DESCRIPTION = (
    "Load the original VST matrix, re-apply deduplication and near-zero-variance\n"
    "filtering, then sweep multiple top-variable-gene caps.\n"
    "Outputs a comparison table (gene_cap_sensitivity_metrics.csv) and figure."
)
_INPUTS_DOC = (
    "ESCA_vst_normalized_matrix.csv   original variance-stabilized expression\n"
    "TCGA_ESCA_Metadata.csv           clinical metadata (for sample alignment)\n"
    "(auto-resolved inside --input-dir; override with --vst-csv)"
)
_OUTPUTS_DOC = (
    "gene_cap_sensitivity_metrics.csv  Jaccard + size metrics across caps\n"
    "figures/fig_gene_cap_sensitivity.*  bar charts"
)
_EXAMPLE = (
    "# Post-hoc sensitivity sweep on original VST data:\n"
    "python -m cage.step2_gene_sensitivity \\\n"
    "  --input-dir . --output-dir outputs/step2_sensitivity \\\n"
    "  --gene-sensitivity-grid 5000 10000 20000 all \\\n"
    "  --near-zero-variance-threshold 0.01"
)


def build_parser() -> argparse.ArgumentParser:
    parser = build_step_parser(
        prog="python -m cage.step2_gene_sensitivity",
        step_title=_STEP_TITLE,
        step_description=_STEP_DESCRIPTION,
        inputs_doc=_INPUTS_DOC,
        outputs_doc=_OUTPUTS_DOC,
        example=_EXAMPLE,
    )
    g = parser.add_argument_group("Sensitivity-analysis options")
    g.add_argument(
        "--vst-csv",
        type=Path,
        default=None,
        metavar="FILE",
        help="Override path to ESCA_vst_normalized_matrix.csv.",
    )
    g.add_argument(
        "--metadata-csv",
        type=Path,
        default=None,
        metavar="FILE",
        help="Override path to TCGA_ESCA_Metadata.csv.",
    )
    g.add_argument(
        "--near-zero-variance-threshold",
        type=float,
        default=0.01,
        metavar="F",
        help="Minimum per-gene variance for inclusion (default: 0.01).",
    )
    g.add_argument(
        "--gene-sensitivity-grid",
        nargs="+",
        default=["5000", "10000", "20000", "all"],
        metavar="N",
        help=(
            "Gene-cap values to sweep. Use integers or 'all'. "
            "(default: 5000 10000 20000 all)"
        ),
    )
    add_rigor_profile_arg(parser)
    return parser


def run_sensitivity(args: argparse.Namespace) -> None:
    """Execute the standalone gene-cap sensitivity analysis."""
    apply_rigor_profile(args)

    input_dir: Path | None = args.input_dir
    vst_csv = cli_args.resolve_path(args.vst_csv, input_dir, "ESCA_vst_normalized_matrix.csv")
    meta_csv = cli_args.resolve_path(args.metadata_csv, input_dir, "TCGA_ESCA_Metadata.csv")

    for label, p in (("vst", vst_csv), ("metadata", meta_csv)):
        if p is None or not p.exists():
            raise FileNotFoundError(
                f"Input {label} file not found: {p} "
                f"(use --input-dir or --{label}-csv)"
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Align to primary samples
    metadata = pp.load_metadata(meta_csv)
    pp.add_harmonized_identifiers(metadata)
    primary_records = pp.filter_primary_samples(metadata)
    primary_barcodes = [r["barcode"] for r in primary_records]

    logger.info("Primary samples aligned: %d", len(primary_barcodes))

    # Load and filter VST matrix
    vst_genes, vst_samples, vst_matrix = pp.load_csv_matrix(vst_csv)
    n_raw = len(vst_genes)
    vst_genes, vst_matrix = pp.align_matrix_to_samples(
        vst_genes, vst_samples, vst_matrix, primary_barcodes,
    )
    vst_genes, vst_matrix = pp.remove_duplicate_genes(vst_genes, vst_matrix)
    n_dedup = len(vst_genes)
    vst_genes, vst_matrix = pp.filter_near_zero_variance(
        vst_genes, vst_matrix, threshold=args.near_zero_variance_threshold,
    )
    n_var = len(vst_genes)

    logger.info(
        "Gene counts | raw=%d  after_dedup=%d  after_var_filter=%d",
        n_raw, n_dedup, n_var,
    )

    style = style_from_args(args)
    paths = {
        "gene_sensitivity": args.output_dir / "gene_cap_sensitivity_metrics.csv",
    }

    _run_sensitivity_grid(
        args.gene_sensitivity_grid,
        vst_genes,
        vst_matrix,
        n_var,
        paths,
        style,
        style.default_formats,
    )
    logger.info("Gene-cap sensitivity analysis complete → %s", args.output_dir)


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    cli_args.apply_thread_limits(args)
    cli_args.ensure_output_dir(args)
    configure_logging(
        args,
        log_file=args.output_dir / "logs" / "step2_gene_sensitivity.log",
    )
    logger.info(
        "CAGE step2-sensitivity invoked | grid=%s output=%s",
        args.gene_sensitivity_grid, args.output_dir,
    )
    apply_rigor_profile(args, parser_defaults={a.dest: a.default for a in parser._actions})
    run_sensitivity(args)


if __name__ == "__main__":
    main()
