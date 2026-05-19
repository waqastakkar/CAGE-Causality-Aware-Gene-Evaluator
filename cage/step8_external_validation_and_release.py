"""CAGE Step 8: Unified external-validation and release CLI (Phase V).

Subcommands
-----------
prepare-geo      Download and prepare GEO datasets (GSE accessions).
validate-geo     Affymetrix multi-cohort DE replication and model transfer.
validate-agilent Agilent GPL18109 cohort validation (GSE53624, GSE53625).
release          Assemble release bundle from Steps 2-7 artifacts.
all              Run prepare-geo → validate-geo → validate-agilent → release.

Usage
-----
python -m cage.step8_external_validation_and_release <subcommand> --help

Examples
--------
# Step 8a — download and prepare GEO datasets (4 validated cohorts):
python -m cage.step8_external_validation_and_release prepare-geo \\
    --gse GSE38129 GSE161533 GSE53624 GSE53625 \\
    --output-dir outputs/step8_geo_prepared

# Step 8b — Affymetrix GEO validation:
python -m cage.step8_external_validation_and_release validate-geo \\
    --geo-dir outputs/step8_geo_prepared \\
    --step5-dir outputs/step5_cdps \\
    --step6-dir outputs/step6_validation \\
    --output-dir outputs/step8_geo_validation \\
    --run-model --step4-dir outputs/step4_deep_model

# Step 8c — Agilent validation:
python -m cage.step8_external_validation_and_release validate-agilent \\
    --geo-dir outputs/step8_geo_prepared \\
    --step5-dir outputs/step5_cdps \\
    --step6-dir outputs/step6_validation \\
    --output-dir outputs/step8_agilent_validation \\
    --run-model --step4-dir outputs/step4_deep_model

# Step 8d — release bundle:
python -m cage.step8_external_validation_and_release release \\
    --step5-dir outputs/step5_cdps \\
    --step6-dir outputs/step6_validation \\
    --step7-dir outputs/final_package \\
    --output-dir outputs/step8_release \\
    --build-reviewer-bundle

# Run all sub-steps:
python -m cage.step8_external_validation_and_release all \\
    --gse GSE38129 GSE161533 GSE53624 GSE53625 \\
    --geo-dir outputs/step8_geo_prepared \\
    --step5-dir outputs/step5_cdps \\
    --step6-dir outputs/step6_validation \\
    --step4-dir outputs/step4_deep_model \\
    --step7-dir outputs/final_package \\
    --output-dir outputs/step8 \\
    --run-model
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from . import cli_args
from .cli_args import (
    add_global_args,
    add_figure_args,
    apply_thread_limits,
    configure_logging,
    ensure_output_dir,
    style_from_args,
    CAGE_BANNER,
)
from .step8_runner import run_step8

logger = logging.getLogger("cage.step8")


# ---------------------------------------------------------------------------
# Per-subcommand parsers
# ---------------------------------------------------------------------------

def _add_prepare_geo_args(p: argparse.ArgumentParser) -> None:
    """Add prepare-geo specific arguments (inline to avoid pandas import at parse time)."""
    g = p.add_argument_group("GEO download options")
    g.add_argument("--gse", nargs="+", required=True, metavar="ACC",
                   help="One or more GEO series accessions, e.g. GSE38129 GSE161533 GSE53624 GSE53625.")
    g.add_argument("--download-supp", action="store_true",
                   help="Download supplementary files when available.")
    g.add_argument("--collapse-method", choices=["median", "mean", "max"], default="median",
                   metavar="METHOD",
                   help="How to collapse multiple probes per gene symbol (default: median).")
    g.add_argument("--continue-on-error", action="store_true", default=True,
                   help="Continue if one accession fails (default: on).")
    g.add_argument("--no-continue-on-error", dest="continue_on_error",
                   action="store_false", help="Abort on first failure.")
    g.add_argument("--max-retries", type=int, default=5, metavar="N",
                   help="Max HTTPS download retries per accession (default: 5).")
    g.add_argument("--retry-delay", type=float, default=10.0, metavar="SECS",
                   help="Initial retry delay in seconds (default: 10).")
    g.add_argument("--platform-auto", action="store_true", default=True,
                   help="Automatically pick dominant platform for probe mapping (default: on).")


def _add_validate_geo_args(p: argparse.ArgumentParser) -> None:
    """Add validate-geo specific arguments (inline to avoid import at parse time)."""
    g = p.add_argument_group("GEO validation inputs")
    g.add_argument("--geo-dir", required=True, type=Path,
                   help="Directory produced by prepare-geo (contains GSE<id>/ subdirs).")
    g.add_argument("--step5-dir", required=True, type=Path,
                   help="Step 5 CDPS outputs (ranked_genes_cdps.csv).")
    g.add_argument("--step6-dir", required=True, type=Path,
                   help="Step 6 validation outputs (differential_expression_results.csv).")
    g.add_argument("--step4-dir", default=None, type=Path,
                   help="Step 4 deep model outputs (checkpoints/). Required for --run-model.")
    g2 = p.add_argument_group("GEO validation options")
    g2.add_argument("--top-k", type=int, default=100, metavar="K",
                    help="Number of top CDPS genes to test for DE replication (default: 100).")
    g2.add_argument("--run-model", action="store_true",
                    help="Also run the trained deep invariant model on each cohort.")
    g2.add_argument("--skip", nargs="*", default=[], metavar="ACC",
                    help="Additional accessions to skip.")


def _add_validate_agilent_args(p: argparse.ArgumentParser) -> None:
    """Add validate-agilent specific arguments (inline to avoid import at parse time)."""
    g = p.add_argument_group("Agilent validation inputs")
    g.add_argument("--geo-dir", required=True, type=Path,
                   help="Root GEO directory (contains GSE53624/, GSE53625/).")
    g.add_argument("--step5-dir", required=True, type=Path,
                   help="Step 5 CDPS outputs (ranked_genes_cdps.csv).")
    g.add_argument("--step6-dir", required=True, type=Path,
                   help="Step 6 validation outputs (differential_expression_results.csv).")
    g.add_argument("--step4-dir", default=None, type=Path,
                   help="Step 4 deep model outputs (checkpoints/). Required for --run-model.")
    g2 = p.add_argument_group("Agilent validation options")
    g2.add_argument("--datasets", nargs="+", default=["GSE53624", "GSE53625"], metavar="ACC",
                    help="Which GEO accessions to process (default: GSE53624 GSE53625).")
    g2.add_argument("--top-k", type=int, default=100, metavar="K",
                    help="Top K CDPS genes to validate (default: 100).")
    g2.add_argument("--run-model", action="store_true",
                    help="Also run the trained deep invariant model on each cohort.")
    g2.add_argument("--collapse", default="median", choices=["median", "mean", "max"],
                    metavar="METHOD",
                    help="Probe-to-gene aggregation method (default: median).")


def _build_release_parser(subs: argparse._SubParsersAction) -> argparse.ArgumentParser:
    p = subs.add_parser(
        "release",
        help="Assemble the release bundle from Steps 2-7 artifacts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            f"{CAGE_BANNER}\n\n"
            "Collect artifacts from Steps 2-7 into a release-grade directory tree\n"
            "with QC checks, claim-boundary language, and optional reviewer bundle."
        ),
    )
    add_global_args(p, require_input_dir=False)
    add_figure_args(p)

    dirs = p.add_argument_group("Per-phase input directories")
    dirs.add_argument("--step2-dir", type=Path, default=None, metavar="DIR")
    dirs.add_argument("--step4-dir", type=Path, default=None, metavar="DIR")
    dirs.add_argument("--step5-dir", type=Path, default=None, metavar="DIR")
    dirs.add_argument("--step6-dir", type=Path, default=None, metavar="DIR")
    dirs.add_argument("--step6b-dir", type=Path, default=None, metavar="DIR",
                      help="Step-6b top-25 prioritization outputs.")
    dirs.add_argument("--step6b-survival-dir", type=Path, default=None, metavar="DIR",
                      help="Step-6b survival/external validation outputs.")
    dirs.add_argument("--step7-dir", type=Path, default=None, metavar="DIR")
    dirs.add_argument("--step8-geo-dir", type=Path, default=None, metavar="DIR",
                      help="Step-8 GEO validation outputs (step8_geo_validation/).")
    dirs.add_argument("--step8-agilent-dir", type=Path, default=None, metavar="DIR",
                      help="Step-8 Agilent validation outputs (step8_agilent_validation/).")

    ext = p.add_argument_group("External cohort inputs (optional)")
    ext.add_argument("--external-counts",       type=Path, default=None, metavar="FILE")
    ext.add_argument("--external-normalized",   type=Path, default=None, metavar="FILE")
    ext.add_argument("--external-metadata",     type=Path, default=None, metavar="FILE")
    ext.add_argument("--external-label-column", default="sample_type", metavar="COL")

    rel = p.add_argument_group("Release bundle options")
    rel.add_argument("--release-bundle-dir", type=Path, default=None, metavar="DIR")
    rel.add_argument("--copy-final-figures",    action="store_true")
    rel.add_argument("--copy-key-tables",       action="store_true")
    rel.add_argument("--build-supplement",      action="store_true")
    rel.add_argument("--build-reviewer-bundle", action="store_true")
    return p


def _build_all_parser(subs: argparse._SubParsersAction) -> argparse.ArgumentParser:
    p = subs.add_parser(
        "all",
        help="Run all Step 8 sub-steps: prepare-geo → validate-geo → validate-agilent → release.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            f"{CAGE_BANNER}\n\n"
            "Runs all four Step 8 sub-steps in sequence.\n"
            "Uses --output-dir as the base; each sub-step writes to a sub-directory."
        ),
    )
    add_global_args(p, require_input_dir=False)
    add_figure_args(p)

    g = p.add_argument_group("GEO preparation (prepare-geo)")
    g.add_argument("--gse", nargs="+", default=None, metavar="ACC",
                   help="GEO accessions to download (required for prepare-geo and validate-geo).")
    g.add_argument("--download-supp", action="store_true",
                   help="Download supplementary GEO files.")
    g.add_argument("--collapse-method", choices=["median", "mean", "max"], default="median")
    g.add_argument("--max-retries", type=int, default=5)
    g.add_argument("--retry-delay", type=float, default=10.0)

    g2 = p.add_argument_group("GEO / Agilent validation (validate-geo, validate-agilent)")
    g2.add_argument("--step5-dir", type=Path, default=None, metavar="DIR", required=True)
    g2.add_argument("--step6-dir", type=Path, default=None, metavar="DIR", required=True)
    g2.add_argument("--step4-dir", type=Path, default=None, metavar="DIR")
    g2.add_argument("--top-k",     type=int, default=100, metavar="K")
    g2.add_argument("--run-model", action="store_true")
    g2.add_argument("--skip",      nargs="*", default=[], metavar="ACC")
    g2.add_argument("--collapse",  choices=["median", "mean", "max"], default="median")

    g3 = p.add_argument_group("Release bundle (release)")
    g3.add_argument("--step2-dir",  type=Path, default=None, metavar="DIR")
    g3.add_argument("--step7-dir",  type=Path, default=None, metavar="DIR")
    g3.add_argument("--copy-final-figures",    action="store_true")
    g3.add_argument("--copy-key-tables",       action="store_true")
    g3.add_argument("--build-supplement",      action="store_true")
    g3.add_argument("--build-reviewer-bundle", action="store_true")
    return p


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level subcommand dispatcher parser."""
    parser = argparse.ArgumentParser(
        prog="python -m cage.step8_external_validation_and_release",
        description=(
            f"{CAGE_BANNER}\n\n"
            "Step 8: Unified external-validation and release CLI.\n\n"
            "Choose a sub-command:\n"
            "  prepare-geo      Download and prepare GEO datasets.\n"
            "  validate-geo     Affymetrix multi-cohort external validation.\n"
            "  validate-agilent Agilent GPL18109 cohort validation.\n"
            "  release          Assemble the final release bundle.\n"
            "  all              Run all four sub-steps in sequence."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subs = parser.add_subparsers(dest="subcommand", metavar="SUBCOMMAND")
    subs.required = True

    # prepare-geo
    p_prep = subs.add_parser(
        "prepare-geo",
        help="Download and prepare GEO datasets for validation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            f"{CAGE_BANNER}\n\n"
            "Step 8a: Download and prepare GEO datasets (SOFT.gz + expression + metadata)."
        ),
    )
    add_global_args(p_prep, require_input_dir=False)
    add_figure_args(p_prep)
    _add_prepare_geo_args(p_prep)

    # validate-geo
    p_geo = subs.add_parser(
        "validate-geo",
        help="Affymetrix GEO multi-cohort DE replication and model transfer.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            f"{CAGE_BANNER}\n\n"
            "Step 8b: Affymetrix GEO multi-cohort external validation."
        ),
    )
    add_global_args(p_geo, require_input_dir=False)
    add_figure_args(p_geo)
    _add_validate_geo_args(p_geo)

    # validate-agilent
    p_agi = subs.add_parser(
        "validate-agilent",
        help="Agilent GPL18109 cohort validation (GSE53624, GSE53625).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            f"{CAGE_BANNER}\n\n"
            "Step 8c: Agilent GPL18109 lncRNA+mRNA ESCC cohort validation."
        ),
    )
    add_global_args(p_agi, require_input_dir=False)
    add_figure_args(p_agi)
    _add_validate_agilent_args(p_agi)

    # release
    _build_release_parser(subs)

    # all
    _build_all_parser(subs)

    return parser


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------

def _run_prepare_geo(args: argparse.Namespace) -> None:
    from .step8_prepare_geo import run_prepare_geo
    configure_logging(args, log_file=args.output_dir / "logs" / "step8_prepare_geo.log")
    logger.info("Step 8 prepare-geo | gse=%s output=%s", args.gse, args.output_dir)
    run_prepare_geo(args)


def _run_validate_geo(args: argparse.Namespace) -> None:
    from .step8_geo_validation import run_geo_validation
    configure_logging(args, log_file=args.output_dir / "logs" / "step8_geo_validation.log")
    logger.info("Step 8 validate-geo | geo_dir=%s output=%s", args.geo_dir, args.output_dir)
    run_geo_validation(args)


def _run_validate_agilent(args: argparse.Namespace) -> None:
    from .step8_agilent_validation import run_agilent_validation
    configure_logging(args, log_file=args.output_dir / "logs" / "step8_agilent_validation.log")
    logger.info("Step 8 validate-agilent | geo_dir=%s output=%s", args.geo_dir, args.output_dir)
    run_agilent_validation(args)


def _run_release(args: argparse.Namespace) -> None:
    configure_logging(args, log_file=args.output_dir / "logs" / "step8_release.log")
    style = style_from_args(args)

    has_external = all(
        p is not None
        for p in (
            getattr(args, "external_counts", None),
            getattr(args, "external_normalized", None),
            getattr(args, "external_metadata", None),
        )
    )
    logger.info(
        "Step 8 release | external_cohort=%s reviewer_bundle=%s",
        has_external, getattr(args, "build_reviewer_bundle", False),
    )

    summary = run_step8(
        step2_dir=getattr(args, "step2_dir", None),
        step4_dir=getattr(args, "step4_dir", None),
        step5_dir=getattr(args, "step5_dir", None),
        step6_dir=getattr(args, "step6_dir", None),
        step7_dir=getattr(args, "step7_dir", None),
        output_dir=args.output_dir,
        step6b_top25_dir=getattr(args, "step6b_dir", None),
        step6b_survival_dir=getattr(args, "step6b_survival_dir", None),
        step8_geo_dir=getattr(args, "step8_geo_dir", None),
        step8_agilent_dir=getattr(args, "step8_agilent_dir", None),
        external_counts=getattr(args, "external_counts", None),
        external_normalized=getattr(args, "external_normalized", None),
        external_metadata=getattr(args, "external_metadata", None),
        external_label_column=getattr(args, "external_label_column", "sample_type"),
        release_bundle_dir=getattr(args, "release_bundle_dir", None),
        copy_final_figures=getattr(args, "copy_final_figures", False),
        copy_key_tables=getattr(args, "copy_key_tables", False),
        build_supplement=getattr(args, "build_supplement", False),
        do_build_reviewer_bundle=getattr(args, "build_reviewer_bundle", False),
        style=style,
        seed=args.seed,
    )
    qc = summary.get("qc", {})
    logger.info(
        "Release done | bundle_files=%d qc=(%dP/%dW/%dM)",
        summary.get("bundle_manifest", {}).get("files_copied", 0),
        qc.get("n_pass", 0), qc.get("n_warn", 0), qc.get("n_missing", 0),
    )


def _run_all(args: argparse.Namespace) -> None:
    """Run all four sub-steps in sequence using subdirectories of --output-dir."""
    import types

    base = args.output_dir
    configure_logging(args, log_file=base / "logs" / "step8_all.log")
    logger.info("Step 8 all | base=%s", base)

    def _child(subdir: str, extra: dict) -> argparse.Namespace:
        """Clone args with output_dir set to a sub-directory."""
        ns = types.SimpleNamespace(**vars(args))
        ns.output_dir = base / subdir
        ns.output_dir.mkdir(parents=True, exist_ok=True)
        for k, v in extra.items():
            setattr(ns, k, v)
        return ns  # type: ignore[return-value]

    geo_prepared_dir = base / "geo_prepared"

    # 1. prepare-geo
    if getattr(args, "gse", None):
        logger.info("--- Step 8 all: prepare-geo ---")
        from .step8_prepare_geo import run_prepare_geo
        run_prepare_geo(_child("geo_prepared", {}))
    else:
        logger.info("Skipping prepare-geo (no --gse provided); using --geo-dir if set.")

    geo_dir = getattr(args, "geo_dir", None) or geo_prepared_dir

    # 2. validate-geo
    logger.info("--- Step 8 all: validate-geo ---")
    from .step8_geo_validation import run_geo_validation
    run_geo_validation(_child("geo_validation", {"geo_dir": geo_dir}))

    # 3. validate-agilent
    logger.info("--- Step 8 all: validate-agilent ---")
    from .step8_agilent_validation import run_agilent_validation
    run_agilent_validation(_child("agilent_validation", {"geo_dir": geo_dir}))

    # 4. release
    logger.info("--- Step 8 all: release ---")
    _run_release(_child("release", {}))

    logger.info("Step 8 all: complete.")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    apply_thread_limits(args)
    ensure_output_dir(args)

    dispatch = {
        "prepare-geo":      _run_prepare_geo,
        "validate-geo":     _run_validate_geo,
        "validate-agilent": _run_validate_agilent,
        "release":          _run_release,
        "all":              _run_all,
    }
    handler = dispatch.get(args.subcommand)
    if handler is None:
        print(f"Unknown subcommand: {args.subcommand}", file=sys.stderr)
        sys.exit(1)
    handler(args)


if __name__ == "__main__":
    main()
