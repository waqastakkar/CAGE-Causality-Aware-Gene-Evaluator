"""CAGE Step 7: Manuscript packaging (Phase V).

Aggregates outputs from every prior phase into a reproducible, publication-
ready package: manuscript figures (SVG, CAGE default styling — configurable
via --font-family / --palette), main and supplementary tables, structured
Markdown methods/results drafts
(with optional HTML / DOCX-ready / LaTeX-ready exports), and reproducibility
manifests (study_summary.json, output/figure/table manifests).

Deliverables
------------
final_figures/      figure1..figure8 SVG (optional PDF/PNG)
final_tables/       table1..table4 + supplementary_table_s1..s6
manuscript/         manuscript_report.md, manuscript_methods.md,
                    manuscript_results_summary.md (optional HTML/DOCX/LaTeX)
supplementary/      supplementary_materials.md + figures/tables
manifests/          output_manifest.csv, figure_manifest.csv,
                    table_manifest.csv, study_summary.json

Run
---
python -m cage.step7_manuscript_packaging --help
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from . import cli_args
from .cli_args import build_step_parser, configure_logging, style_from_args
from .step7_runner import run_step7_packaging

logger = logging.getLogger("cage.step7")

_STEP_TITLE = "Phase V - Manuscript figure/table/report packaging."
_STEP_DESCRIPTION = (
    "Collects artifacts from steps 2-6b (and optionally step 8), assembles\n"
    "Figures 1-8+ and main/supplementary tables (S1-S10), and emits manuscript-ready\n"
    "Markdown (with optional HTML / DOCX-ready / LaTeX templates) plus a\n"
    "reproducibility manifest. Explicit claim boundaries ('candidate driver\n"
    "genes' / 'computationally prioritized') are preserved throughout."
)
_INPUTS_DOC = (
    "--step2-dir .. --step6-dir (required)\n"
    "--step6b-dir (optional, step-6b top-25 prioritization figures/tables/reports)\n"
    "--step6b-survival-dir (optional, step-6b survival/external KM/boxplot outputs)\n"
    "--step8-dir (optional, for external validation packaging)\n"
    "Falls back to per-phase subdirectories inside --input-dir if given."
)
_OUTPUTS_DOC = (
    "final_figures/figure1..figure8+.svg (+ optional pdf/png)\n"
    "final_tables/table1_cohort_summary.csv ... table4_validation_augmented_genes.csv\n"
    "final_tables/supplementary_table_s1..s10.csv\n"
    "manuscript/manuscript_report.md, manuscript_methods.md, manuscript_results_summary.md\n"
    "supplementary/supplementary_materials.md\n"
    "manifests/study_summary.json, output_manifest.csv,\n"
    "         figure_manifest.csv, table_manifest.csv"
)
_EXAMPLE = (
    "python -m cage.step7_manuscript_packaging \\\n"
    "  --step2-dir outputs/step2_cohort \\\n"
    "  --step3-dir outputs/step3_baselines \\\n"
    "  --step4-dir outputs/step4_deep_model \\\n"
    "  --step5-dir outputs/step5_cdps \\\n"
    "  --step6-dir outputs/step6_validation \\\n"
    "  --step6b-dir outputs/step6b_top25_prioritization \\\n"
    "  --step6b-survival-dir outputs/step6b_survival_external \\\n"
    "  --step8-dir outputs/step8_external_validation \\\n"
    "  --output-dir outputs/final_package \\\n"
    "  --export-markdown --export-html \\\n"
    "  --copy-final-figures --copy-key-tables --build-supplement"
)


def build_parser() -> argparse.ArgumentParser:
    parser = build_step_parser(
        prog="python -m cage.step7_manuscript_packaging",
        step_title=_STEP_TITLE,
        step_description=_STEP_DESCRIPTION,
        inputs_doc=_INPUTS_DOC,
        outputs_doc=_OUTPUTS_DOC,
        example=_EXAMPLE,
        require_input_dir=False,
    )

    dirs = parser.add_argument_group("Per-phase input directories")
    dirs.add_argument("--step2-dir", type=Path, default=None, metavar="DIR",
                      help="Step-2 cohort outputs (required for Figures 1-2, Table 1).")
    dirs.add_argument("--step3-dir", type=Path, default=None, metavar="DIR",
                      help="Step-3 baseline outputs (for Figure 4 / Table 2).")
    dirs.add_argument("--step4-dir", type=Path, default=None, metavar="DIR",
                      help="Step-4 deep-model outputs (for Figures 3-4).")
    dirs.add_argument("--step5-dir", type=Path, default=None, metavar="DIR",
                      help="Step-5 CDPS outputs (for Figure 5 / Table 3).")
    dirs.add_argument("--step6-dir", type=Path, default=None, metavar="DIR",
                      help="Step-6 validation outputs (for Figures 6-7 / Table 4).")
    dirs.add_argument("--step6b-dir", type=Path, default=None, metavar="DIR",
                      help="Step-6b top-25 prioritization outputs (tables, figures, reports).")
    dirs.add_argument("--step6b-survival-dir", type=Path, default=None, metavar="DIR",
                      help="Step-6b survival/external validation outputs (KM, boxplots, GEO panels).")
    dirs.add_argument("--step8-dir", type=Path, default=None, metavar="DIR",
                      help="Step-8 external validation outputs (optional, for Figure 8).")

    exports = parser.add_argument_group("Manuscript exports")
    exports.add_argument("--export-markdown",   action="store_true",
                         help="Emit manuscript_report.md, manuscript_methods.md, manuscript_results_summary.md.")
    exports.add_argument("--export-html",       action="store_true",
                         help="Also render Markdown drafts to HTML.")
    exports.add_argument("--export-docx-ready", action="store_true",
                         help="Produce Markdown tuned for Pandoc -> DOCX conversion.")
    exports.add_argument("--export-latex-ready", action="store_true",
                         help="Produce LaTeX-ready figure/table stubs.")

    assembly = parser.add_argument_group("Bundle assembly")
    assembly.add_argument("--copy-final-figures", action="store_true",
                          help="Copy Figure 1-8 SVGs (and other formats) into final_figures/.")
    assembly.add_argument("--copy-key-tables",    action="store_true",
                          help="Copy Table 1-4 CSVs into final_tables/.")
    assembly.add_argument("--build-supplement",   action="store_true",
                          help="Assemble the supplementary/ bundle (tables S1-S6 + supplementary figures).")
    assembly.add_argument("--build-reviewer-bundle", action="store_true",
                          help="Build a reviewer-friendly zipped bundle with diffs, logs, and manifests.")

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cli_args.apply_thread_limits(args)
    cli_args.ensure_output_dir(args)
    configure_logging(args, log_file=args.output_dir / "logs" / "step7_manuscript_packaging.log")
    style = style_from_args(args)

    logger.info(
        "CAGE step 7 invoked | exports=(md=%s html=%s docx=%s latex=%s) "
        "copy_figs=%s copy_tables=%s supplement=%s reviewer=%s",
        args.export_markdown, args.export_html, args.export_docx_ready, args.export_latex_ready,
        args.copy_final_figures, args.copy_key_tables, args.build_supplement, args.build_reviewer_bundle,
    )
    logger.debug("figure style: %s", style.as_dict())

    summary = run_step7_packaging(
        step2_dir=args.step2_dir,
        step3_dir=args.step3_dir,
        step4_dir=args.step4_dir,
        step5_dir=args.step5_dir,
        step6_dir=args.step6_dir,
        step6b_top25_dir=args.step6b_dir,
        step6b_survival_dir=args.step6b_survival_dir,
        output_dir=args.output_dir,
        step8_dir=args.step8_dir,
        export_markdown=args.export_markdown,
        export_html=args.export_html,
        export_docx_ready=args.export_docx_ready,
        export_latex_ready=args.export_latex_ready,
        copy_final_figures=args.copy_final_figures,
        copy_key_tables=args.copy_key_tables,
        build_supplement=args.build_supplement,
        build_reviewer_bundle=args.build_reviewer_bundle,
        style=style,
        seed=args.seed,
    )

    n_tables = sum(1 for v in summary.get("tables_written", {}).values() if v > 0)
    logger.info(
        "Step 7 done | tables=%d supp=%d drafts=%d figs_ok=%d figs_skip=%d",
        n_tables,
        len(summary.get("supplementary_tables", [])),
        len(summary.get("manuscript_drafts", [])),
        len(summary.get("figures", {}).get("generated", [])),
        len(summary.get("figures", {}).get("skipped", [])),
    )


if __name__ == "__main__":
    main()
