"""CAGE Step 3: Grouped baseline models.

Trains patient-grouped baseline classifiers (logistic ridge and elastic-net
logistic by default) on the curated cohort emitted by Step 2, using the Step-2
outer folds so no patient leaks between train and test. Produces reference
metrics and out-of-fold predictions that Phase II and Phase V comparisons
consume.

Deliverables
------------
baseline_oof_predictions.csv    sample_barcode, patient, y_true, <model>_prob
baseline_per_fold_metrics.csv   per-model, per-outer-fold AUROC/AUPRC/...
baseline_summary_metrics.csv    overall-OOF and mean-of-folds with CIs
baseline_feature_importance.csv  top genes per model (rank, importance)
baseline_subgroup_metrics.csv   (optional) per-environment metrics
baseline_calibration.csv        (optional) reliability-diagram table
phase3_summary.json             configuration + metrics + figure manifest
figures/  ROC, PR, calibration, model-comparison (SVG + optional PDF/PNG)

Run
---
python -m cage.step3_grouped_baselines --help
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from . import cli_args, preprocess_esca as pp
from .cli_args import (
    add_rigor_profile_arg, apply_rigor_profile,
    build_step_parser, configure_logging, style_from_args,
)
from . import step3_runner as runner

logger = logging.getLogger("cage.step3")


def _format_probability(value: float) -> str:
    """Preserve near-0/1 probability ordering for ROC plots and audits."""
    return "" if (value != value) else f"{float(value):.17g}"


_STEP_TITLE = "Grouped baseline classifiers over patient-safe CV folds."
_STEP_DESCRIPTION = (
    "Trains patient-grouped baseline models (logistic and elastic-net)\n"
    "on normalized_primary_matrix.csv using the outer folds produced\n"
    "in Step 2. Produces reference metrics and OOF predictions consumed by\n"
    "Phase II / Phase V comparisons."
)
_INPUTS_DOC = (
    "normalized_primary_matrix.csv (from step 2)\n"
    "master_samples_primary.csv    (from step 2)\n"
    "grouped_outer_folds.csv       (from step 2)\n"
    "(auto-resolved inside --input-dir; override with --step2-dir)"
)
_OUTPUTS_DOC = (
    "baseline_oof_predictions.csv\n"
    "baseline_per_fold_metrics.csv\n"
    "baseline_summary_metrics.csv\n"
    "baseline_feature_importance.csv\n"
    "baseline_subgroup_metrics.csv (if --run-subgroup-sensitivity)\n"
    "baseline_calibration.csv       (if --run-calibration)\n"
    "phase3_summary.json\n"
    "figures/ : ROC, PR, calibration, model comparison"
)
_EXAMPLE = (
    "python -m cage.step3_grouped_baselines \\\n"
    "  --input-dir outputs/step2_cohort \\\n"
    "  --output-dir outputs/step3_baselines \\\n"
    "  --models logistic elasticnet \\\n"
    "  --run-calibration --run-subgroup-sensitivity"
)


def build_parser() -> argparse.ArgumentParser:
    parser = build_step_parser(
        prog="python -m cage.step3_grouped_baselines",
        step_title=_STEP_TITLE,
        step_description=_STEP_DESCRIPTION,
        inputs_doc=_INPUTS_DOC,
        outputs_doc=_OUTPUTS_DOC,
        example=_EXAMPLE,
    )
    phase = parser.add_argument_group("Step-3 phase-specific options")
    phase.add_argument(
        "--step2-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Override --input-dir when locating step-2 artifacts.",
    )
    phase.add_argument(
        "--models",
        nargs="+",
        default=["logistic", "elasticnet"],
        choices=["logistic", "elasticnet", "rf", "decision_tree"],
        help="Baseline classifiers to fit (default: logistic elasticnet).",
    )
    phase.add_argument(
        "--decision-threshold",
        type=float,
        default=0.5,
        metavar="F",
        help="Probability threshold for binary predictions (default: 0.5).",
    )
    phase.add_argument(
        "--top-n-importances",
        type=int,
        default=50,
        metavar="N",
        help="Keep the top-N genes per model in baseline_feature_importance.csv (default: 50).",
    )
    phase.add_argument(
        "--bootstrap-ci-n",
        type=int,
        default=500,
        metavar="N",
        help="Bootstrap resamples for overall-OOF confidence intervals (default: 500).",
    )
    phase.add_argument(
        "--calibration-bins",
        type=int,
        default=10,
        metavar="N",
        help="Number of reliability-diagram bins when --run-calibration is set (default: 10).",
    )
    phase.add_argument(
        "--calibration-strategy",
        choices=("uniform", "quantile"),
        default="uniform",
        help="Calibration binning strategy (default: uniform).",
    )
    phase.add_argument(
        "--environments",
        nargs="+",
        default=["smoking", "sex", "histology", "country", "stage"],
        metavar="NAME",
        help="Environment columns to stratify OOF metrics on (default matches step 2).",
    )
    phase.add_argument(
        "--run-calibration",
        action="store_true",
        help="Compute and plot probability calibration (Brier, reliability).",
    )
    phase.add_argument(
        "--run-subgroup-sensitivity",
        action="store_true",
        help="Report per-environment (histology/sex/smoking/...) subgroup metrics.",
    )
    add_rigor_profile_arg(parser)
    return parser


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def _output_paths(args: argparse.Namespace) -> dict[str, Path]:
    out = args.output_dir
    return {
        "oof": out / "baseline_oof_predictions.csv",
        "train": out / "baseline_train_predictions.csv",
        "per_fold": out / "baseline_per_fold_metrics.csv",
        "summary_metrics": out / "baseline_summary_metrics.csv",
        "importance": out / "baseline_feature_importance.csv",
        "subgroup": out / "baseline_subgroup_metrics.csv",
        "calibration": out / "baseline_calibration.csv",
        "summary_json": out / "phase3_summary.json",
    }


def _check_overwrite(args: argparse.Namespace, paths: dict[str, Path]) -> None:
    existing = [p for p in paths.values() if p.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Output files already exist; pass --overwrite to regenerate:\n  "
            + "\n  ".join(str(p) for p in existing)
        )


def _resolve_step2_dir(args: argparse.Namespace) -> Path:
    step2_dir: Path | None = args.step2_dir or args.input_dir
    if step2_dir is None:
        raise FileNotFoundError(
            "Step-2 directory not set; pass --input-dir or --step2-dir."
        )
    if not step2_dir.exists():
        raise FileNotFoundError(f"Step-2 directory does not exist: {step2_dir}")
    return step2_dir


def run_step3(args: argparse.Namespace) -> dict:
    """Execute the step-3 baseline pipeline and return a summary dict."""
    apply_rigor_profile(args)
    paths = _output_paths(args)
    _check_overwrite(args, paths)
    step2_dir = _resolve_step2_dir(args)

    # -------------------------------------------------------------------
    # 1. Load step-2 artifacts + align
    # -------------------------------------------------------------------
    artifacts = runner.load_step2_artifacts(step2_dir)
    aligned = runner.align_arrays(artifacts)
    X = aligned["X"]
    y = aligned["y"]
    groups = aligned["groups"]
    outer_fold = aligned["outer_fold"]
    master_rows = aligned["master_rows"]
    sample_barcodes = aligned["sample_barcodes"]
    gene_names = aligned["gene_names"]

    logger.info(
        "Input cohort: %d samples (%d tumor / %d normal) x %d genes, %d patients, %d outer folds",
        X.shape[0], int((y == 1).sum()), int((y == 0).sum()), X.shape[1],
        len(set(str(g) for g in groups)), len(set(int(f) for f in outer_fold)),
    )

    # -------------------------------------------------------------------
    # 2. Patient-grouped CV loop
    # -------------------------------------------------------------------
    cv_results = runner.run_grouped_cv(
        X, y, outer_fold, groups,
        model_names=args.models,
        seed=args.seed,
        decision_threshold=args.decision_threshold,
    )
    oof_probs = cv_results["oof_probs"]
    train_predictions = cv_results["train_predictions"]
    per_fold_rows = cv_results["per_fold_metrics"]
    feature_importances = cv_results["feature_importances"]
    model_configs = cv_results["model_configs"]

    # -------------------------------------------------------------------
    # 3. Persist OOF predictions
    # -------------------------------------------------------------------
    oof_rows = []
    for i, bc in enumerate(sample_barcodes):
        row: dict[str, object] = {
            "sample_barcode": bc,
            "patient_barcode": str(groups[i]),
            "outer_fold": int(outer_fold[i]),
            "y_true": int(y[i]),
        }
        for name in args.models:
            probs = oof_probs[name]
            val = probs[i]
            row[f"{name}_prob"] = _format_probability(float(val))
        oof_rows.append(row)
    pp.write_csv_records(paths["oof"], oof_rows)

    # -------------------------------------------------------------------
    # 3b. Persist train-fold predictions for diagnostic train ROC curves
    # -------------------------------------------------------------------
    train_rows = []
    for rec in train_predictions:
        i = int(rec["sample_index"])
        train_rows.append({
            "sample_barcode": sample_barcodes[i],
            "patient_barcode": str(groups[i]),
            "outer_fold": int(rec["outer_fold"]),
            "model": str(rec["model"]),
            "y_true": int(rec["y_true"]),
            "train_prob": _format_probability(float(rec["train_prob"])),
        })
    pp.write_csv_records(paths["train"], train_rows)

    # -------------------------------------------------------------------
    # 4. Per-fold metrics CSV
    # -------------------------------------------------------------------
    per_fold_csv = []
    for r in per_fold_rows:
        row = dict(r)
        for k, v in list(row.items()):
            if isinstance(v, float):
                row[k] = "" if (v != v) else f"{v:.6f}"
        per_fold_csv.append(row)
    pp.write_csv_records(paths["per_fold"], per_fold_csv)

    # -------------------------------------------------------------------
    # 5. Summary metrics (overall-OOF + mean-of-folds)
    # -------------------------------------------------------------------
    summary_rows = runner.summarize_oof_metrics(
        y=y,
        oof_probs=oof_probs,
        per_fold_rows=per_fold_rows,
        decision_threshold=args.decision_threshold,
        bootstrap_ci_n=args.bootstrap_ci_n,
        seed=args.seed,
    )
    summary_csv = []
    for r in summary_rows:
        row = dict(r)
        for k, v in list(row.items()):
            if isinstance(v, float):
                row[k] = "" if (v != v) else f"{v:.6f}"
        summary_csv.append(row)
    # Union of all keys so header is stable across aggregation rows
    all_keys: list[str] = []
    for r in summary_csv:
        for k in r:
            if k not in all_keys:
                all_keys.append(k)
    pp.write_csv_records(paths["summary_metrics"], summary_csv, fieldnames=all_keys)

    # -------------------------------------------------------------------
    # 6. Feature importances (top-N per model)
    # -------------------------------------------------------------------
    importance_rows = runner.aggregate_feature_importances(
        gene_names=gene_names,
        importances=feature_importances,
        top_n=args.top_n_importances,
    )
    pp.write_csv_records(paths["importance"], importance_rows)

    # -------------------------------------------------------------------
    # 7. Optional: subgroup sensitivity, calibration
    # -------------------------------------------------------------------
    subgroup_rows: list[dict] = []
    if args.run_subgroup_sensitivity:
        subgroup_rows = runner.compute_subgroup_metrics(
            master_rows=master_rows,
            y=y,
            oof_probs=oof_probs,
            env_names=args.environments,
            decision_threshold=args.decision_threshold,
        )
        if subgroup_rows:
            formatted = []
            for r in subgroup_rows:
                row = dict(r)
                for k, v in list(row.items()):
                    if isinstance(v, float):
                        row[k] = "" if (v != v) else f"{v:.6f}"
                formatted.append(row)
            pp.write_csv_records(paths["subgroup"], formatted)
        else:
            logger.warning("Subgroup sensitivity requested but no env_* columns found.")

    calibration_rows: list[dict] = []
    if args.run_calibration:
        calibration_rows = runner.compute_calibration(
            y=y,
            oof_probs=oof_probs,
            n_bins=args.calibration_bins,
            strategy=args.calibration_strategy,
        )
        if calibration_rows:
            formatted = []
            for r in calibration_rows:
                row = dict(r)
                for k, v in list(row.items()):
                    if isinstance(v, float):
                        row[k] = "" if (v != v) else f"{v:.6f}"
                formatted.append(row)
            pp.write_csv_records(paths["calibration"], formatted)

    # -------------------------------------------------------------------
    # 8. Figures (optional; graceful skip without matplotlib)
    # -------------------------------------------------------------------
    style = style_from_args(args)
    generated_figs, skipped_figs = runner.generate_step3_figures(
        y=y,
        oof_probs=oof_probs,
        train_predictions=train_predictions,
        per_fold_metrics=per_fold_rows,
        output_dir=args.output_dir,
        style=style,
        formats=style.default_formats,
        run_calibration=args.run_calibration,
        feature_importances=feature_importances,
        gene_names=gene_names,
        subgroup_rows=subgroup_rows if subgroup_rows else None,
    )
    for f in generated_figs:
        logger.info("figure OK: %s", f)
    for name, reason in skipped_figs:
        logger.warning("figure SKIPPED: %s (%s)", name, reason)

    # -------------------------------------------------------------------
    # 9. Phase 3 summary JSON
    # -------------------------------------------------------------------
    # Build a compact metric overview keyed by model
    overall_overview: dict[str, dict[str, float]] = {}
    for r in summary_rows:
        if r.get("aggregation") != "overall_oof":
            continue
        overall_overview[str(r["model"])] = {
            k: (None if isinstance(v, float) and v != v else v)
            for k, v in r.items()
            if k in {
                "auroc", "auroc_ci_lower", "auroc_ci_upper",
                "auprc", "auprc_ci_lower", "auprc_ci_upper",
                "balanced_accuracy", "f1", "brier", "log_loss",
                "sensitivity", "specificity", "n", "n_positive", "n_negative",
            }
        }

    summary = {
        "phase": "II (baselines)",
        "step": "step3_grouped_baselines",
        "cohort": {
            "n_samples": int(X.shape[0]),
            "n_tumor": int((y == 1).sum()),
            "n_normal": int((y == 0).sum()),
            "n_patients": len(set(str(g) for g in groups)),
            "n_genes": int(X.shape[1]),
            "n_outer_folds": len(cv_results["folds_used"]),
        },
        "config": {
            "models": list(args.models),
            "decision_threshold": float(args.decision_threshold),
            "top_n_importances": int(args.top_n_importances),
            "bootstrap_ci_n": int(args.bootstrap_ci_n),
            "calibration": args.run_calibration,
            "calibration_bins": int(args.calibration_bins),
            "calibration_strategy": args.calibration_strategy,
            "subgroup_sensitivity": args.run_subgroup_sensitivity,
            "environments": list(args.environments),
            "seed": int(args.seed),
            "step2_dir": str(step2_dir),
        },
        "model_configs": model_configs,
        "overall_oof_metrics": overall_overview,
        "figures": {
            "generated": generated_figs,
            "skipped": [{"name": n, "reason": r} for n, r in skipped_figs],
        },
        "style": style.as_dict(),
    }
    pp.write_json(paths["summary_json"], summary)

    # Flight-recorder log lines so reviewers can grep the run status.
    for name, overview in overall_overview.items():
        auc = overview.get("auroc", float("nan"))
        ap = overview.get("auprc", float("nan"))
        bac = overview.get("balanced_accuracy", float("nan"))
        brier = overview.get("brier", float("nan"))
        try:
            logger.info(
                "Overall OOF [%s]  AUROC=%.4f  AUPRC=%.4f  BAC=%.4f  Brier=%.4f",
                name, float(auc), float(ap), float(bac), float(brier),
            )
        except Exception:
            logger.info("Overall OOF [%s]  metrics=%s", name, overview)

    return summary


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    cli_args.apply_thread_limits(args)
    cli_args.ensure_output_dir(args)
    configure_logging(args, log_file=args.output_dir / "logs" / "step3_grouped_baselines.log")

    logger.info(
        "CAGE step 3 invoked | models=%s seed=%s threads=%s output=%s",
        args.models, args.seed, args.n_threads, args.output_dir,
    )
    apply_rigor_profile(args, parser_defaults={a.dest: a.default for a in parser._actions})
    run_step3(args)


if __name__ == "__main__":
    main()
