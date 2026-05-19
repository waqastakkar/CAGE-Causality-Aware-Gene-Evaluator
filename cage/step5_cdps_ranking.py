"""CAGE Step 5: Causality-Aware Priority Score ranking (Phase III).

Aggregates attribution scores, gate weights, cross-fold stability,
environment invariance, and in-silico perturbation effects into a single
Causality-Aware Priority Score (CDPS) per gene and emits ranked gene
tables for downstream biological validation.

Default CDPS composition:
    CDPS = 0.30 * attribution + 0.20 * gate + 0.20 * stability
         + 0.15 * invariance   + 0.15 * perturbation

Deliverables
------------
ranked_genes_cdps.csv        full genome-wide CDPS ranking
top25_genes_cdps.csv         top 25 CDPS genes (main text table)
top100_genes_cdps.csv        top 100 CDPS genes (supplementary)
gene_attribution_scores.csv  attribution component
gene_stability_scores.csv    stability component
gene_invariance_scores.csv   invariance component
gene_perturbation_scores.csv perturbation component (if enabled)
figures/                     heatmaps, scatter, CDPS component barplots

Run
---
python -m cage.step5_cdps_ranking --help
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

from . import cli_args, preprocess_esca as pp
from . import step5_runner as runner
from .cli_args import (
    add_rigor_profile_arg, apply_rigor_profile,
    build_step_parser, configure_logging, style_from_args,
)

logger = logging.getLogger("cage.step5")

_STEP_TITLE = "Phase III - Causality-Aware Priority Score (CDPS) ranking."
_STEP_DESCRIPTION = (
    "Integrates deep-model signals (attribution, gate, stability,\n"
    "invariance, perturbation) into a configurable CDPS ranking with\n"
    "reproducible fold/seed aggregation and publication-grade figures."
)
_INPUTS_DOC = (
    "deep_oof_predictions.csv   (from step 4)\n"
    "gate_weights.csv           (from step 4)\n"
    "latent_embeddings.csv      (from step 4)\n"
    "deep_summary_metrics.csv   (from step 4)\n"
    "checkpoints/               (from step 4, for attribution + perturbation)\n"
    "normalized_primary_matrix.csv, master_samples_primary.csv (from step 2)"
)
_OUTPUTS_DOC = (
    "ranked_genes_cdps.csv\n"
    "top25_genes_cdps.csv\n"
    "top100_genes_cdps.csv\n"
    "gene_attribution_scores.csv, gene_stability_scores.csv,\n"
    "gene_invariance_scores.csv, gene_perturbation_scores.csv\n"
    "figures/ : stability heatmap, attribution vs perturbation scatter,\n"
    "          CDPS component barplots"
)
_EXAMPLE = (
    "python -m cage.step5_cdps_ranking \\\n"
    "  --step2-dir outputs/step2_cohort \\\n"
    "  --step4-dir outputs/step4_deep_model \\\n"
    "  --output-dir outputs/step5_cdps \\\n"
    "  --attribution-method integrated-gradients \\\n"
    "  --top-ks 25 --run-perturbation"
)


def build_parser() -> argparse.ArgumentParser:
    parser = build_step_parser(
        prog="python -m cage.step5_cdps_ranking",
        step_title=_STEP_TITLE,
        step_description=_STEP_DESCRIPTION,
        inputs_doc=_INPUTS_DOC,
        outputs_doc=_OUTPUTS_DOC,
        example=_EXAMPLE,
        require_input_dir=False,
    )
    phase = parser.add_argument_group("Step-5 input directories")
    phase.add_argument(
        "--step2-dir", type=Path, default=None, metavar="DIR",
        help="Directory with step-2 cohort outputs (fallback: --input-dir).",
    )
    phase.add_argument(
        "--step4-dir", type=Path, default=None, metavar="DIR",
        help="Directory with step-4 deep-model outputs (required if --input-dir unset).",
    )
    phase.add_argument(
        "--env-names", nargs="+",
        default=["smoking", "sex", "histology", "country", "stage"],
        metavar="NAME",
        help="Environment columns used for the invariance component "
             "(default: smoking sex histology country stage).",
    )

    attr = parser.add_argument_group("Attribution")
    attr.add_argument(
        "--attribution-method",
        choices=["integrated-gradients", "grad-x-input", "gate-weight"],
        default="integrated-gradients",
        help="Attribution algorithm for the attribution component (default: integrated-gradients).",
    )
    attr.add_argument(
        "--attribution-sample-cap-per-class", type=int, default=100, metavar="N",
        help="Cap samples/class for attribution (default: 100).",
    )
    attr.add_argument(
        "--ig-steps", type=int, default=50, metavar="N",
        help="Trapezoidal steps for Integrated Gradients (default: 50).",
    )

    stab = parser.add_argument_group("Stability / normalization")
    stab.add_argument(
        "--stability-top-frac", type=float, default=0.05, metavar="F",
        help="Top fraction of genes (by |attribution|) counted per-fold for "
             "selection-frequency stability (default: 0.05).",
    )
    stab.add_argument(
        "--normalization", choices=["minmax", "rank"], default="minmax",
        help="Per-component normalization before CDPS blending (default: minmax).",
    )

    weights = parser.add_argument_group("CDPS component weights (default: 0.30/0.20/0.20/0.15/0.15)")
    weights.add_argument("--w-attribution", type=float, default=0.30, metavar="F")
    weights.add_argument("--w-gate",        type=float, default=0.20, metavar="F")
    weights.add_argument("--w-stability",   type=float, default=0.20, metavar="F")
    weights.add_argument("--w-invariance",  type=float, default=0.15, metavar="F")
    weights.add_argument("--w-perturbation", type=float, default=0.15, metavar="F")

    rank = parser.add_argument_group("Ranking output")
    rank.add_argument(
        "--top-ks", nargs="+", type=int, default=[25], metavar="K",
        help="Top-K gene tables to emit (default: 25).",
    )

    pert = parser.add_argument_group("In-silico perturbation")
    pert_group = pert.add_mutually_exclusive_group()
    pert_group.add_argument(
        "--run-perturbation", dest="run_perturbation", action="store_true",
        help="Enable tumor <-> normal expression perturbation analysis (default).",
    )
    pert_group.add_argument(
        "--no-perturbation", dest="run_perturbation", action="store_false",
        help="Skip perturbation; zero the perturbation component.",
    )
    pert.set_defaults(run_perturbation=True)
    add_rigor_profile_arg(parser)
    return parser


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def _output_paths(args: argparse.Namespace) -> Dict[str, Path]:
    out = args.output_dir
    paths: Dict[str, Path] = {
        "ranked": out / "ranked_genes_cdps.csv",
        "attribution": out / "gene_attribution_scores.csv",
        "stability": out / "gene_stability_scores.csv",
        "invariance": out / "gene_invariance_scores.csv",
        "perturbation": out / "gene_perturbation_scores.csv",
        "summary_json": out / "phase3_summary.json",
    }
    for k in args.top_ks:
        paths[f"top{int(k)}"] = out / f"top{int(k)}_genes_cdps.csv"
    return paths


def _check_overwrite(args: argparse.Namespace, paths: Mapping[str, Path]) -> None:
    existing = [p for p in paths.values() if p.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Output files already exist; pass --overwrite to regenerate:\n  "
            + "\n  ".join(str(p) for p in existing)
        )


def _resolve_dirs(args: argparse.Namespace) -> Dict[str, Path]:
    step2_dir: Path | None = args.step2_dir or args.input_dir
    step4_dir: Path | None = args.step4_dir or args.input_dir
    if step2_dir is None:
        raise FileNotFoundError(
            "Step-2 directory not set; pass --input-dir or --step2-dir."
        )
    if step4_dir is None:
        raise FileNotFoundError(
            "Step-4 directory not set; pass --input-dir or --step4-dir."
        )
    if not step2_dir.exists():
        raise FileNotFoundError(f"Step-2 directory does not exist: {step2_dir}")
    if not step4_dir.exists():
        raise FileNotFoundError(f"Step-4 directory does not exist: {step4_dir}")
    return {"step2": step2_dir, "step4": step4_dir}


def _fmt_f(v: Any) -> Any:
    if isinstance(v, float):
        return "" if (v != v) else f"{v:.6f}"
    return v


def _fmt_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    return [{k: _fmt_f(v) for k, v in r.items()} for r in rows]


def _build_attribution_rows(
    *,
    gene_names: Sequence[str],
    per_fold_mean_abs: np.ndarray,
    per_fold_signed_mean: np.ndarray,
    fold_ids: Sequence[int],
    method: str,
) -> List[Dict[str, Any]]:
    P = len(gene_names)
    mean = per_fold_mean_abs.mean(axis=0) if per_fold_mean_abs.size else np.zeros(P)
    std = per_fold_mean_abs.std(axis=0, ddof=0) if per_fold_mean_abs.size else np.zeros(P)
    signed = per_fold_signed_mean.mean(axis=0) if per_fold_signed_mean.size else np.zeros(P)
    rows: List[Dict[str, Any]] = []
    for j in range(P):
        row: Dict[str, Any] = {
            "gene": gene_names[j],
            "method": method,
            "attribution_mean_abs": float(mean[j]),
            "attribution_std_abs": float(std[j]),
            "attribution_signed_mean": float(signed[j]),
        }
        for k_idx, fid in enumerate(fold_ids):
            row[f"attribution_fold_{int(fid)}"] = float(per_fold_mean_abs[k_idx, j]) \
                if k_idx < per_fold_mean_abs.shape[0] else float("nan")
        rows.append(row)
    return rows


def _build_stability_rows(
    *,
    gene_names: Sequence[str],
    stability_out: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    P = len(gene_names)
    freq = stability_out["selection_frequency"]
    std = stability_out["attribution_std"]
    mean = stability_out["attribution_mean"]
    rel_std = stability_out["relative_std"]
    stability = stability_out["stability"]
    rows: List[Dict[str, Any]] = []
    for j in range(P):
        rows.append({
            "gene": gene_names[j],
            "selection_frequency": float(freq[j]),
            "attribution_mean": float(mean[j]),
            "attribution_std": float(std[j]),
            "relative_std": float(rel_std[j]),
            "stability": float(stability[j]),
        })
    return rows


def _build_invariance_rows(
    *,
    gene_names: Sequence[str],
    invariance_out: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    P = len(gene_names)
    avg_std = invariance_out["avg_env_std"]
    invariance = invariance_out["invariance"]
    env_used = invariance_out["env_used"]
    env_used_str = ";".join(env_used) if env_used else ""
    rows: List[Dict[str, Any]] = []
    for j in range(P):
        rows.append({
            "gene": gene_names[j],
            "env_columns_used": env_used_str,
            "avg_env_abs_attribution_std": float(avg_std[j]),
            "invariance": float(invariance[j]),
        })
    return rows


def _build_perturbation_rows(
    *,
    gene_names: Sequence[str],
    perturbation_out: Mapping[str, Any],
    fold_ids: Sequence[int],
) -> List[Dict[str, Any]]:
    P = len(gene_names)
    per_fold = perturbation_out["per_fold"]
    pooled = perturbation_out["perturbation"]
    rows: List[Dict[str, Any]] = []
    for j in range(P):
        row: Dict[str, Any] = {
            "gene": gene_names[j],
            "perturbation_mean_abs_dp": float(pooled[j]),
        }
        for k_idx, fid in enumerate(fold_ids):
            if k_idx < per_fold.shape[0]:
                row[f"perturbation_fold_{int(fid)}"] = float(per_fold[k_idx, j])
            else:
                row[f"perturbation_fold_{int(fid)}"] = float("nan")
        rows.append(row)
    return rows


def _ranked_records_to_rows(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    # Convert runner.records (already numeric) to the CSV-ready shape.
    return [dict(r) for r in records]


def run_step5(args: argparse.Namespace) -> Dict[str, Any]:
    """Execute the Step-5 CDPS pipeline and return the JSON summary."""
    apply_rigor_profile(args)
    paths = _output_paths(args)
    _check_overwrite(args, paths)
    dirs = _resolve_dirs(args)

    weights = {
        "attribution": float(args.w_attribution),
        "gate": float(args.w_gate),
        "stability": float(args.w_stability),
        "invariance": float(args.w_invariance),
        "perturbation": float(args.w_perturbation),
    }

    logger.info(
        "Resolving directories | step2=%s step4=%s output=%s",
        dirs["step2"], dirs["step4"], args.output_dir,
    )
    logger.info(
        "CDPS config | attribution=%s cap=%d ig_steps=%d perturbation=%s "
        "normalization=%s stability_top_frac=%.3f",
        args.attribution_method,
        int(args.attribution_sample_cap_per_class),
        int(args.ig_steps),
        bool(args.run_perturbation),
        str(args.normalization),
        float(args.stability_top_frac),
    )

    result = runner.run_step5_cdps(
        step2_dir=dirs["step2"],
        step4_dir=dirs["step4"],
        output_dir=args.output_dir,
        attribution_method=args.attribution_method,
        cap_per_class=int(args.attribution_sample_cap_per_class),
        ig_steps=int(args.ig_steps),
        run_perturbation=bool(args.run_perturbation),
        env_names=list(args.env_names),
        weights=weights,
        normalization=str(args.normalization),
        stability_top_frac=float(args.stability_top_frac),
        top_ks=list(args.top_ks),
        seed=int(args.seed),
    )

    gene_names: List[str] = result["gene_names"]
    fold_ids: List[int] = result["fold_ids"]
    attribution_out = result["attribution_out"]
    stability_out = result["stability_out"]
    invariance_out = result["invariance_out"]
    perturbation_out = result["perturbation_out"]
    gate_out = result["gate_out"]
    ranking = result["ranking"]

    # ------------------------------------------------------------------
    # 1. Full ranked table
    # ------------------------------------------------------------------
    ranked_rows = _ranked_records_to_rows(ranking["records"])
    pp.write_csv_records(paths["ranked"], _fmt_rows(ranked_rows))

    # ------------------------------------------------------------------
    # 2. Top-K tables
    # ------------------------------------------------------------------
    for k in args.top_ks:
        k = int(k)
        top_rows = ranked_rows[: min(k, len(ranked_rows))]
        pp.write_csv_records(paths[f"top{k}"], _fmt_rows(top_rows))

    # ------------------------------------------------------------------
    # 3. Per-component tables
    # ------------------------------------------------------------------
    attr_rows = _build_attribution_rows(
        gene_names=gene_names,
        per_fold_mean_abs=attribution_out["per_fold_mean_abs"],
        per_fold_signed_mean=attribution_out["per_fold_signed_mean"],
        fold_ids=fold_ids,
        method=attribution_out["method"],
    )
    pp.write_csv_records(paths["attribution"], _fmt_rows(attr_rows))

    stab_rows = _build_stability_rows(gene_names=gene_names, stability_out=stability_out)
    pp.write_csv_records(paths["stability"], _fmt_rows(stab_rows))

    inv_rows = _build_invariance_rows(gene_names=gene_names, invariance_out=invariance_out)
    pp.write_csv_records(paths["invariance"], _fmt_rows(inv_rows))

    if args.run_perturbation:
        pert_rows = _build_perturbation_rows(
            gene_names=gene_names, perturbation_out=perturbation_out, fold_ids=fold_ids,
        )
        pp.write_csv_records(paths["perturbation"], _fmt_rows(pert_rows))

    # ------------------------------------------------------------------
    # 4. Figures
    # ------------------------------------------------------------------
    style = style_from_args(args)
    generated_figs, skipped_figs = runner.generate_step5_figures(
        ranking=ranking,
        per_fold_attribution=attribution_out["per_fold_mean_abs"],
        gene_names=gene_names,
        output_dir=args.output_dir,
        style=style,
        formats=style.default_formats,
        top_k_heat=min(25, max(args.top_ks) if args.top_ks else 25),
    )
    for f in generated_figs:
        logger.info("figure OK: %s", f)
    for name, reason in skipped_figs:
        logger.warning("figure SKIPPED: %s (%s)", name, reason)

    # ------------------------------------------------------------------
    # 5. Phase-3 summary JSON
    # ------------------------------------------------------------------
    summary = {
        "phase": "III (CDPS ranking)",
        "step": "step5_cdps_ranking",
        "cohort": {
            "n_genes": int(len(gene_names)),
            "n_samples": int(len(result["sample_barcodes"])),
            "n_outer_folds": int(len(fold_ids)),
        },
        "config": result["config"],
        "weights_effective": ranking["weights_effective"],
        "weights_raw": ranking["weights_raw"],
        "normalization": ranking["normalization"],
        "top_k_tables": [int(k) for k in args.top_ks],
        "invariance_env_used": list(invariance_out["env_used"]),
        "gate_overview": {
            "mean_across_folds_max": float(gate_out["mean"].max()) if gate_out["mean"].size else 0.0,
            "mean_across_folds_min": float(gate_out["mean"].min()) if gate_out["mean"].size else 0.0,
            "mean_across_folds_median": float(np.median(gate_out["mean"])) if gate_out["mean"].size else 0.0,
        },
        "top_genes": [
            {
                "rank": int(rec["rank"]),
                "gene": str(rec["gene"]),
                "cdps": float(rec["cdps"]),
                "attribution_norm": float(rec["attribution_norm"]),
                "gate_norm": float(rec["gate_norm"]),
                "stability_norm": float(rec["stability_norm"]),
                "invariance_norm": float(rec["invariance_norm"]),
                "perturbation_norm": float(rec["perturbation_norm"]),
            }
            for rec in ranking["records"][: max(args.top_ks) if args.top_ks else 25]
        ],
        "figures": {
            "generated": generated_figs,
            "skipped": [{"name": n, "reason": r} for n, r in skipped_figs],
        },
        "style": style.as_dict(),
    }
    pp.write_json(paths["summary_json"], summary)

    # Flight-recorder line
    top1 = ranking["records"][0] if ranking["records"] else None
    if top1 is not None:
        logger.info(
            "CDPS top-1: %s (cdps=%.4f) | weights=%s",
            top1["gene"], float(top1["cdps"]), ranking["weights_effective"],
        )
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    cli_args.apply_thread_limits(args)
    cli_args.ensure_output_dir(args)
    configure_logging(args, log_file=args.output_dir / "logs" / "step5_cdps_ranking.log")
    style = style_from_args(args)

    logger.info(
        "CAGE step 5 invoked | attribution=%s weights=(%.2f,%.2f,%.2f,%.2f,%.2f)",
        args.attribution_method,
        args.w_attribution, args.w_gate, args.w_stability,
        args.w_invariance, args.w_perturbation,
    )
    logger.debug("figure style: %s", style.as_dict())

    apply_rigor_profile(args, parser_defaults={a.dest: a.default for a in parser._actions})
    run_step5(args)


if __name__ == "__main__":
    main()
