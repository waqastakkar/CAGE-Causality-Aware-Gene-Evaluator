"""CAGE Step 4: Sparse invariant autoencoder-classifier (Phase II).

Trains a confounder-aware sparse autoencoder-classifier with a learnable
per-gene gate, a disease-classification head, an adversarial confounder
module, an optional reconstruction decoder, and an invariance regularizer
over environment strata. Produces out-of-fold predictions, latent
embeddings, gate weights, and model checkpoints for CDPS ranking.

Deliverables
------------
deep_oof_predictions.csv     per-sample tumor probability (OOF, all folds)
gate_weights.csv             learned sparse gate weights per gene/fold (long)
latent_embeddings.csv        per-sample latent vectors
deep_per_fold_metrics.csv    per-fold AUROC/AUPRC/BAC/Brier/...
deep_summary_metrics.csv     overall-OOF + mean-of-folds with CIs
deep_subgroup_metrics.csv    (if --run-subgroup-sensitivity)
deep_calibration.csv         (if --run-calibration)
deep_training_history.csv    per-fold per-epoch training / validation loss
checkpoints/                 trained model weights per fold (npz)
phase2_summary.json          configuration + metrics + figure manifest
figures/                     latent PCA, gate distributions, ROC/PR

Run
---
python -m cage.step4_sparse_invariant_model --help
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

from . import cli_args, preprocess_esca as pp
from . import step3_runner as step3
from . import step4_runner as runner
from .cli_args import (
    add_rigor_profile_arg, apply_rigor_profile,
    build_step_parser, configure_logging, style_from_args,
)
from .step4_runner import TrainingConfig

logger = logging.getLogger("cage.step4")

_STEP_TITLE = "Phase II - Sparse, confounder-aware invariant deep model."
_STEP_DESCRIPTION = (
    "Trains the sparse invariant adversarial autoencoder-classifier over\n"
    "patient-grouped nested CV folds. Combines classification, sparsity,\n"
    "adversarial confounder, and invariance losses (optional reconstruction)\n"
    "to produce latent embeddings, gene gate weights, and OOF predictions."
)
_INPUTS_DOC = (
    "normalized_primary_matrix.csv (from step 2)\n"
    "master_samples_primary.csv    (from step 2)\n"
    "grouped_outer_folds.csv       (from step 2)\n"
    "(optional) step 3 baseline predictions for comparison logging"
)
_OUTPUTS_DOC = (
    "deep_oof_predictions.csv\n"
    "gate_weights.csv\n"
    "latent_embeddings.csv\n"
    "deep_per_fold_metrics.csv\n"
    "deep_summary_metrics.csv\n"
    "deep_subgroup_metrics.csv (if --run-subgroup-sensitivity)\n"
    "deep_calibration.csv      (if --run-calibration)\n"
    "deep_training_history.csv\n"
    "checkpoints/ : trained models per fold (npz)\n"
    "phase2_summary.json\n"
    "figures/ : latent PCA, gate distributions, ROC/PR"
)
_EXAMPLE = (
    "python -m cage.step4_sparse_invariant_model \\\n"
    "  --input-dir outputs/step2_cohort \\\n"
    "  --output-dir outputs/step4_deep_model \\\n"
    "  --latent-dim 48 --n-epochs 150 --batch-size 64 \\\n"
    "  --sparsity-lambda 1e-3 --adv-lambda 0.5 --invariance-lambda 0.1 \\\n"
    "  --run-calibration --run-subgroup-sensitivity"
)


def build_parser() -> argparse.ArgumentParser:
    parser = build_step_parser(
        prog="python -m cage.step4_sparse_invariant_model",
        step_title=_STEP_TITLE,
        step_description=_STEP_DESCRIPTION,
        inputs_doc=_INPUTS_DOC,
        outputs_doc=_OUTPUTS_DOC,
        example=_EXAMPLE,
    )
    phase = parser.add_argument_group("Step-4 phase-specific options")
    phase.add_argument(
        "--step2-dir", type=Path, default=None, metavar="DIR",
        help="Override --input-dir when locating step-2 artifacts.",
    )
    phase.add_argument(
        "--confounder-column", type=str, default="env_histology", metavar="COL",
        help=(
            "Master-table column to treat as the adversary's target "
            "(default: env_histology). Accepts raw categorical columns "
            "(e.g. histology, sex, smoking_history) or env_* indicators."
        ),
    )
    phase.add_argument(
        "--environment-column", type=str, default="env_sex", metavar="COL",
        help=(
            "Master-table column whose strata drive the latent invariance "
            "penalty (default: env_sex). Accepts raw or env_* columns."
        ),
    )
    phase.add_argument(
        "--environments",
        nargs="+",
        default=["smoking", "sex", "histology", "country", "stage"],
        metavar="NAME",
        help="Environment columns used for subgroup-sensitivity metrics "
             "(default matches Step 2 env strata).",
    )
    phase.add_argument(
        "--decision-threshold", type=float, default=0.5, metavar="F",
        help="Probability threshold for binary predictions (default: 0.5).",
    )
    phase.add_argument(
        "--bootstrap-ci-n", type=int, default=500, metavar="N",
        help="Bootstrap resamples for overall-OOF confidence intervals (default: 500).",
    )
    phase.add_argument(
        "--calibration-bins", type=int, default=10, metavar="N",
        help="Reliability-diagram bin count when --run-calibration is set (default: 10).",
    )
    phase.add_argument(
        "--calibration-strategy", choices=("uniform", "quantile"), default="uniform",
        help="Calibration binning strategy (default: uniform).",
    )
    phase.add_argument(
        "--run-calibration", action="store_true",
        help="Compute and plot probability calibration (Brier, reliability).",
    )
    phase.add_argument(
        "--run-subgroup-sensitivity", action="store_true",
        help="Report per-environment subgroup metrics for the deep model.",
    )
    phase.add_argument(
        "--save-latent-embeddings", dest="save_latent", action="store_true",
        default=True,
        help="Save per-sample OOF latent coordinates (default: on).",
    )
    phase.add_argument(
        "--no-save-latent-embeddings", dest="save_latent", action="store_false",
        help="Skip writing latent_embeddings.csv (saves disk on large cohorts).",
    )

    arch = parser.add_argument_group("Model architecture")
    arch.add_argument(
        "--latent-dim", type=int, default=48, metavar="D",
        help="Latent bottleneck dimension (default: 48).",
    )
    arch.add_argument(
        "--hidden-dims", nargs="+", type=int, default=[256, 96], metavar="D",
        help="Encoder hidden dimensions, outer-to-inner (default: 256 96).",
    )
    arch.add_argument(
        "--sparsity-type", choices=["l1", "hard-concrete"], default="l1",
        help="Per-gene gate regularization type (default: l1).",
    )
    arch.add_argument(
        "--dropout", type=float, default=0.1, metavar="F",
        help="Dropout probability in encoder hidden layers (default: 0.1).",
    )
    decoder = arch.add_mutually_exclusive_group()
    decoder.add_argument(
        "--use-decoder", dest="use_decoder", action="store_true",
        help="Enable reconstruction decoder (default).",
    )
    decoder.add_argument(
        "--no-decoder", dest="use_decoder", action="store_false",
        help="Disable reconstruction decoder (classification-only).",
    )
    arch.set_defaults(use_decoder=True)

    train = parser.add_argument_group("Training")
    train.add_argument("--n-epochs", type=int, default=150, metavar="N",
                       help="Maximum training epochs per fold (default: 150).")
    train.add_argument("--batch-size", type=int, default=64, metavar="N",
                       help="Mini-batch size (default: 64).")
    train.add_argument("--lr", type=float, default=1e-3, metavar="F",
                       help="AdamW learning rate (default: 1e-3).")
    train.add_argument("--weight-decay", type=float, default=1e-4, metavar="F",
                       help="AdamW weight decay (default: 1e-4).")
    train.add_argument("--patience", type=int, default=15, metavar="N",
                       help="Early-stopping patience on validation AUROC (default: 15).")
    train.add_argument("--adv-ramp-epochs", type=int, default=5, metavar="N",
                       help="Linearly ramp the adversarial weight from 0 over N epochs (default: 5).")
    train.add_argument("--grad-clip-norm", type=float, default=5.0, metavar="F",
                       help="Global L2 gradient-norm clip per batch; <=0 disables (default: 5.0).")
    train.add_argument("--mixed-precision", action="store_true",
                       help="Reserved flag (numpy backend is always float64; no-op).")
    train.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                       help="Reserved flag (numpy backend only); default: auto.")

    losses = parser.add_argument_group("Loss weights")
    losses.add_argument("--sparsity-lambda", type=float, default=1e-3, metavar="F",
                        help="Sparsity penalty weight (default: 1e-3).")
    losses.add_argument("--adv-lambda", type=float, default=0.5, metavar="F",
                        help="Adversarial confounder loss weight (default: 0.5).")
    losses.add_argument("--invariance-lambda", type=float, default=0.1, metavar="F",
                        help="Environment invariance penalty weight (default: 0.1).")
    losses.add_argument("--recon-lambda", type=float, default=0.1, metavar="F",
                        help="Reconstruction loss weight (default: 0.1).")

    attr = parser.add_argument_group("Attribution preparation")
    attr.add_argument(
        "--attribution-sample-cap-per-class", type=int, default=100, metavar="N",
        help="Cap samples/class used for integrated-gradient attribution to manage memory (default: 100).",
    )
    add_rigor_profile_arg(parser)
    return parser


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def _output_paths(args: argparse.Namespace) -> Dict[str, Path]:
    out = args.output_dir
    return {
        "oof": out / "deep_oof_predictions.csv",
        "train": out / "deep_train_predictions.csv",
        "per_fold": out / "deep_per_fold_metrics.csv",
        "summary_metrics": out / "deep_summary_metrics.csv",
        "gate": out / "gate_weights.csv",
        "latent": out / "latent_embeddings.csv",
        "history": out / "deep_training_history.csv",
        "subgroup": out / "deep_subgroup_metrics.csv",
        "calibration": out / "deep_calibration.csv",
        "summary_json": out / "phase2_summary.json",
        "checkpoint_dir": out / "checkpoints",
    }


def _check_overwrite(args: argparse.Namespace, paths: Mapping[str, Path]) -> None:
    to_check = [p for k, p in paths.items() if k != "checkpoint_dir"]
    existing = [p for p in to_check if p.exists()]
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


def _build_config(args: argparse.Namespace) -> TrainingConfig:
    return TrainingConfig(
        latent_dim=int(args.latent_dim),
        hidden_dims=tuple(int(h) for h in args.hidden_dims),
        sparsity_type=str(args.sparsity_type),
        use_decoder=bool(args.use_decoder),
        dropout=float(args.dropout),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        n_epochs=int(args.n_epochs),
        batch_size=int(args.batch_size),
        patience=int(args.patience),
        lambda_recon=float(args.recon_lambda),
        lambda_sparsity=float(args.sparsity_lambda),
        lambda_adv=float(args.adv_lambda),
        lambda_inv=float(args.invariance_lambda),
        adv_ramp_epochs=int(args.adv_ramp_epochs),
        grad_clip_norm=float(args.grad_clip_norm),
        seed=int(args.seed),
    )


def _format_float(v: Any) -> Any:
    if isinstance(v, float):
        return "" if (v != v) else f"{v:.6f}"
    return v


def _format_probability(v: float) -> str:
    """Write prediction probabilities without collapsing near-ties."""
    return "" if (v != v) else f"{float(v):.17g}"


def _format_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    formatted: List[Dict[str, Any]] = []
    for r in rows:
        formatted.append({k: _format_float(v) for k, v in r.items()})
    return formatted


def _build_oof_rows(
    *,
    sample_barcodes: Sequence[str],
    groups: np.ndarray,
    outer_fold: np.ndarray,
    y: np.ndarray,
    oof_probs: np.ndarray,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for i, bc in enumerate(sample_barcodes):
        p = oof_probs[i]
        rows.append({
            "sample_barcode": bc,
            "patient_barcode": str(groups[i]),
            "outer_fold": int(outer_fold[i]),
            "y_true": int(y[i]),
            "deep_prob": _format_probability(float(p)),
        })
    return rows


def _build_gate_rows(
    gate_weights_per_fold: np.ndarray,
    fold_ids: Sequence[int],
    gene_names: Sequence[str],
) -> List[Dict[str, Any]]:
    K, P = gate_weights_per_fold.shape
    rows: List[Dict[str, Any]] = []
    mean_across = gate_weights_per_fold.mean(axis=0)
    std_across = gate_weights_per_fold.std(axis=0)
    # Ranking by mean across folds (descending)
    order = np.argsort(-mean_across, kind="mergesort")
    ranks = np.empty(P, dtype=np.int64)
    ranks[order] = np.arange(1, P + 1)
    for j in range(P):
        row: Dict[str, Any] = {
            "gene": gene_names[j],
            "rank_by_mean": int(ranks[j]),
            "gate_mean_across_folds": f"{float(mean_across[j]):.6f}",
            "gate_std_across_folds": f"{float(std_across[j]):.6f}",
        }
        for k_idx, fid in enumerate(fold_ids):
            row[f"gate_fold_{int(fid)}"] = f"{float(gate_weights_per_fold[k_idx, j]):.6f}"
        rows.append(row)
    return rows


def _build_latent_rows(
    *,
    sample_barcodes: Sequence[str],
    groups: np.ndarray,
    outer_fold: np.ndarray,
    y: np.ndarray,
    oof_latents: np.ndarray,
) -> List[Dict[str, Any]]:
    n, d = oof_latents.shape
    rows: List[Dict[str, Any]] = []
    for i, bc in enumerate(sample_barcodes):
        row: Dict[str, Any] = {
            "sample_barcode": bc,
            "patient_barcode": str(groups[i]),
            "outer_fold": int(outer_fold[i]),
            "y_true": int(y[i]),
        }
        for j in range(d):
            v = oof_latents[i, j]
            row[f"z{j+1:02d}"] = "" if (v != v) else f"{float(v):.6f}"
        rows.append(row)
    return rows


def _summarize_per_fold(per_fold: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Compute mean/std across folds for headline metrics."""
    keys = ["auroc", "auprc", "balanced_accuracy", "f1", "brier",
            "log_loss", "sensitivity", "specificity"]
    summary: Dict[str, Any] = {"n_folds": len(per_fold)}
    for k in keys:
        vals = np.array([float(r.get(k, float("nan"))) for r in per_fold],
                        dtype=np.float64)
        finite = vals[np.isfinite(vals)]
        if finite.size:
            summary[f"{k}_mean"] = float(finite.mean())
            summary[f"{k}_std"] = float(finite.std(ddof=0))
            summary[f"{k}_n_valid_folds"] = int(finite.size)
        else:
            summary[f"{k}_mean"] = float("nan")
            summary[f"{k}_std"] = float("nan")
            summary[f"{k}_n_valid_folds"] = 0
    return summary


def _summarize_overall_oof(
    y: np.ndarray,
    oof_probs: np.ndarray,
    *,
    decision_threshold: float,
    bootstrap_ci_n: int,
    seed: int,
) -> Dict[str, Any]:
    from . import metrics as mx

    valid = ~np.isnan(oof_probs)
    y_valid = y[valid]
    p_valid = oof_probs[valid]
    y_pred = (p_valid >= decision_threshold).astype(np.int64)

    auc_ci = mx.bootstrap_ci(mx.auroc, y_valid, p_valid,
                             n_boot=bootstrap_ci_n, seed=seed)
    ap_ci = mx.bootstrap_ci(mx.auprc, y_valid, p_valid,
                            n_boot=bootstrap_ci_n, seed=seed + 1)
    sens, spec = mx.sensitivity_specificity(y_valid, y_pred)
    return {
        "aggregation": "overall_oof",
        "n": int(valid.sum()),
        "n_positive": int((y_valid == 1).sum()),
        "n_negative": int((y_valid == 0).sum()),
        "threshold": float(decision_threshold),
        "auroc": mx.auroc(y_valid, p_valid),
        "auroc_ci_lower": auc_ci["lower"],
        "auroc_ci_upper": auc_ci["upper"],
        "auprc": mx.auprc(y_valid, p_valid),
        "auprc_ci_lower": ap_ci["lower"],
        "auprc_ci_upper": ap_ci["upper"],
        "balanced_accuracy": mx.balanced_accuracy(y_valid, y_pred),
        "f1": mx.f1_score(y_valid, y_pred),
        "brier": mx.brier_score(y_valid, p_valid),
        "log_loss": mx.log_loss(y_valid, p_valid),
        "sensitivity": sens,
        "specificity": spec,
    }


def run_step4(args: argparse.Namespace) -> Dict[str, Any]:
    """Execute the Step-4 sparse-invariant pipeline and return a summary dict."""
    apply_rigor_profile(args)
    paths = _output_paths(args)
    _check_overwrite(args, paths)
    step2_dir = _resolve_step2_dir(args)
    paths["checkpoint_dir"].mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------
    # 1. Load step-2 artifacts + align
    # -------------------------------------------------------------------
    artifacts = step3.load_step2_artifacts(step2_dir)
    aligned = step3.align_arrays(artifacts)
    X: np.ndarray = aligned["X"]
    y: np.ndarray = aligned["y"]
    groups: np.ndarray = aligned["groups"]
    outer_fold: np.ndarray = aligned["outer_fold"]
    master_rows = aligned["master_rows"]
    fold_rows = aligned["fold_rows"]
    sample_barcodes = aligned["sample_barcodes"]
    gene_names = aligned["gene_names"]

    step3.assert_no_patient_leakage(groups, outer_fold)

    logger.info(
        "Input cohort: %d samples (%d tumor / %d normal) x %d genes, %d patients, %d outer folds",
        X.shape[0], int((y == 1).sum()), int((y == 0).sum()), X.shape[1],
        len(set(str(g) for g in groups)), len(set(int(f) for f in outer_fold)),
    )

    # -------------------------------------------------------------------
    # 2. Build config and run CV training
    # -------------------------------------------------------------------
    config = _build_config(args)
    logger.info(
        "Deep-model config | latent=%d hidden=%s sparsity=%s decoder=%s "
        "lr=%.1e wd=%.1e epochs=%d batch=%d patience=%d",
        config.latent_dim, config.hidden_dims, config.sparsity_type,
        config.use_decoder, config.lr, config.weight_decay,
        config.n_epochs, config.batch_size, config.patience,
    )
    logger.info(
        "Loss weights | recon=%.3g sparsity=%.3g adv=%.3g inv=%.3g (ramp=%d)",
        config.lambda_recon, config.lambda_sparsity, config.lambda_adv,
        config.lambda_inv, config.adv_ramp_epochs,
    )

    # Model size report — written before training so reviewers can inspect compute demands
    n_genes_actual = X.shape[1]
    n_samples_actual = X.shape[0]
    n_params = (
        n_genes_actual
        + n_genes_actual * (config.hidden_dims[0] if config.hidden_dims else 128)
        + (config.hidden_dims[0] if config.hidden_dims else 128) * config.latent_dim
        + config.latent_dim * 2
    )
    mem_mb = (n_params * 8 * 3) / 1e6
    n_outer = len(set(int(f) for f in outer_fold))
    time_per_epoch_s = 0.5 * (n_genes_actual / 1000) * (n_samples_actual / 200)
    total_hours = (time_per_epoch_s * config.n_epochs * n_outer) / 3600
    model_size_report = {
        "n_genes_input": n_genes_actual,
        "n_samples": n_samples_actual,
        "n_outer_folds": n_outer,
        "latent_dim": config.latent_dim,
        "hidden_dims": list(config.hidden_dims),
        "n_epochs": config.n_epochs,
        "estimated_trainable_params": n_params,
        "estimated_memory_mb": round(mem_mb, 1),
        "estimated_training_hours_4core": round(total_hours, 1),
        "rigor_profile": getattr(args, "rigor_profile", "standard"),
    }
    pp.write_json(paths["summary_json"].parent / "model_size_report.json", model_size_report)
    logger.info("Model size report written | genes=%d params=%d est_mem=%.0f MB",
                n_genes_actual, n_params, mem_mb)
    if mem_mb > 4000:
        logger.warning("MODEL SIZE: %.0f MB for %d genes. Consider --n-top-variable-genes 20000.",
                       mem_mb, n_genes_actual)
    if total_hours > 24:
        logger.warning("RUNTIME: estimated %.1f hours. Plan compute time accordingly.", total_hours)

    cv = runner.run_step4_cv(
        X=X, y=y, outer_fold=outer_fold, groups=groups,
        master_rows=master_rows, fold_rows=fold_rows,
        gene_names=gene_names, sample_barcodes=sample_barcodes,
        config=config,
        confounder_column=args.confounder_column,
        environment_column=args.environment_column,
        checkpoint_dir=paths["checkpoint_dir"],
    )

    oof_probs: np.ndarray = cv["oof_probs"]
    train_predictions = cv["train_predictions"]
    oof_latents: np.ndarray = cv["oof_latents"]
    gate_per_fold: np.ndarray = cv["gate_weights_per_fold"]
    per_fold_metrics = cv["per_fold_metrics"]
    training_history = cv["training_history"]
    fold_info = cv["fold_info"]
    fold_ids = cv["folds_used"]

    # -------------------------------------------------------------------
    # 3. Persist OOF predictions
    # -------------------------------------------------------------------
    oof_rows = _build_oof_rows(
        sample_barcodes=sample_barcodes, groups=groups,
        outer_fold=outer_fold, y=y, oof_probs=oof_probs,
    )
    pp.write_csv_records(paths["oof"], oof_rows)

    # -------------------------------------------------------------------
    # 3b. Persist train-fold predictions for diagnostic train ROC curves
    # -------------------------------------------------------------------
    train_rows: List[Dict[str, Any]] = []
    for rec in train_predictions:
        i = int(rec["sample_index"])
        p = float(rec["deep_train_prob"])
        train_rows.append({
            "sample_barcode": sample_barcodes[i],
            "patient_barcode": str(groups[i]),
            "outer_fold": int(rec["outer_fold"]),
            "y_true": int(rec["y_true"]),
            "deep_train_prob": _format_probability(p),
        })
    pp.write_csv_records(paths["train"], train_rows)

    # -------------------------------------------------------------------
    # 4. Per-fold metrics CSV
    # -------------------------------------------------------------------
    per_fold_csv = _format_rows(per_fold_metrics)
    pp.write_csv_records(paths["per_fold"], per_fold_csv)

    # -------------------------------------------------------------------
    # 5. Summary metrics (overall OOF + mean-of-folds)
    # -------------------------------------------------------------------
    overall = _summarize_overall_oof(
        y=y, oof_probs=oof_probs,
        decision_threshold=args.decision_threshold,
        bootstrap_ci_n=args.bootstrap_ci_n,
        seed=args.seed,
    )
    mean_row = {"aggregation": "mean_of_folds", **_summarize_per_fold(per_fold_metrics)}
    summary_rows = [{"model": "deep", **overall}, {"model": "deep", **mean_row}]
    summary_csv = _format_rows(summary_rows)
    # Union of keys so the header is stable across rows
    all_keys: List[str] = []
    for r in summary_csv:
        for k in r:
            if k not in all_keys:
                all_keys.append(k)
    pp.write_csv_records(paths["summary_metrics"], summary_csv, fieldnames=all_keys)

    # -------------------------------------------------------------------
    # 6. Gate weights (long form, one row per gene with per-fold columns)
    # -------------------------------------------------------------------
    gate_rows = _build_gate_rows(gate_per_fold, fold_ids, gene_names)
    pp.write_csv_records(paths["gate"], gate_rows)

    # -------------------------------------------------------------------
    # 7. Latent embeddings (optional)
    # -------------------------------------------------------------------
    if args.save_latent:
        latent_rows = _build_latent_rows(
            sample_barcodes=sample_barcodes, groups=groups,
            outer_fold=outer_fold, y=y, oof_latents=oof_latents,
        )
        pp.write_csv_records(paths["latent"], latent_rows)

    # -------------------------------------------------------------------
    # 8. Training history
    # -------------------------------------------------------------------
    hist_rows = _format_rows(training_history)
    pp.write_csv_records(paths["history"], hist_rows)

    # -------------------------------------------------------------------
    # 9. Optional: subgroup sensitivity + calibration
    # -------------------------------------------------------------------
    subgroup_rows: List[Dict[str, Any]] = []
    if args.run_subgroup_sensitivity:
        subgroup_rows = step3.compute_subgroup_metrics(
            master_rows=master_rows,
            y=y,
            oof_probs={"deep": oof_probs},
            env_names=args.environments,
            decision_threshold=args.decision_threshold,
        )
        if subgroup_rows:
            pp.write_csv_records(paths["subgroup"], _format_rows(subgroup_rows))
        else:
            logger.warning("Subgroup sensitivity requested but no env_* columns found.")

    calibration_rows: List[Dict[str, Any]] = []
    if args.run_calibration:
        calibration_rows = step3.compute_calibration(
            y=y,
            oof_probs={"deep": oof_probs},
            n_bins=args.calibration_bins,
            strategy=args.calibration_strategy,
        )
        if calibration_rows:
            pp.write_csv_records(paths["calibration"], _format_rows(calibration_rows))

    # -------------------------------------------------------------------
    # 10. Figures (graceful skip without matplotlib)
    # -------------------------------------------------------------------
    style = style_from_args(args)
    generated_figs, skipped_figs = runner.generate_step4_figures(
        y=y,
        oof_probs=oof_probs,
        train_predictions=train_predictions,
        oof_latents=oof_latents,
        gate_weights_per_fold=gate_per_fold,
        master_rows=master_rows,
        output_dir=args.output_dir,
        style=style,
        formats=style.default_formats,
        training_history=training_history,
        per_fold_metrics=per_fold_metrics,
        gene_names=gene_names,
        oof_fold_ids=outer_fold,
        subgroup_rows=subgroup_rows if subgroup_rows else None,
    )
    for f in generated_figs:
        logger.info("figure OK: %s", f)
    for name, reason in skipped_figs:
        logger.warning("figure SKIPPED: %s (%s)", name, reason)

    # -------------------------------------------------------------------
    # 11. Phase-2 summary JSON
    # -------------------------------------------------------------------
    overall_overview = {
        k: (None if (isinstance(v, float) and v != v) else v)
        for k, v in overall.items()
        if k not in {"aggregation"}
    }

    summary = {
        "phase": "II (sparse invariant deep model)",
        "step": "step4_sparse_invariant_model",
        "cohort": {
            "n_samples": int(X.shape[0]),
            "n_tumor": int((y == 1).sum()),
            "n_normal": int((y == 0).sum()),
            "n_patients": len(set(str(g) for g in groups)),
            "n_genes": int(X.shape[1]),
            "n_outer_folds": len(fold_ids),
        },
        "config": {
            "confounder_column": str(args.confounder_column),
            "environment_column": str(args.environment_column),
            "environments": list(args.environments),
            "decision_threshold": float(args.decision_threshold),
            "bootstrap_ci_n": int(args.bootstrap_ci_n),
            "calibration": args.run_calibration,
            "calibration_bins": int(args.calibration_bins),
            "calibration_strategy": args.calibration_strategy,
            "subgroup_sensitivity": args.run_subgroup_sensitivity,
            "save_latent_embeddings": bool(args.save_latent),
            "seed": int(args.seed),
            "step2_dir": str(step2_dir),
            "mixed_precision": bool(args.mixed_precision),
            "device": str(args.device),
            "attribution_sample_cap_per_class": int(args.attribution_sample_cap_per_class),
        },
        "model_config": config.as_dict(),
        "confounder_levels": list(cv.get("confounder_levels") or []),
        "environment_levels": list(cv.get("environment_levels") or []),
        "overall_oof_metrics": overall_overview,
        "mean_of_folds": {k: (None if (isinstance(v, float) and v != v) else v)
                          for k, v in mean_row.items() if k != "aggregation"},
        "fold_info": [
            {
                "fold_id": int(info.get("fold_id", -1)),
                "best_epoch": int(info.get("best_epoch", -1)),
                "best_val_auroc": float(info.get("best_val_auroc", float("nan"))),
                "best_val_bac": float(info.get("best_val_bac", float("nan"))),
                "best_val_loss": float(info.get("best_val_loss", float("nan"))),
                "gate_mean": float(info.get("gate_mean", float("nan"))),
                "gate_median": float(info.get("gate_median", float("nan"))),
                "gate_frac_below_0_1": float(info.get("gate_frac_below_0_1", float("nan"))),
                "gate_frac_below_0_01": float(info.get("gate_frac_below_0_01", float("nan"))),
            }
            for info in fold_info
        ],
        "gate_aggregates": {
            "mean_across_folds_top10": [
                {"gene": gene_names[int(idx)],
                 "gate_mean": float(gate_per_fold.mean(axis=0)[int(idx)])}
                for idx in np.argsort(-gate_per_fold.mean(axis=0), kind="mergesort")[:10]
            ],
            "frac_genes_mean_below_0_1": float(
                (gate_per_fold.mean(axis=0) < 0.1).mean()
            ),
            "frac_genes_mean_below_0_01": float(
                (gate_per_fold.mean(axis=0) < 0.01).mean()
            ),
        },
        "checkpoints": [
            str((paths["checkpoint_dir"] / f"fold_{int(fid)}.npz").name)
            for fid in fold_ids
        ],
        "figures": {
            "generated": generated_figs,
            "skipped": [{"name": n, "reason": r} for n, r in skipped_figs],
        },
        "style": style.as_dict(),
    }
    pp.write_json(paths["summary_json"], summary)

    # Flight-recorder log line
    try:
        logger.info(
            "Overall OOF [deep]  AUROC=%.4f  AUPRC=%.4f  BAC=%.4f  Brier=%.4f",
            float(overall["auroc"]), float(overall["auprc"]),
            float(overall["balanced_accuracy"]), float(overall["brier"]),
        )
    except Exception:
        logger.info("Overall OOF [deep]  metrics=%s", overall)

    return summary


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    cli_args.apply_thread_limits(args)
    cli_args.ensure_output_dir(args)
    configure_logging(args, log_file=args.output_dir / "logs" / "step4_sparse_invariant_model.log")

    logger.info(
        "CAGE step 4 invoked | latent=%s epochs=%s device=%s seed=%s",
        args.latent_dim, args.n_epochs, args.device, args.seed,
    )
    apply_rigor_profile(args, parser_defaults={a.dest: a.default for a in parser._actions})
    run_step4(args)


if __name__ == "__main__":
    main()
