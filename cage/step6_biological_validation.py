"""CAGE Step 6: Biological and statistical validation (Phase IV).

Validates CDPS-ranked candidate driver genes with raw-count differential
expression, pathway/gene-set enrichment, PPI network support, and
clinicopathologic association (including optional subgroup robustness
and survival analyses). Produces a final weighted validation-augmented
gene ranking and publication-grade figures.

Deliverables
------------
differential_expression_results.csv     DE across tumor vs normal
differential_expression_top_hits.csv    DE filtered by FDR/effect-size
top_cdps_de_support.csv                 CDPS x DE merge
enrichment_results_hallmark.csv         Hallmark enrichment (if enabled)
enrichment_results_reactome.csv         Reactome enrichment (if enabled)
enrichment_results_kegg_go.csv          KEGG/GO enrichment (if enabled)
enrichment_summary_top_pathways.csv
top_gene_pathway_membership.csv
network_gene_support.csv                PPI support (if --run-network)
network_edges_top_genes.csv             PPI induced edges (if --run-network)
clinical_association_results.csv        stage/histology/residual
survival_gene_summary.csv               (if --run-survival)
subgroup_sensitivity_summary.csv        (if --run-subgroup-sensitivity)
final_validated_gene_ranking.csv        weighted integrated ranking
phase4_summary.json                     config + component overview
figures/                                volcano, enrichment dot plots,
                                        network subgraph, clinical bars

Run
---
python -m cage.step6_biological_validation --help
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from . import cli_args, preprocess_esca as pp
from . import step6_runner as runner
from .cli_args import build_step_parser, configure_logging, style_from_args

logger = logging.getLogger("cage.step6")

_STEP_TITLE = "Phase IV - Biological & statistical validation."
_STEP_DESCRIPTION = (
    "Differential expression, pathway enrichment, network support, and\n"
    "clinicopathologic association for CDPS-ranked candidate drivers.\n"
    "Integrates all components into a final weighted validation ranking."
)
_INPUTS_DOC = (
    "ranked_genes_cdps.csv, top25/top100 (from step 5)\n"
    "counts_primary_matrix.csv            (from step 2)\n"
    "normalized_primary_matrix.csv        (from step 2)\n"
    "master_samples_primary.csv           (from step 2)\n"
    "(optional) GMT files for enrichment, PPI edge list for networks"
)
_OUTPUTS_DOC = (
    "differential_expression_results.csv, differential_expression_top_hits.csv\n"
    "top_cdps_de_support.csv\n"
    "enrichment_results_*.csv, enrichment_summary_top_pathways.csv\n"
    "top_gene_pathway_membership.csv\n"
    "network_gene_support.csv, network_edges_top_genes.csv\n"
    "clinical_association_results.csv, survival_gene_summary.csv\n"
    "subgroup_sensitivity_summary.csv\n"
    "final_validated_gene_ranking.csv\n"
    "phase4_summary.json\n"
    "figures/ : volcano, CDPS vs lfc scatter, enrichment dot plot,\n"
    "           network subgraph, clinical bars"
)
_EXAMPLE = (
    "python -m cage.step6_biological_validation \\\n"
    "  --step2-dir outputs/step2_cohort \\\n"
    "  --step5-dir outputs/step5_cdps \\\n"
    "  --output-dir outputs/step6_validation \\\n"
    "  --run-enrichment --run-network --run-survival --run-subgroup-sensitivity \\\n"
    "  --gmt-hallmark data/h.all.v2023.1.Hs.symbols.gmt"
)


def build_parser() -> argparse.ArgumentParser:
    parser = build_step_parser(
        prog="python -m cage.step6_biological_validation",
        step_title=_STEP_TITLE,
        step_description=_STEP_DESCRIPTION,
        inputs_doc=_INPUTS_DOC,
        outputs_doc=_OUTPUTS_DOC,
        example=_EXAMPLE,
        require_input_dir=False,
    )

    io = parser.add_argument_group("Step-6 input directories")
    io.add_argument(
        "--step2-dir", type=Path, default=None, metavar="DIR",
        help="Directory with step-2 cohort outputs.",
    )
    io.add_argument(
        "--step5-dir", type=Path, default=None, metavar="DIR",
        help="Directory with step-5 CDPS outputs.",
    )
    io.add_argument(
        "--env-names", nargs="+",
        default=["smoking", "sex", "histology", "country", "stage"],
        metavar="NAME",
        help="Environment columns used for clinical + subgroup analyses "
             "(default: smoking sex histology country stage).",
    )

    toggles = parser.add_argument_group("Optional analyses")
    toggles.add_argument("--run-enrichment", action="store_true",
                         help="Run Hallmark/Reactome/KEGG-GO/ImmuneSigDB enrichment on top genes.")
    toggles.add_argument("--run-network", action="store_true",
                         help="Run PPI network-level support analysis (requires --ppi-edge-list).")
    toggles.add_argument("--run-survival", action="store_true",
                         help="Run univariate survival analysis on tumor cohort for top genes.")
    toggles.add_argument("--run-subgroup-sensitivity", action="store_true",
                         help="Compute per-environment tumor-vs-normal consistency for top genes.")

    de = parser.add_argument_group("Differential expression")
    de.add_argument("--de-method", choices=["nb", "welch", "ranksum"], default="welch",
                    help="DE test method (default: welch; NB auto-falls-back to Welch).")
    de.add_argument("--fdr-threshold", type=float, default=0.05, metavar="F",
                    help="FDR (Benjamini-Hochberg) threshold for DE significance (default: 0.05).")
    de.add_argument("--lfc-threshold", type=float, default=1.0, metavar="F",
                    help="Absolute effect-size threshold (normalized z-units) for DE top hits (default: 1.0).")
    de.add_argument("--min-count", type=int, default=10, metavar="N",
                    help="Minimum count for DE gene filter (informational; applied in step 2).")
    de.add_argument("--min-samples-per-gene", type=int, default=10, metavar="N",
                    help="Minimum samples required at --min-count (informational).")

    enrich = parser.add_argument_group("Enrichment inputs (optional GMT overrides)")
    enrich.add_argument("--gmt-hallmark",  type=Path, default=None, metavar="FILE",
                        help="Hallmark MSigDB .gmt file.")
    enrich.add_argument("--gmt-reactome",  type=Path, default=None, metavar="FILE",
                        help="Reactome .gmt file.")
    enrich.add_argument("--gmt-kegg-go",   type=Path, default=None, metavar="FILE",
                        help="Combined KEGG+GO .gmt file.")
    enrich.add_argument("--gmt-immune",    type=Path, default=None, metavar="FILE",
                        help="MSigDB C7 ImmuneSigDB .gmt file (immunologic signatures).")
    enrich.add_argument("--top-k-for-enrichment", type=int, default=25, metavar="K",
                        help="Top-K CDPS genes used as the enrichment foreground (default: 25).")
    enrich.add_argument("--top-k-for-clinical", type=int, default=25, metavar="K",
                        help="Top-K CDPS genes used for clinical / survival / subgroup analyses (default: 25).")

    net = parser.add_argument_group("Network inputs (optional)")
    net.add_argument("--ppi-edge-list", type=Path, default=None, metavar="FILE",
                     help="Two-column edge list (gene\\tgene) for PPI analysis.")
    net.add_argument("--download-string-ppi", action="store_true",
                     help="Download PPI interactions for the top-K CDPS genes from the STRING "
                          "REST API (https://string-db.org). Requires internet access. "
                          "Saves the edge list to <output-dir>/network/string_ppi_edges.tsv "
                          "which is then passed automatically to --ppi-edge-list. "
                          "Ignored if --ppi-edge-list is also provided.")
    net.add_argument("--string-score-threshold", type=int, default=400, metavar="INT",
                     help="STRING combined score threshold 0-1000 "
                          "(400=medium, 700=high, 900=very high; default: 400).")
    net.add_argument("--string-species", type=int, default=9606, metavar="INT",
                     help="NCBI taxonomy ID for STRING query (default: 9606 = Homo sapiens).")

    final = parser.add_argument_group("Final validation score weights")
    final.add_argument("--w-de",        type=float, default=0.35, metavar="F")
    final.add_argument("--w-enrichment", type=float, default=0.20, metavar="F")
    final.add_argument("--w-network",   type=float, default=0.15, metavar="F")
    final.add_argument("--w-clinical",  type=float, default=0.15, metavar="F")
    final.add_argument("--w-subgroup",  type=float, default=0.10, metavar="F")
    final.add_argument("--w-external",  type=float, default=0.05, metavar="F")

    return parser


# ---------------------------------------------------------------------------
# STRING PPI download helper
# ---------------------------------------------------------------------------


def _download_string_edges_for_step6(
    genes: List[str],
    *,
    species: int = 9606,
    score_threshold: int = 400,
    save_path: Path,
) -> Optional[Path]:
    """Download STRING PPI edges for the given genes and save as TSV.

    Uses the STRING REST API (no registration required).
    Returns the saved path on success, None on failure.
    The saved file has columns: gene1, gene2, score (tab-separated).
    Compatible with Step 6's --ppi-edge-list format.
    """
    import csv as _csv
    import urllib.parse
    import urllib.request

    identifiers = "\r".join(genes)
    params = urllib.parse.urlencode({
        "identifiers": identifiers,
        "species": str(species),
        "required_score": str(score_threshold),
        "caller_identity": "cage_pipeline",
    })
    url = "https://string-db.org/api/tsv/network?" + params

    logger.info(
        "Querying STRING API for %d genes (species=%d, score>=%d)...",
        len(genes), species, score_threshold,
    )
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "CAGE/1.0"})
        with urllib.request.urlopen(req, timeout=90) as resp:
            content = resp.read().decode("utf-8")
    except Exception as exc:
        logger.warning("STRING API request failed: %s — network step will be skipped.", exc)
        return None

    lines = content.strip().split("\n")
    if len(lines) < 2:
        logger.warning("STRING API returned no interactions for the queried genes.")
        return None

    header = lines[0].split("\t")
    try:
        ia = header.index("preferredName_A")
        ib = header.index("preferredName_B")
        isc = header.index("score")
    except ValueError:
        logger.warning("Unexpected STRING API column header: %s", header)
        return None

    save_path.parent.mkdir(parents=True, exist_ok=True)
    n_edges = 0
    with open(str(save_path), "w", newline="", encoding="utf-8") as fh:
        writer = _csv.writer(fh, delimiter="\t")
        writer.writerow(["gene1", "gene2", "score"])
        for line in lines[1:]:
            parts = line.split("\t")
            if len(parts) <= max(ia, ib, isc):
                continue
            g1, g2 = parts[ia].strip(), parts[ib].strip()
            score = parts[isc].strip()
            if g1 and g2 and g1 != g2:
                writer.writerow([g1, g2, score])
                n_edges += 1

    logger.info(
        "STRING: %d edges downloaded and saved to %s", n_edges, save_path
    )
    return save_path if n_edges > 0 else None


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------


def _output_paths(args: argparse.Namespace) -> Dict[str, Path]:
    out = args.output_dir
    return {
        "de_results": out / "differential_expression_results.csv",
        "de_top_hits": out / "differential_expression_top_hits.csv",
        "cdps_de": out / "top_cdps_de_support.csv",
        "de_qc_json": out / "de_qc_summary.json",
        "enrich_hallmark": out / "enrichment_results_hallmark.csv",
        "enrich_reactome": out / "enrichment_results_reactome.csv",
        "enrich_kegg_go": out / "enrichment_results_kegg_go.csv",
        "enrich_immune": out / "enrichment_results_immune.csv",
        "enrich_summary": out / "enrichment_summary_top_pathways.csv",
        "enrich_membership": out / "top_gene_pathway_membership.csv",
        "network_rows": out / "network_gene_support.csv",
        "network_edges": out / "network_edges_top_genes.csv",
        "network_summary": out / "network_summary.json",
        "clinical": out / "clinical_association_results.csv",
        "survival": out / "survival_gene_summary.csv",
        "subgroup": out / "subgroup_sensitivity_summary.csv",
        "final": out / "final_validated_gene_ranking.csv",
        "summary_json": out / "phase4_summary.json",
    }


def _check_overwrite(args: argparse.Namespace, paths: Mapping[str, Path]) -> None:
    existing = [p for p in paths.values() if p.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Output files already exist; pass --overwrite to regenerate:\n  "
            + "\n  ".join(str(p) for p in existing)
        )


def _resolve_dirs(args: argparse.Namespace) -> Dict[str, Path]:
    step2_dir: Path | None = args.step2_dir or args.input_dir
    step5_dir: Path | None = args.step5_dir or args.input_dir
    if step2_dir is None:
        raise FileNotFoundError("Step-2 directory not set; pass --input-dir or --step2-dir.")
    if step5_dir is None:
        raise FileNotFoundError("Step-5 directory not set; pass --input-dir or --step5-dir.")
    if not step2_dir.exists():
        raise FileNotFoundError(f"Step-2 directory does not exist: {step2_dir}")
    if not step5_dir.exists():
        raise FileNotFoundError(f"Step-5 directory does not exist: {step5_dir}")
    return {"step2": step2_dir, "step5": step5_dir}


def _fmt_f(v: Any) -> Any:
    if isinstance(v, float):
        return "" if (v != v) else f"{v:.6g}"
    return v


def _fmt_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    return [{k: _fmt_f(v) for k, v in r.items()} for r in rows]


def run_step6(args: argparse.Namespace) -> Dict[str, Any]:
    """Execute the Step-6 validation pipeline and return the JSON summary."""
    paths = _output_paths(args)
    _check_overwrite(args, paths)
    dirs = _resolve_dirs(args)

    weights = {
        "de": float(args.w_de),
        "enrichment": float(args.w_enrichment),
        "network": float(args.w_network),
        "clinical": float(args.w_clinical),
        "subgroup": float(args.w_subgroup),
        "external": float(args.w_external),
    }

    # ------------------------------------------------------------------
    # STRING PPI auto-download (runs before validation so edge list is ready)
    # ------------------------------------------------------------------
    ppi_edge_list = args.ppi_edge_list
    if getattr(args, "download_string_ppi", False) and ppi_edge_list is None:
        # Read top-K gene names from step5 ranked list
        ranked_path = dirs["step5"] / "ranked_genes_cdps.csv"
        top25_path = dirs["step5"] / "top25_genes_cdps.csv"
        top_genes_for_string: List[str] = []
        for candidate in (ranked_path, top25_path):
            if candidate.is_file():
                import csv as _csv_mod
                with open(str(candidate), "r", newline="", encoding="utf-8") as _fh:
                    _reader = _csv_mod.DictReader(_fh)
                    for _row in _reader:
                        _g = (_row.get("gene") or "").strip()
                        if _g:
                            top_genes_for_string.append(_g)
                        if len(top_genes_for_string) >= int(args.top_k_for_enrichment):
                            break
                if top_genes_for_string:
                    break

        if top_genes_for_string:
            _string_save = args.output_dir / "network" / "string_ppi_edges.tsv"
            _downloaded = _download_string_edges_for_step6(
                top_genes_for_string,
                species=getattr(args, "string_species", 9606),
                score_threshold=getattr(args, "string_score_threshold", 400),
                save_path=_string_save,
            )
            if _downloaded is not None:
                ppi_edge_list = _downloaded
                logger.info(
                    "STRING edge list ready: %s — passing to --ppi-edge-list", ppi_edge_list
                )
        else:
            logger.warning(
                "--download-string-ppi requested but no CDPS gene list found in %s",
                dirs["step5"],
            )

    logger.info(
        "Step 6 | step2=%s step5=%s output=%s | enrich=%s network=%s survival=%s subgroup=%s",
        dirs["step2"], dirs["step5"], args.output_dir,
        args.run_enrichment, args.run_network, args.run_survival,
        args.run_subgroup_sensitivity,
    )

    result = runner.run_step6_validation(
        step2_dir=dirs["step2"],
        step5_dir=dirs["step5"],
        output_dir=args.output_dir,
        de_method=args.de_method,
        fdr_threshold=float(args.fdr_threshold),
        lfc_threshold=float(args.lfc_threshold),
        top_k_for_enrichment=int(args.top_k_for_enrichment),
        top_k_for_clinical=int(args.top_k_for_clinical),
        gmt_hallmark=args.gmt_hallmark,
        gmt_reactome=args.gmt_reactome,
        gmt_kegg_go=args.gmt_kegg_go,
        gmt_immune=args.gmt_immune,
        ppi_edge_list=ppi_edge_list,
        env_names=list(args.env_names),
        run_enrichment_flag=bool(args.run_enrichment),
        run_network_flag=bool(args.run_network),
        run_survival_flag=bool(args.run_survival),
        run_subgroup_flag=bool(args.run_subgroup_sensitivity),
        weights=weights,
        seed=int(args.seed),
    )

    # ------------------------------------------------------------------
    # 1. DE outputs
    # ------------------------------------------------------------------
    de_rows = result["de"]["rows"]
    pp.write_csv_records(paths["de_results"], _fmt_rows(de_rows))
    top_hits = [r for r in de_rows if int(r.get("sig_fdr_and_effect", 0) or 0) == 1]
    # Sort top hits by ascending FDR
    top_hits.sort(key=lambda r: (r.get("fdr_bh", 1.0), -abs(r.get("effect_size_norm", 0.0))))
    pp.write_csv_records(paths["de_top_hits"], _fmt_rows(top_hits))
    pp.write_csv_records(paths["cdps_de"], _fmt_rows(result["cdps_de_support"]))
    pp.write_json(paths["de_qc_json"], result["de"]["qc"])

    # ------------------------------------------------------------------
    # 2. Enrichment outputs
    # ------------------------------------------------------------------
    if args.run_enrichment:
        for source, rows in result["enrichment"]["per_source_rows"].items():
            target_key = {
                "hallmark": "enrich_hallmark",
                "reactome": "enrich_reactome",
                "kegg_go": "enrich_kegg_go",
                "immune": "enrich_immune",
            }.get(source)
            if target_key is None:
                continue
            pp.write_csv_records(paths[target_key], _fmt_rows(rows))
        pp.write_csv_records(paths["enrich_summary"], _fmt_rows(result["enrichment"]["summary_rows"]))
        pp.write_csv_records(paths["enrich_membership"], _fmt_rows(result["enrichment"]["membership_rows"]))
        for source, reason in result["enrichment"]["skipped_sources"]:
            logger.warning("enrichment SKIPPED [%s]: %s", source, reason)

    # ------------------------------------------------------------------
    # 3. Network outputs
    # ------------------------------------------------------------------
    if args.run_network:
        if result["network"]["skipped_reason"]:
            logger.warning("network SKIPPED: %s", result["network"]["skipped_reason"])
        else:
            pp.write_csv_records(paths["network_rows"], _fmt_rows(result["network"]["rows"]))
            pp.write_csv_records(paths["network_edges"], _fmt_rows(result["network"]["edges_top"]))
            pp.write_json(paths["network_summary"], result["network"]["summary"])

    # ------------------------------------------------------------------
    # 4. Clinical
    # ------------------------------------------------------------------
    pp.write_csv_records(paths["clinical"], _fmt_rows(result["clinical"]["rows"]))

    # ------------------------------------------------------------------
    # 5. Survival
    # ------------------------------------------------------------------
    if args.run_survival:
        if result["survival"]["skipped_reason"] and result["survival"]["skipped_reason"] != "not requested":
            logger.warning("survival SKIPPED: %s", result["survival"]["skipped_reason"])
        pp.write_csv_records(paths["survival"], _fmt_rows(result["survival"]["rows"]))

    # ------------------------------------------------------------------
    # 6. Subgroup
    # ------------------------------------------------------------------
    if args.run_subgroup_sensitivity:
        pp.write_csv_records(paths["subgroup"], _fmt_rows(result["subgroup"]["rows"]))

    # ------------------------------------------------------------------
    # 7. Final integrated ranking
    # ------------------------------------------------------------------
    pp.write_csv_records(paths["final"], _fmt_rows(result["integrated"]["records"]))

    # ------------------------------------------------------------------
    # 8. Figures
    # ------------------------------------------------------------------
    style = style_from_args(args)
    _enr_per_source = result["enrichment"].get("per_source_rows", {})
    generated_figs, skipped_figs = runner.generate_step6_figures(
        de_rows=de_rows,
        cdps_top_genes=result["cdps_top_genes"],
        enrichment_summary_rows=result["enrichment"]["summary_rows"],
        network_summary=result["network"]["summary"],
        final_records=result["integrated"]["records"],
        output_dir=args.output_dir,
        style=style,
        formats=style.default_formats,
        fdr_threshold=float(args.fdr_threshold),
        lfc_threshold=float(args.lfc_threshold),
        clinical_assoc_rows=result["clinical"]["rows"] or None,
        hallmark_rows=_enr_per_source.get("hallmark") or None,
        reactome_rows=_enr_per_source.get("reactome") or None,
        immune_rows=_enr_per_source.get("immune") or None,
        survival_rows=result["survival"]["rows"] if args.run_survival else None,
        subgroup_sensitivity_rows=(
            result["subgroup"]["rows"] if args.run_subgroup_sensitivity else None
        ),
        pathway_membership_rows=result["enrichment"].get("membership_rows") or None,
    )
    for f in generated_figs:
        logger.info("figure OK: %s", f)
    for name, reason in skipped_figs:
        logger.warning("figure SKIPPED: %s (%s)", name, reason)

    # ------------------------------------------------------------------
    # 9. Phase-4 summary JSON
    # ------------------------------------------------------------------
    summary = {
        "phase": "IV (biological & statistical validation)",
        "step": "step6_biological_validation",
        "cohort": result["cohort"],
        "config": result["config"],
        "weights_effective": result["integrated"]["weights_effective"],
        "weights_raw": result["integrated"]["weights_raw"],
        "de_qc": result["de"]["qc"],
        "enrichment_summary": {
            "n_sources": len(result["enrichment"]["per_source_rows"]),
            "sources": list(result["enrichment"]["per_source_rows"].keys()),
            "skipped": [{"source": s, "reason": r}
                        for s, r in result["enrichment"]["skipped_sources"]],
            "n_membership_rows": len(result["enrichment"]["membership_rows"]),
        },
        "network_summary": result["network"]["summary"] if not result["network"]["skipped_reason"]
        else {"skipped_reason": result["network"]["skipped_reason"]},
        "survival_summary": {
            "n_analyzed": int(result["survival"].get("n_analyzed", 0) or 0),
            "n_gene_rows": len(result["survival"]["rows"]),
            "skipped_reason": result["survival"].get("skipped_reason"),
        },
        "subgroup_summary": {
            "n_gene_rows": len(result["subgroup"]["rows"]),
        },
        "clinical_summary": {
            "n_gene_rows": len(result["clinical"]["rows"]),
        },
        "top_final_genes": [
            {
                "rank_final": int(rec["rank_final"]),
                "gene": str(rec["gene"]),
                "final_score": float(rec["final_score"]),
                "cdps_rank": int(rec["cdps_rank"]),
                "cdps": float(rec["cdps"]),
                "validation_score": float(rec["validation_score"]),
                "de_norm": float(rec["de_norm"]),
                "enrichment_norm": float(rec["enrichment_norm"]),
                "network_norm": float(rec["network_norm"]),
                "clinical_norm": float(rec["clinical_norm"]),
                "subgroup_norm": float(rec["subgroup_norm"]),
                "external_norm": float(rec["external_norm"]),
            }
            for rec in result["integrated"]["records"][:25]
        ],
        "figures": {
            "generated": generated_figs,
            "skipped": [{"name": n, "reason": r} for n, r in skipped_figs],
        },
        "style": style.as_dict(),
    }
    pp.write_json(paths["summary_json"], summary)

    # Flight-recorder
    top1 = result["integrated"]["records"][0] if result["integrated"]["records"] else None
    if top1 is not None:
        logger.info(
            "Step 6 top-1 validated: %s (final=%.4f, cdps=%.4f, val=%.4f) | sig_de=%d",
            top1["gene"], top1["final_score"], top1["cdps"], top1["validation_score"],
            int(result["de"]["qc"]["n_sig_fdr_and_effect"]),
        )
    return summary


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cli_args.apply_thread_limits(args)
    cli_args.ensure_output_dir(args)
    configure_logging(args, log_file=args.output_dir / "logs" / "step6_biological_validation.log")
    style = style_from_args(args)

    logger.info(
        "CAGE step 6 invoked | enrich=%s network=%s survival=%s subgroup=%s",
        args.run_enrichment, args.run_network, args.run_survival,
        args.run_subgroup_sensitivity,
    )
    logger.debug("figure style: %s", style.as_dict())
    run_step6(args)


if __name__ == "__main__":
    main()
