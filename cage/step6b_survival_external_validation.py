"""
Step 6b (addon) — Survival Analysis & External Cohort Validation for top-25 CDPS genes.

Performs:
  1. Kaplan-Meier survival analysis (TCGA tumor samples, median expression split)
  2. Tumor vs normal expression boxplots (TCGA)
  3. External GEO cohort validation (GSE53624, GSE53625, GSE161533, GSE38129)
  4. External concordance summary heatmap
  5. Master evidence table merging all results

Outputs individual SVG per gene AND combined publication panels:
  figures/km/         <gene>_KM_curve.svg  (× 25 individual)
  figures/km/         combined_KM_all25.svg  (5 × 5 panel)
  figures/boxplots/   <gene>_TCGA_tumor_vs_normal.svg  (× 25 individual)
  figures/boxplots/   combined_TCGA_boxplots_all25.svg  (5 × 5 panel)
  figures/external/   <GSE>_<gene>_boxplot.svg  (individual)
  figures/external/   <GSE>_all25_boxplots.svg  (per-dataset panel, 5 × 5)

Usage:
  python -m cage.step6b_survival_external_validation \\
    --expression-matrix outputs/step2_cohort/normalized_primary_matrix.csv \\
    --metadata          outputs/step2_cohort/master_samples_primary.csv \\
    --top25-ranking     outputs/step6b_top25_prioritization/tables/top25_final_priority_ranking.csv \\
    --agilent-replication outputs/step8_agilent_validation/combined_de_replication.csv \\
    --geo-replication   outputs/step8_geo_validation/combined_de_replication.csv \\
    --geo-prepared-dir  outputs/step8_geo_prepared \\
    --output-dir        outputs/step6b_survival_external
"""

from __future__ import annotations

import csv
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CAGE default palette (consistent with rest of pipeline)
# ---------------------------------------------------------------------------
_COLOR_TUMOR   = "#E64B35"
_COLOR_NORMAL  = "#4DBBD5"
_COLOR_HIGH    = "#E64B35"
_COLOR_LOW     = "#4DBBD5"
_COLOR_CONCORD = "#00A087"
_COLOR_DISCORD = "#F39B7F"
_COLOR_ABSENT  = "#D0D0D0"

# GEO dataset labels shown in combined panels
_GEO_LABELS: Dict[str, str] = {
    "GSE53624":  "GSE53624 (Agilent, n=119+119)",
    "GSE53625":  "GSE53625 (Agilent, n=179+179)",
    "GSE161533": "GSE161533 (Affymetrix, n=56+28)",
    "GSE38129":  "GSE38129 (Affymetrix, n=30+30)",
}

# Default publication font settings (sans-serif, journal-agnostic)
_FONT_FAMILY = "DejaVu Sans"
_FONT_SIZE_TITLE  = 9
_FONT_SIZE_LABEL  = 8
_FONT_SIZE_TICK   = 7
_FONT_SIZE_ANNOT  = 7
_FONT_WEIGHT = "normal"


# ---------------------------------------------------------------------------
# matplotlib publication style helper
# ---------------------------------------------------------------------------

def _set_pub_rcparams() -> None:
    """Apply default publication rcParams for all figure text."""
    try:
        import matplotlib as mpl
        mpl.rcParams.update({
            "font.family":      "sans-serif",
            "font.sans-serif":  [_FONT_FAMILY, "Arial", "Helvetica", "Liberation Sans", "sans-serif"],
            "font.weight":      _FONT_WEIGHT,
            "font.size":        _FONT_SIZE_LABEL,
            "axes.titlesize":   _FONT_SIZE_TITLE,
            "axes.titleweight": _FONT_WEIGHT,
            "axes.labelsize":   _FONT_SIZE_LABEL,
            "axes.labelweight": _FONT_WEIGHT,
            "xtick.labelsize":  _FONT_SIZE_TICK,
            "ytick.labelsize":  _FONT_SIZE_TICK,
            "legend.fontsize":  _FONT_SIZE_ANNOT,
            "figure.dpi":       150,
            "axes.linewidth":   0.8,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "pdf.fonttype":     42,   # embeddable TrueType in PDF
            "svg.fonttype":     "none",
        })
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Pure-NumPy statistics helpers
# ---------------------------------------------------------------------------

def _km_estimator(times: List[float], events: List[int]) -> Tuple[List[float], List[float]]:
    """Kaplan-Meier product-limit estimator. Returns (event_times, survival)."""
    n = len(times)
    if n == 0:
        return [], []
    pairs = sorted(zip(times, events), key=lambda x: x[0])
    unique_t: List[float] = []
    survival: List[float] = []
    S = 1.0
    at_risk = n
    i = 0
    while i < n:
        t = pairs[i][0]
        d = 0
        censored = 0
        j = i
        while j < n and pairs[j][0] == t:
            d += pairs[j][1]
            censored += 1 - pairs[j][1]
            j += 1
        if d > 0:
            S = S * (1.0 - d / at_risk)
            unique_t.append(t)
            survival.append(S)
        at_risk -= (d + censored)
        i = j
    return unique_t, survival


def _logrank_test(t1: List[float], e1: List[int],
                  t2: List[float], e2: List[int]) -> Tuple[float, float]:
    """Log-rank (Mantel-Cox) test. Returns (chi2, p_value) with df=1."""
    if len(t1) < 2 or len(t2) < 2:
        return float("nan"), float("nan")
    all_times = sorted(
        set(t for t, e in zip(t1, e1) if e == 1) |
        set(t for t, e in zip(t2, e2) if e == 1)
    )
    if not all_times:
        return float("nan"), float("nan")
    O1 = E1 = V1 = 0.0
    for t in all_times:
        n1 = sum(1 for x in t1 if x >= t)
        n2 = sum(1 for x in t2 if x >= t)
        d1 = sum(1 for x, e in zip(t1, e1) if x == t and e == 1)
        d2 = sum(1 for x, e in zip(t2, e2) if x == t and e == 1)
        n_tot = n1 + n2
        d_tot = d1 + d2
        if n_tot < 2 or d_tot == 0:
            continue
        O1 += d1
        E1 += n1 * d_tot / n_tot
        if n_tot > 1:
            V1 += (n1 * n2 * d_tot * (n_tot - d_tot)) / (n_tot * n_tot * (n_tot - 1))
    if V1 <= 0:
        return float("nan"), float("nan")
    chi2 = (O1 - E1) ** 2 / V1
    try:
        p = math.erfc(math.sqrt(chi2 / 2))
    except (ValueError, OverflowError):
        p = float("nan")
    return chi2, p


# ---------------------------------------------------------------------------
# CSV / IO helpers
# ---------------------------------------------------------------------------

def _safe_read_csv(path: Optional[Path]) -> Optional[List[Dict[str, str]]]:
    if path is None or not path.exists():
        return None
    try:
        with open(path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except Exception as exc:
        logger.warning("Cannot read %s: %s", path, exc)
        return None


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    logger.info("Wrote %s (%d rows)", path, len(rows))


def _to_float(val: Any, default: float = float("nan")) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _fmt(val: float, decimals: int = 4) -> str:
    if math.isnan(val):
        return "NA"
    return f"{val:.{decimals}f}"


def _median(vals: List[float]) -> float:
    s = sorted(v for v in vals if not math.isnan(v))
    if not s:
        return float("nan")
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def _mean(vals: List[float]) -> float:
    clean = [v for v in vals if not math.isnan(v)]
    return sum(clean) / len(clean) if clean else float("nan")


# ---------------------------------------------------------------------------
# Expression matrix loader
# ---------------------------------------------------------------------------

def _load_expression_matrix(path: Path) -> Tuple[List[str], Dict[str, List[float]], List[str]]:
    rows = _safe_read_csv(path)
    if rows is None or not rows:
        return [], {}, []
    cols = list(rows[0].keys())
    sample_col = cols[0]
    gene_names = cols[1:]
    sample_barcodes = [r[sample_col] for r in rows]
    gene_expr = {g: [_to_float(r.get(g, "")) for r in rows] for g in gene_names}
    return sample_barcodes, gene_expr, gene_names


def _load_metadata(path: Path) -> Dict[str, Dict[str, str]]:
    rows = _safe_read_csv(path)
    if rows is None:
        return {}
    first_col = list(rows[0].keys())[0] if rows else "sample_barcode"
    return {r[first_col]: r for r in rows}


def _load_geo_matrix(geo_dir: Path, gse_id: str) -> Tuple[Dict[str, List[float]], List[str]]:
    mat_path = geo_dir / gse_id / "processed" / "expression_gene_matrix.csv"
    rows = _safe_read_csv(mat_path)
    if rows is None or not rows:
        return {}, []
    cols = list(rows[0].keys())
    gene_col = cols[0]
    gsm_ids = cols[1:]
    gene_expr: Dict[str, List[float]] = {}
    for r in rows:
        gene = r[gene_col].strip()
        if gene:
            gene_expr[gene] = [_to_float(r.get(g, "")) for g in gsm_ids]

    # Detect matrices written with numeric probe IDs instead of HGNC gene symbols.
    # This happens for Agilent GPL18109 datasets before step8_agilent_validation is run.
    # Numeric-ID matrices cannot serve target gene lookups; log a warning and return empty.
    if gene_expr:
        sample_keys = list(gene_expr.keys())[:30]
        numeric_frac = sum(1 for k in sample_keys if k.isdigit()) / len(sample_keys)
        if numeric_frac > 0.5:
            logger.warning(
                "%s: expression_gene_matrix.csv uses numeric probe IDs (e.g. %s). "
                "Run step8_agilent_validation first to generate an HGNC-symbol-indexed "
                "gene matrix; all target genes will show 'Not measured' until then.",
                gse_id, sample_keys[:3],
            )
            return {}, gsm_ids

    return gene_expr, gsm_ids


def _load_geo_metadata(geo_dir: Path, gse_id: str) -> Dict[str, str]:
    meta_path = geo_dir / gse_id / "metadata" / "sample_metadata_inferred.csv"
    rows = _safe_read_csv(meta_path)
    if rows is None:
        return {}
    return {r.get("gsm_id", ""): r.get("sample_type_inferred", "") for r in rows}


# ---------------------------------------------------------------------------
# Box statistics
# ---------------------------------------------------------------------------

def _boxplot_stats(vals: List[float]) -> Dict:
    s = sorted(v for v in vals if not math.isnan(v))
    if not s:
        return {}
    n = len(s)
    med = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2
    q1 = s[n // 4]
    q3 = s[(3 * n) // 4]
    iqr = q3 - q1
    lo = q1 - 1.5 * iqr
    hi = q3 + 1.5 * iqr
    whisker_lo = next((x for x in s if x >= lo), s[0])
    whisker_hi = next((x for x in reversed(s) if x <= hi), s[-1])
    outliers = [x for x in s if x < whisker_lo or x > whisker_hi]
    return {"q1": q1, "med": med, "q3": q3,
            "wlo": whisker_lo, "whi": whisker_hi,
            "outliers": outliers, "n": n}


def _jitter(n: int, width: float = 0.22, seed: int = 42) -> List[float]:
    state = seed
    result = []
    for _ in range(n):
        state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
        result.append(((state / 0xFFFFFFFF) - 0.5) * 2 * width)
    return result


def _draw_box(ax: Any, x: float, stats: Dict, color: str, width: float = 0.5) -> None:
    if not stats:
        return
    try:
        from matplotlib.patches import FancyBboxPatch
    except ImportError:
        return
    hw = width / 2
    ax.add_patch(FancyBboxPatch(
        (x - hw, stats["q1"]), width, stats["q3"] - stats["q1"],
        boxstyle="square,pad=0", linewidth=0.8,
        edgecolor="#222222", facecolor=color, alpha=0.72))
    ax.plot([x - hw, x + hw], [stats["med"], stats["med"]],
            color="#111111", linewidth=1.6)
    ax.plot([x, x], [stats["q3"], stats["whi"]], color="#444444", linewidth=0.8)
    ax.plot([x, x], [stats["q1"], stats["wlo"]], color="#444444", linewidth=0.8)
    for cap_y in (stats["whi"], stats["wlo"]):
        ax.plot([x - hw * 0.45, x + hw * 0.45], [cap_y, cap_y],
                color="#444444", linewidth=0.8)
    if stats.get("outliers"):
        ax.scatter([x] * len(stats["outliers"]), stats["outliers"],
                   color=color, s=8, zorder=5, alpha=0.55,
                   edgecolors="#333333", linewidths=0.4)


def _style_ax(ax: Any, title: str = "", xlabel: str = "", ylabel: str = "") -> None:
    """Apply uniform bold Times New Roman styling to an axis."""
    kw = {"fontfamily": _FONT_FAMILY, "fontweight": _FONT_WEIGHT}
    if title:
        ax.set_title(title, fontsize=_FONT_SIZE_TITLE, **kw)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=_FONT_SIZE_LABEL, **kw)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=_FONT_SIZE_LABEL, **kw)
    for item in ax.get_xticklabels() + ax.get_yticklabels():
        item.set_fontfamily(_FONT_FAMILY)
        item.set_fontweight(_FONT_WEIGHT)
        item.set_fontsize(_FONT_SIZE_TICK)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)


def _save_fig(fig: Any, out_path: Path) -> bool:
    try:
        from cage.publication_style import save_figure
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_figure(fig, out_path)
    except Exception:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return True


# ---------------------------------------------------------------------------
# Individual KM curve
# ---------------------------------------------------------------------------

def _draw_km_on_ax(ax: Any, gene: str,
                   high_t: List[float], high_e: List[int],
                   low_t:  List[float], low_e:  List[int],
                   p_value: float, compact: bool = False) -> None:
    """Draw one KM curve onto ax (used for individual and panel)."""
    km_high_t, km_high_s = _km_estimator(high_t, high_e)
    km_low_t,  km_low_s  = _km_estimator(low_t,  low_e)

    def _step(ax, times, surv, color, label):
        if not times:
            return
        ax.step([0] + list(times), [1.0] + list(surv),
                where="post", color=color, label=label, linewidth=1.4)

    _step(ax, km_high_t, km_high_s, _COLOR_HIGH, f"High (n={len(high_t)})")
    _step(ax, km_low_t,  km_low_s,  _COLOR_LOW,  f"Low  (n={len(low_t)})")

    # Censoring ticks
    for t, e, km_t, km_s, col in [
        (high_t, high_e, km_high_t, km_high_s, _COLOR_HIGH),
        (low_t,  low_e,  km_low_t,  km_low_s,  _COLOR_LOW),
    ]:
        for obs_t, obs_e in zip(t, e):
            if obs_e == 0:
                s_val = 1.0
                for kt, ks in zip(km_t, km_s):
                    if kt <= obs_t:
                        s_val = ks
                ax.plot(obs_t, s_val, "+", color=col, markersize=3, alpha=0.6,
                        markeredgewidth=0.8)

    p_txt = f"p={p_value:.3f}" if not math.isnan(p_value) else "p=NA"
    ax.text(0.97, 0.97, p_txt,
            transform=ax.transAxes, ha="right", va="top",
            fontsize=_FONT_SIZE_ANNOT,
            fontfamily=_FONT_FAMILY, fontweight=_FONT_WEIGHT,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85,
                      edgecolor="#cccccc", linewidth=0.5))
    ax.set_xlim(left=0)
    ax.set_ylim(-0.05, 1.05)

    title_fs = _FONT_SIZE_ANNOT if compact else _FONT_SIZE_TITLE
    label_fs = _FONT_SIZE_ANNOT if compact else _FONT_SIZE_LABEL
    _style_ax(ax,
              title=gene,
              xlabel="Days" if not compact else "",
              ylabel="Overall Survival" if not compact else "")
    ax.title.set_fontsize(title_fs)
    if not compact:
        leg = ax.legend(fontsize=_FONT_SIZE_ANNOT, loc="lower left",
                        frameon=False, prop={"family": _FONT_FAMILY, "weight": _FONT_WEIGHT})


def _plot_km_curve(gene: str,
                   high_t: List[float], high_e: List[int],
                   low_t:  List[float], low_e:  List[int],
                   p_value: float, out_path: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    _set_pub_rcparams()
    fig, ax = plt.subplots(figsize=(4.0, 3.2))
    _draw_km_on_ax(ax, gene, high_t, high_e, low_t, low_e, p_value, compact=False)
    leg = ax.legend(fontsize=_FONT_SIZE_ANNOT, loc="lower left", frameon=False)
    if leg:
        for t in leg.get_texts():
            t.set_fontfamily(_FONT_FAMILY)
            t.set_fontweight(_FONT_WEIGHT)
    plt.tight_layout()
    _save_fig(fig, out_path)
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# Combined KM panel (5 × 5)
# ---------------------------------------------------------------------------

def _plot_km_combined(km_data: List[Tuple[str, List, List, List, List, float]],
                      out_path: Path) -> bool:
    """5 × 5 panel of all 25 KM curves.

    km_data: list of (gene, high_t, high_e, low_t, low_e, p_value).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError:
        return False

    _set_pub_rcparams()
    n = len(km_data)
    ncols = 5
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 2.6, nrows * 2.4),
                             squeeze=False)

    for idx, (gene, high_t, high_e, low_t, low_e, pval) in enumerate(km_data):
        ax = axes[idx // ncols][idx % ncols]
        _draw_km_on_ax(ax, gene, high_t, high_e, low_t, low_e, pval, compact=True)

    # Hide any unused axes
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    # Shared legend at figure level
    legend_elements = [
        Line2D([0], [0], color=_COLOR_HIGH, linewidth=1.4, label="High expression"),
        Line2D([0], [0], color=_COLOR_LOW,  linewidth=1.4, label="Low expression"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2,
               fontsize=_FONT_SIZE_ANNOT, frameon=False,
               prop={"family": _FONT_FAMILY, "weight": _FONT_WEIGHT},
               bbox_to_anchor=(0.5, 0.0))

    fig.text(0.5, 1.0,
             "Kaplan–Meier Overall Survival — Top-25 CDPS Genes (TCGA-ESCA, median split)",
             ha="center", va="top", fontsize=_FONT_SIZE_TITLE + 1,
             fontfamily=_FONT_FAMILY, fontweight=_FONT_WEIGHT)

    # Shared axis labels
    fig.text(0.5, -0.01, "Days", ha="center", fontsize=_FONT_SIZE_LABEL,
             fontfamily=_FONT_FAMILY, fontweight=_FONT_WEIGHT)
    fig.text(-0.01, 0.5, "Overall Survival Probability",
             va="center", rotation="vertical", fontsize=_FONT_SIZE_LABEL,
             fontfamily=_FONT_FAMILY, fontweight=_FONT_WEIGHT)

    plt.tight_layout(rect=[0.02, 0.04, 1.0, 0.97])
    _save_fig(fig, out_path)
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# Individual TCGA boxplot
# ---------------------------------------------------------------------------

def _draw_boxplot_on_ax(ax: Any, gene: str,
                        tumor_vals: List[float], normal_vals: List[float],
                        compact: bool = False) -> None:
    tumor_clean  = [v for v in tumor_vals  if not math.isnan(v)]
    normal_clean = [v for v in normal_vals if not math.isnan(v)]

    for x, vals, color, seed in [(1, tumor_clean, _COLOR_TUMOR, 1),
                                  (2, normal_clean, _COLOR_NORMAL, 2)]:
        if not vals:
            continue
        jit = _jitter(len(vals), seed=seed)
        ax.scatter([x + j for j in jit], vals,
                   color=color, s=8 if compact else 12,
                   alpha=0.38, zorder=3, edgecolors="none")
        _draw_box(ax, x, _boxplot_stats(vals), color,
                  width=0.42 if compact else 0.50)

    ax.set_xticks([1, 2])
    lbl_kw = {"fontfamily": _FONT_FAMILY, "fontweight": _FONT_WEIGHT,
               "fontsize": _FONT_SIZE_ANNOT if compact else _FONT_SIZE_TICK}
    ax.set_xticklabels(
        [f"T\n(n={len(tumor_clean)})" if compact else f"Tumor\n(n={len(tumor_clean)})",
         f"N\n(n={len(normal_clean)})" if compact else f"Normal\n(n={len(normal_clean)})"],
        **lbl_kw)
    ax.set_xlim(0.3, 2.7)
    _style_ax(ax, title=gene,
              ylabel="" if compact else "Normalized expression")
    ax.title.set_fontsize(_FONT_SIZE_ANNOT if compact else _FONT_SIZE_TITLE)


def _plot_boxplot_tcga(gene: str,
                       tumor_vals: List[float], normal_vals: List[float],
                       out_path: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    _set_pub_rcparams()
    fig, ax = plt.subplots(figsize=(3.0, 3.8))
    _draw_boxplot_on_ax(ax, gene, tumor_vals, normal_vals, compact=False)
    plt.tight_layout()
    _save_fig(fig, out_path)
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# Combined TCGA boxplot panel (5 × 5)
# ---------------------------------------------------------------------------

def _plot_boxplot_tcga_combined(box_data: List[Tuple[str, List, List]],
                                out_path: Path) -> bool:
    """5 × 5 panel of all 25 TCGA tumor vs normal boxplots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except ImportError:
        return False

    _set_pub_rcparams()
    n = len(box_data)
    ncols = 5
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 2.2, nrows * 2.6),
                             squeeze=False)

    for idx, (gene, t_vals, n_vals) in enumerate(box_data):
        ax = axes[idx // ncols][idx % ncols]
        _draw_boxplot_on_ax(ax, gene, t_vals, n_vals, compact=True)

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    legend_elements = [
        Patch(facecolor=_COLOR_TUMOR,  label="Tumor",  edgecolor="#333333", linewidth=0.5),
        Patch(facecolor=_COLOR_NORMAL, label="Normal", edgecolor="#333333", linewidth=0.5),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2,
               fontsize=_FONT_SIZE_ANNOT, frameon=False,
               prop={"family": _FONT_FAMILY, "weight": _FONT_WEIGHT},
               bbox_to_anchor=(0.5, 0.0))

    fig.text(0.5, 1.0,
             "Tumor vs Normal Expression — Top-25 CDPS Genes (TCGA-ESCA)",
             ha="center", va="top", fontsize=_FONT_SIZE_TITLE + 1,
             fontfamily=_FONT_FAMILY, fontweight=_FONT_WEIGHT)
    fig.text(-0.01, 0.5, "Normalized expression",
             va="center", rotation="vertical", fontsize=_FONT_SIZE_LABEL,
             fontfamily=_FONT_FAMILY, fontweight=_FONT_WEIGHT)

    plt.tight_layout(rect=[0.02, 0.04, 1.0, 0.97])
    _save_fig(fig, out_path)
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# Individual GEO boxplot
# ---------------------------------------------------------------------------

def _draw_geo_box_on_ax(ax: Any, gene: str, gse_id: str,
                        tumor_vals: List[float], normal_vals: List[float],
                        compact: bool = False) -> None:
    tumor_clean  = [v for v in tumor_vals  if not math.isnan(v)]
    normal_clean = [v for v in normal_vals if not math.isnan(v)]

    if not tumor_clean and not normal_clean:
        ax.text(0.5, 0.5, "Not\nmeasured",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=_FONT_SIZE_ANNOT, fontfamily=_FONT_FAMILY,
                fontweight=_FONT_WEIGHT, color="#888888")
        _style_ax(ax, title=gene)
        ax.title.set_fontsize(_FONT_SIZE_ANNOT if compact else _FONT_SIZE_TITLE)
        return

    for x, vals, color, seed in [(1, tumor_clean, _COLOR_TUMOR, 3),
                                  (2, normal_clean, _COLOR_NORMAL, 4)]:
        if not vals:
            continue
        jit = _jitter(len(vals), seed=seed)
        ax.scatter([x + j for j in jit], vals,
                   color=color, s=8 if compact else 12,
                   alpha=0.38, zorder=3, edgecolors="none")
        _draw_box(ax, x, _boxplot_stats(vals), color,
                  width=0.42 if compact else 0.50)

    ax.set_xticks([1, 2])
    lbl_kw = {"fontfamily": _FONT_FAMILY, "fontweight": _FONT_WEIGHT,
               "fontsize": _FONT_SIZE_ANNOT if compact else _FONT_SIZE_TICK}
    ax.set_xticklabels(
        [f"T\n(n={len(tumor_clean)})" if compact else f"Tumor\n(n={len(tumor_clean)})",
         f"N\n(n={len(normal_clean)})" if compact else f"Normal\n(n={len(normal_clean)})"],
        **lbl_kw)
    ax.set_xlim(0.3, 2.7)
    _style_ax(ax, title=gene,
              ylabel="" if compact else "Expression")
    ax.title.set_fontsize(_FONT_SIZE_ANNOT if compact else _FONT_SIZE_TITLE)


def _plot_boxplot_geo(gene: str, gse_id: str,
                      tumor_vals: List[float], normal_vals: List[float],
                      out_path: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    _set_pub_rcparams()
    fig, ax = plt.subplots(figsize=(3.0, 3.8))
    _draw_geo_box_on_ax(ax, gene, gse_id, tumor_vals, normal_vals, compact=False)
    _style_ax(ax, title=f"{gene} — {gse_id}")
    plt.tight_layout()
    _save_fig(fig, out_path)
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# Combined per-dataset external boxplot panel (5 × 5 per GSE)
# ---------------------------------------------------------------------------

def _plot_geo_dataset_combined(
        gse_id: str,
        geo_box_data: List[Tuple[str, List[float], List[float]]],
        out_path: Path) -> bool:
    """One combined figure per GEO dataset showing all 25 genes (5 × 5 grid).

    geo_box_data: list of (gene, tumor_vals, normal_vals).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except ImportError:
        return False

    _set_pub_rcparams()
    n = len(geo_box_data)
    ncols = 5
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 2.2, nrows * 2.6),
                             squeeze=False)

    for idx, (gene, t_vals, n_vals) in enumerate(geo_box_data):
        ax = axes[idx // ncols][idx % ncols]
        _draw_geo_box_on_ax(ax, gene, gse_id, t_vals, n_vals, compact=True)

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    legend_elements = [
        Patch(facecolor=_COLOR_TUMOR,  label="Tumor",  edgecolor="#333333", linewidth=0.5),
        Patch(facecolor=_COLOR_NORMAL, label="Normal", edgecolor="#333333", linewidth=0.5),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2,
               fontsize=_FONT_SIZE_ANNOT, frameon=False,
               prop={"family": _FONT_FAMILY, "weight": _FONT_WEIGHT},
               bbox_to_anchor=(0.5, 0.0))

    dataset_label = _GEO_LABELS.get(gse_id, gse_id)
    fig.text(0.5, 1.0,
             f"External Validation — Top-25 CDPS Genes\n{dataset_label}",
             ha="center", va="top", fontsize=_FONT_SIZE_TITLE + 1,
             fontfamily=_FONT_FAMILY, fontweight=_FONT_WEIGHT)
    fig.text(-0.01, 0.5, "Expression",
             va="center", rotation="vertical", fontsize=_FONT_SIZE_LABEL,
             fontfamily=_FONT_FAMILY, fontweight=_FONT_WEIGHT)

    plt.tight_layout(rect=[0.02, 0.04, 1.0, 0.96])
    _save_fig(fig, out_path)
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# External concordance heatmap
# ---------------------------------------------------------------------------

def _plot_concordance_heatmap(genes: List[str],
                              datasets: List[str],
                              matrix: List[List[str]],
                              out_path: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        from matplotlib.patches import Patch
    except ImportError:
        return False

    _set_pub_rcparams()
    cell_val = {"concordant": 1, "discordant": 0, "absent": 2}
    numeric = [[cell_val.get(matrix[g][d], 2) for d in range(len(datasets))]
               for g in range(len(genes))]

    cmap = ListedColormap([_COLOR_DISCORD, _COLOR_CONCORD, _COLOR_ABSENT])
    fig_h = max(5, len(genes) * 0.38 + 1.8)
    fig_w = max(5, len(datasets) * 1.6 + 2.2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    ax.imshow(numeric, cmap=cmap, vmin=0, vmax=2, aspect="auto")

    mark = {"concordant": "+", "discordant": "-", "absent": "."}
    for gi in range(len(genes)):
        for di in range(len(datasets)):
            val = matrix[gi][di]
            col = "white" if val != "absent" else "#777777"
            ax.text(di, gi, mark.get(val, "—"), ha="center", va="center",
                    fontsize=_FONT_SIZE_LABEL,
                    fontfamily=_FONT_FAMILY, fontweight=_FONT_WEIGHT, color=col)

    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels([_GEO_LABELS.get(d, d) for d in datasets],
                       fontsize=_FONT_SIZE_TICK,
                       fontfamily=_FONT_FAMILY, fontweight=_FONT_WEIGHT)
    ax.set_yticks(range(len(genes)))
    ax.set_yticklabels(genes, fontsize=_FONT_SIZE_TICK,
                       fontfamily=_FONT_FAMILY, fontweight=_FONT_WEIGHT)
    _style_ax(ax, title="External Replication Concordance — Top-25 CDPS Genes")

    legend_elements = [
        Patch(facecolor=_COLOR_CONCORD, label="Concordant"),
        Patch(facecolor=_COLOR_DISCORD, label="Discordant"),
        Patch(facecolor=_COLOR_ABSENT,  label="Not measured"),
    ]
    leg = ax.legend(handles=legend_elements, bbox_to_anchor=(1.01, 1),
                    loc="upper left", fontsize=_FONT_SIZE_ANNOT, frameon=True)
    if leg:
        for t in leg.get_texts():
            t.set_fontfamily(_FONT_FAMILY)
            t.set_fontweight(_FONT_WEIGHT)

    plt.tight_layout()
    _save_fig(fig, out_path)
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# Survival summary bar chart
# ---------------------------------------------------------------------------

def _plot_survival_summary(survival_rows: List[Dict[str, Any]], out_path: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    _set_pub_rcparams()
    valid = [(r["gene"], _to_float(r.get("logrank_p", "")))
             for r in survival_rows
             if not math.isnan(_to_float(r.get("logrank_p", "")))]
    if not valid:
        return False

    valid.sort(key=lambda x: x[1])
    genes, pvals = zip(*valid)
    neg_log_p = [-math.log10(max(p, 1e-10)) for p in pvals]
    threshold = -math.log10(0.05)

    colors = [_COLOR_HIGH if p < 0.05 else "#AAAAAA" for p in pvals]
    fig, ax = plt.subplots(figsize=(6.0, max(4.5, len(genes) * 0.38 + 1.2)))
    ax.barh(range(len(genes)), neg_log_p, color=colors, edgecolor="none", height=0.72)
    ax.axvline(threshold, color="#444444", linestyle="--", linewidth=0.9, alpha=0.8)

    # Annotate the threshold line
    ax.text(threshold + 0.05, len(genes) - 0.5, "p = 0.05",
            fontsize=_FONT_SIZE_ANNOT, fontfamily=_FONT_FAMILY,
            fontweight=_FONT_WEIGHT, va="top", color="#444444")

    ax.set_yticks(range(len(genes)))
    ax.set_yticklabels(genes, fontsize=_FONT_SIZE_TICK,
                       fontfamily=_FONT_FAMILY, fontweight=_FONT_WEIGHT)
    _style_ax(ax,
              title="Overall Survival Association — Top-25 CDPS Genes (TCGA-ESCA)",
              xlabel="−log₁₀(p)",
              ylabel="")

    plt.tight_layout()
    _save_fig(fig, out_path)
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run(args: Any) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(name)s | %(message)s",
        stream=sys.stderr,
    )

    out_dir = Path(args.output_dir)
    for sub in ("figures/km", "figures/boxplots", "figures/external",
                "figures", "tables", "reports"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load top-25 gene list
    # ------------------------------------------------------------------
    top25_path = Path(args.top25_ranking) if args.top25_ranking else None
    ranking_rows = _safe_read_csv(top25_path)
    if ranking_rows:
        genes = [r["gene"] for r in ranking_rows if r.get("gene")]
        logger.info("Loaded %d genes from ranking file", len(genes))
    else:
        logger.warning("No ranking file — falling back to step5 CDPS list")
        step5_path = Path(args.output_dir).parent / "step5_gene_scoring" / "ranked_genes_cdps.csv"
        step5_rows = _safe_read_csv(step5_path)
        genes = [r["gene"] for r in (step5_rows or [])[:25] if r.get("gene")]

    if not genes:
        logger.error("Gene list is empty — provide --top25-ranking")
        return 1
    logger.info("Processing %d genes: %s ...", len(genes), ", ".join(genes[:5]))

    # ------------------------------------------------------------------
    # 2. Load TCGA expression & metadata
    # ------------------------------------------------------------------
    expr_path = Path(args.expression_matrix) if args.expression_matrix else None
    meta_path = Path(args.metadata) if args.metadata else None

    sample_barcodes, gene_expr, _ = (
        _load_expression_matrix(expr_path)
        if expr_path and expr_path.exists() else ([], {}, [])
    )
    metadata = _load_metadata(meta_path) if meta_path and meta_path.exists() else {}

    if not sample_barcodes:
        logger.warning("No TCGA expression loaded — survival/TCGA figures skipped")

    tumor_barcodes = [bc for bc in sample_barcodes
                      if str(metadata.get(bc, {}).get("label", "")).lower() in {"tumor", "1"}]
    normal_barcodes = [bc for bc in sample_barcodes
                       if str(metadata.get(bc, {}).get("label", "")).lower() in {"normal", "0"}]
    barcode_to_idx = {bc: i for i, bc in enumerate(sample_barcodes)}
    logger.info("TCGA: %d tumor, %d normal", len(tumor_barcodes), len(normal_barcodes))

    # ------------------------------------------------------------------
    # 3. Survival analysis (KM per gene) + individual figures
    # ------------------------------------------------------------------
    survival_rows: List[Dict[str, Any]] = []
    km_data_for_panel: List[Tuple] = []   # for combined panel

    for gene in genes:
        row: Dict[str, Any] = {"gene": gene}
        expr_vals = gene_expr.get(gene, [])

        if not expr_vals or not tumor_barcodes:
            row.update({"n_tumor": 0, "median_cutoff": "NA", "n_high": 0, "n_low": 0,
                        "logrank_chi2": "NA", "logrank_p": "NA", "hazard_direction": "NA"})
            survival_rows.append(row)
            continue

        tumor_idxs = [barcode_to_idx[bc] for bc in tumor_barcodes if bc in barcode_to_idx]
        tumor_expr = [expr_vals[i] for i in tumor_idxs if not math.isnan(expr_vals[i])]
        valid_tumor_bc = [tumor_barcodes[j] for j, bc in enumerate(tumor_barcodes)
                          if bc in barcode_to_idx and
                          not math.isnan(expr_vals[barcode_to_idx[bc]])]

        row["n_tumor"] = len(tumor_expr)
        if len(tumor_expr) < 4:
            row.update({"median_cutoff": "NA", "n_high": 0, "n_low": 0,
                        "logrank_chi2": "NA", "logrank_p": "NA", "hazard_direction": "NA"})
            survival_rows.append(row)
            continue

        med_cut = _median(tumor_expr)
        row["median_cutoff"] = _fmt(med_cut)

        high_t: List[float] = []
        high_e: List[int]  = []
        low_t:  List[float] = []
        low_e:  List[int]  = []

        for bc, val in zip(valid_tumor_bc, tumor_expr):
            meta = metadata.get(bc, {})
            vs   = meta.get("vital_status", "")
            dtd  = _to_float(meta.get("days_to_death", ""))
            dtlf = _to_float(meta.get("days_to_last_follow_up", ""))
            if math.isnan(dtd) and math.isnan(dtlf):
                continue
            obs_time = dtd if not math.isnan(dtd) else dtlf
            event = 1 if str(vs).lower() == "dead" else 0
            if val >= med_cut:
                high_t.append(obs_time)
                high_e.append(event)
            else:
                low_t.append(obs_time)
                low_e.append(event)

        row["n_high"] = len(high_t)
        row["n_low"]  = len(low_t)

        if len(high_t) < 2 or len(low_t) < 2:
            row.update({"logrank_chi2": "NA", "logrank_p": "NA", "hazard_direction": "NA"})
            survival_rows.append(row)
            km_data_for_panel.append((gene, high_t, high_e, low_t, low_e, float("nan")))
            continue

        chi2, p = _logrank_test(high_t, high_e, low_t, low_e)
        row["logrank_chi2"] = _fmt(chi2)
        row["logrank_p"]    = _fmt(p, 6)

        med_high = _median(high_t)
        med_low  = _median(low_t)
        if not math.isnan(med_high) and not math.isnan(med_low):
            row["hazard_direction"] = "low_better" if med_high < med_low else "high_better"
        else:
            row["hazard_direction"] = "NA"

        survival_rows.append(row)
        km_data_for_panel.append((gene, high_t, high_e, low_t, low_e, p))

        # Individual KM figure
        km_path = out_dir / "figures" / "km" / f"{gene}_KM_curve.svg"
        ok = _plot_km_curve(gene, high_t, high_e, low_t, low_e, p, km_path)
        if ok:
            logger.info("KM figure: %s (p=%.4f)", gene,
                        p if not math.isnan(p) else -1)

    _write_csv(out_dir / "tables" / "top25_survival_summary.csv", survival_rows,
               ["gene", "n_tumor", "median_cutoff", "n_high", "n_low",
                "logrank_chi2", "logrank_p", "hazard_direction"])

    # Combined KM panel
    if km_data_for_panel:
        ok = _plot_km_combined(km_data_for_panel,
                               out_dir / "figures" / "km" / "combined_KM_all25.svg")
        if ok:
            logger.info("Combined KM panel saved (%d genes)", len(km_data_for_panel))

    # Survival summary bar chart
    _plot_survival_summary(survival_rows,
                           out_dir / "figures" / "top25_survival_summary_plot.svg")

    # ------------------------------------------------------------------
    # 4. TCGA Tumor vs Normal boxplots
    # ------------------------------------------------------------------
    tcga_directions: Dict[str, str] = {}
    box_data_for_panel: List[Tuple] = []

    for gene in genes:
        expr_vals = gene_expr.get(gene, [])
        t_vals = ([expr_vals[barcode_to_idx[bc]] for bc in tumor_barcodes
                   if bc in barcode_to_idx] if expr_vals else [])
        n_vals = ([expr_vals[barcode_to_idx[bc]] for bc in normal_barcodes
                   if bc in barcode_to_idx] if expr_vals else [])

        m_t = _mean(t_vals)
        m_n = _mean(n_vals)
        if not math.isnan(m_t) and not math.isnan(m_n):
            tcga_directions[gene] = "up" if m_t > m_n else "down"
        else:
            tcga_directions[gene] = "absent"

        box_data_for_panel.append((gene, t_vals, n_vals))

        bp_path = out_dir / "figures" / "boxplots" / f"{gene}_TCGA_tumor_vs_normal.svg"
        _plot_boxplot_tcga(gene, t_vals, n_vals, bp_path)
        logger.info("TCGA boxplot: %s (dir=%s)", gene, tcga_directions[gene])

    # Combined TCGA boxplot panel
    if box_data_for_panel:
        ok = _plot_boxplot_tcga_combined(
            box_data_for_panel,
            out_dir / "figures" / "boxplots" / "combined_TCGA_boxplots_all25.svg")
        if ok:
            logger.info("Combined TCGA boxplot panel saved")

    # ------------------------------------------------------------------
    # 5. External GEO validation
    # ------------------------------------------------------------------
    geo_dir = Path(args.geo_prepared_dir) if args.geo_prepared_dir else None
    agilent_path = Path(args.agilent_replication) if args.agilent_replication else None
    geo_rep_path = Path(args.geo_replication) if args.geo_replication else None

    agilent_rows = _safe_read_csv(agilent_path) or []
    geo_rep_rows = _safe_read_csv(geo_rep_path) or []
    all_de_rows  = agilent_rows + geo_rep_rows

    de_index: Dict[Tuple[str, str], Dict] = {}
    for r in all_de_rows:
        de_index[(r.get("accession", ""), r.get("gene", ""))] = r

    all_datasets = sorted(set(r.get("accession", "") for r in all_de_rows
                              if r.get("accession"))) or list(_GEO_LABELS.keys())
    logger.info("External datasets: %s", all_datasets)

    external_rows: List[Dict[str, Any]] = []
    concordance_matrix: List[List[str]] = []

    # Cache GEO expression matrices (load once per dataset)
    geo_matrices: Dict[str, Tuple[Dict, List]] = {}
    geo_metas:    Dict[str, Dict]               = {}
    if geo_dir and geo_dir.exists():
        for gse_id in all_datasets:
            geo_matrices[gse_id] = _load_geo_matrix(geo_dir, gse_id)
            geo_metas[gse_id]    = _load_geo_metadata(geo_dir, gse_id)
            geo_expr_ds, gsm_ids_ds = geo_matrices[gse_id]
            n_found = sum(1 for g in genes if g in geo_expr_ds)
            if not geo_expr_ds:
                logger.warning(
                    "%s: no usable gene matrix — all %d target genes will show 'Not measured'",
                    gse_id, len(genes),
                )
            elif n_found < len(genes):
                missing_genes = [g for g in genes if g not in geo_expr_ds]
                logger.info(
                    "%s: %d/%d target genes found in expression matrix; "
                    "%d absent (likely non-coding genes not on this platform): %s",
                    gse_id, n_found, len(genes), len(missing_genes),
                    ", ".join(missing_genes),
                )

    # Per-dataset gene expression data for combined panels
    # {gse_id: [(gene, tumor_vals, normal_vals), ...]}
    geo_panel_data: Dict[str, List] = {gse: [] for gse in all_datasets}

    for gene in genes:
        gene_concordance_row: List[str] = []
        for gse_id in all_datasets:
            de_row = de_index.get((gse_id, gene))
            in_ds  = str((de_row or {}).get("in_dataset", "0"))
            if de_row is None or in_ds != "1":
                gene_concordance_row.append("absent")
            else:
                status = "concordant" if str(de_row.get("concordant", "0")) == "1" else "discordant"
                gene_concordance_row.append(status)
                external_rows.append({
                    "gene":           gene,
                    "dataset":        gse_id,
                    "ext_n_tumor":    de_row.get("ext_n_tumor",    "NA"),
                    "ext_n_normal":   de_row.get("ext_n_normal",   "NA"),
                    "ext_mean_tumor": de_row.get("ext_mean_tumor", "NA"),
                    "ext_mean_normal":de_row.get("ext_mean_normal","NA"),
                    "ext_effect":     de_row.get("ext_effect",     "NA"),
                    "ext_direction":  de_row.get("ext_direction",  "NA"),
                    "ref_direction":  de_row.get("ref_direction",  "NA"),
                    "concordant":     de_row.get("concordant",     ""),
                    "tcga_direction": tcga_directions.get(gene, "absent"),
                })

            # Extract per-gene expression for combined panel
            if gse_id in geo_matrices:
                geo_expr, gsm_ids = geo_matrices[gse_id]
                geo_meta = geo_metas.get(gse_id, {})
                gene_geo_vals = geo_expr.get(gene, [float("nan")] * len(gsm_ids))
                t_geo = [gene_geo_vals[i] for i, g in enumerate(gsm_ids)
                         if geo_meta.get(g, "").lower() == "tumor"]
                n_geo = [gene_geo_vals[i] for i, g in enumerate(gsm_ids)
                         if geo_meta.get(g, "").lower() == "normal"]
                geo_panel_data[gse_id].append((gene, t_geo, n_geo))

                # Individual gene-dataset figure
                bp_path = out_dir / "figures" / "external" / f"{gse_id}_{gene}_boxplot.svg"
                _plot_boxplot_geo(gene, gse_id, t_geo, n_geo, bp_path)

        concordance_matrix.append(gene_concordance_row)

    # Combined per-dataset panels
    for gse_id in all_datasets:
        pdata = geo_panel_data.get(gse_id, [])
        if pdata:
            panel_path = out_dir / "figures" / "external" / f"{gse_id}_all25_boxplots.svg"
            ok = _plot_geo_dataset_combined(gse_id, pdata, panel_path)
            if ok:
                logger.info("Combined external panel: %s (%d genes)", gse_id, len(pdata))

    ext_fields = ["gene", "dataset", "ext_n_tumor", "ext_n_normal",
                  "ext_mean_tumor", "ext_mean_normal", "ext_effect",
                  "ext_direction", "ref_direction", "concordant", "tcga_direction"]
    _write_csv(out_dir / "tables" / "top25_external_validation.csv",
               external_rows, ext_fields)

    # Concordance heatmap
    if genes and all_datasets:
        _plot_concordance_heatmap(
            genes, all_datasets, concordance_matrix,
            out_dir / "figures" / "top25_external_concordance_heatmap.svg")

    # ------------------------------------------------------------------
    # 6. Master evidence table
    # ------------------------------------------------------------------
    surv_index = {r["gene"]: r for r in survival_rows}

    def _ext_summary(gene: str) -> Dict:
        gr = [r for r in external_rows if r["gene"] == gene]
        n_t = len(gr)
        n_c = sum(1 for r in gr if str(r.get("concordant", "0")) == "1")
        return {"n_ext_datasets_tested": n_t,
                "n_ext_concordant": n_c,
                "ext_concordance_rate": _fmt(n_c / n_t if n_t else float("nan"), 3)}

    master_rows = []
    for i, gene in enumerate(genes):
        sr = surv_index.get(gene, {})
        ex = _ext_summary(gene)
        rr = (ranking_rows[i] if ranking_rows and i < len(ranking_rows) else {})
        master_rows.append({
            "priority_rank":       i + 1,
            "gene":                gene,
            "evidence_tier":       rr.get("evidence_tier", ""),
            "final_priority_score":rr.get("final_priority_score", ""),
            "cdps_score":          rr.get("cdps_score", ""),
            "tcga_direction":      tcga_directions.get(gene, "absent"),
            "n_tumor_surv":        sr.get("n_tumor", 0),
            "km_median_cutoff":    sr.get("median_cutoff", "NA"),
            "km_n_high":           sr.get("n_high", 0),
            "km_n_low":            sr.get("n_low", 0),
            "km_logrank_p":        sr.get("logrank_p", "NA"),
            "km_hazard_direction": sr.get("hazard_direction", "NA"),
            **ex,
        })

    master_fields = ["priority_rank", "gene", "evidence_tier", "final_priority_score",
                     "cdps_score", "tcga_direction",
                     "n_tumor_surv", "km_median_cutoff", "km_n_high", "km_n_low",
                     "km_logrank_p", "km_hazard_direction",
                     "n_ext_datasets_tested", "n_ext_concordant", "ext_concordance_rate"]
    _write_csv(out_dir / "tables" / "top25_full_validation_summary.csv",
               master_rows, master_fields)

    # ------------------------------------------------------------------
    # 7. Reports
    # ------------------------------------------------------------------
    n_sig = sum(1 for r in survival_rows
                if not math.isnan(_to_float(r.get("logrank_p", "")))
                and _to_float(r.get("logrank_p", "")) < 0.05)
    n_concord_genes = sum(
        1 for g in genes
        if any(str(r.get("concordant", "0")) == "1"
               for r in external_rows if r["gene"] == g))

    methods_text = f"""## Methods — Survival Analysis & External Cohort Validation

### Kaplan-Meier Survival Analysis
Overall survival was analysed for each of the top-25 CDPS-prioritised genes using TCGA-ESCA
(n={len(tumor_barcodes)} primary tumour samples with available survival data). Quantile-normalised
expression values were used. Each gene's samples were dichotomised at the gene-specific median
expression value into high- and low-expression groups. Survival curves were estimated by the
Kaplan-Meier product-limit method. Statistical significance was assessed by the log-rank
(Mantel-Cox) test implemented in pure NumPy. p < 0.05 was considered nominally significant.

### Tumour vs Adjacent Normal Expression
TCGA-ESCA primary tumour (n={len(tumor_barcodes)}) and matched adjacent normal
(n={len(normal_barcodes)}) samples were compared using log₂-normalised expression values.
Direction was determined by the sign of the group-mean difference (tumour − normal).

### External Cohort Validation
Expression concordance was assessed in {len(all_datasets)} independent GEO datasets:
{", ".join(all_datasets)}.
Pre-computed differential-expression replication results from Step 8 of the CAGE pipeline
were used. Concordance was defined as matching direction of differential expression between
TCGA-ESCA and each external cohort. Individual and combined panel figures (5 × 5 grid) were
generated per dataset using the CAGE publication-style defaults (DejaVu Sans, SVG primary
format; configurable via --font-family / --palette).
"""

    results_text = f"""## Results — Survival Analysis & External Cohort Validation

### Kaplan-Meier Summary
| Metric | Value |
|--------|-------|
| Genes analysed | {len(genes)} |
| Genes with valid KM data | {sum(1 for r in survival_rows if r.get('n_high', 0))} |
| Nominally significant (p < 0.05) | {n_sig} |

### External Validation Summary
| Metric | Value |
|--------|-------|
| External datasets | {len(all_datasets)} ({', '.join(all_datasets)}) |
| Genes concordant in ≥1 cohort | {n_concord_genes}/{len(genes)} |

### Top Survival-Associated Genes
"""
    sig_genes = sorted(
        [(r["gene"], _to_float(r.get("logrank_p", "")))
         for r in survival_rows
         if not math.isnan(_to_float(r.get("logrank_p", "")))],
        key=lambda x: x[1])
    results_text += "| Gene | Log-rank p | Hazard direction |\n|------|------------|------------------|\n"
    for g, p in sig_genes[:10]:
        hd = surv_index.get(g, {}).get("hazard_direction", "NA")
        results_text += f"| {g} | {p:.4f} | {hd} |\n"

    (out_dir / "reports" / "survival_methods.md").write_text(methods_text, encoding="utf-8")
    (out_dir / "reports" / "external_validation_summary.md").write_text(results_text, encoding="utf-8")

    # ------------------------------------------------------------------
    # 8. Reproducibility record
    # ------------------------------------------------------------------
    import datetime, platform
    repro: Dict[str, Any] = {
        "timestamp": datetime.datetime.now().isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "numpy_version": "unknown",
        "matplotlib_version": "unknown",
        "n_genes": len(genes),
        "n_tcga_tumor": len(tumor_barcodes),
        "n_tcga_normal": len(normal_barcodes),
        "external_datasets": all_datasets,
    }
    try:
        import numpy as np
        repro["numpy_version"] = np.__version__
    except ImportError:
        pass
    try:
        import matplotlib
        repro["matplotlib_version"] = matplotlib.__version__
    except ImportError:
        pass
    (out_dir / "reproducibility.json").write_text(
        json.dumps(repro, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    # 9. Manifest
    # ------------------------------------------------------------------
    manifest_entries = [
        {"file": str(p.relative_to(out_dir)), "type": p.suffix.lstrip("."),
         "size_bytes": str(p.stat().st_size)}
        for p in sorted(out_dir.rglob("*"))
        if p.is_file() and p.name != "manifest.csv"
    ]
    _write_csv(out_dir / "manifest.csv", manifest_entries,
               ["file", "type", "size_bytes"])

    logger.info("Step 6b survival/external validation complete → %s", out_dir)
    logger.info("  Genes processed: %d", len(genes))
    logger.info("  KM individual figures: %d", len(km_data_for_panel))
    logger.info("  Sig. survival genes (p<0.05): %d", n_sig)
    logger.info("  Concordant in ≥1 GEO cohort: %d/%d", n_concord_genes, len(genes))
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    from cage.cli_args import build_step_parser
    return build_step_parser(
        prog="step6b_survival_external_validation",
        step_title="Step 6b — Survival Analysis & External Cohort Validation",
        step_description=(
            "Kaplan-Meier survival analysis, TCGA tumour vs normal boxplots, and "
            "GEO external cohort validation for the top-25 CDPS-prioritised genes. "
            "Generates individual SVG figures and combined 5×5 publication panels."
        ),
        inputs_doc=(
            "  --expression-matrix   normalized_primary_matrix.csv\n"
            "  --metadata            master_samples_primary.csv\n"
            "  --top25-ranking       top25_final_priority_ranking.csv\n"
            "  --agilent-replication combined_de_replication.csv (step8_agilent)\n"
            "  --geo-replication     combined_de_replication.csv (step8_geo)\n"
            "  --geo-prepared-dir    outputs/step8_geo_prepared/"
        ),
        outputs_doc=(
            "  figures/km/<gene>_KM_curve.svg            (individual, × 25)\n"
            "  figures/km/combined_KM_all25.svg          (5 × 5 panel)\n"
            "  figures/boxplots/<gene>_TCGA_*.svg        (individual, × 25)\n"
            "  figures/boxplots/combined_TCGA_*.svg      (5 × 5 panel)\n"
            "  figures/external/<GSE>_<gene>_boxplot.svg (individual per gene+dataset)\n"
            "  figures/external/<GSE>_all25_boxplots.svg (per-dataset 5 × 5 panel)\n"
            "  figures/top25_external_concordance_heatmap.svg\n"
            "  figures/top25_survival_summary_plot.svg\n"
            "  tables/top25_survival_summary.csv\n"
            "  tables/top25_external_validation.csv\n"
            "  tables/top25_full_validation_summary.csv"
        ),
        example=(
            "python -m cage.step6b_survival_external_validation \\\n"
            "  --expression-matrix outputs/step2_cohort/normalized_primary_matrix.csv \\\n"
            "  --metadata outputs/step2_cohort/master_samples_primary.csv \\\n"
            "  --top25-ranking "
            "outputs/step6b_top25_prioritization/tables/top25_final_priority_ranking.csv \\\n"
            "  --agilent-replication "
            "outputs/step8_agilent_validation/combined_de_replication.csv \\\n"
            "  --geo-replication "
            "outputs/step8_geo_validation/combined_de_replication.csv \\\n"
            "  --geo-prepared-dir outputs/step8_geo_prepared \\\n"
            "  --output-dir outputs/step6b_survival_external"
        ),
        require_input_dir=False,
    )


def _add_input_args(p: Any) -> None:
    inp = p.add_argument_group("Input files")
    inp.add_argument("--expression-matrix", metavar="CSV",
                     help="normalized_primary_matrix.csv (samples × genes)")
    inp.add_argument("--metadata", metavar="CSV",
                     help="master_samples_primary.csv with vital_status columns")
    inp.add_argument("--top25-ranking", metavar="CSV",
                     help="top25_final_priority_ranking.csv from Step 6b prioritization")
    inp.add_argument("--agilent-replication", metavar="CSV",
                     help="combined_de_replication.csv from step8_agilent_validation")
    inp.add_argument("--geo-replication", metavar="CSV",
                     help="combined_de_replication.csv from step8_geo_validation")
    inp.add_argument("--geo-prepared-dir", metavar="DIR",
                     help="Directory containing step8_geo_prepared/<GSE>/ subdirs")


def main(argv=None) -> int:
    p = build_parser()
    _add_input_args(p)
    args = p.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
