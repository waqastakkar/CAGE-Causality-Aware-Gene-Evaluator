"""Create the composite public validation figure from Step 9 outputs."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D


GENES = ["FOXS1", "ESM1", "KIF2C"]

MARKER_CATEGORY_ORDER = [
    "immune checkpoints",
    "chemokines",
    "endothelial/angiogenesis",
    "stromal markers",
    "epithelial marker",
    "proliferation markers",
    "immune cell markers",
]

MARKER_GROUPS = {
    "PDCD1": "immune checkpoints",
    "CD274": "immune checkpoints",
    "CTLA4": "immune checkpoints",
    "LAG3": "immune checkpoints",
    "TIGIT": "immune checkpoints",
    "HAVCR2": "immune checkpoints",
    "PDCD1LG2": "immune checkpoints",
    "IDO1": "immune checkpoints",
    "CXCL9": "chemokines",
    "CXCL10": "chemokines",
    "PECAM1": "endothelial/angiogenesis",
    "VWF": "endothelial/angiogenesis",
    "VEGFA": "endothelial/angiogenesis",
    "COL1A1": "stromal markers",
    "ACTA2": "stromal markers",
    "EPCAM": "epithelial marker",
    "MKI67": "proliferation markers",
    "TOP2A": "proliferation markers",
    "CD8A": "immune cell markers",
    "CD8B": "immune cell markers",
    "CD4": "immune cell markers",
    "MS4A1": "immune cell markers",
    "CD68": "immune cell markers",
    "CD163": "immune cell markers",
    "ITGAM": "immune cell markers",
    "FCGR3A": "immune cell markers",
    "NKG7": "immune cell markers",
    "GNLY": "immune cell markers",
    "FOXP3": "immune cell markers",
}

GROUP_COLORS = {
    "immune checkpoints": "#E64B35",
    "chemokines": "#F39B7F",
    "endothelial/angiogenesis": "#00A087",
    "stromal markers": "#7E6148",
    "epithelial marker": "#4DBBD5",
    "proliferation markers": "#3C5488",
    "immune cell markers": "#8491B4",
}


def apply_publication_style() -> None:
    for font_path in (
        Path("/mnt/c/Windows/Fonts/times.ttf"),
        Path("/mnt/c/Windows/Fonts/timesbd.ttf"),
        Path("C:/Windows/Fonts/times.ttf"),
        Path("C:/Windows/Fonts/timesbd.ttf"),
    ):
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
            "font.weight": "bold",
            "mathtext.default": "bf",
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
            "axes.linewidth": 1.1,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 9,
        }
    )
    mpl.colormaps.register(
        LinearSegmentedColormap.from_list("public_validation_diverging", ["#3C5488", "#FFFFFF", "#E64B35"]),
        force=True,
    )
    mpl.colormaps.register(
        LinearSegmentedColormap.from_list("public_validation_sequential", ["#F7F7F7", "#D1E5F0", "#4DBBD5", "#00A087", "#E64B35"]),
        force=True,
    )


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.08,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=18,
        fontweight="bold",
        va="top",
        ha="left",
        color="black",
    )


def force_times_new_roman_bold(fig: plt.Figure) -> None:
    """Make every visible text element Times New Roman and bold."""
    for text in fig.findobj(mpl.text.Text):
        text.set_fontfamily("Times New Roman")
        text.set_fontweight("bold")
        text.set_color("black")


def draw_panel_a(ax: plt.Axes, step9_dir: Path) -> None:
    data = pd.read_csv(step9_dir / "cbioportal" / "cbioportal_alteration_frequency_plot_data.csv")
    data = data.set_index("gene").reindex(GENES)
    cols = ["Mutation", "CNA", "mRNA z >= 2", "Any alteration"]
    colors = ["#E64B35", "#4DBBD5", "#00A087", "#3C5488"]
    values = data[cols].astype(float) * 100.0
    x = np.arange(len(values.index))
    width = 0.18
    offsets = (np.arange(len(cols)) - (len(cols) - 1) / 2) * width
    for i, col in enumerate(cols):
        bars = ax.bar(x + offsets[i], values[col], width=width, label=col, color=colors[i], edgecolor="black", linewidth=0.4)
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.35,
                f"{height:.1f}",
                ha="center",
                va="bottom",
                fontsize=7,
                fontweight="bold",
            )
    ax.set_title("cBioPortal TCGA-ESCA Genomic And mRNA Support", fontsize=11, pad=10)
    ax.set_ylabel("Frequency (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(values.index)
    ax.set_ylim(0, max(10, float(values.max().max()) + 4))
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.7)
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="upper left")
    add_panel_label(ax, "A")


def draw_panel_b(ax: plt.Axes, step9_dir: Path) -> None:
    data = pd.read_csv(step9_dir / "hpa" / "hpa_candidate_evidence_plot_data.csv")
    data = data.set_index("gene").reindex(GENES)
    matrix = data.astype(float)
    im = ax.imshow(matrix.values, cmap="public_validation_sequential", vmin=0, vmax=1, aspect="auto")
    ax.set_title("Human Protein Atlas evidence availability", fontsize=11, pad=10)
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(matrix.index, fontsize=9)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, "Yes" if matrix.iloc[i, j] > 0 else "No", ha="center", va="center", fontsize=7)
    cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["No", "Yes"])
    cbar.ax.tick_params(labelsize=7)
    add_panel_label(ax, "B")


def marker_order(markers: Sequence[str]) -> list[str]:
    available = set(markers)
    ordered: list[str] = []
    for group in MARKER_CATEGORY_ORDER:
        group_markers = [m for m, g in MARKER_GROUPS.items() if g == group and m in available]
        ordered.extend(group_markers)
    ordered.extend([m for m in markers if m not in ordered])
    return ordered


def draw_panel_c(ax: plt.Axes, step9_dir: Path) -> None:
    corr = pd.read_csv(step9_dir / "immune" / "immune_marker_gene_correlations_fdr.csv")
    markers = marker_order(corr["marker_gene"].dropna().astype(str).unique().tolist())
    matrix = corr.pivot(index="candidate_gene", columns="marker_gene", values="spearman_r")
    matrix = matrix.reindex(index=GENES, columns=markers)
    im = ax.imshow(matrix.values.astype(float), cmap="public_validation_diverging", vmin=-0.75, vmax=0.75, aspect="auto")
    ax.set_title(
        "Candidate gene correlations with immune, stromal, endothelial, and proliferation markers",
        fontsize=12,
        pad=22,
    )
    ax.set_yticks(np.arange(len(GENES)))
    ax.set_yticklabels(GENES, fontsize=10)
    ax.set_xticks(np.arange(len(markers)))
    ax.set_xticklabels(markers, rotation=55, ha="right", fontsize=8)
    ax.tick_params(axis="x", pad=2)

    group_spans: list[tuple[str, int, int]] = []
    start = 0
    while start < len(markers):
        group = MARKER_GROUPS.get(markers[start], "other")
        end = start
        while end + 1 < len(markers) and MARKER_GROUPS.get(markers[end + 1], "other") == group:
            end += 1
        group_spans.append((group, start, end))
        start = end + 1

    for group, start, end in group_spans:
        if start > 0:
            ax.axvline(start - 0.5, color="black", linewidth=0.5, alpha=0.45)
    legend_handles = [
        Line2D([0], [0], color=GROUP_COLORS[group], lw=4, label=group)
        for group in MARKER_CATEGORY_ORDER
        if any(MARKER_GROUPS.get(marker) == group for marker in markers)
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.06),
        ncol=4,
        frameon=False,
        fontsize=7,
        handlelength=1.4,
        columnspacing=1.0,
    )

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix.iloc[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6.2, color="black")

    cbar = plt.colorbar(im, ax=ax, fraction=0.018, pad=0.012)
    cbar.set_label("Spearman rho", fontsize=9, fontweight="bold")
    cbar.ax.tick_params(labelsize=8)
    add_panel_label(ax, "C")


def draw_panel_d(ax: plt.Axes, step9_dir: Path) -> None:
    summary = pd.read_csv(step9_dir / "integrated" / "candidate_gene_public_validation_summary.csv")
    summary = summary.set_index("gene").reindex(GENES)
    immune = pd.read_csv(step9_dir / "immune" / "immune_marker_gene_correlations_fdr.csv")
    max_immune = immune.groupby("candidate_gene")["spearman_r"].apply(lambda s: float(s.abs().max()))
    data = pd.DataFrame(
        {
            "HPA evidence availability": summary["hpa_support_summary"].map(lambda x: 0.0 if str(x) in {"not available", "not found"} else 1.0),
            "cBioPortal any alteration": pd.to_numeric(summary["cbioportal_any_alteration_frequency"], errors="coerce").fillna(0.0),
            "cBioPortal mRNA z-score >= 2": pd.to_numeric(summary["cbioportal_mrna_upregulated_frequency"], errors="coerce").fillna(0.0),
            "Max immune/TME marker correlation": max_immune.reindex(GENES).fillna(0.0),
            "Single-cell data availability": summary["singlecell_celltype_localization"].map(lambda x: 0.0 if str(x) == "not available" else 1.0),
        },
        index=GENES,
    )
    im = ax.imshow(data.values.astype(float), cmap="public_validation_sequential", vmin=0, vmax=1, aspect="auto")
    ax.set_title("Integrated public validation summary", fontsize=12, pad=10)
    ax.set_yticks(np.arange(len(data.index)))
    ax.set_yticklabels(data.index, fontsize=10)
    ax.set_xticks(np.arange(len(data.columns)))
    ax.set_xticklabels(data.columns, rotation=25, ha="right", fontsize=8)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    cbar = plt.colorbar(im, ax=ax, fraction=0.018, pad=0.012)
    cbar.set_label("Evidence metric", fontsize=9, fontweight="bold")
    cbar.ax.tick_params(labelsize=8)
    data.reset_index(names="gene").to_csv(step9_dir / "public_validation_integrated_panel_plot_data.csv", index=False)
    add_panel_label(ax, "D")


def write_legend(path: Path) -> None:
    legend = (
        "Public biological validation of FOXS1, ESM1, and KIF2C in ESCA.\n"
        "(A) cBioPortal analysis showing mutation, copy-number alteration, mRNA upregulation, "
        "and overall alteration frequencies in TCGA-ESCA. (B) Human Protein Atlas evidence "
        "availability across protein, RNA, subcellular, reliability, and prognostic annotation "
        "categories. (C) Correlation of candidate genes with immune, stromal, endothelial, and "
        "proliferation markers, supporting distinct biological association patterns for FOXS1, "
        "ESM1, and KIF2C. (D) Integrated public validation summary combining HPA evidence "
        "availability, cBioPortal alteration support, immune/TME association, and single-cell "
        "data availability. Together, these data support biologically distinct ESCA-associated "
        "axes for FOXS1, ESM1, and KIF2C.\n"
    )
    path.write_text(legend, encoding="utf-8")


def normalize_svg_fonts_for_illustrator(svg_path: Path) -> None:
    """Rewrite Matplotlib SVG font shorthand into Illustrator-friendly CSS."""
    text = svg_path.read_text(encoding="utf-8")

    def replace_font(match: re.Match[str]) -> str:
        size = match.group("size")
        anchor = match.group("anchor") or ""
        extra = match.group("extra") or ""
        parts = [
            f"font-size: {size}px",
            "font-family: TimesNewRomanPS-BoldMT, 'Times New Roman'",
            "font-weight: 700",
        ]
        if anchor:
            parts.append(f"text-anchor: {anchor}")
        if extra:
            parts.append(extra.strip().rstrip(";"))
        return 'style="' + "; ".join(parts) + '"'

    text = re.sub(
        r'style="font:\s*700\s+(?P<size>[\d.]+)px\s+\'Times New Roman\';'
        r'(?:\s*text-anchor:\s*(?P<anchor>[^;"]+))?'
        r'(?P<extra>[^"]*)"',
        replace_font,
        text,
    )
    text = re.sub(
        r'style="font:\s*700\s+(?P<size>[\d.]+)px\s+\'Times New Roman\''
        r'(?:;\s*text-anchor:\s*(?P<anchor>[^;"]+))?'
        r'(?P<extra>[^"]*)"',
        replace_font,
        text,
    )
    text = text.replace(
        "font-family: Times New Roman",
        "font-family: TimesNewRomanPS-BoldMT, 'Times New Roman'",
    )
    svg_path.write_text(text, encoding="utf-8")


def build_figure(step9_dir: Path, output_dir: Path) -> None:
    apply_publication_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(14.5, 13.2), facecolor="white", constrained_layout=False)
    grid = GridSpec(
        3,
        2,
        figure=fig,
        height_ratios=[1.0, 1.95, 1.08],
        width_ratios=[1.0, 1.0],
        hspace=0.55,
        wspace=0.28,
        left=0.055,
        right=0.965,
        top=0.935,
        bottom=0.08,
    )
    fig.suptitle(
        "Public biological validation of FOXS1, ESM1, and KIF2C in ESCA",
        fontsize=17,
        fontweight="bold",
        y=0.985,
    )

    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, :])
    ax_d = fig.add_subplot(grid[2, :])

    draw_panel_a(ax_a, step9_dir)
    draw_panel_b(ax_b, step9_dir)
    draw_panel_c(ax_c, step9_dir)
    draw_panel_d(ax_d, step9_dir)
    force_times_new_roman_bold(fig)

    svg = output_dir / "public_biological_validation_composite.svg"
    png = output_dir / "public_biological_validation_composite.png"
    fig.savefig(svg, format="svg")
    fig.savefig(png, format="png", dpi=300)
    plt.close(fig)
    normalize_svg_fonts_for_illustrator(svg)
    write_legend(output_dir / "public_biological_validation_composite_legend.md")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cage-public-validation-figure",
        description="Create the composite public validation figure.",
    )
    parser.add_argument("--step9-dir", type=Path, default=Path("outputs/step9_public_biological_validation"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/step9_public_biological_validation/composite"))
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    build_figure(args.step9_dir, args.output_dir)


if __name__ == "__main__":
    main()
