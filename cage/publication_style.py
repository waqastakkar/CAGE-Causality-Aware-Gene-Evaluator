"""CAGE publication styling utilities.

Centralizes publication-grade figure standards used across all phases of the
CAGE pipeline. Every phase that produces figures imports from this module so
styling stays consistent and reproducible.

Standards enforced:

* Primary output format: SVG; optional high-DPI PDF/PNG for presentations.
* Configurable font family with graceful fallback. The default is a widely
  available sans-serif (DejaVu Sans) so figures render reliably on any system;
  users can override at the CLI to match journal requirements.
* CAGE default categorical palette with semantic role helpers
  (tumor/normal, up/down-regulated, enriched, validated, subgroup).
  Colours are colour-blind-friendly where feasible.
* Clean, minimal design: balanced whitespace, tight layout, high-DPI
  rasterization when exporting PDF/PNG.
* Configurable font sizes, figure dimensions, and resolution via
  :class:`PublicationStyle`.
* Deterministic, logged figure saves through :func:`save_figure`.

Typical use::

    from cage.publication_style import (
        PublicationStyle, apply_style, save_figure, cage_palette,
    )

    style = PublicationStyle(font_family="DejaVu Sans", base_font_size=10)
    apply_style(style)

    fig, ax = plt.subplots(figsize=style.figsize("single"))
    ax.scatter(x, y, color=cage_palette()["tumor"])
    save_figure(fig, "outputs/figures/fig1_scatter", style=style,
                formats=("svg", "pdf"))
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

logger = logging.getLogger("cage.publication_style")


# ---------------------------------------------------------------------------
# Font stack
# ---------------------------------------------------------------------------

#: Preferred font families, in priority order. Sans-serif default for
#: cross-platform reliability; serif fallbacks are included for users who
#: explicitly request a serif family at the CLI.
DEFAULT_FONT_FAMILIES: tuple[str, ...] = (
    "DejaVu Sans",
    "Arial",
    "Helvetica",
    "Liberation Sans",
    "sans-serif",
)

# Kept as an alias for backward compatibility with older callers that
# referenced the serif fallback list directly.
ROMAN_SERIF_FAMILIES: tuple[str, ...] = (
    "Times New Roman",
    "Times",
    "STIXGeneral",
    "DejaVu Serif",
)


# ---------------------------------------------------------------------------
# CAGE default palette
# ---------------------------------------------------------------------------

#: Categorical palette used across CAGE figures. Colour-blind-friendly where
#: feasible; order is stable so figures render deterministically.
CAGE_PALETTE_CATEGORICAL: tuple[str, ...] = (
    "#E64B35",  # vermilion
    "#4DBBD5",  # sky
    "#00A087",  # teal
    "#3C5488",  # indigo
    "#F39B7F",  # coral
    "#8491B4",  # slate
    "#91D1C2",  # mint
    "#DC0000",  # crimson
    "#7E6148",  # bronze
    "#B09C85",  # sand
)

#: Semantic role -> hex color. Keeps meaning consistent across all phases.
CAGE_PALETTE_SEMANTIC: dict[str, str] = {
    "tumor": "#E64B35",
    "normal": "#4DBBD5",
    "up": "#E64B35",
    "down": "#3C5488",
    "nonsig": "#B0B0B0",
    "enriched": "#00A087",
    "validated": "#F39B7F",
    "highlight": "#DC0000",
    "subgroup_a": "#3C5488",
    "subgroup_b": "#F39B7F",
    "subgroup_c": "#00A087",
    "subgroup_d": "#8491B4",
}

#: Continuous diverging colormap stops (low -> mid -> high). Suitable for
#: log2 fold-change heatmaps; register once per :func:`apply_style` call.
DIVERGING_STOPS: tuple[str, str, str] = ("#3C5488", "#FFFFFF", "#E64B35")

#: Continuous sequential colormap stops for non-negative magnitudes
#: (e.g., attribution magnitude, CDPS component scores).
SEQUENTIAL_STOPS: tuple[str, ...] = (
    "#F7F7F7",
    "#D1E5F0",
    "#4DBBD5",
    "#00A087",
    "#E64B35",
)


def cage_palette(kind: str = "semantic") -> dict[str, str] | tuple[str, ...]:
    """Return the CAGE default palette.

    Parameters
    ----------
    kind:
        ``"semantic"`` for role -> color mapping (default),
        ``"categorical"`` for an ordered tuple of hex colors.
    """
    if kind == "semantic":
        return dict(CAGE_PALETTE_SEMANTIC)
    if kind == "categorical":
        return tuple(CAGE_PALETTE_CATEGORICAL)
    raise ValueError(
        f"Unknown palette kind {kind!r}; expected 'semantic' or 'categorical'."
    )


# Backward-compatible alias for any external callers.
nature_palette = cage_palette


# ---------------------------------------------------------------------------
# Style configuration
# ---------------------------------------------------------------------------


@dataclass
class PublicationStyle:
    """Configurable styling parameters for all CAGE figures.

    All fields are plain Python values so the config can be serialized to
    JSON/YAML alongside figure outputs for full reproducibility.
    """

    font_family: str = "DejaVu Sans"
    font_fallbacks: tuple[str, ...] = DEFAULT_FONT_FAMILIES
    bold: bool = False
    base_font_size: float = 10.0
    title_font_size: float = 12.0
    axis_label_font_size: float = 10.0
    tick_label_font_size: float = 9.0
    legend_font_size: float = 9.0
    panel_label_font_size: float = 12.0
    colorbar_label_font_size: float = 9.0
    annotation_font_size: float = 8.0

    axis_linewidth: float = 1.4
    tick_major_width: float = 1.2
    tick_minor_width: float = 0.8
    tick_major_length: float = 4.0
    tick_minor_length: float = 2.5

    figure_dpi: int = 300
    raster_dpi: int = 600  # when exporting PDF/PNG for print

    single_column_width_in: float = 3.5
    double_column_width_in: float = 7.2
    default_aspect: float = 0.75

    palette: str = "cage"
    diverging_stops: tuple[str, str, str] = DIVERGING_STOPS
    sequential_stops: tuple[str, ...] = SEQUENTIAL_STOPS

    default_formats: tuple[str, ...] = ("svg",)
    tight_layout: bool = True

    # Free-form extras so downstream phases can pin additional parameters
    # (e.g., UMAP neighbors, heatmap clustering) without subclassing.
    extras: dict[str, Any] = field(default_factory=dict)

    # --- helpers -----------------------------------------------------------

    def figsize(self, width: str | float = "single", aspect: float | None = None) -> tuple[float, float]:
        """Return a ``(width, height)`` tuple in inches.

        ``width`` accepts ``"single"``, ``"double"``, or a numeric inch value.
        ``aspect`` defaults to :attr:`default_aspect` (height / width).
        """
        if isinstance(width, str):
            if width == "single":
                w = self.single_column_width_in
            elif width == "double":
                w = self.double_column_width_in
            else:
                raise ValueError(f"Unknown width preset {width!r}.")
        else:
            w = float(width)
        a = self.default_aspect if aspect is None else float(aspect)
        return w, w * a

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict of this style."""
        d = asdict(self)
        d["font_fallbacks"] = list(self.font_fallbacks)
        d["diverging_stops"] = list(self.diverging_stops)
        d["sequential_stops"] = list(self.sequential_stops)
        d["default_formats"] = list(self.default_formats)
        return d


# ---------------------------------------------------------------------------
# rcParams application
# ---------------------------------------------------------------------------


def apply_style(style: PublicationStyle | None = None) -> PublicationStyle:
    """Apply the publication style to matplotlib's rcParams.

    Safe to call in scripts that do not have matplotlib installed: the
    import is deferred and a clear error is raised so callers can catch it.
    Returns the applied :class:`PublicationStyle` for chaining.
    """
    style = style or PublicationStyle()

    try:
        import matplotlib as mpl
        from matplotlib.colors import LinearSegmentedColormap
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise ImportError(
            "matplotlib is required to apply CAGE publication style."
        ) from exc

    family_stack = [style.font_family, *style.font_fallbacks]
    weight = "bold" if style.bold else "normal"

    serif_families = {"Times New Roman", "Times", "STIXGeneral", "DejaVu Serif", "serif"}
    is_serif = style.font_family in serif_families
    generic_family = "serif" if is_serif else "sans-serif"
    family_key = "font.serif" if is_serif else "font.sans-serif"

    mpl.rcParams.update(
        {
            # Typography
            "font.family": generic_family,
            family_key: list(dict.fromkeys(family_stack)),
            "font.weight": weight,
            "font.size": style.base_font_size,
            "axes.titlesize": style.title_font_size,
            "axes.titleweight": weight,
            "axes.labelsize": style.axis_label_font_size,
            "axes.labelweight": weight,
            "xtick.labelsize": style.tick_label_font_size,
            "ytick.labelsize": style.tick_label_font_size,
            "legend.fontsize": style.legend_font_size,
            "figure.titlesize": style.title_font_size,
            "figure.titleweight": weight,
            # Axes / ticks
            "axes.linewidth": style.axis_linewidth,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.width": style.tick_major_width,
            "ytick.major.width": style.tick_major_width,
            "xtick.minor.width": style.tick_minor_width,
            "ytick.minor.width": style.tick_minor_width,
            "xtick.major.size": style.tick_major_length,
            "ytick.major.size": style.tick_major_length,
            "xtick.minor.size": style.tick_minor_length,
            "ytick.minor.size": style.tick_minor_length,
            "xtick.direction": "out",
            "ytick.direction": "out",
            # Figure / saving
            "figure.dpi": style.figure_dpi,
            "savefig.dpi": style.raster_dpi,
            "savefig.bbox": "tight",
            "savefig.transparent": False,
            "svg.fonttype": "none",  # keep text as text in SVG
            "pdf.fonttype": 42,       # TrueType for editable PDFs
            "ps.fonttype": 42,
            # Legend
            "legend.frameon": False,
            "legend.handlelength": 1.6,
        }
    )

    # Register semantic colormaps under stable names so callers can use
    # ``cmap="cage_diverging"`` / ``"cage_sequential"`` directly.
    _register_cmap(
        "cage_diverging",
        LinearSegmentedColormap.from_list("cage_diverging", list(style.diverging_stops)),
    )
    _register_cmap(
        "cage_sequential",
        LinearSegmentedColormap.from_list("cage_sequential", list(style.sequential_stops)),
    )

    logger.debug("Applied CAGE publication style: %s", style.as_dict())
    return style


def _register_cmap(name: str, cmap: Any) -> None:
    """Register a colormap under ``name``, tolerant of matplotlib versions."""
    try:
        import matplotlib as mpl

        # matplotlib >= 3.9 prefers mpl.colormaps.register
        if hasattr(mpl, "colormaps") and hasattr(mpl.colormaps, "register"):
            try:
                mpl.colormaps.register(cmap=cmap, name=name, force=True)
                return
            except TypeError:
                mpl.colormaps.register(cmap=cmap, name=name)
                return
        # Legacy API
        from matplotlib import cm  # type: ignore

        cm.register_cmap(name=name, cmap=cmap)  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - best-effort registration
        logger.warning("Could not register colormap %s; using inline only.", name)


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------


def save_figure(
    fig: Any,
    path: str | Path,
    *,
    style: PublicationStyle | None = None,
    formats: Sequence[str] | None = None,
    metadata: Mapping[str, Any] | None = None,
    close: bool = True,
) -> list[Path]:
    """Save a figure to one or more formats with CAGE conventions.

    Writes SVG by default (primary format per plane.md). Additional PDF/PNG
    exports use high-DPI rasterization. Every save is logged, and an
    optional sidecar ``<basename>.style.json`` is written capturing the
    style parameters and any user metadata for full reproducibility.

    Parameters
    ----------
    fig:
        A matplotlib ``Figure``.
    path:
        Base path (with or without extension). The stem is used and the
        extension is replaced with each requested format.
    style:
        Applied style. If ``None``, a default :class:`PublicationStyle`
        is instantiated for metadata (rcParams are not re-applied).
    formats:
        Iterable of file extensions (``"svg"``, ``"pdf"``, ``"png"``).
        Defaults to the style's :attr:`PublicationStyle.default_formats`.
    metadata:
        Extra JSON-serializable metadata to stash in the sidecar (e.g.,
        data source, phase, caption).
    close:
        Close the figure after saving (default True).
    """
    style = style or PublicationStyle()
    fmts = tuple(formats) if formats is not None else style.default_formats
    base = Path(path).with_suffix("")
    base.parent.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    skipped: list[tuple[str, str]] = []
    for fmt in fmts:
        fmt_l = fmt.lower().lstrip(".")
        out = base.with_suffix(f".{fmt_l}")
        try:
            save_kwargs: dict[str, Any] = {"format": fmt_l}
            if fmt_l in {"pdf", "png"}:
                save_kwargs["dpi"] = style.raster_dpi
            if style.tight_layout:
                save_kwargs["bbox_inches"] = "tight"
            fig.savefig(out, **save_kwargs)
            written.append(out)
            logger.info("Saved figure: %s", out)
        except Exception as exc:
            skipped.append((fmt_l, str(exc)))
            logger.warning("Skipped figure format %s for %s: %s", fmt_l, out, exc)

    sidecar = base.with_suffix(".style.json")
    try:
        sidecar.write_text(
            json.dumps(
                {
                    "base_path": str(base),
                    "formats_written": [p.name for p in written],
                    "formats_skipped": skipped,
                    "style": style.as_dict(),
                    "metadata": dict(metadata) if metadata else {},
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    except Exception as exc:  # pragma: no cover - sidecar is best-effort
        logger.warning("Could not write style sidecar %s: %s", sidecar, exc)

    if close:
        try:
            import matplotlib.pyplot as plt

            plt.close(fig)
        except Exception:  # pragma: no cover
            pass

    return written


# ---------------------------------------------------------------------------
# Convenience: semantic color lookup
# ---------------------------------------------------------------------------


def semantic_color(role: str, *, default: str = "#4D4D4D") -> str:
    """Return a CAGE-palette hex color for a semantic role.

    Unknown roles fall back to a neutral gray, and a warning is logged so
    mismatches surface during review rather than silently rendering.
    """
    color = CAGE_PALETTE_SEMANTIC.get(role)
    if color is None:
        logger.warning("Unknown semantic color role %r; using fallback.", role)
        return default
    return color


def categorical_colors(n: int) -> list[str]:
    """Return ``n`` distinct CAGE-palette categorical colors (cycled)."""
    if n <= 0:
        return []
    base = list(CAGE_PALETTE_CATEGORICAL)
    return [base[i % len(base)] for i in range(n)]


# ---------------------------------------------------------------------------
# Figure-generation logging helper
# ---------------------------------------------------------------------------


def log_figure_status(
    generated: Iterable[str],
    skipped: Iterable[tuple[str, str]] = (),
    *,
    logger_name: str = "cage.figures",
) -> None:
    """Emit a compact summary of which figures were generated vs skipped.

    Used at the end of each phase script to satisfy plane.md's logging
    requirement ("which figures were successfully generated and which
    were skipped due to missing data").
    """
    phase_logger = logging.getLogger(logger_name)
    for name in generated:
        phase_logger.info("figure OK: %s", name)
    for name, reason in skipped:
        phase_logger.warning("figure SKIPPED: %s (%s)", name, reason)


__all__ = [
    "DEFAULT_FONT_FAMILIES",
    "ROMAN_SERIF_FAMILIES",
    "CAGE_PALETTE_CATEGORICAL",
    "CAGE_PALETTE_SEMANTIC",
    "DIVERGING_STOPS",
    "SEQUENTIAL_STOPS",
    "PublicationStyle",
    "apply_style",
    "save_figure",
    "cage_palette",
    "nature_palette",  # backward-compatible alias
    "semantic_color",
    "categorical_colors",
    "log_figure_status",
]
