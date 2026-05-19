"""CAGE shared CLI conventions.

All pipeline steps expose a consistent command-line surface through the
helpers in this module. Every script uses :func:`build_step_parser` (which
wires in :func:`add_global_args` and :func:`add_figure_args`) so the flag
names, defaults, and ``--help`` branding stay identical across phases.

Global options
--------------
``--input-dir``, ``--output-dir``, ``--seed``, ``--n-threads``, ``--overwrite``
``--log-level`` (extra convenience; defaults to ``INFO``)

Figure options
--------------
``--figure-format`` primary format (default ``svg``)
``--extra-figure-format`` additional formats, repeatable (e.g. ``pdf``, ``png``)
``--font-family`` (default ``"DejaVu Sans"``)
``--font-size`` configurable base font size (default ``10``)
``--palette`` (default ``"cage"``)

Phase-specific flags are added inside each step script.
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing
import os
import sys
from pathlib import Path
from typing import Iterable, Sequence

from . import __version__
from .publication_style import PublicationStyle

logger = logging.getLogger("cage.cli")


# ---------------------------------------------------------------------------
# Branding & constants
# ---------------------------------------------------------------------------

CAGE_BANNER = (
    "================================================================\n"
    f"        CAGE: Causality-Aware Gene Evaluator  v{__version__}\n"
    "----------------------------------------------------------------\n"
    "  This tool prioritizes top candidate genes using causality-\n"
    "  aware, environment-aware, multi-layer analysis.\n"
    "================================================================"
)

FIGURE_STANDARDS_NOTE = """\
Figure output follows publication-grade defaults:
  * Primary format: SVG (DejaVu Sans typography, CAGE default palette)
  * Optional high-DPI PDF/PNG for presentation/supplementary use
  * Font family and palette are user-configurable via --font-family/--palette
  * Styling pinned centrally via cage.publication_style
"""

DEFAULT_SEED = 2026
FIGURE_FORMAT_CHOICES = ("svg", "pdf", "png")
PALETTE_CHOICES = ("cage", "nature")  # "nature" kept as backward-compatible alias


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_n_threads() -> int:
    """Return a conservative default for ``--n-threads``.

    Uses half the visible CPU count (floor 1) so the pipeline is a polite
    citizen on shared workstations. The user can always override.
    """
    try:
        cpu = multiprocessing.cpu_count()
    except NotImplementedError:  # pragma: no cover
        cpu = 1
    return max(1, cpu // 2)


def _format_help_block(title: str, body: str) -> str:
    """Indent a multi-line block under a titled section for the epilog."""
    indented = "\n".join("  " + line if line else "" for line in body.splitlines())
    return f"{title}:\n{indented}"


# ---------------------------------------------------------------------------
# Argument groups
# ---------------------------------------------------------------------------


def add_global_args(
    parser: argparse.ArgumentParser,
    *,
    require_input_dir: bool = True,
) -> argparse._ArgumentGroup:
    """Add the CAGE global options to ``parser``.

    ``require_input_dir`` can be toggled off for scripts (e.g. step 7) that
    take per-phase input directories instead of a single ``--input-dir``.
    """
    g = parser.add_argument_group("Global options")
    g.add_argument(
        "--input-dir",
        type=Path,
        required=require_input_dir,
        default=None,
        metavar="DIR",
        help="Directory containing inputs for this step.",
    )
    g.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        metavar="DIR",
        help="Directory where outputs (tables, figures, logs) are written.",
    )
    g.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        metavar="INT",
        help=f"Random seed for reproducibility (default: {DEFAULT_SEED}).",
    )
    g.add_argument(
        "--n-threads",
        type=int,
        default=_default_n_threads(),
        metavar="INT",
        help=(
            "Number of worker threads/processes for parallelizable steps "
            f"(default: {_default_n_threads()})."
        ),
    )
    g.add_argument(
        "--overwrite",
        action="store_true",
        help="Safely regenerate outputs even if they already exist.",
    )
    g.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity (default: INFO).",
    )
    return g


def add_figure_args(parser: argparse.ArgumentParser) -> argparse._ArgumentGroup:
    """Add the CAGE figure options."""
    g = parser.add_argument_group("Figure options")
    g.add_argument(
        "--figure-format",
        choices=FIGURE_FORMAT_CHOICES,
        default="svg",
        help="Primary figure output format (default: svg).",
    )
    g.add_argument(
        "--extra-figure-format",
        action="append",
        choices=FIGURE_FORMAT_CHOICES,
        default=[],
        metavar="FMT",
        help=(
            "Additional figure format to emit alongside the primary. "
            "Repeatable: --extra-figure-format pdf --extra-figure-format png"
        ),
    )
    g.add_argument(
        "--font-family",
        default="DejaVu Sans",
        metavar="NAME",
        help=(
            "Preferred figure font family (default: DejaVu Sans). "
            "Any font installed on the system is accepted."
        ),
    )
    g.add_argument(
        "--font-size",
        type=float,
        default=10.0,
        metavar="PT",
        help="Base font size in points (default: 10).",
    )
    g.add_argument(
        "--palette",
        choices=PALETTE_CHOICES,
        default="cage",
        help="Categorical palette to use (default: cage).",
    )
    return g


# ---------------------------------------------------------------------------
# Post-parse helpers
# ---------------------------------------------------------------------------


def style_from_args(args: argparse.Namespace) -> PublicationStyle:
    """Construct a :class:`PublicationStyle` from parsed arguments.

    Merges the primary ``--figure-format`` with any ``--extra-figure-format``
    values, preserves order, and deduplicates so SVG stays first.
    """
    fmts: list[str] = [args.figure_format]
    for extra in getattr(args, "extra_figure_format", []) or []:
        if extra not in fmts:
            fmts.append(extra)
    return PublicationStyle(
        font_family=args.font_family,
        base_font_size=float(args.font_size),
        axis_label_font_size=float(args.font_size),
        tick_label_font_size=max(1.0, float(args.font_size) - 1.0),
        legend_font_size=max(1.0, float(args.font_size) - 1.0),
        title_font_size=float(args.font_size) + 2.0,
        panel_label_font_size=float(args.font_size) + 2.0,
        annotation_font_size=max(1.0, float(args.font_size) - 2.0),
        colorbar_label_font_size=max(1.0, float(args.font_size) - 1.0),
        palette=args.palette,
        default_formats=tuple(fmts),
    )


def configure_logging(args: argparse.Namespace, *, log_file: Path | None = None) -> None:
    """Configure root logging based on ``--log-level``.

    If ``log_file`` is supplied (typically ``<output_dir>/logs/<step>.log``)
    we attach an additional file handler so each run has a durable trail.
    """
    level_name = getattr(args, "log_level", "INFO")
    level = getattr(logging, level_name.upper(), logging.INFO)
    fmt = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    root = logging.getLogger()
    # Reset handlers so repeat CLI invocations don't stack duplicates.
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(level)
    stream = logging.StreamHandler(stream=sys.stderr)
    stream.setFormatter(logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S"))
    root.addHandler(stream)
    if log_file is not None:
        try:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
            fh.setFormatter(logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S"))
            root.addHandler(fh)
        except Exception as exc:  # pragma: no cover
            logger.warning("Could not attach log file %s: %s", log_file, exc)


def apply_thread_limits(args: argparse.Namespace) -> None:
    """Propagate ``--n-threads`` to common numerical libraries.

    Setting these env vars early (before numpy/torch import) is the most
    reliable way to actually cap BLAS/OpenMP thread counts in practice.
    """
    n = str(max(1, int(getattr(args, "n_threads", 1))))
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ.setdefault(var, n)


# ---------------------------------------------------------------------------
# Parser factory
# ---------------------------------------------------------------------------


def build_step_parser(
    *,
    prog: str,
    step_title: str,
    step_description: str,
    inputs_doc: str,
    outputs_doc: str,
    example: str,
    require_input_dir: bool = True,
    extra_epilog: str = "",
) -> argparse.ArgumentParser:
    """Build a branded CAGE parser wired with global + figure args.

    Step scripts call this factory, then add their own ``add_argument_group``
    for phase-specific flags. The returned parser uses
    :class:`argparse.RawDescriptionHelpFormatter` so the banner, inputs,
    outputs, and example block render exactly as written.
    """
    description = f"{CAGE_BANNER}\n\n{step_title}\n\n{step_description}"
    epilog_blocks = [
        _format_help_block("Inputs", inputs_doc.rstrip()),
        _format_help_block("Outputs", outputs_doc.rstrip()),
        _format_help_block("Example", example.rstrip()),
        FIGURE_STANDARDS_NOTE.rstrip(),
    ]
    if extra_epilog.strip():
        epilog_blocks.append(extra_epilog.rstrip())
    epilog = "\n\n".join(epilog_blocks)

    parser = argparse.ArgumentParser(
        prog=prog,
        description=description,
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=True,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"CAGE {__version__}",
        help="Print CAGE version and exit.",
    )
    add_global_args(parser, require_input_dir=require_input_dir)
    add_figure_args(parser)
    return parser


# ---------------------------------------------------------------------------
# Rigor profile
# ---------------------------------------------------------------------------

#: Settings overridden by ``--rigor-profile all_genes``.
#: Keys match argparse dest names; values are applied only when the arg
#: still holds its parser-level default (i.e. the user did not pass it).
_ALL_GENES_OVERRIDES: dict[str, object] = {
    # Step 2
    "use_all_filtered_genes": True,
    "run_cohort_flow_figure": True,
    # Step 3 / 4
    "bootstrap_ci_n": 1000,
    "run_calibration": True,
    "run_subgroup_sensitivity": True,
    # Step 5
    "run_perturbation": True,
    # All steps
    "save_config_hash": True,
}


def add_rigor_profile_arg(parser: argparse.ArgumentParser) -> None:
    """Add ``--rigor-profile`` to *parser* (call from each step's ``build_parser``)."""
    parser.add_argument(
        "--rigor-profile",
        dest="rigor_profile",
        choices=("standard", "all_genes"),
        default="standard",
        metavar="PROFILE",
        help=(
            "Preset analysis rigor level. "
            "'standard' keeps per-flag defaults. "
            "'all_genes' enables all rigorous options: --use-all-filtered-genes, "
            "--run-calibration, --run-subgroup-sensitivity, --run-perturbation, "
            "--bootstrap-ci-n 1000. "
            "Individual flags still override this preset when explicitly passed. "
            "(default: standard)"
        ),
    )


def apply_rigor_profile(args: argparse.Namespace, *, parser_defaults: dict | None = None) -> None:
    """Apply ``--rigor-profile`` overrides to *args* in-place.

    Only touches attributes that are still at their parser-level default so
    that explicit CLI flags always win over the profile.  Call this inside
    each step's ``main()`` after ``parse_args()``, before any logic runs.

    Parameters
    ----------
    args:
        Parsed namespace to modify.
    parser_defaults:
        Map of dest → default value as returned by
        ``{a.dest: a.default for a in parser._actions}``.
        When supplied, only attributes whose current value matches the parser
        default are overridden (= user did not supply them).
        When omitted, overrides are applied unconditionally for keys that do
        not already have a truthy value.
    """
    if getattr(args, "rigor_profile", "standard") != "all_genes":
        return

    for dest, value in _ALL_GENES_OVERRIDES.items():
        if not hasattr(args, dest):
            continue
        current = getattr(args, dest)
        if parser_defaults is not None:
            default = parser_defaults.get(dest)
            if current != default:
                continue  # user passed this flag explicitly
        else:
            if current:
                continue  # already truthy — don't overwrite
        setattr(args, dest, value)

    # When use_all_filtered_genes is now True, push n_top_variable_genes to
    # a sentinel that the step-2 code interprets as "no cap".
    if getattr(args, "use_all_filtered_genes", False):
        if parser_defaults is not None:
            default_n = parser_defaults.get("n_top_variable_genes", 5000)
            if getattr(args, "n_top_variable_genes", default_n) == default_n:
                setattr(args, "n_top_variable_genes", 999_999)
        else:
            setattr(args, "n_top_variable_genes", 999_999)

    logger.info(
        "Rigor profile 'all_genes' applied — "
        "use_all_filtered_genes=%s  bootstrap_ci_n=%s  "
        "run_calibration=%s  run_subgroup_sensitivity=%s  run_perturbation=%s",
        getattr(args, "use_all_filtered_genes", "N/A"),
        getattr(args, "bootstrap_ci_n", "N/A"),
        getattr(args, "run_calibration", "N/A"),
        getattr(args, "run_subgroup_sensitivity", "N/A"),
        getattr(args, "run_perturbation", "N/A"),
    )


# ---------------------------------------------------------------------------
# Small utilities reused across steps
# ---------------------------------------------------------------------------


def ensure_output_dir(args: argparse.Namespace) -> Path:
    """Create ``args.output_dir`` (and a ``logs`` subdir) and return it."""
    out: Path = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    return out


def resolve_path(
    explicit: Path | None,
    fallback_dir: Path | None,
    filename: str,
) -> Path | None:
    """Resolve a path argument: explicit > ``fallback_dir/filename`` > None."""
    if explicit is not None:
        return Path(explicit)
    if fallback_dir is not None:
        return Path(fallback_dir) / filename
    return None


def flags_as_dict(args: argparse.Namespace, keys: Sequence[str]) -> dict[str, object]:
    """Materialize a subset of parsed args into a plain dict (for logs/JSON)."""
    return {k: getattr(args, k, None) for k in keys}


__all__ = [
    "CAGE_BANNER",
    "FIGURE_STANDARDS_NOTE",
    "DEFAULT_SEED",
    "FIGURE_FORMAT_CHOICES",
    "PALETTE_CHOICES",
    "add_global_args",
    "add_figure_args",
    "build_step_parser",
    "style_from_args",
    "configure_logging",
    "apply_thread_limits",
    "ensure_output_dir",
    "resolve_path",
    "flags_as_dict",
    "add_rigor_profile_arg",
    "apply_rigor_profile",
]
