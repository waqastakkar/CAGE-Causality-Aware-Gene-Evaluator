"""CLI smoke tests — verify --help does not crash and parsers are wired up.

These tests do NOT run the full pipeline (no real data required). They only
verify that:
  1. Each step's build_parser() returns an ArgumentParser.
  2. --help exits with code 0 (argparse contract).
  3. Essential flags are present in each step's parser.
"""

from __future__ import annotations

import subprocess
import sys

import pytest


# ---------------------------------------------------------------------------
# build_parser import tests (no subprocess needed)
# ---------------------------------------------------------------------------

def test_step2_parser_builds():
    from cage.step2_build_cohort import build_parser
    p = build_parser()
    assert p is not None
    # Verify leakage-guard flags exist
    actions = {a.dest for a in p._actions}
    assert "no_zscore" in actions, "--no-zscore / --global-zscore dest must be 'no_zscore'"

def test_step2_no_zscore_default_true():
    """The leakage-guard default: global z-scoring must be DISABLED by default."""
    from cage.step2_build_cohort import build_parser
    p = build_parser()
    defaults = vars(p.parse_args([
        "--input-dir", ".", "--output-dir", "/tmp/test_step2"
    ]))
    assert defaults["no_zscore"] is True, (
        "--no-zscore must default to True (global z-score disabled). "
        "This guards against test-fold leakage."
    )

def test_step3_parser_builds():
    from cage.step3_grouped_baselines import build_parser
    p = build_parser()
    assert p is not None

def test_step4_parser_builds():
    from cage.step4_sparse_invariant_model import build_parser
    p = build_parser()
    assert p is not None

def test_step5_parser_builds():
    from cage.step5_cdps_ranking import build_parser
    p = build_parser()
    assert p is not None


# ---------------------------------------------------------------------------
# --help exits 0 (subprocess, catches import-time errors too)
# ---------------------------------------------------------------------------

STEP_MODULES = [
    "cage.step2_build_cohort",
    "cage.step3_grouped_baselines",
    "cage.step4_sparse_invariant_model",
    "cage.step5_cdps_ranking",
]

@pytest.mark.parametrize("module", STEP_MODULES)
def test_help_exits_zero(module):
    result = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"{module} --help exited with code {result.returncode}.\n"
        f"stderr: {result.stderr[:500]}"
    )
    assert "usage" in result.stdout.lower() or "Usage" in result.stdout, (
        f"{module} --help output did not contain 'usage'"
    )


# ---------------------------------------------------------------------------
# Module-level import sanity
# ---------------------------------------------------------------------------

def test_metrics_importable():
    from cage import metrics
    assert hasattr(metrics, "auroc")
    assert hasattr(metrics, "bootstrap_ci")

def test_baseline_models_importable():
    from cage import baseline_models
    assert hasattr(baseline_models, "MODEL_REGISTRY")
    assert len(baseline_models.MODEL_REGISTRY) >= 4

def test_deep_model_utils_importable():
    from cage import deep_model_utils
    assert hasattr(deep_model_utils, "standardize_fit")
    assert hasattr(deep_model_utils, "standardize_apply")
    assert hasattr(deep_model_utils, "AdamW")
    assert hasattr(deep_model_utils, "FeatureGate")

def test_preprocess_esca_importable():
    from cage import preprocess_esca
    assert hasattr(preprocess_esca, "build_patient_grouped_folds")
    assert hasattr(preprocess_esca, "zscore_normalize")

def test_step5_runner_importable():
    from cage import step5_runner
    assert hasattr(step5_runner, "normalize_scores")
    assert hasattr(step5_runner, "build_cdps_ranking")
