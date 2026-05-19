"""Tests for cage.metrics — all numpy-only metric functions."""

from __future__ import annotations

import numpy as np
import pytest

from cage import metrics as mx


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _perfect_classifier(n_pos=30, n_neg=10):
    y = np.array([1] * n_pos + [0] * n_neg)
    s = np.array([0.9] * n_pos + [0.1] * n_neg, dtype=np.float64)
    return y, s


def _random_classifier(n=100, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    s = rng.uniform(size=n)
    return y.astype(np.int64), s


# ---------------------------------------------------------------------------
# auroc
# ---------------------------------------------------------------------------

class TestAuroc:

    def test_perfect_auroc(self):
        y, s = _perfect_classifier()
        assert mx.auroc(y, s) == pytest.approx(1.0, abs=1e-6)

    def test_random_auroc_near_half(self):
        aurocs = []
        for seed in range(20):
            y, s = _random_classifier(seed=seed)
            if len(np.unique(y)) > 1:
                aurocs.append(mx.auroc(y, s))
        assert abs(np.mean(aurocs) - 0.5) < 0.1

    def test_worst_classifier_auroc(self):
        y, s = _perfect_classifier()
        assert mx.auroc(y, 1 - s) == pytest.approx(0.0, abs=1e-6)

    def test_single_class_returns_nan(self):
        y = np.ones(10, dtype=np.int64)
        s = np.random.rand(10)
        result = mx.auroc(y, s)
        assert np.isnan(result), "Single-class AUROC should be NaN"


# ---------------------------------------------------------------------------
# auprc
# ---------------------------------------------------------------------------

class TestAuprc:

    def test_perfect_auprc(self):
        y, s = _perfect_classifier()
        assert mx.auprc(y, s) == pytest.approx(1.0, abs=1e-6)

    def test_auprc_positive(self):
        y, s = _random_classifier()
        if len(np.unique(y)) > 1:
            result = mx.auprc(y, s)
            assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# brier_score
# ---------------------------------------------------------------------------

class TestBrier:

    def test_perfect_brier_zero(self):
        y = np.array([1, 1, 0, 0])
        s = np.array([1.0, 1.0, 0.0, 0.0])
        assert mx.brier_score(y, s) == pytest.approx(0.0, abs=1e-9)

    def test_worst_brier_one(self):
        y = np.array([1, 1, 0, 0])
        s = np.array([0.0, 0.0, 1.0, 1.0])
        assert mx.brier_score(y, s) == pytest.approx(1.0, abs=1e-9)

    def test_brier_in_range(self):
        y, s = _random_classifier()
        b = mx.brier_score(y.astype(float), s)
        assert 0.0 <= b <= 1.0


# ---------------------------------------------------------------------------
# balanced_accuracy
# ---------------------------------------------------------------------------

class TestBalancedAccuracy:

    def test_perfect(self):
        y = np.array([1, 1, 0, 0])
        p = np.array([1, 1, 0, 0])
        assert mx.balanced_accuracy(y, p) == pytest.approx(1.0)

    def test_all_wrong(self):
        y = np.array([1, 1, 0, 0])
        p = np.array([0, 0, 1, 1])
        assert mx.balanced_accuracy(y, p) == pytest.approx(0.0)

    def test_random_chance_near_half(self):
        rng = np.random.default_rng(7)
        y = rng.integers(0, 2, size=200)
        p = rng.integers(0, 2, size=200)
        bac = mx.balanced_accuracy(y, p)
        assert 0.3 < bac < 0.7


# ---------------------------------------------------------------------------
# confusion_counts / sensitivity_specificity
# ---------------------------------------------------------------------------

class TestConfusion:

    def test_known_confusion(self):
        y    = np.array([1, 1, 0, 0, 1])
        pred = np.array([1, 0, 0, 1, 1])
        tp, fp, tn, fn = mx.confusion_counts(y, pred)
        assert tp == 2 and fp == 1 and tn == 1 and fn == 1

    def test_sensitivity_specificity(self):
        y    = np.array([1, 1, 0, 0, 1])
        pred = np.array([1, 0, 0, 1, 1])
        sens, spec = mx.sensitivity_specificity(y, pred)
        assert sens == pytest.approx(2 / 3, rel=1e-5)
        assert spec == pytest.approx(1 / 2, rel=1e-5)


# ---------------------------------------------------------------------------
# f1_score
# ---------------------------------------------------------------------------

class TestF1:

    def test_perfect_f1(self):
        y = np.array([1, 1, 0, 0])
        assert mx.f1_score(y, y) == pytest.approx(1.0)

    def test_zero_f1(self):
        y    = np.array([1, 1, 0, 0])
        pred = np.array([0, 0, 0, 0])
        assert mx.f1_score(y, pred) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# log_loss
# ---------------------------------------------------------------------------

class TestLogLoss:

    def test_perfect_log_loss_near_zero(self):
        y = np.array([1, 1, 0, 0])
        s = np.array([0.9999, 0.9999, 0.0001, 0.0001])
        assert mx.log_loss(y, s) < 0.01

    def test_log_loss_non_negative(self):
        y, s = _random_classifier()
        assert mx.log_loss(y, s) >= 0.0


# ---------------------------------------------------------------------------
# calibration_curve
# ---------------------------------------------------------------------------

class TestCalibrationCurve:

    def test_shape_and_range(self):
        y, s = _perfect_classifier()
        bins = mx.calibration_curve(y, s, n_bins=5)
        assert isinstance(bins, list)
        for row in bins:
            assert "lower" in row and "upper" in row
            assert "mean_predicted_prob" in row and "fraction_positives" in row

    def test_uniform_strategy(self):
        y, s = _random_classifier()
        bins = mx.calibration_curve(y, s, n_bins=10, strategy="uniform")
        assert len(bins) <= 10


# ---------------------------------------------------------------------------
# bootstrap_ci
# ---------------------------------------------------------------------------

class TestBootstrapCI:

    def test_ci_contains_true_value(self):
        """Bootstrap CI on a near-perfect classifier should bound AUROC near 1."""
        y, s = _perfect_classifier(n_pos=40, n_neg=10)
        result = mx.bootstrap_ci(mx.auroc, y, s, n_boot=200, seed=0, level=0.95)
        lo, hi = result["lower"], result["upper"]
        assert lo >= 0.9, f"95% CI lower bound {lo:.3f} should be >= 0.9 for a perfect classifier"
        assert lo <= hi, "CI lower must be <= upper"

    def test_ci_returns_dict_with_required_keys(self):
        y, s = _perfect_classifier()
        result = mx.bootstrap_ci(mx.auroc, y, s, n_boot=100, seed=0)
        for key in ("estimate", "lower", "upper", "level", "n_boot"):
            assert key in result, f"bootstrap_ci result missing key: {key}"

    def test_ci_width_decreases_with_samples(self):
        """Wider CI with fewer samples."""
        rng = np.random.default_rng(3)
        y_small = np.array([1] * 10 + [0] * 5)
        s_small = rng.uniform(size=15)
        y_large = np.array([1] * 60 + [0] * 40)
        s_large = rng.uniform(size=100)

        if len(np.unique(y_small)) == 1 or len(np.unique(y_large)) == 1:
            pytest.skip("Degenerate labels in synthetic data")

        r_s = mx.bootstrap_ci(mx.auroc, y_small, s_small, n_boot=200, seed=0)
        r_l = mx.bootstrap_ci(mx.auroc, y_large, s_large, n_boot=200, seed=0)
        width_small = r_s["upper"] - r_s["lower"]
        width_large = r_l["upper"] - r_l["lower"]
        assert width_small >= width_large * 0.5, \
            "Smaller sample should tend to have wider CI"
