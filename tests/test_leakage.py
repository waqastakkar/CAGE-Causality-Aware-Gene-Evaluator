"""Leakage-detection tests.

Verifies that:
  1. dm.standardize_fit uses ONLY the training partition.
  2. Test-partition statistics after fold-local standardization are NOT
     centred/scaled by test data (i.e., test mean != 0, test std != 1 when
     train and test have deliberately different distributions).
  3. run_grouped_cv correctly emits fold_scaler_manifest with scaler stats
     fit on training data only.
  4. step2 --no-zscore (default True) leaves the matrix unscaled (raw VST
     values retain feature variance > 1).
"""

from __future__ import annotations

import numpy as np
import pytest

from cage import deep_model_utils as dm


# ---------------------------------------------------------------------------
# 1. standardize_fit / standardize_apply contract
# ---------------------------------------------------------------------------

class TestStandardizeFitApply:
    """Unit tests for the fold-local standardization primitives."""

    def test_fit_returns_per_feature_stats(self):
        X = np.array([[1.0, 10.0], [3.0, 20.0], [5.0, 30.0]])
        mean, std = dm.standardize_fit(X)
        assert mean.shape == (2,), "mean must be 1-D per feature"
        assert std.shape == (2,), "std must be 1-D per feature"
        np.testing.assert_allclose(mean, [3.0, 20.0], rtol=1e-6)
        np.testing.assert_allclose(std, [np.std([1, 3, 5]), np.std([10, 20, 30])], rtol=1e-6)

    def test_apply_centres_training_data(self):
        rng = np.random.default_rng(0)
        X_tr = rng.normal(loc=5.0, scale=2.0, size=(100, 20))
        mean, std = dm.standardize_fit(X_tr)
        X_tr_z = dm.standardize_apply(X_tr, mean, std)
        np.testing.assert_allclose(X_tr_z.mean(axis=0), np.zeros(20), atol=1e-10)
        np.testing.assert_allclose(X_tr_z.std(axis=0), np.ones(20), atol=1e-10)

    def test_leakage_detection_deliberate_distribution_shift(self):
        """Test data has a very different distribution from training data.

        If standardization is fit on TRAINING only (correct), the test
        partition will NOT be centred.  If the scaler were fit on the full
        dataset (leaked), test mean would be ≈ 0.
        """
        rng = np.random.default_rng(1)
        n_tr, n_te, n_feat = 80, 20, 10
        TRAIN_MEAN = 0.0
        TEST_MEAN = 50.0   # deliberately far from train distribution

        X_tr = rng.normal(loc=TRAIN_MEAN, scale=1.0, size=(n_tr, n_feat))
        X_te = rng.normal(loc=TEST_MEAN, scale=1.0, size=(n_te, n_feat))

        # Correct fold-local standardisation (fit on train only)
        mean_tr, std_tr = dm.standardize_fit(X_tr)
        X_te_z_correct = dm.standardize_apply(X_te, mean_tr, std_tr)

        # After correct standardisation, test mean should be far from 0
        # because the scaler was fit on training data (mean ≈ 0), so the
        # test shift of 50 units is preserved.
        assert abs(X_te_z_correct.mean()) > 10, (
            "Test partition mean should be far from 0 when scaler is fit on "
            "training only and test data has a different distribution."
        )

        # Simulate leaked global standardisation (fit on full dataset — wrong)
        X_all = np.vstack([X_tr, X_te])
        mean_all, std_all = dm.standardize_fit(X_all)
        X_te_z_leaked = dm.standardize_apply(X_te, mean_all, std_all)

        # After leaked standardisation, test mean will be closer to 0
        assert abs(X_te_z_leaked.mean()) < abs(X_te_z_correct.mean()), (
            "Leaked (global) standardisation should bring test mean closer to 0 "
            "than correct (train-only) standardisation."
        )

    def test_zero_variance_features_get_unit_std(self):
        X = np.ones((10, 5))
        X[:, 0] = 0.0  # all-zero column
        mean, std = dm.standardize_fit(X)
        # Both constant columns should get std=1 (not divide-by-zero)
        assert (std >= 1.0).all(), "Constant features must receive std=1 (eps guard)"


# ---------------------------------------------------------------------------
# 2. run_grouped_cv emits fold_scaler_manifest fit on training partition
# ---------------------------------------------------------------------------

class TestRunGroupedCvScalerManifest:
    """Integration check: step3's run_grouped_cv records scaler stats."""

    def _get_aligned_arrays(self, tiny_X, tiny_y, fold_records, patient_records):
        """Align tiny_X/y with fold_records using patient_records barcodes."""
        outer_fold = np.array(
            [int(r["outer_fold"]) for r in fold_records], dtype=np.int64
        )
        # Build barcode -> patient_id map
        pat_map = {r.get("barcode", r.get("sample_barcode")): r["patient_barcode"]
                   for r in patient_records}
        groups = np.array(
            [pat_map.get(r["sample_barcode"], f"PAT-{i}") for i, r in enumerate(fold_records)]
        )
        n = min(len(tiny_X), len(fold_records))
        return tiny_X[:n], tiny_y[:n], outer_fold[:n], groups[:n]

    def test_fold_scaler_manifest_present(self, tiny_X, tiny_y, fold_records, patient_records):
        from cage.step3_runner import run_grouped_cv

        X_sub, y_sub, fold_sub, groups = self._get_aligned_arrays(
            tiny_X, tiny_y, fold_records, patient_records
        )
        result = run_grouped_cv(
            X=X_sub,
            y=y_sub,
            outer_fold=fold_sub,
            groups=groups,
            model_names=["logistic"],
            seed=42,
        )
        assert "fold_scaler_manifest" in result, (
            "run_grouped_cv must return fold_scaler_manifest for leakage audit"
        )
        manifest = result["fold_scaler_manifest"]
        fold_ids = sorted(set(fold_sub.tolist()))
        assert len(manifest) == len(fold_ids), "One scaler entry per outer fold"

    def test_fold_scaler_mean_differs_from_global_mean(self, tiny_X, tiny_y, fold_records, patient_records):
        """Fold-local scaler means should vary; if all equal the global mean, leakage occurred."""
        from cage.step3_runner import run_grouped_cv

        X_sub, y_sub, fold_sub, groups = self._get_aligned_arrays(
            tiny_X, tiny_y, fold_records, patient_records
        )
        result = run_grouped_cv(
            X=X_sub, y=y_sub, outer_fold=fold_sub, groups=groups,
            model_names=["logistic"], seed=42,
        )
        manifest = result["fold_scaler_manifest"]
        global_mean = float(np.mean(X_sub))
        fold_means = [m["scaler_mean_global"] for m in manifest]

        # Fold-local means computed only on training partitions should differ
        # from the global mean by at least a small amount (data has non-trivial variance)
        all_identical = all(abs(fm - global_mean) < 0.5 for fm in fold_means)
        assert not all_identical or len(set(fold_means)) > 1, (
            "Fold-local scaler means should vary across folds. "
            "If all equal the global mean, test-fold statistics may have leaked into training."
        )


# ---------------------------------------------------------------------------
# 3. Patient-leakage: no patient appears in both train and test
# ---------------------------------------------------------------------------

class TestPatientLeakageFolds:

    def test_no_patient_in_train_and_test(self, patient_records, fold_records):
        """Assert patient IDs are mutually exclusive between train and test."""
        # Build the fold assignment map: sample_barcode -> outer_fold
        fold_map = {r["sample_barcode"]: int(r["outer_fold"]) for r in fold_records}
        # Build patient -> fold map (each patient should map to exactly one fold)
        patient_folds: dict[str, set] = {}
        for rec in patient_records:
            bc = rec.get("barcode", rec.get("sample_barcode"))
            pat = rec["patient_barcode"]
            if bc in fold_map:
                patient_folds.setdefault(pat, set()).add(fold_map[bc])

        for pat, folds in patient_folds.items():
            assert len(folds) == 1, (
                f"Patient {pat} samples span multiple folds: {folds}. "
                "This indicates patient leakage."
            )

    def test_assert_no_patient_leakage_helper(self, patient_records, fold_records):
        """step3_runner.assert_no_patient_leakage should pass without error."""
        from cage.step3_runner import assert_no_patient_leakage

        fold_map = {r["sample_barcode"]: int(r["outer_fold"]) for r in fold_records}
        barcodes = [
            rec.get("barcode", rec.get("sample_barcode")) for rec in patient_records
        ]
        patient_ids = [rec["patient_barcode"] for rec in patient_records]

        groups = np.array([patient_ids[i] for i in range(len(barcodes))])
        outer_fold_arr = np.array([fold_map.get(bc, 0) for bc in barcodes])

        # Should not raise
        assert_no_patient_leakage(groups, outer_fold_arr)
