"""Tests for patient-grouped fold assignment (preprocess_esca)."""

from __future__ import annotations

import numpy as np
import pytest
from collections import Counter

from cage.preprocess_esca import build_patient_grouped_folds


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simple_records(n_tumor=40, n_normal=10, n_patients=20, seed=0):
    import random
    rng = random.Random(seed)
    records = []
    pats = [f"PAT-{i:03d}" for i in range(n_patients)]
    normal_pats = pats[:5]

    for i in range(n_tumor):
        pat = pats[i % n_patients]
        records.append({
            "barcode": f"{pat}-T-{i:04d}",
            "sample_barcode": f"{pat}-T-{i:04d}",
            "patient_barcode": pat,
            "label": "Tumor",
        })
    for i in range(n_normal):
        pat = normal_pats[i % len(normal_pats)]
        records.append({
            "barcode": f"{pat}-N-{i:04d}",
            "sample_barcode": f"{pat}-N-{i:04d}",
            "patient_barcode": pat,
            "label": "Normal",
        })
    return records


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBuildPatientGroupedFolds:

    def test_returns_list_of_dicts(self):
        recs = _simple_records()
        folds = build_patient_grouped_folds(recs, n_outer=5, n_inner=3, seed=0)
        assert isinstance(folds, list)
        assert all(isinstance(r, dict) for r in folds)

    def test_all_samples_assigned(self):
        recs = _simple_records(n_tumor=40, n_normal=10)
        folds = build_patient_grouped_folds(recs, n_outer=5, n_inner=3, seed=0)
        barcodes_in = {r["barcode"] for r in recs}
        barcodes_out = {r["sample_barcode"] for r in folds}
        assert barcodes_in == barcodes_out, "Every input sample must appear in fold output"

    def test_outer_fold_values_in_range(self):
        n_outer = 5
        recs = _simple_records()
        folds = build_patient_grouped_folds(recs, n_outer=n_outer, n_inner=3, seed=0)
        fold_values = {int(r["outer_fold"]) for r in folds}
        assert fold_values.issubset(set(range(n_outer)))

    def test_no_patient_spans_multiple_folds(self):
        recs = _simple_records(n_tumor=50, n_normal=10, n_patients=25)
        folds = build_patient_grouped_folds(recs, n_outer=5, n_inner=3, seed=0)
        fold_map = {r["sample_barcode"]: int(r["outer_fold"]) for r in folds}

        patient_folds: dict[str, set] = {}
        for rec in recs:
            bc = rec["barcode"]
            pat = rec["patient_barcode"]
            if bc in fold_map:
                patient_folds.setdefault(pat, set()).add(fold_map[bc])

        for pat, folds_set in patient_folds.items():
            assert len(folds_set) == 1, (
                f"Patient {pat} appears in multiple folds: {folds_set} — patient leakage!"
            )

    def test_all_folds_have_samples(self):
        recs = _simple_records(n_tumor=50, n_normal=10, n_patients=25)
        folds = build_patient_grouped_folds(recs, n_outer=5, n_inner=3, seed=0)
        fold_counts = Counter(int(r["outer_fold"]) for r in folds)
        assert len(fold_counts) == 5, "All 5 outer folds must be populated"
        for k, cnt in fold_counts.items():
            assert cnt > 0, f"Fold {k} is empty"

    def test_deterministic_with_same_seed(self):
        recs = _simple_records()
        folds_a = build_patient_grouped_folds(recs, n_outer=5, n_inner=3, seed=99)
        folds_b = build_patient_grouped_folds(recs, n_outer=5, n_inner=3, seed=99)
        assignments_a = {r["sample_barcode"]: r["outer_fold"] for r in folds_a}
        assignments_b = {r["sample_barcode"]: r["outer_fold"] for r in folds_b}
        assert assignments_a == assignments_b

    def test_different_seed_gives_different_folds(self):
        recs = _simple_records(n_patients=20)
        folds_a = build_patient_grouped_folds(recs, n_outer=5, n_inner=3, seed=1)
        folds_b = build_patient_grouped_folds(recs, n_outer=5, n_inner=3, seed=2)
        a = {r["sample_barcode"]: r["outer_fold"] for r in folds_a}
        b = {r["sample_barcode"]: r["outer_fold"] for r in folds_b}
        # At least some assignments must differ with different seeds
        assert a != b, "Different seeds should (with high probability) give different fold assignments"

    def test_inner_fold_columns_present(self):
        recs = _simple_records()
        folds = build_patient_grouped_folds(recs, n_outer=5, n_inner=3, seed=0)
        first = folds[0]
        # Inner fold columns are named inner_fold_outer0 .. inner_fold_outer4
        for outer in range(5):
            col = f"inner_fold_outer{outer}"
            assert col in first, f"Expected column {col} missing from fold record"

    def test_normal_patients_spread_across_folds(self):
        """Patients with normals should appear in different folds, not all in one."""
        recs = _simple_records(n_tumor=40, n_normal=10, n_patients=20)
        folds = build_patient_grouped_folds(recs, n_outer=5, n_inner=3, seed=0)
        fold_map = {r["sample_barcode"]: int(r["outer_fold"]) for r in folds}

        normal_patient_folds = set()
        for rec in recs:
            if rec["label"] == "Normal":
                bc = rec["barcode"]
                if bc in fold_map:
                    normal_patient_folds.add(fold_map[bc])

        assert len(normal_patient_folds) > 1, (
            "Normal samples should be spread across multiple folds so every "
            "train split has normals."
        )
