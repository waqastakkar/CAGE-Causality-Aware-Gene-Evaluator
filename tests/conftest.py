"""Shared fixtures for CAGE test suite.

All fixtures use synthetic data only — no real TCGA files required.
"""

from __future__ import annotations

import random
from typing import List, Dict

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Numeric fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture(scope="session")
def tiny_X(rng) -> np.ndarray:
    """60 samples × 50 features; tumor block has higher mean than normal."""
    X_tumor = rng.normal(loc=3.0, scale=1.0, size=(45, 50))
    X_normal = rng.normal(loc=0.0, scale=1.0, size=(15, 50))
    return np.vstack([X_tumor, X_normal]).astype(np.float64)


@pytest.fixture(scope="session")
def tiny_y() -> np.ndarray:
    return np.array([1] * 45 + [0] * 15, dtype=np.int64)


@pytest.fixture(scope="session")
def tiny_labels() -> list[str]:
    return ["Tumor"] * 45 + ["Normal"] * 15


# ---------------------------------------------------------------------------
# Metadata / fold fixtures
# ---------------------------------------------------------------------------

def _make_patient_records(
    n_tumor: int = 45,
    n_normal: int = 15,
    n_patients: int = 40,
    seed: int = 42,
) -> List[Dict[str, str]]:
    """Synthetic metadata records that look like step2 output."""
    rng_py = random.Random(seed)
    records = []

    patient_ids = [f"TCGA-ZZ-{i:04d}" for i in range(n_patients)]
    # First 10 patients have matched normals
    normal_patients = patient_ids[:10]
    tumor_patients = patient_ids[10:]

    barcode_counter = 0

    # Tumor samples — spread across all patients
    tumor_patients_all = patient_ids[:]
    for i in range(n_tumor):
        pat = tumor_patients_all[i % len(tumor_patients_all)]
        barcode = f"{pat}-01A-{barcode_counter:04d}"
        barcode_counter += 1
        records.append({
            "barcode": barcode,
            "sample_barcode": barcode,
            "patient_barcode": pat,
            "label": "Tumor",
            "sample_type": "Primary Tumor",
            "gender": rng_py.choice(["Male", "Female"]),
            "tobacco_smoking_status": rng_py.choice(["Current reformed smoker for < or = 15 years", "Lifelong Non-smoker"]),
            "paper_Histological.Type": rng_py.choice(["Esophagus Squamous Cell Carcinoma", "Esophagus Adenocarcinoma"]),
            "paper_Country": rng_py.choice(["China", "United States"]),
            "paper_Pathologic.stage": rng_py.choice(["Stage I", "Stage II", "Stage III"]),
        })

    # Normal samples — from the first 10 patients only
    for i in range(n_normal):
        pat = normal_patients[i % len(normal_patients)]
        barcode = f"{pat}-11A-{barcode_counter:04d}"
        barcode_counter += 1
        records.append({
            "barcode": barcode,
            "sample_barcode": barcode,
            "patient_barcode": pat,
            "label": "Normal",
            "sample_type": "Solid Tissue Normal",
            "gender": rng_py.choice(["Male", "Female"]),
            "tobacco_smoking_status": rng_py.choice(["Current reformed smoker for < or = 15 years", "Lifelong Non-smoker"]),
            "paper_Histological.Type": rng_py.choice(["Esophagus Squamous Cell Carcinoma", "Esophagus Adenocarcinoma"]),
            "paper_Country": rng_py.choice(["China", "United States"]),
            "paper_Pathologic.stage": rng_py.choice(["Stage I", "Stage II", "Stage III"]),
        })

    return records


@pytest.fixture(scope="session")
def patient_records() -> List[Dict[str, str]]:
    return _make_patient_records()


@pytest.fixture(scope="session")
def fold_records(patient_records) -> List[Dict[str, str]]:
    from cage.preprocess_esca import build_patient_grouped_folds
    return build_patient_grouped_folds(patient_records, n_outer=5, n_inner=3, seed=42)
