"""Tests for cage.baseline_models — the numpy-only classifiers."""

from __future__ import annotations

import numpy as np
import pytest

from cage.baseline_models import (
    LogisticRidge,
    ElasticNetLogistic,
    DecisionTreeBinary,
    RandomForestBinary,
    build_model,
    MODEL_REGISTRY,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def binary_data():
    rng = np.random.default_rng(42)
    n, p = 80, 30
    X = np.vstack([
        rng.normal(loc=1.0, scale=1.0, size=(60, p)),  # class 1
        rng.normal(loc=-1.0, scale=1.0, size=(20, p)),  # class 0
    ]).astype(np.float64)
    y = np.array([1] * 60 + [0] * 20, dtype=np.int64)
    return X, y


# ---------------------------------------------------------------------------
# Interface contract (all models)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_name", list(MODEL_REGISTRY.keys()))
class TestModelInterface:

    def test_fit_returns_self(self, model_name, binary_data):
        X, y = binary_data
        model = build_model(model_name, seed=0)
        result = model.fit(X, y)
        assert result is model

    def test_predict_proba_shape(self, model_name, binary_data):
        X, y = binary_data
        model = build_model(model_name, seed=0)
        model.fit(X, y)
        probs = model.predict_proba(X)
        assert probs.shape == (X.shape[0],), "predict_proba must return 1-D array"

    def test_predict_proba_in_zero_one(self, model_name, binary_data):
        X, y = binary_data
        model = build_model(model_name, seed=0)
        model.fit(X, y)
        probs = model.predict_proba(X)
        assert np.all((probs >= 0.0) & (probs <= 1.0)), "Probabilities must be in [0, 1]"

    def test_predict_binary_output(self, model_name, binary_data):
        X, y = binary_data
        model = build_model(model_name, seed=0)
        model.fit(X, y)
        preds = model.predict(X)
        assert set(np.unique(preds)).issubset({0, 1}), "Predictions must be 0 or 1"

    def test_feature_importances_shape(self, model_name, binary_data):
        X, y = binary_data
        model = build_model(model_name, seed=0)
        model.fit(X, y)
        imp = model.feature_importances()
        assert imp.shape == (X.shape[1],), "feature_importances must have one value per feature"

    def test_feature_importances_non_negative(self, model_name, binary_data):
        X, y = binary_data
        model = build_model(model_name, seed=0)
        model.fit(X, y)
        imp = model.feature_importances()
        assert np.all(imp >= 0.0), "Feature importances must be non-negative"

    def test_deterministic_with_seed(self, model_name, binary_data):
        X, y = binary_data
        m1 = build_model(model_name, seed=7).fit(X, y)
        m2 = build_model(model_name, seed=7).fit(X, y)
        np.testing.assert_array_equal(
            m1.predict_proba(X), m2.predict_proba(X),
            err_msg=f"{model_name}: same seed must give identical predictions"
        )

    def test_get_config_returns_dict(self, model_name, binary_data):
        X, y = binary_data
        model = build_model(model_name, seed=0).fit(X, y)
        cfg = model.get_config()
        assert isinstance(cfg, dict)
        assert "name" in cfg


# ---------------------------------------------------------------------------
# Correctness: separable data should yield AUROC >> 0.5
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_name", list(MODEL_REGISTRY.keys()))
def test_model_learns_separable_data(model_name):
    from cage.metrics import auroc
    rng = np.random.default_rng(0)
    n, p = 100, 10
    X = np.vstack([
        rng.normal(loc=3.0, scale=0.5, size=(70, p)),
        rng.normal(loc=-3.0, scale=0.5, size=(30, p)),
    ]).astype(np.float64)
    y = np.array([1] * 70 + [0] * 30, dtype=np.int64)

    model = build_model(model_name, seed=42).fit(X, y)
    probs = model.predict_proba(X)
    auc = auroc(y, probs)
    assert auc > 0.7, f"{model_name} AUROC={auc:.3f} on separable data — model may not be fitting"


# ---------------------------------------------------------------------------
# Class weight: balanced weighting should not crash
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_name", ["logistic", "elasticnet"])
def test_balanced_class_weight(model_name, binary_data):
    X, y = binary_data
    model = build_model(model_name, seed=0, class_weight="balanced")
    model.fit(X, y)
    probs = model.predict_proba(X)
    assert probs.shape == (X.shape[0],)


# ---------------------------------------------------------------------------
# build_model: unknown name raises
# ---------------------------------------------------------------------------

def test_build_model_unknown_name():
    with pytest.raises((KeyError, ValueError)):
        build_model("nonexistent_model_xyz", seed=0)
