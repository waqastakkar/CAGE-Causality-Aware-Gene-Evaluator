"""Tests for cage.deep_model_utils — primitives and modules."""

from __future__ import annotations

import numpy as np
import pytest

from cage import deep_model_utils as dm


# ---------------------------------------------------------------------------
# Activations
# ---------------------------------------------------------------------------

class TestSigmoid:

    def test_sigmoid_zero(self):
        assert dm.sigmoid(np.array([0.0]))[0] == pytest.approx(0.5, abs=1e-9)

    def test_sigmoid_large_positive(self):
        assert dm.sigmoid(np.array([100.0]))[0] == pytest.approx(1.0, abs=1e-6)

    def test_sigmoid_large_negative(self):
        assert dm.sigmoid(np.array([-100.0]))[0] == pytest.approx(0.0, abs=1e-6)

    def test_sigmoid_range(self):
        x = np.linspace(-10, 10, 100)
        s = dm.sigmoid(x)
        assert np.all((s >= 0) & (s <= 1))


class TestSoftmax:

    def test_softmax_sums_to_one(self):
        x = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
        s = dm.softmax(x)
        np.testing.assert_allclose(s.sum(axis=1), [1.0, 1.0], atol=1e-9)

    def test_softmax_max_gets_highest_prob(self):
        x = np.array([[0.0, 0.0, 10.0]])
        s = dm.softmax(x)
        assert np.argmax(s[0]) == 2


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

class TestBCE:

    def test_perfect_prediction_near_zero(self):
        logits = np.array([10.0, 10.0, -10.0, -10.0])
        y = np.array([1.0, 1.0, 0.0, 0.0])
        loss = dm.bce_with_logits(logits, y)
        assert float(loss) < 0.01

    def test_worst_prediction_high_loss(self):
        logits = np.array([-10.0, -10.0, 10.0, 10.0])
        y = np.array([1.0, 1.0, 0.0, 0.0])
        loss = dm.bce_with_logits(logits, y)
        assert float(loss) > 5.0

    def test_bce_non_negative(self):
        rng = np.random.default_rng(0)
        logits = rng.normal(size=50)
        y = rng.integers(0, 2, size=50).astype(float)
        assert dm.bce_with_logits(logits, y) >= 0.0


class TestMSELoss:

    def test_zero_loss_on_perfect(self):
        pred = np.array([1.0, 2.0, 3.0])
        assert dm.mse_loss(pred, pred) == pytest.approx(0.0, abs=1e-9)

    def test_known_mse(self):
        pred = np.array([0.0, 0.0, 0.0])
        target = np.array([1.0, 1.0, 1.0])
        assert dm.mse_loss(pred, target) == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Linear layer
# ---------------------------------------------------------------------------

class TestLinear:

    def test_forward_shape(self):
        rng = np.random.default_rng(0)
        layer = dm.Linear(10, 4, rng=rng)
        X = rng.normal(size=(5, 10))
        out = layer.forward(X)
        assert out.shape == (5, 4)

    def test_forward_backward_grad_shape(self):
        rng = np.random.default_rng(1)
        layer = dm.Linear(10, 4, rng=rng)
        X = rng.normal(size=(8, 10))
        out = layer.forward(X)
        grad_out = rng.normal(size=out.shape)
        grad_in = layer.backward(grad_out)
        assert grad_in.shape == X.shape
        assert "W" in layer.grads
        assert "b" in layer.grads

    def test_params_keys(self):
        rng = np.random.default_rng(0)
        layer = dm.Linear(5, 3, rng=rng)
        assert set(layer.params.keys()) == {"W", "b"}


# ---------------------------------------------------------------------------
# AdamW optimizer
# ---------------------------------------------------------------------------

class TestAdamW:

    def test_loss_decreases_on_linear_regression(self):
        rng = np.random.default_rng(0)
        n, p = 50, 10
        W_true = rng.normal(size=(p, 1))
        X = rng.normal(size=(n, p))
        y = X @ W_true + rng.normal(scale=0.01, size=(n, 1))

        layer = dm.Linear(p, 1, rng=rng)
        opt = dm.AdamW(layer.params, lr=0.01, weight_decay=0.0)

        losses = []
        for _ in range(200):
            # Zero grads before backward
            for k in layer.grads:
                layer.grads[k][:] = 0.0
            pred = layer.forward(X)
            loss = dm.mse_loss(pred, y)
            losses.append(float(loss))
            grad_out = dm.mse_loss_grad(pred, y)
            layer.backward(grad_out)
            opt.step(layer.params, layer.grads)

        assert losses[-1] < losses[0], "AdamW should reduce MSE loss"
        assert losses[-1] < losses[0] * 0.5, f"Final loss {losses[-1]:.4f} should be well below initial {losses[0]:.4f}"


# ---------------------------------------------------------------------------
# FeatureGate
# ---------------------------------------------------------------------------

class TestFeatureGate:

    @pytest.mark.parametrize("sparsity_type", ["l1", "hard-concrete"])
    def test_gate_output_shape(self, sparsity_type):
        rng = np.random.default_rng(0)
        gate = dm.FeatureGate(n_features=20, sparsity_type=sparsity_type, rng=rng)
        X = rng.normal(size=(10, 20))
        out = gate.forward(X, training=True)
        assert out.shape == X.shape

    @pytest.mark.parametrize("sparsity_type", ["l1", "hard-concrete"])
    def test_gate_output_in_range(self, sparsity_type):
        rng = np.random.default_rng(0)
        gate = dm.FeatureGate(n_features=20, sparsity_type=sparsity_type, rng=rng)
        X = np.ones((5, 20))
        out = gate.forward(X, training=False)
        assert not np.any(np.isnan(out)), "Gate output must not contain NaN"

    def test_sparsity_penalty_non_negative(self):
        gate = dm.FeatureGate(n_features=30, sparsity_type="l1")
        penalty = gate.sparsity_penalty()
        assert float(penalty) >= 0.0


# ---------------------------------------------------------------------------
# standardize_fit / standardize_apply (also tested in test_leakage.py)
# ---------------------------------------------------------------------------

class TestStandardize:

    def test_apply_is_inverse_of_fit(self):
        rng = np.random.default_rng(5)
        X = rng.normal(loc=3.0, scale=2.0, size=(40, 8))
        mean, std = dm.standardize_fit(X)
        Z = dm.standardize_apply(X, mean, std)
        X_back = Z * std[None, :] + mean[None, :]
        np.testing.assert_allclose(X_back, X, atol=1e-9)
