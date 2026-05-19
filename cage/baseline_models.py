"""CAGE pure-numpy baseline classifiers.

Implements the reference binary classifiers that Step 3 evaluates against
the patient-grouped CV splits produced in Step 2, without requiring
scikit-learn. Every model exposes the minimal ``fit(X, y) /
predict_proba(X) / predict(X)`` interface and supports deterministic
seeding and ``class_weight="balanced"`` (the ESCA cohort is ~92% tumor).

Models
------
LogisticRidge          L2-penalized logistic regression, Adam full-batch GD
ElasticNetLogistic     L1+L2 logistic regression via FISTA proximal gradient
DecisionTreeBinary     Gini-split binary tree with depth / sample guards
RandomForestBinary     Bagging of DecisionTreeBinary with sqrt feature subsample

Each model is lightweight, small-sample-friendly (155 samples x ~5000
genes), and reports feature importances compatible with downstream CDPS
comparisons.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger("cage.baseline_models")

__all__ = [
    "BaselineModel",
    "LogisticRidge",
    "ElasticNetLogistic",
    "DecisionTreeBinary",
    "RandomForestBinary",
    "build_model",
    "MODEL_REGISTRY",
]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _sigmoid(z: np.ndarray) -> np.ndarray:
    # Numerically stable sigmoid via separate branches.
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def _balanced_weights(y: np.ndarray) -> np.ndarray:
    """Per-sample weights matching sklearn's ``class_weight='balanced'``.

    w_c = n_samples / (n_classes * n_c) so minority samples dominate.
    """
    y = np.asarray(y).ravel().astype(np.int64)
    n = y.size
    classes, counts = np.unique(y, return_counts=True)
    w = np.zeros(n, dtype=np.float64)
    for c, cnt in zip(classes, counts):
        if cnt == 0:
            continue
        w[y == c] = n / (classes.size * cnt)
    return w


def _resolve_sample_weights(
    y: np.ndarray,
    class_weight: Optional[str | Dict[int, float]],
    sample_weight: Optional[np.ndarray],
) -> np.ndarray:
    """Combine ``class_weight`` with an explicit per-sample weight vector."""
    n = y.size
    if class_weight == "balanced":
        w = _balanced_weights(y)
    elif isinstance(class_weight, dict):
        w = np.array([class_weight.get(int(yi), 1.0) for yi in y], dtype=np.float64)
    else:
        w = np.ones(n, dtype=np.float64)
    if sample_weight is not None:
        w = w * np.asarray(sample_weight, dtype=np.float64).ravel()
    return w


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class BaselineModel:
    """Common interface so step-3 can iterate models uniformly."""

    name: str = "base"

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> "BaselineModel":
        raise NotImplementedError

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(np.int64)

    def feature_importances(self) -> np.ndarray:
        """Return a non-negative importance per input feature."""
        raise NotImplementedError

    def get_config(self) -> Dict[str, object]:
        """Return a JSON-serializable summary for manifests / logging."""
        return {"name": self.name}


# ---------------------------------------------------------------------------
# Logistic ridge (L2)
# ---------------------------------------------------------------------------


class LogisticRidge(BaselineModel):
    """L2-penalized binary logistic regression via full-batch Adam GD.

    Parameters
    ----------
    C:
        Inverse regularization strength (matches sklearn semantics):
        smaller values = stronger penalty. L2 lambda = 1 / (C * n).
    learning_rate, beta1, beta2, epsilon:
        Adam optimizer settings.
    max_iter:
        Maximum number of gradient steps.
    tol:
        Early-stopping tolerance on relative loss change.
    class_weight:
        ``"balanced"`` or dict; ``None`` disables.
    random_state:
        Unused here (deterministic by default); kept for API symmetry.
    """

    name = "logistic"

    def __init__(
        self,
        C: float = 1.0,
        learning_rate: float = 0.05,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        max_iter: int = 500,
        tol: float = 1e-6,
        class_weight: Optional[str | Dict[int, float]] = "balanced",
        random_state: Optional[int] = None,
    ) -> None:
        self.C = float(C)
        self.learning_rate = float(learning_rate)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.epsilon = float(epsilon)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.class_weight = class_weight
        self.random_state = random_state
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0
        self.n_iter_: int = 0
        self.loss_history_: List[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> "LogisticRidge":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        n, p = X.shape
        w = _resolve_sample_weights(y, self.class_weight, sample_weight)
        # Normalize sample weights so they sum to n (comparable to class-weight sklearn)
        if w.sum() > 0:
            w = w * (n / w.sum())

        lam = 1.0 / max(self.C * n, 1e-12)
        theta = np.zeros(p, dtype=np.float64)
        b = 0.0

        m_theta = np.zeros(p, dtype=np.float64)
        v_theta = np.zeros(p, dtype=np.float64)
        m_b = 0.0
        v_b = 0.0
        prev_loss = math.inf
        self.loss_history_.clear()

        for t in range(1, self.max_iter + 1):
            z = X @ theta + b
            p_hat = _sigmoid(z)
            resid = p_hat - y
            # Weighted log-loss + L2 on theta (not on bias)
            # Guard against log(0)
            eps = 1e-12
            ll = -(w * (y * np.log(np.clip(p_hat, eps, 1.0))
                        + (1.0 - y) * np.log(np.clip(1.0 - p_hat, eps, 1.0)))).sum() / max(w.sum(), 1.0)
            reg = 0.5 * lam * float(theta @ theta)
            loss = ll + reg
            self.loss_history_.append(loss)

            grad_theta = (X.T @ (w * resid)) / max(w.sum(), 1.0) + lam * theta
            grad_b = float((w * resid).sum()) / max(w.sum(), 1.0)

            # Adam update
            m_theta = self.beta1 * m_theta + (1.0 - self.beta1) * grad_theta
            v_theta = self.beta2 * v_theta + (1.0 - self.beta2) * (grad_theta ** 2)
            m_b = self.beta1 * m_b + (1.0 - self.beta1) * grad_b
            v_b = self.beta2 * v_b + (1.0 - self.beta2) * (grad_b ** 2)
            mh_theta = m_theta / (1.0 - self.beta1 ** t)
            vh_theta = v_theta / (1.0 - self.beta2 ** t)
            mh_b = m_b / (1.0 - self.beta1 ** t)
            vh_b = v_b / (1.0 - self.beta2 ** t)

            theta = theta - self.learning_rate * mh_theta / (np.sqrt(vh_theta) + self.epsilon)
            b = b - self.learning_rate * mh_b / (math.sqrt(vh_b) + self.epsilon)

            self.n_iter_ = t
            if t > 3 and prev_loss - loss < self.tol * max(1.0, abs(prev_loss)):
                break
            prev_loss = loss

        self.coef_ = theta
        self.intercept_ = float(b)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("LogisticRidge must be fit before predict_proba.")
        z = np.asarray(X, dtype=np.float64) @ self.coef_ + self.intercept_
        return _sigmoid(z)

    def feature_importances(self) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("Fit the model before accessing feature_importances.")
        return np.abs(self.coef_)

    def get_config(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "C": self.C,
            "learning_rate": self.learning_rate,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "class_weight": self.class_weight,
            "n_iter_": self.n_iter_,
        }


# ---------------------------------------------------------------------------
# Elastic-net logistic (L1 + L2) via FISTA
# ---------------------------------------------------------------------------


class ElasticNetLogistic(BaselineModel):
    """Binary logistic regression with mixed L1/L2 penalty (FISTA).

    The penalty follows the standard elastic-net parameterization:
        alpha * (l1_ratio * ||theta||_1 + (1 - l1_ratio) * 0.5 * ||theta||_2^2)
    with alpha = 1 / (C * n).
    """

    name = "elasticnet"

    def __init__(
        self,
        C: float = 1.0,
        l1_ratio: float = 0.5,
        learning_rate: Optional[float] = None,
        max_iter: int = 600,
        tol: float = 1e-6,
        class_weight: Optional[str | Dict[int, float]] = "balanced",
        random_state: Optional[int] = None,
    ) -> None:
        self.C = float(C)
        self.l1_ratio = float(l1_ratio)
        self.learning_rate = learning_rate
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.class_weight = class_weight
        self.random_state = random_state
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0
        self.n_iter_: int = 0
        self.loss_history_: List[float] = []

    @staticmethod
    def _soft_threshold(x: np.ndarray, lam: float) -> np.ndarray:
        return np.sign(x) * np.maximum(np.abs(x) - lam, 0.0)

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> "ElasticNetLogistic":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        n, p = X.shape
        w = _resolve_sample_weights(y, self.class_weight, sample_weight)
        if w.sum() > 0:
            w = w * (n / w.sum())

        alpha = 1.0 / max(self.C * n, 1e-12)
        lam_l1 = alpha * self.l1_ratio
        lam_l2 = alpha * (1.0 - self.l1_ratio)

        # Lipschitz constant upper bound for logistic: 0.25 * ||X||_2^2 + lam_l2
        # where ||X||_2 is the operator norm. Approximate via column norm bound.
        if self.learning_rate is None:
            # Estimate via power iteration on X^T W X; fall back to Frobenius bound
            col_norm_sq = float(np.sum((X * w[:, None]) * X) / max(w.sum(), 1.0))
            L = 0.25 * col_norm_sq / max(p, 1) + lam_l2
            # Use conservative step: 1 / L
            step = 1.0 / max(L, 1e-6)
        else:
            step = float(self.learning_rate)

        theta = np.zeros(p, dtype=np.float64)
        b = 0.0
        # FISTA auxiliary variables
        theta_y = theta.copy()
        b_y = b
        t_k = 1.0
        prev_loss = math.inf
        self.loss_history_.clear()
        sum_w = max(w.sum(), 1.0)

        for it in range(1, self.max_iter + 1):
            z = X @ theta_y + b_y
            p_hat = _sigmoid(z)
            resid = p_hat - y

            grad_theta = (X.T @ (w * resid)) / sum_w + lam_l2 * theta_y
            grad_b = float((w * resid).sum()) / sum_w

            theta_next = theta_y - step * grad_theta
            theta_next = self._soft_threshold(theta_next, step * lam_l1)
            b_next = b_y - step * grad_b

            t_next = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t_k * t_k))
            theta_y = theta_next + ((t_k - 1.0) / t_next) * (theta_next - theta)
            b_y = b_next + ((t_k - 1.0) / t_next) * (b_next - b)

            # Track loss on the averaged (theta_next) point
            z2 = X @ theta_next + b_next
            p2 = _sigmoid(z2)
            eps = 1e-12
            ll = -(w * (y * np.log(np.clip(p2, eps, 1.0))
                        + (1.0 - y) * np.log(np.clip(1.0 - p2, eps, 1.0)))).sum() / sum_w
            reg = lam_l1 * float(np.abs(theta_next).sum()) + 0.5 * lam_l2 * float(theta_next @ theta_next)
            loss = ll + reg
            self.loss_history_.append(loss)

            theta, b, t_k = theta_next, b_next, t_next
            self.n_iter_ = it
            if it > 3 and prev_loss - loss < self.tol * max(1.0, abs(prev_loss)):
                break
            prev_loss = loss

        self.coef_ = theta
        self.intercept_ = float(b)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("ElasticNetLogistic must be fit before predict_proba.")
        z = np.asarray(X, dtype=np.float64) @ self.coef_ + self.intercept_
        return _sigmoid(z)

    def feature_importances(self) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("Fit the model before accessing feature_importances.")
        return np.abs(self.coef_)

    def get_config(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "C": self.C,
            "l1_ratio": self.l1_ratio,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "class_weight": self.class_weight,
            "n_iter_": self.n_iter_,
            "n_nonzero": int((self.coef_ != 0).sum()) if self.coef_ is not None else 0,
        }


# ---------------------------------------------------------------------------
# Decision tree (binary, Gini-split)
# ---------------------------------------------------------------------------


@dataclass
class _TreeNode:
    feature: int = -1
    threshold: float = float("nan")
    left: Optional["_TreeNode"] = None
    right: Optional["_TreeNode"] = None
    is_leaf: bool = True
    probability: float = 0.5
    n_samples: int = 0
    weighted_samples: float = 0.0
    impurity: float = 0.0
    impurity_reduction: float = 0.0  # for feature importance accumulation


class DecisionTreeBinary(BaselineModel):
    """Binary classification tree with weighted Gini splits.

    Supports depth and sample-size guards plus random feature subsampling
    for use inside :class:`RandomForestBinary`.
    """

    name = "decision_tree"

    def __init__(
        self,
        max_depth: int = 6,
        min_samples_split: int = 4,
        min_samples_leaf: int = 1,
        max_features: Optional[str | int | float] = None,
        class_weight: Optional[str | Dict[int, float]] = "balanced",
        random_state: Optional[int] = None,
    ) -> None:
        self.max_depth = int(max_depth)
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_features = max_features
        self.class_weight = class_weight
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)
        self.root_: Optional[_TreeNode] = None
        self.n_features_: int = 0
        self.feature_importances_raw_: Optional[np.ndarray] = None

    # ----- fitting ---------------------------------------------------------

    def _n_features_to_use(self) -> int:
        p = self.n_features_
        mf = self.max_features
        if mf is None:
            return p
        if isinstance(mf, str):
            if mf == "sqrt":
                return max(1, int(math.sqrt(p)))
            if mf == "log2":
                return max(1, int(math.log2(max(p, 2))))
            raise ValueError(f"Unknown max_features {mf!r}")
        if isinstance(mf, float):
            return max(1, int(mf * p))
        return max(1, min(int(mf), p))

    @staticmethod
    def _weighted_gini(pos_w: float, neg_w: float) -> float:
        total = pos_w + neg_w
        if total <= 0:
            return 0.0
        p = pos_w / total
        return 1.0 - p * p - (1.0 - p) ** 2

    def _best_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        w: np.ndarray,
    ) -> Tuple[int, float, float]:
        """Return (feature, threshold, impurity_reduction) or (-1, nan, 0)."""
        n, p = X.shape
        total_pos_w = float((w * y).sum())
        total_neg_w = float((w * (1.0 - y)).sum())
        parent_imp = self._weighted_gini(total_pos_w, total_neg_w)
        total_w = total_pos_w + total_neg_w
        if total_w <= 0 or parent_imp == 0.0:
            return -1, float("nan"), 0.0

        n_use = self._n_features_to_use()
        if n_use < p:
            feat_idx = self._rng.choice(p, size=n_use, replace=False)
        else:
            feat_idx = np.arange(p)

        best_feat = -1
        best_thr = float("nan")
        best_gain = 0.0

        for f in feat_idx:
            col = X[:, f]
            order = np.argsort(col, kind="mergesort")
            col_s = col[order]
            y_s = y[order]
            w_s = w[order]

            # Cumulative weighted pos / neg for candidate split at position i
            # (left = samples [0..i], right = [i+1..n-1])
            cum_pos = np.cumsum(w_s * y_s)
            cum_neg = np.cumsum(w_s * (1.0 - y_s))
            cum_w = cum_pos + cum_neg

            for i in range(n - 1):
                # Skip if the next value is tied -> threshold would be ambiguous
                if col_s[i + 1] == col_s[i]:
                    continue
                left_w = cum_w[i]
                right_w = total_w - left_w
                if left_w < self.min_samples_leaf or right_w < self.min_samples_leaf:
                    # With uniform weights min_samples_leaf is a count threshold;
                    # weighted setting treats it as min weighted count.
                    continue
                left_pos = cum_pos[i]
                left_neg = cum_neg[i]
                right_pos = total_pos_w - left_pos
                right_neg = total_neg_w - left_neg
                left_imp = self._weighted_gini(left_pos, left_neg)
                right_imp = self._weighted_gini(right_pos, right_neg)
                child_imp = (left_w * left_imp + right_w * right_imp) / total_w
                gain = parent_imp - child_imp
                if gain > best_gain:
                    best_gain = gain
                    best_feat = int(f)
                    best_thr = 0.5 * (col_s[i] + col_s[i + 1])

        return best_feat, best_thr, best_gain

    def _build(
        self,
        X: np.ndarray,
        y: np.ndarray,
        w: np.ndarray,
        depth: int,
    ) -> _TreeNode:
        total_pos_w = float((w * y).sum())
        total_neg_w = float((w * (1.0 - y)).sum())
        total_w = total_pos_w + total_neg_w
        prob = total_pos_w / total_w if total_w > 0 else 0.5
        node = _TreeNode(
            is_leaf=True,
            probability=prob,
            n_samples=int(y.size),
            weighted_samples=total_w,
            impurity=self._weighted_gini(total_pos_w, total_neg_w),
        )

        if depth >= self.max_depth:
            return node
        if y.size < self.min_samples_split:
            return node
        if node.impurity == 0.0:
            return node

        feat, thr, gain = self._best_split(X, y, w)
        if feat < 0 or gain <= 0.0:
            return node

        mask = X[:, feat] <= thr
        if mask.sum() == 0 or mask.sum() == y.size:
            return node

        node.is_leaf = False
        node.feature = feat
        node.threshold = thr
        node.impurity_reduction = gain * total_w  # weight by node mass for importance
        node.left = self._build(X[mask], y[mask], w[mask], depth + 1)
        node.right = self._build(X[~mask], y[~mask], w[~mask], depth + 1)
        return node

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> "DecisionTreeBinary":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        self.n_features_ = X.shape[1]
        w = _resolve_sample_weights(y, self.class_weight, sample_weight)
        self._rng = np.random.default_rng(self.random_state)
        self.root_ = self._build(X, y, w, depth=0)
        self.feature_importances_raw_ = self._collect_importances()
        return self

    # ----- prediction ------------------------------------------------------

    def _predict_proba_single(self, x: np.ndarray) -> float:
        node = self.root_
        assert node is not None
        while not node.is_leaf:
            if x[node.feature] <= node.threshold:
                node = node.left  # type: ignore[assignment]
            else:
                node = node.right  # type: ignore[assignment]
            assert node is not None
        return node.probability

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.root_ is None:
            raise RuntimeError("DecisionTreeBinary must be fit before predict_proba.")
        X = np.asarray(X, dtype=np.float64)
        return np.array([self._predict_proba_single(x) for x in X], dtype=np.float64)

    # ----- importances -----------------------------------------------------

    def _collect_importances(self) -> np.ndarray:
        imp = np.zeros(self.n_features_, dtype=np.float64)

        def _walk(node: Optional[_TreeNode]) -> None:
            if node is None or node.is_leaf:
                return
            imp[node.feature] += node.impurity_reduction
            _walk(node.left)
            _walk(node.right)

        _walk(self.root_)
        total = imp.sum()
        if total > 0:
            imp = imp / total
        return imp

    def feature_importances(self) -> np.ndarray:
        if self.feature_importances_raw_ is None:
            raise RuntimeError("Fit the model before accessing feature_importances.")
        return self.feature_importances_raw_.copy()

    def get_config(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "max_features": self.max_features,
            "class_weight": self.class_weight,
        }


# ---------------------------------------------------------------------------
# Random forest
# ---------------------------------------------------------------------------


class RandomForestBinary(BaselineModel):
    """Bagged binary decision forest with sqrt feature subsampling."""

    name = "rf"

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 6,
        min_samples_split: int = 4,
        min_samples_leaf: int = 1,
        max_features: str | int | float = "sqrt",
        class_weight: Optional[str | Dict[int, float]] = "balanced",
        bootstrap: bool = True,
        random_state: Optional[int] = None,
    ) -> None:
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_features = max_features
        self.class_weight = class_weight
        self.bootstrap = bool(bootstrap)
        self.random_state = random_state
        self.trees_: List[DecisionTreeBinary] = []
        self.oob_indices_: List[np.ndarray] = []
        self.n_features_: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> "RandomForestBinary":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        n, p = X.shape
        self.n_features_ = p
        rng = np.random.default_rng(self.random_state)
        self.trees_.clear()
        self.oob_indices_.clear()

        for t in range(self.n_estimators):
            if self.bootstrap:
                idx = rng.integers(0, n, size=n)
                in_bag_mask = np.zeros(n, dtype=bool)
                in_bag_mask[idx] = True
                self.oob_indices_.append(np.where(~in_bag_mask)[0])
                Xb, yb = X[idx], y[idx]
                wb = None if sample_weight is None else np.asarray(sample_weight).ravel()[idx]
            else:
                Xb, yb = X, y
                wb = sample_weight
                self.oob_indices_.append(np.array([], dtype=np.int64))

            # Derive a reproducible child seed for the tree
            child_seed = int(rng.integers(0, 2**31 - 1))
            tree = DecisionTreeBinary(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                class_weight=self.class_weight,
                random_state=child_seed,
            )
            tree.fit(Xb, yb, sample_weight=wb)
            self.trees_.append(tree)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.trees_:
            raise RuntimeError("RandomForestBinary must be fit before predict_proba.")
        X = np.asarray(X, dtype=np.float64)
        probs = np.zeros(X.shape[0], dtype=np.float64)
        for tree in self.trees_:
            probs += tree.predict_proba(X)
        return probs / len(self.trees_)

    def feature_importances(self) -> np.ndarray:
        if not self.trees_:
            raise RuntimeError("Fit the model before accessing feature_importances.")
        imp = np.zeros(self.n_features_, dtype=np.float64)
        for tree in self.trees_:
            imp += tree.feature_importances()
        imp = imp / len(self.trees_)
        total = imp.sum()
        if total > 0:
            imp = imp / total
        return imp

    def get_config(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "max_features": self.max_features,
            "class_weight": self.class_weight,
            "bootstrap": self.bootstrap,
        }


# ---------------------------------------------------------------------------
# Registry / builder
# ---------------------------------------------------------------------------


MODEL_REGISTRY: Dict[str, Callable[..., BaselineModel]] = {
    "logistic": LogisticRidge,
    "elasticnet": ElasticNetLogistic,
    "rf": RandomForestBinary,
    "decision_tree": DecisionTreeBinary,
}


def build_model(name: str, seed: Optional[int] = None, **overrides) -> BaselineModel:
    """Construct a baseline model by name with sensible default hyperparams.

    Step-3 default hyperparameters
    ------------------------------
    logistic     : C=1.0, class_weight='balanced', max_iter=500
    elasticnet   : C=1.0, l1_ratio=0.5, class_weight='balanced'
    rf           : n_estimators=200, max_depth=6, max_features='sqrt'
    decision_tree: max_depth=6, class_weight='balanced'

    Unrecognized model names raise ``ValueError`` (prevents silent typos).
    """
    key = name.lower()
    if key not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown baseline model {name!r}; available: {sorted(MODEL_REGISTRY)}"
        )
    defaults: Dict[str, Dict[str, object]] = {
        "logistic": {
            "C": 1.0,
            "max_iter": 500,
            "tol": 1e-6,
            "class_weight": "balanced",
            "random_state": seed,
        },
        "elasticnet": {
            "C": 1.0,
            "l1_ratio": 0.5,
            "max_iter": 600,
            "tol": 1e-6,
            "class_weight": "balanced",
            "random_state": seed,
        },
        "rf": {
            "n_estimators": 200,
            "max_depth": 6,
            "min_samples_split": 4,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "class_weight": "balanced",
            "bootstrap": True,
            "random_state": seed,
        },
        "decision_tree": {
            "max_depth": 6,
            "min_samples_split": 4,
            "min_samples_leaf": 1,
            "max_features": None,
            "class_weight": "balanced",
            "random_state": seed,
        },
    }
    params = dict(defaults.get(key, {}))
    params.update(overrides)
    return MODEL_REGISTRY[key](**params)


# ---------------------------------------------------------------------------
# Optional external wrappers (sklearn / XGBoost / LightGBM)
# Registered into MODEL_REGISTRY only when the dependency is importable.
# ---------------------------------------------------------------------------


class _SklearnWrapper(BaselineModel):
    """Thin adapter for scikit-learn classifiers."""

    def __init__(self, clf_class, **kwargs) -> None:
        self._clf = clf_class(**kwargs)
        self.name = getattr(self._clf, "__class__", type(self._clf)).__name__.lower()

    def fit(self, X, y, sample_weight=None):
        kw = {}
        if sample_weight is not None:
            kw["sample_weight"] = sample_weight
        self._clf.fit(X, y, **kw)
        return self

    def predict_proba(self, X):
        return self._clf.predict_proba(X)[:, 1]

    def feature_importances(self):
        if hasattr(self._clf, "coef_"):
            return np.abs(self._clf.coef_).ravel()
        if hasattr(self._clf, "feature_importances_"):
            return self._clf.feature_importances_.ravel()
        return np.zeros(1)

    def get_config(self):
        return {"name": self.name, "params": self._clf.get_params()}


def _try_register_sklearn() -> None:
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier
        from sklearn.svm import SVC

        def _make_sklearn_lr(random_state=None, **kw):
            defaults = {"C": 1.0, "max_iter": 1000, "solver": "saga",
                        "class_weight": "balanced", "random_state": random_state}
            defaults.update(kw)
            return _SklearnWrapper(LogisticRegression, **defaults)

        def _make_sklearn_gbt(random_state=None, **kw):
            defaults = {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05,
                        "subsample": 0.8, "random_state": random_state}
            defaults.update(kw)
            return _SklearnWrapper(GradientBoostingClassifier, **defaults)

        def _make_sklearn_extratrees(random_state=None, **kw):
            defaults = {"n_estimators": 200, "max_depth": 8, "class_weight": "balanced",
                        "random_state": random_state}
            defaults.update(kw)
            return _SklearnWrapper(ExtraTreesClassifier, **defaults)

        MODEL_REGISTRY.setdefault("sklearn_lr", _make_sklearn_lr)
        MODEL_REGISTRY.setdefault("sklearn_gbt", _make_sklearn_gbt)
        MODEL_REGISTRY.setdefault("sklearn_extratrees", _make_sklearn_extratrees)
        logger.debug("scikit-learn optional baselines registered: sklearn_lr, sklearn_gbt, sklearn_extratrees")
    except ImportError:
        pass


def _try_register_xgboost() -> None:
    try:
        from xgboost import XGBClassifier

        def _make_xgb(random_state=None, **kw):
            defaults = {
                "n_estimators": 300, "max_depth": 4, "learning_rate": 0.05,
                "subsample": 0.8, "colsample_bytree": 0.8,
                "scale_pos_weight": 6,   # approx 92% tumor -> ~11:1 imbalance
                "use_label_encoder": False, "eval_metric": "logloss",
                "seed": random_state,
            }
            defaults.update(kw)
            return _SklearnWrapper(XGBClassifier, **defaults)

        MODEL_REGISTRY.setdefault("xgboost", _make_xgb)
        logger.debug("XGBoost optional baseline registered: xgboost")
    except ImportError:
        pass


def _try_register_lightgbm() -> None:
    try:
        from lightgbm import LGBMClassifier

        def _make_lgbm(random_state=None, **kw):
            defaults = {
                "n_estimators": 300, "max_depth": 4, "learning_rate": 0.05,
                "subsample": 0.8, "class_weight": "balanced",
                "random_state": random_state, "verbose": -1,
            }
            defaults.update(kw)
            return _SklearnWrapper(LGBMClassifier, **defaults)

        MODEL_REGISTRY.setdefault("lightgbm", _make_lgbm)
        logger.debug("LightGBM optional baseline registered: lightgbm")
    except ImportError:
        pass


# Register optional models at import time (silent if deps missing).
_try_register_sklearn()
_try_register_xgboost()
_try_register_lightgbm()
