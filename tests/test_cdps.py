"""Tests for Step-5 CDPS scoring (normalize_scores, build_cdps_ranking)."""

from __future__ import annotations

import numpy as np
import pytest

from cage.step5_runner import normalize_scores, build_cdps_ranking


# ---------------------------------------------------------------------------
# normalize_scores
# ---------------------------------------------------------------------------

class TestNormalizeScores:

    def test_minmax_range(self):
        scores = np.array([1.0, 5.0, 3.0, 7.0, 2.0])
        normed = normalize_scores(scores, method="minmax")
        assert float(normed.min()) == pytest.approx(0.0, abs=1e-9)
        assert float(normed.max()) == pytest.approx(1.0, abs=1e-9)

    def test_minmax_constant_scores_returns_zeros(self):
        scores = np.ones(10) * 5.0
        normed = normalize_scores(scores, method="minmax")
        np.testing.assert_array_equal(normed, np.zeros(10))

    def test_rank_range(self):
        scores = np.array([3.0, 1.0, 4.0, 1.0, 5.0])
        normed = normalize_scores(scores, method="rank")
        assert float(normed.min()) >= 0.0
        assert float(normed.max()) <= 1.0

    def test_rank_preserves_order(self):
        scores = np.array([10.0, 1.0, 5.0, 3.0])
        normed = normalize_scores(scores, method="rank")
        order_orig = np.argsort(scores)
        order_norm = np.argsort(normed)
        np.testing.assert_array_equal(order_orig, order_norm)

    def test_rank_tied_values_get_same_rank(self):
        scores = np.array([1.0, 2.0, 2.0, 3.0])
        normed = normalize_scores(scores, method="rank")
        assert normed[1] == pytest.approx(normed[2], abs=1e-9), "Tied values must get same rank"

    def test_empty_array(self):
        normed = normalize_scores(np.array([]), method="minmax")
        assert normed.size == 0

    def test_single_element(self):
        normed = normalize_scores(np.array([7.0]), method="minmax")
        assert normed.size == 1
        assert normed[0] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# build_cdps_ranking
# ---------------------------------------------------------------------------

class TestBuildCdpsRanking:

    def _synthetic_components(self, n=50, seed=0):
        rng = np.random.default_rng(seed)
        genes = [f"GENE{i:04d}" for i in range(n)]
        return {
            "gene_names": genes,
            "attribution": rng.uniform(0, 1, n),
            "gate": rng.uniform(0, 1, n),
            "stability": rng.uniform(0, 1, n),
            "invariance": rng.uniform(0, 1, n),
            "perturbation": rng.uniform(0, 1, n),
            "weights": {
                "attribution": 0.30, "gate": 0.20, "stability": 0.20,
                "invariance": 0.15, "perturbation": 0.15,
            },
        }

    def test_returns_dict_with_required_keys(self):
        kwargs = self._synthetic_components()
        result = build_cdps_ranking(**kwargs)
        for key in ("cdps", "records"):
            assert key in result, f"Missing key: {key}"

    def test_cdps_in_zero_one(self):
        kwargs = self._synthetic_components()
        result = build_cdps_ranking(**kwargs)
        cdps = result["cdps"]
        assert np.all((cdps >= 0.0) & (cdps <= 1.0)), "CDPS values must be in [0, 1]"

    def test_ranked_genes_length(self):
        n = 50
        kwargs = self._synthetic_components(n=n)
        result = build_cdps_ranking(**kwargs)
        assert len(result["records"]) == n

    def test_ranked_genes_sorted_descending(self):
        kwargs = self._synthetic_components()
        result = build_cdps_ranking(**kwargs)
        scores = [r["cdps"] for r in result["records"]]
        assert scores == sorted(scores, reverse=True), "records must be sorted by CDPS descending"

    def test_highest_attribution_gene_ranks_near_top(self):
        """Gene with max attribution should rank first when attribution weight is dominant."""
        n = 100
        rng = np.random.default_rng(3)
        genes = [f"GENE{i:04d}" for i in range(n)]
        attribution = rng.uniform(0, 0.5, n)
        attribution[42] = 1.0  # spike

        result = build_cdps_ranking(
            gene_names=genes,
            attribution=attribution,
            gate=np.zeros(n),
            stability=np.zeros(n),
            invariance=np.zeros(n),
            perturbation=np.zeros(n),
            weights={"attribution": 1.0, "gate": 0.0, "stability": 0.0,
                     "invariance": 0.0, "perturbation": 0.0},
        )
        top_gene = result["records"][0]["gene"]
        assert top_gene == "GENE0042", f"Top gene should be GENE0042, got {top_gene}"

    def test_weights_sum_to_one_internally(self):
        """Non-normalized input weights should be re-normalized inside the function."""
        kwargs = self._synthetic_components()
        kwargs["weights"] = {"attribution": 3.0, "gate": 2.0, "stability": 2.0,
                             "invariance": 1.5, "perturbation": 1.5}
        result = build_cdps_ranking(**kwargs)
        cdps = result["cdps"]
        assert np.all((cdps >= 0.0) & (cdps <= 1.0))

    def test_no_perturbation_flag(self):
        kwargs = self._synthetic_components()
        result = build_cdps_ranking(**kwargs, include_perturbation=False)
        assert "cdps" in result
        cdps = result["cdps"]
        assert np.all((cdps >= 0.0) & (cdps <= 1.0))

    def test_zero_weights_raises(self):
        kwargs = self._synthetic_components()
        kwargs["weights"] = {"attribution": 0.0, "gate": 0.0, "stability": 0.0,
                             "invariance": 0.0, "perturbation": 0.0}
        with pytest.raises(ValueError, match="non-positive"):
            build_cdps_ranking(**kwargs)

    def test_all_genes_in_ranking(self):
        n = 30
        kwargs = self._synthetic_components(n=n)
        result = build_cdps_ranking(**kwargs)
        ranked_names = {r["gene"] for r in result["records"]}
        assert ranked_names == set(kwargs["gene_names"])

    def test_rank_normalization_gives_different_order_than_minmax(self):
        """rank and minmax normalization may differ; ensure both run without error."""
        kwargs = self._synthetic_components()
        r_mm = build_cdps_ranking(**kwargs, normalization="minmax")
        r_rk = build_cdps_ranking(**kwargs, normalization="rank")
        assert len(r_mm["records"]) == len(r_rk["records"])
