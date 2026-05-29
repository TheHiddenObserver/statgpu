"""
Tests for the covariance module (EmpiricalCovariance, LedoitWolf, OAS).

Validates that statgpu implementations match sklearn within tolerance.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from sklearn.covariance import (
    EmpiricalCovariance as SKEmpirical,
    LedoitWolf as SKLedoitWolf,
    OAS as SKOAS,
)

from statgpu.covariance import EmpiricalCovariance, LedoitWolf, OAS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.RandomState(42)


@pytest.fixture
def sample_data(rng):
    """n=200, p=10 Gaussian data."""
    X = rng.randn(200, 10)
    # Add some correlation structure
    L = rng.randn(10, 10)
    X = X @ L
    return X


@pytest.fixture
def small_data(rng):
    """n=50, p=5 data."""
    X = rng.randn(50, 5)
    L = rng.randn(5, 5)
    X = X @ L
    return X


@pytest.fixture
def test_data(rng):
    """n=50, p=10 test data (same p as sample_data)."""
    return rng.randn(50, 10)


@pytest.fixture
def wide_data(rng):
    """n=30, p=50 (p > n) data -- tests shrinkage necessity."""
    return rng.randn(30, 50)


# ---------------------------------------------------------------------------
# EmpiricalCovariance
# ---------------------------------------------------------------------------

class TestEmpiricalCovariance:
    def test_covariance_matches_sklearn(self, sample_data):
        sk = SKEmpirical().fit(sample_data)
        sg = EmpiricalCovariance().fit(sample_data)

        assert_allclose(sg.covariance_, sk.covariance_, rtol=1e-10)
        assert_allclose(sg.location_, sk.location_, rtol=1e-10)

    def test_precision_matches_sklearn(self, sample_data):
        sk = SKEmpirical().fit(sample_data)
        sg = EmpiricalCovariance().fit(sample_data)

        assert_allclose(sg.precision_, sk.precision_, rtol=1e-5)

    def test_assume_centered(self, sample_data):
        sk = SKEmpirical(assume_centered=True).fit(sample_data)
        sg = EmpiricalCovariance(assume_centered=True).fit(sample_data)

        assert_allclose(sg.covariance_, sk.covariance_, rtol=1e-10)
        assert_allclose(sg.location_, np.zeros(sample_data.shape[1]), atol=1e-15)

    def test_mahalanobis_matches_sklearn(self, sample_data, test_data):
        sk = SKEmpirical().fit(sample_data)
        sg = EmpiricalCovariance().fit(sample_data)

        mahal_sk = sk.mahalanobis(test_data)
        mahal_sg = sg.mahalanobis(test_data)

        assert_allclose(mahal_sg, mahal_sk, rtol=1e-5)

    def test_score_matches_sklearn(self, sample_data, test_data):
        sk = SKEmpirical().fit(sample_data)
        sg = EmpiricalCovariance().fit(sample_data)

        score_sk = sk.score(test_data)
        score_sg = sg.score(test_data)

        assert_allclose(score_sg, score_sk, rtol=1e-5)

    def test_predict_returns_mahalanobis(self, sample_data, test_data):
        sg = EmpiricalCovariance().fit(sample_data)

        pred = sg.predict(test_data)
        mahal = sg.mahalanobis(test_data)

        assert_allclose(pred, mahal, rtol=1e-15)

    def test_1d_input(self, rng):
        x = rng.randn(100)
        sg = EmpiricalCovariance().fit(x)
        assert sg.covariance_.shape == (1, 1)
        assert sg.location_.shape == (1,)

    def test_check_is_fitted(self):
        sg = EmpiricalCovariance()
        with pytest.raises(RuntimeError, match="not fitted"):
            sg.mahalanobis(np.zeros((5, 3)))

    def test_get_params(self):
        sg = EmpiricalCovariance(assume_centered=True, device="cpu")
        params = sg.get_params()
        assert params["assume_centered"] is True
        assert params["device"] == "cpu"


# ---------------------------------------------------------------------------
# LedoitWolf
# ---------------------------------------------------------------------------

class TestLedoitWolf:
    def test_covariance_matches_sklearn(self, sample_data):
        sk = SKLedoitWolf().fit(sample_data)
        sg = LedoitWolf().fit(sample_data)

        assert_allclose(sg.covariance_, sk.covariance_, rtol=1e-5)

    def test_shrinkage_matches_sklearn(self, sample_data):
        sk = SKLedoitWolf().fit(sample_data)
        sg = LedoitWolf().fit(sample_data)

        assert_allclose(sg.shrinkage_, sk.shrinkage_, rtol=1e-5)

    def test_precision_matches_sklearn(self, sample_data):
        sk = SKLedoitWolf().fit(sample_data)
        sg = LedoitWolf().fit(sample_data)

        assert_allclose(sg.precision_, sk.precision_, rtol=1e-3)

    def test_location_matches_sklearn(self, sample_data):
        sk = SKLedoitWolf().fit(sample_data)
        sg = LedoitWolf().fit(sample_data)

        assert_allclose(sg.location_, sk.location_, rtol=1e-10)

    def test_mahalanobis_matches_sklearn(self, sample_data, test_data):
        sk = SKLedoitWolf().fit(sample_data)
        sg = LedoitWolf().fit(sample_data)

        mahal_sk = sk.mahalanobis(test_data)
        mahal_sg = sg.mahalanobis(test_data)

        assert_allclose(mahal_sg, mahal_sk, rtol=1e-3)

    def test_score_matches_sklearn(self, sample_data, test_data):
        sk = SKLedoitWolf().fit(sample_data)
        sg = LedoitWolf().fit(sample_data)

        score_sk = sk.score(test_data)
        score_sg = sg.score(test_data)

        assert_allclose(score_sg, score_sk, rtol=1e-3)

    def test_wide_data_shrinkage(self, wide_data):
        """With p > n, shrinkage should be substantial."""
        sg = LedoitWolf().fit(wide_data)
        assert 0.0 < sg.shrinkage_ <= 1.0

    def test_shrinkage_range(self, sample_data):
        sg = LedoitWolf().fit(sample_data)
        assert 0.0 <= sg.shrinkage_ <= 1.0

    def test_assume_centered(self, sample_data):
        sk = SKLedoitWolf(assume_centered=True).fit(sample_data)
        sg = LedoitWolf(assume_centered=True).fit(sample_data)

        assert_allclose(sg.covariance_, sk.covariance_, rtol=1e-5)
        assert_allclose(sg.shrinkage_, sk.shrinkage_, rtol=1e-5)

    def test_small_sample(self, rng):
        """n=10, p=5 -- extreme shrinkage regime."""
        X = rng.randn(10, 5)
        sk = SKLedoitWolf().fit(X)
        sg = LedoitWolf().fit(X)

        assert_allclose(sg.shrinkage_, sk.shrinkage_, rtol=1e-5)
        assert_allclose(sg.covariance_, sk.covariance_, rtol=1e-5)


# ---------------------------------------------------------------------------
# OAS
# ---------------------------------------------------------------------------

class TestOAS:
    def test_covariance_matches_sklearn(self, sample_data):
        sk = SKOAS().fit(sample_data)
        sg = OAS().fit(sample_data)

        assert_allclose(sg.covariance_, sk.covariance_, rtol=1e-5)

    def test_shrinkage_matches_sklearn(self, sample_data):
        sk = SKOAS().fit(sample_data)
        sg = OAS().fit(sample_data)

        assert_allclose(sg.shrinkage_, sk.shrinkage_, rtol=1e-10)

    def test_precision_matches_sklearn(self, sample_data):
        sk = SKOAS().fit(sample_data)
        sg = OAS().fit(sample_data)

        assert_allclose(sg.precision_, sk.precision_, rtol=1e-3)

    def test_location_matches_sklearn(self, sample_data):
        sk = SKOAS().fit(sample_data)
        sg = OAS().fit(sample_data)

        assert_allclose(sg.location_, sk.location_, rtol=1e-10)

    def test_mahalanobis_matches_sklearn(self, sample_data, test_data):
        sk = SKOAS().fit(sample_data)
        sg = OAS().fit(sample_data)

        mahal_sk = sk.mahalanobis(test_data)
        mahal_sg = sg.mahalanobis(test_data)

        assert_allclose(mahal_sg, mahal_sk, rtol=1e-3)

    def test_score_matches_sklearn(self, sample_data, test_data):
        sk = SKOAS().fit(sample_data)
        sg = OAS().fit(sample_data)

        score_sk = sk.score(test_data)
        score_sg = sg.score(test_data)

        assert_allclose(score_sg, score_sk, rtol=1e-3)

    def test_shrinkage_range(self, sample_data):
        sg = OAS().fit(sample_data)
        assert 0.0 <= sg.shrinkage_ <= 1.0

    def test_assume_centered(self, sample_data):
        sk = SKOAS(assume_centered=True).fit(sample_data)
        sg = OAS(assume_centered=True).fit(sample_data)

        assert_allclose(sg.covariance_, sk.covariance_, rtol=1e-5)
        assert_allclose(sg.shrinkage_, sk.shrinkage_, rtol=1e-5)

    def test_small_sample(self, rng):
        """n=10, p=5 -- extreme shrinkage regime."""
        X = rng.randn(10, 5)
        sk = SKOAS().fit(X)
        sg = OAS().fit(X)

        assert_allclose(sg.shrinkage_, sk.shrinkage_, rtol=1e-5)
        assert_allclose(sg.covariance_, sk.covariance_, rtol=1e-5)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_sample_raises(self):
        sg = EmpiricalCovariance()
        with pytest.raises(ValueError, match="at least 2"):
            sg.fit(np.zeros((1, 5)))

    def test_two_samples(self, rng):
        X = rng.randn(2, 3)
        sg = EmpiricalCovariance().fit(X)
        assert sg.covariance_.shape == (3, 3)

    def test_identity_covariance(self, rng):
        """For data with identity covariance, shrinkage should be valid."""
        n, p = 500, 5
        X = rng.randn(n, p)
        sg = LedoitWolf().fit(X)
        # Shrinkage is clamped to [0, 1]; for identity cov the optimal
        # shrinkage can be 1.0 (full shrinkage toward target) — matches sklearn.
        assert 0.0 <= sg.shrinkage_ <= 1.0

    def test_fitted_flag(self):
        sg = EmpiricalCovariance()
        assert not sg._fitted
        sg.fit(np.random.randn(50, 5))
        assert sg._fitted


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
