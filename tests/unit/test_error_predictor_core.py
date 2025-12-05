# tests/unit/test_error_predictor_core.py
"""
Tests for core error predictor models (SIR, PCE).
"""
import numpy as np
from sklearn.metrics import r2_score

from wifa_uq.postprocessing.error_predictor.error_predictor import (
    SIRPolynomialRegressor,
    PCERegressor,
)


class TestSIRPolynomialRegressor:
    """Tests for SIRPolynomialRegressor."""

    def test_learns_simple_linear_relation(self):
        """Should learn y = 2*x0 - 3*x1 + noise with high R²."""
        rng = np.random.default_rng(0)
        X = rng.normal(size=(200, 2))
        y = 2.0 * X[:, 0] - 3.0 * X[:, 1] + 0.1 * rng.normal(size=200)

        model = SIRPolynomialRegressor(n_directions=1, degree=1)
        model.fit(X, y)
        y_pred = model.predict(X)

        assert r2_score(y, y_pred) > 0.9

    def test_feature_importance_available_after_fit(self):
        """get_feature_importance should return Series with feature names."""
        rng = np.random.default_rng(42)
        X = rng.normal(size=(100, 3))
        y = X[:, 0] + 0.1 * rng.normal(size=100)

        model = SIRPolynomialRegressor(n_directions=1, degree=2)
        model.fit(X, y)
        
        importance = model.get_feature_importance(["f1", "f2", "f3"])
        
        assert len(importance) == 3
        assert list(importance.index) == ["f1", "f2", "f3"]


class TestPCERegressor:
    """Tests for PCERegressor."""

    def test_refuses_too_many_features_by_default(self):
        """Should raise ValueError when features exceed max_features."""
        X = np.random.randn(50, 6)  # 6 > default max_features=5
        y = X[:, 0] ** 2

        pce = PCERegressor(degree=3, max_features=5, allow_high_dim=False)
        
        try:
            pce.fit(X, y)
            assert False, "Expected ValueError for too many features"
        except ValueError as e:
            assert "refused to run" in str(e)

    def test_learns_simple_polynomial(self):
        """PCE with degree 2 should learn y = x² almost perfectly."""
        x = np.linspace(-1, 1, 30)
        y = x ** 2
        X = x.reshape(-1, 1)

        pce = PCERegressor(
            degree=2,
            marginals="normal",
            copula="independent",
            q=1.0,
            max_features=5,
            allow_high_dim=True,
        )
        pce.fit(X, y)
        y_pred = pce.predict(X)

        assert r2_score(y, y_pred) > 0.98

    def test_allows_high_dim_when_enabled(self):
        """Should work with >5 features when allow_high_dim=True."""
        X = np.random.randn(100, 8)
        y = X[:, 0] + X[:, 1]

        pce = PCERegressor(
            degree=2,
            max_features=5,
            allow_high_dim=True,  # Override safety limit
        )
        pce.fit(X, y)  # Should not raise
        y_pred = pce.predict(X)
        
        assert len(y_pred) == 100
