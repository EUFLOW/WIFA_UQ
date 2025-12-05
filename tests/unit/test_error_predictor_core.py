# tests/unit/test_error_predictor_core.py
import numpy as np
from sklearn.metrics import r2_score

from wifa_uq.postprocessing.error_predictor.error_predictor import (
    SIRPolynomialRegressor,
    PCERegressor,
)


def test_sir_polynomial_regressor_learns_simple_relation():
    # y = 2 * x0 - 3 * x1 + noise
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 2))
    y = 2.0 * X[:, 0] - 3.0 * X[:, 1] + 0.1 * rng.normal(size=200)

    model = SIRPolynomialRegressor(n_directions=1, degree=1)
    model.fit(X, y)
    y_pred = model.predict(X)

    assert r2_score(y, y_pred) > 0.9


def test_pce_regressor_refuses_too_many_features():
    X = np.random.randn(50, 6)  # 6 > default max_features=5
    y = X[:, 0] ** 2

    pce = PCERegressor(degree=3, max_features=5, allow_high_dim=False)
    try:
        pce.fit(X, y)
    except ValueError as e:
        assert "refused to run" in str(e)
    else:
        raise AssertionError("Expected ValueError for too many features")


def test_pce_regressor_on_simple_polynomial():
    # 1D: y = x^2, PCE with degree 2 should learn it almost perfectly
    x = np.linspace(-1, 1, 30)
    y = x ** 2
    X = x.reshape(-1, 1)

    pce = PCERegressor(
        degree=2,
        marginals="normal",
        copula="independent",
        q=1.0,
        max_features=5,
        allow_high_dim=True,  # not needed here but explicit
    )
    pce.fit(X, y)
    y_pred = pce.predict(X)

    assert r2_score(y, y_pred) > 0.98

