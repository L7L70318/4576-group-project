import numpy as np
import pandas as pd

from project.asset_pricing import (
    TEST_ASSETS,
    _gmm_moments,
    _pack,
    automatic_nw_lags,
    fama_macbeth,
    long_run_covariance,
)


def test_long_run_covariance_is_symmetric():
    values = np.arange(60, dtype=float).reshape(20, 3)
    covariance = long_run_covariance(values, nlags=3)
    assert covariance.shape == (3, 3)
    assert np.allclose(covariance, covariance.T)


def test_fama_macbeth_uses_no_intercept_cross_section():
    rng = np.random.default_rng(4576)
    factor_values = rng.normal(size=(240, 2)) * 0.03
    factor_values -= factor_values.mean(axis=0)
    factors = pd.DataFrame(factor_values, columns=["F1", "F2"])
    betas = np.array([[0.5, -0.2], [1.0, 0.3], [-0.4, 1.2]])
    target_lambda = np.array([0.01, 0.02])
    factors += target_lambda
    assets = pd.DataFrame(factors.to_numpy() @ betas.T, columns=["A", "B", "C"])
    result = fama_macbeth(assets, factors)
    assert np.allclose(result.lambda_estimate, target_lambda, atol=1e-12)
    assert np.allclose(result.pricing_errors, 0, atol=1e-12)
    assert result.r2_ols == 1


def test_equation_four_moment_count_matches_paper():
    rng = np.random.default_rng(1)
    returns = rng.normal(size=(50, len(TEST_ASSETS)))
    factors = rng.normal(size=(50, 3))
    mean = factors.mean(axis=0)
    covariance = np.cov(factors, rowvar=False, ddof=0)
    theta = _pack(np.ones(3), mean, covariance)
    moments = _gmm_moments(theta, returns, factors)
    assert len(theta) == 12
    assert moments.shape == (50, 21)


def test_newey_west_lag_rule_is_deterministic():
    assert automatic_nw_lags(318) == 5
