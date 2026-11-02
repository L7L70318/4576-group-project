import numpy as np
import pandas as pd

from project.asset_pricing import (
    TEST_ASSETS,
    _gmm_moments,
    _pack,
    automatic_nw_lags,
    fama_macbeth,
    long_run_covariance,
    table8_equity_volatility_results,
    time_series_table,
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


def test_time_series_table_reports_fit_and_joint_alpha_test():
    rng = np.random.default_rng(4576)
    factors = pd.DataFrame(rng.normal(size=(240, 2)), columns=["AVG", "CARRY"])
    assets = pd.DataFrame(
        {
            "A": 0.01 + factors.AVG + rng.normal(scale=0.1, size=len(factors)),
            "B": -0.02 + factors.CARRY + rng.normal(scale=0.1, size=len(factors)),
            "C": factors.AVG - factors.CARRY + rng.normal(scale=0.1, size=len(factors)),
        }
    )

    table, joint = time_series_table(assets, factors, "test_model")

    assert list(table.asset) == ["A", "B", "C"]
    assert {"alpha", "beta_AVG", "beta_CARRY", "adjusted_r2_percent"} <= set(
        table.columns
    )
    assert table.n_months.eq(240).all()
    assert joint.loc[0, "joint_alpha_df"] == 3
    assert 0 <= joint.loc[0, "joint_alpha_pvalue"] <= 1


def test_table8_uses_rank_carry_legs_and_common_months():
    dates = pd.date_range("2000-01-31", periods=120, freq="ME")
    rng = np.random.default_rng(8)
    delta_vol = rng.normal(scale=0.002, size=len(dates))
    avg = rng.normal(scale=0.03, size=len(dates))
    carry_long = 0.01 - 2 * delta_vol + rng.normal(scale=0.02, size=len(dates))
    carry_short = rng.normal(scale=0.02, size=len(dates))
    carry = carry_long + carry_short
    assets = pd.DataFrame(
        rng.normal(size=(len(dates), 4)) * 0.03,
        index=dates,
        columns=["P1", "P2", "P3", "P4"],
    )
    factors = pd.DataFrame({"AVG": avg, "CARRY": carry}, index=dates)
    carry_returns = pd.DataFrame(
        {"C5": carry, "C5_long": carry_long, "C5_short": carry_short}, index=dates
    )
    equity = pd.DataFrame(
        {
            "vol_equity_g20_proxy": 0.01 + np.cumsum(delta_vol),
            "delta_vol_equity_g20_proxy": delta_vol,
            "countries_mean": 19.0,
            "countries_min": 18,
            "days": 21,
        },
        index=dates,
    )
    assets.loc[dates[0], "P1"] = np.nan

    result = table8_equity_volatility_results(assets, factors, carry_returns, equity)

    assert set(result["panel_a"].asset) == {"CARRY", "CARRY_LONG", "CARRY_SHORT"}
    assert result["panel_a"].n_months.eq(120).all()
    assert result["panel_b_model"].loc[0, "n_months"] == 119
    assert result["panel_b_pricing"].asset.tolist() == ["P1", "P2", "P3", "P4"]
