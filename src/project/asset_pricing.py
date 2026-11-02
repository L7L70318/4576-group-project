"""Baseline commodity asset-pricing replication for Section 5.1."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

from .carry_discrepancy import apply_paper_windows, construct_carry
from .data import construct_monthly, momentum_signal, monthly_snapshots
from .paths import (
    PROCESSING_EXPLORATION,
    REPLICATION_FIGURES,
    REPLICATION_TABLES,
    ROOT,
    load_config,
)
from .portfolios import make_portfolios

PAPER_START = pd.Timestamp("1970-01-31")
PAPER_END = pd.Timestamp("2011-09-30")
TEST_ASSETS = [
    "P1",
    "P2",
    "P3",
    "P4",
    "Q1",
    "Q2",
    "Q3",
    "Q4",
    "Q5",
    "Agriculture",
    "Livestock",
    "Metal",
]
FACTORS = ["AVG", "CARRY", "CMOM"]


PUBLISHED_TABLE3 = {
    "three_factor": {
        "lambda_AVG": 0.005,
        "lambda_CARRY": 0.018,
        "lambda_CMOM": 0.012,
        "b_AVG": 2.276,
        "b_CARRY": 3.954,
        "b_CMOM": 1.067,
        "r2_gls_percent": 96.3,
        "r2_ols_percent": 93.9,
        "chi2_nw": 11.91,
        "chi2_shanken": 11.40,
        "hj_distance": 0.006,
    },
    "omit_momentum": {
        "lambda_AVG": 0.005,
        "lambda_CARRY": 0.020,
        "b_AVG": 2.394,
        "b_CARRY": 4.661,
        "r2_gls_percent": 82.9,
        "r2_ols_percent": 91.3,
        "chi2_nw": 23.87,
        "chi2_shanken": 22.70,
        "hj_distance": 0.008,
    },
    "omit_carry": {
        "lambda_AVG": 0.006,
        "lambda_CMOM": 0.014,
        "b_AVG": 2.891,
        "b_CMOM": 2.300,
        "r2_gls_percent": 67.7,
        "r2_ols_percent": 77.3,
        "chi2_nw": 29.67,
        "chi2_shanken": 28.65,
        "hj_distance": 0.012,
    },
}


def automatic_nw_lags(nobs: int) -> int:
    """Common Newey-West (1994) automatic Bartlett-lag rule."""
    return int(np.floor(4 * (nobs / 100) ** (2 / 9)))


def long_run_covariance(values: np.ndarray, nlags: int | None = None) -> np.ndarray:
    """Bartlett-kernel long-run covariance of a T by K array."""
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    centered = values - values.mean(axis=0)
    nobs = len(centered)
    nlags = automatic_nw_lags(nobs) if nlags is None else nlags
    covariance = centered.T @ centered / nobs
    for lag in range(1, min(nlags, nobs - 1) + 1):
        weight = 1 - lag / (nlags + 1)
        gamma = centered[lag:].T @ centered[:-lag] / nobs
        covariance += weight * (gamma + gamma.T)
    return covariance


def _pvalues(estimates: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    standard_errors = np.sqrt(np.clip(np.diag(covariance), 0, None))
    statistics = np.divide(
        estimates,
        standard_errors,
        out=np.full_like(estimates, np.nan, dtype=float),
        where=standard_errors > 0,
    )
    return 2 * stats.norm.sf(np.abs(statistics))


def construct_baseline_returns(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Construct the paper's 12 test assets and rank-based primary factors."""
    returns, weights = make_portfolios(panel, carry_policy="rank")
    assets = returns[
        ["P1", "P2", "P3", "P4", "Q1", "Q2", "Q3", "Q4", "Q5",
         "sector_agriculture", "sector_livestock", "sector_metals"]
    ].rename(
        columns={
            "sector_agriculture": "Agriculture",
            "sector_livestock": "Livestock",
            "sector_metals": "Metal",
        }
    )
    factors = returns[["AVG", "C5", "M5"]].rename(
        columns={"C5": "CARRY", "M5": "CMOM"}
    )
    return assets[TEST_ASSETS], factors[FACTORS], weights


def coverage_table(frame: pd.DataFrame, kind: str) -> pd.DataFrame:
    rows = []
    calendar_months = len(pd.date_range(PAPER_START, PAPER_END, freq="ME"))
    for column in frame:
        values = frame[column].dropna()
        rows.append(
            {
                "kind": kind,
                "series": column,
                "valid_months": len(values),
                "calendar_months": calendar_months,
                "coverage_percent": len(values) / calendar_months * 100,
                "start": values.index.min() if len(values) else pd.NaT,
                "end": values.index.max() if len(values) else pd.NaT,
                "annualized_mean_percent": values.mean() * 1200 if len(values) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def factor_summary_table(factors: pd.DataFrame) -> pd.DataFrame:
    """Submission summary statistics for AVG, CARRY, and CMOM."""
    rows = []
    for factor in FACTORS:
        values = factors[factor].dropna()
        annualized_sd = values.std(ddof=1) * np.sqrt(12)
        rows.append(
            {
                "factor": factor,
                "n_months": len(values),
                "start": values.index.min(),
                "end": values.index.max(),
                "annualized_mean_percent": values.mean() * 1200,
                "annualized_sd_percent": annualized_sd * 100,
                "annualized_sharpe": values.mean() * 12 / annualized_sd,
                "monthly_skewness": values.skew(),
                "positive_months_percent": values.gt(0).mean() * 100,
            }
        )
    return pd.DataFrame(rows)


def construction_validation(
    panel: pd.DataFrame, weights: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Audit equal weights and ordering of P/Q portfolio memberships."""
    relevant = [*TEST_ASSETS, "AVG", "C5", "M5"]
    selected = weights[weights.portfolio.isin(relevant)].copy()
    monthly = (
        selected.groupby(["date", "portfolio"])
        .weight.agg(
            n_members="size",
            net_weight="sum",
            gross_weight=lambda values: values.abs().sum(),
            min_abs_weight=lambda values: values.abs().min(),
            max_abs_weight=lambda values: values.abs().max(),
        )
        .reset_index()
    )
    monthly["equal_absolute_weights"] = np.isclose(
        monthly.min_abs_weight, monthly.max_abs_weight, atol=1e-12
    )
    monthly["expected_net_weight"] = np.where(
        monthly.portfolio.isin(["C5", "M5"]), 0.0, 1.0
    )
    monthly["net_weight_valid"] = np.isclose(
        monthly.net_weight, monthly.expected_net_weight, atol=1e-12
    )

    signals = panel[["date", "paper_code", "slope"]].merge(
        momentum_signal(panel), on=["date", "paper_code"], how="left"
    )
    membership = selected.merge(signals, on=["date", "paper_code"], how="left")
    checks = []
    for date, group in membership.groupby("date"):
        portfolios = {name: values for name, values in group.groupby("portfolio")}
        for left, right in [("P1", "P2"), ("P2", "P3"), ("P3", "P4")]:
            if left in portfolios and right in portfolios:
                checks.append(
                    {
                        "date": date,
                        "check": f"{left}_slope_le_{right}",
                        "valid": portfolios[left].slope.max()
                        <= portfolios[right].slope.min() + 1e-12,
                    }
                )
        for left, right in [(f"Q{k}", f"Q{k + 1}") for k in range(1, 5)]:
            if left in portfolios and right in portfolios:
                checks.append(
                    {
                        "date": date,
                        "check": f"{left}_momentum_le_{right}",
                        "valid": portfolios[left].momentum.max()
                        <= portfolios[right].momentum.min() + 1e-12,
                    }
                )
        if "P1" in portfolios and "P2" in portfolios:
            checks.append(
                {
                    "date": date,
                    "check": "backwardation_portfolios_negative",
                    "valid": max(
                        portfolios["P1"].slope.max(), portfolios["P2"].slope.max()
                    )
                    < 0,
                }
            )
        if "P3" in portfolios and "P4" in portfolios:
            checks.append(
                {
                    "date": date,
                    "check": "contango_portfolios_positive",
                    "valid": min(
                        portfolios["P3"].slope.min(), portfolios["P4"].slope.min()
                    )
                    > 0,
                }
            )
    checks = pd.DataFrame(checks)
    sorting_summary = (
        checks.groupby("check")
        .valid.agg(months_checked="size", violations=lambda values: (~values).sum())
        .reset_index()
    )
    return monthly, sorting_summary


@dataclass
class FamaMacBethResult:
    factors: list[str]
    betas: np.ndarray
    alphas_ts: np.ndarray
    lambda_estimate: np.ndarray
    covariance_nw: np.ndarray
    covariance_shanken: np.ndarray
    fitted: np.ndarray
    pricing_errors: np.ndarray
    r2_ols: float
    r2_gls: float
    chi2_nw: float
    chi2_nw_pvalue: float
    chi2_shanken: float
    chi2_shanken_pvalue: float
    nw_lags: int


def time_series_regressions(
    assets: pd.DataFrame, factors: pd.DataFrame, model: str
) -> pd.DataFrame:
    """Estimate first-pass alphas/betas with Newey-West inference."""
    coefficients, _ = time_series_table(assets, factors, model)
    parameter_columns = ["alpha", *[f"beta_{factor}" for factor in factors]]
    rows = []
    for result in coefficients.itertuples(index=False):
        for parameter in parameter_columns:
            rows.append(
                {
                    "model": model,
                    "asset": result.asset,
                    "parameter": parameter,
                    "estimate_monthly": getattr(result, parameter),
                    "nw_standard_error": getattr(result, f"{parameter}_nw_standard_error"),
                    "nw_pvalue": getattr(result, f"{parameter}_nw_pvalue"),
                    "n_months": result.n_months,
                    "nw_lags": result.nw_lags,
                    "adjusted_r2_percent": result.adjusted_r2_percent,
                }
            )
    return pd.DataFrame(rows)


def time_series_table(
    assets: pd.DataFrame, factors: pd.DataFrame, model: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build paper-style time-series results and the joint alpha test.

    The joint test uses the HAC covariance of the stacked OLS intercept
    influence functions, retaining contemporaneous and serial correlation
    across all test-asset residuals.
    """
    factor_values = factors.to_numpy(float)
    design = np.column_stack([np.ones(len(factors)), factor_values])
    nobs = len(design)
    nparameters = design.shape[1]
    nw_lags = automatic_nw_lags(nobs)
    bread = np.linalg.pinv(design.T @ design / nobs)
    rows = []
    residual_columns = []
    alphas = []
    for asset in assets:
        values = assets[asset].to_numpy(float)
        estimates = np.linalg.lstsq(design, values, rcond=None)[0]
        residuals = values - design @ estimates
        residual_columns.append(residuals)
        alphas.append(estimates[0])
        scores = design * residuals[:, None]
        covariance = bread @ long_run_covariance(scores, nw_lags) @ bread / nobs
        standard_errors = np.sqrt(np.clip(np.diag(covariance), 0, None))
        statistics = np.divide(
            estimates,
            standard_errors,
            out=np.full_like(estimates, np.nan),
            where=standard_errors > 0,
        )
        pvalues = 2 * stats.norm.sf(np.abs(statistics))
        total_sum_squares = np.sum((values - values.mean()) ** 2)
        residual_sum_squares = residuals @ residuals
        adjusted_r2 = (
            1 - (residual_sum_squares / (nobs - nparameters))
            / (total_sum_squares / (nobs - 1))
            if total_sum_squares > 0 and nobs > nparameters
            else np.nan
        )
        row = {
            "model": model,
            "asset": asset,
            "n_months": nobs,
            "nw_lags": nw_lags,
            "adjusted_r2_percent": adjusted_r2 * 100,
        }
        for index, parameter in enumerate(["alpha", *[f"beta_{x}" for x in factors]]):
            row[parameter] = estimates[index]
            row[f"{parameter}_nw_standard_error"] = standard_errors[index]
            row[f"{parameter}_nw_pvalue"] = pvalues[index]
        rows.append(row)

    residual_matrix = np.column_stack(residual_columns)
    alpha_influence = residual_matrix * (design @ bread[:, 0])[:, None]
    alpha_covariance = long_run_covariance(alpha_influence, nw_lags) / nobs
    alpha_values = np.asarray(alphas)
    joint_alpha_chi2 = float(
        alpha_values @ np.linalg.pinv(alpha_covariance) @ alpha_values
    )
    joint = pd.DataFrame(
        [
            {
                "model": model,
                "n_months": nobs,
                "nw_lags": nw_lags,
                "joint_alpha_chi2": joint_alpha_chi2,
                "joint_alpha_df": len(alpha_values),
                "joint_alpha_pvalue": stats.chi2.sf(
                    joint_alpha_chi2, len(alpha_values)
                ),
            }
        ]
    )
    return pd.DataFrame(rows), joint


def table8_equity_volatility_results(
    assets: pd.DataFrame,
    factors: pd.DataFrame,
    carry_returns: pd.DataFrame,
    equity_volatility: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Reconstruct Table 8 with rank Carry and the 19-country equity-vol proxy."""
    equity = equity_volatility.copy()
    equity.index = pd.to_datetime(equity.index)
    equity.index.name = "date"
    delta_name = "DELTA_VOL_EQUITY"
    delta = equity["delta_vol_equity_g20_proxy"].rename(delta_name)

    panel_a_sample = pd.concat(
        [
            carry_returns[["C5", "C5_long", "C5_short"]].rename(
                columns={
                    "C5": "CARRY",
                    "C5_long": "CARRY_LONG",
                    "C5_short": "CARRY_SHORT",
                }
            ),
            delta,
            equity[["vol_equity_g20_proxy", "countries_mean", "countries_min", "days"]],
        ],
        axis=1,
    ).dropna(subset=["CARRY", "CARRY_LONG", "CARRY_SHORT", delta_name])
    panel_a, _ = time_series_table(
        panel_a_sample[["CARRY", "CARRY_LONG", "CARRY_SHORT"]],
        panel_a_sample[[delta_name]],
        "rank_carry_on_equity_volatility_innovation",
    )
    panel_a.insert(0, "carry_policy", "unconditional_rank_bottom_top_5")

    panel_b_sample = pd.concat(
        [assets[["P1", "P2", "P3", "P4"]], factors[["AVG"]], delta, equity], axis=1
    ).dropna(subset=["P1", "P2", "P3", "P4", "AVG", delta_name])
    panel_b_assets = panel_b_sample[["P1", "P2", "P3", "P4"]]
    panel_b_factors = panel_b_sample[["AVG", delta_name]]
    fmb = fama_macbeth(panel_b_assets, panel_b_factors)
    nw_pvalues = _pvalues(fmb.lambda_estimate, fmb.covariance_nw)
    shanken_pvalues = _pvalues(fmb.lambda_estimate, fmb.covariance_shanken)
    panel_b_model = pd.DataFrame(
        [
            {
                "carry_policy": "unconditional_rank_bottom_top_5",
                "equity_volatility_proxy": "19_country_g20_available_country_mean",
                "n_months": len(panel_b_sample),
                "start": panel_b_sample.index.min(),
                "end": panel_b_sample.index.max(),
                "nw_lags": fmb.nw_lags,
                "lambda_AVG_monthly": fmb.lambda_estimate[0],
                "lambda_AVG_nw_pvalue": nw_pvalues[0],
                "lambda_AVG_shanken_pvalue": shanken_pvalues[0],
                "lambda_DELTA_VOL_EQUITY_monthly": fmb.lambda_estimate[1],
                "lambda_DELTA_VOL_EQUITY_nw_pvalue": nw_pvalues[1],
                "lambda_DELTA_VOL_EQUITY_shanken_pvalue": shanken_pvalues[1],
                "r2_ols_percent": fmb.r2_ols * 100,
                "r2_gls_percent": fmb.r2_gls * 100,
                "pricing_error_chi2_nw": fmb.chi2_nw,
                "pricing_error_chi2_nw_pvalue": fmb.chi2_nw_pvalue,
                "pricing_error_chi2_shanken": fmb.chi2_shanken,
                "pricing_error_chi2_shanken_pvalue": fmb.chi2_shanken_pvalue,
            }
        ]
    )
    panel_b_pricing = pd.DataFrame(
        {
            "asset": panel_b_assets.columns,
            "realized_monthly": panel_b_assets.mean().to_numpy(),
            "fitted_monthly": fmb.fitted,
            "pricing_error_monthly": fmb.pricing_errors,
        }
    )
    panel_b_pricing.insert(0, "carry_policy", "unconditional_rank_bottom_top_5")
    for column in ["realized_monthly", "fitted_monthly", "pricing_error_monthly"]:
        panel_b_pricing[column.replace("_monthly", "_annualized_percent")] = (
            panel_b_pricing[column] * 1200
        )

    sample_coverage = pd.DataFrame(
        [
            {
                "panel": "A_time_series",
                "n_months": len(panel_a_sample),
                "start": panel_a_sample.index.min(),
                "end": panel_a_sample.index.max(),
                "countries_mean_average": panel_a_sample.countries_mean.mean(),
                "countries_mean_minimum": panel_a_sample.countries_mean.min(),
                "countries_mean_maximum": panel_a_sample.countries_mean.max(),
            },
            {
                "panel": "B_cross_section",
                "n_months": len(panel_b_sample),
                "start": panel_b_sample.index.min(),
                "end": panel_b_sample.index.max(),
                "countries_mean_average": panel_b_sample.countries_mean.mean(),
                "countries_mean_minimum": panel_b_sample.countries_mean.min(),
                "countries_mean_maximum": panel_b_sample.countries_mean.max(),
            },
        ]
    )
    return {
        "panel_a": panel_a,
        "panel_a_sample": panel_a_sample,
        "panel_b_model": panel_b_model,
        "panel_b_pricing": panel_b_pricing,
        "panel_b_sample": panel_b_sample,
        "sample_coverage": sample_coverage,
    }


def fama_macbeth(assets: pd.DataFrame, factors: pd.DataFrame) -> FamaMacBethResult:
    """Two-pass beta pricing with a no-intercept second-pass regression."""
    returns = assets.to_numpy(float)
    factor_values = factors.to_numpy(float)
    nobs, nassets = returns.shape
    design = np.column_stack([np.ones(nobs), factor_values])
    coefficients = np.linalg.lstsq(design, returns, rcond=None)[0]
    alphas_ts = coefficients[0]
    betas = coefficients[1:].T

    mean_returns = returns.mean(axis=0)
    inverse_betas = np.linalg.pinv(betas.T @ betas) @ betas.T
    lambda_estimate = inverse_betas @ mean_returns
    fitted = betas @ lambda_estimate
    pricing_errors = mean_returns - fitted

    monthly_lambdas = returns @ inverse_betas.T
    nw_lags = automatic_nw_lags(nobs)
    covariance_nw = long_run_covariance(monthly_lambdas, nw_lags) / nobs

    factor_covariance = np.cov(factor_values, rowvar=False, ddof=0)
    if factor_values.shape[1] == 1:
        factor_covariance = np.atleast_2d(factor_covariance)
    time_series_residuals = returns - design @ coefficients
    residual_lrcov = long_run_covariance(time_series_residuals, nw_lags)
    residual_lambda_covariance = inverse_betas @ residual_lrcov @ inverse_betas.T / nobs
    shanken_multiplier = 1 + lambda_estimate @ np.linalg.pinv(factor_covariance) @ lambda_estimate
    covariance_shanken = (
        shanken_multiplier * residual_lambda_covariance + factor_covariance / nobs
    )

    projection = betas @ inverse_betas
    pricing_error_series = returns @ (np.eye(nassets) - projection).T
    covariance_alpha_nw = long_run_covariance(pricing_error_series, nw_lags) / nobs
    covariance_alpha_shanken = shanken_multiplier * covariance_alpha_nw
    rank = max(nassets - len(lambda_estimate), 1)
    chi2_nw = float(pricing_errors @ np.linalg.pinv(covariance_alpha_nw) @ pricing_errors)
    chi2_shanken = float(
        pricing_errors @ np.linalg.pinv(covariance_alpha_shanken) @ pricing_errors
    )

    denominator_ols = mean_returns @ mean_returns
    r2_ols = 1 - pricing_errors @ pricing_errors / denominator_ols
    # Lewellen-Nagel-Shanken Prescription 3 weights the cross section by the
    # covariance matrix of first-pass time-series residuals.
    residual_covariance = np.cov(time_series_residuals, rowvar=False, ddof=1)
    gls_weight = np.linalg.pinv(residual_covariance)
    denominator_gls = mean_returns @ gls_weight @ mean_returns
    r2_gls = 1 - pricing_errors @ gls_weight @ pricing_errors / denominator_gls

    return FamaMacBethResult(
        factors=list(factors.columns),
        betas=betas,
        alphas_ts=alphas_ts,
        lambda_estimate=lambda_estimate,
        covariance_nw=covariance_nw,
        covariance_shanken=covariance_shanken,
        fitted=fitted,
        pricing_errors=pricing_errors,
        r2_ols=float(r2_ols),
        r2_gls=float(r2_gls),
        chi2_nw=chi2_nw,
        chi2_nw_pvalue=float(stats.chi2.sf(chi2_nw, rank)),
        chi2_shanken=chi2_shanken,
        chi2_shanken_pvalue=float(stats.chi2.sf(chi2_shanken, rank)),
        nw_lags=nw_lags,
    )


def _vech(matrix: np.ndarray) -> np.ndarray:
    return matrix[np.triu_indices_from(matrix)]


def _unvech(values: np.ndarray, size: int) -> np.ndarray:
    matrix = np.zeros((size, size))
    indices = np.triu_indices(size)
    matrix[indices] = values
    matrix[(indices[1], indices[0])] = values
    return matrix


def _pack(b: np.ndarray, mean: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    return np.concatenate([b, mean, _vech(covariance)])


def _unpack(theta: np.ndarray, nfactors: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    b = theta[:nfactors]
    mean = theta[nfactors:2 * nfactors]
    covariance = _unvech(theta[2 * nfactors:], nfactors)
    return b, mean, covariance


def _gmm_moments(theta: np.ndarray, returns: np.ndarray, factors: np.ndarray) -> np.ndarray:
    nfactors = factors.shape[1]
    b, mean, covariance = _unpack(theta, nfactors)
    centered = factors - mean
    sdf = 1 - centered @ b
    pricing = returns * sdf[:, None]
    covariance_moments = np.column_stack(
        [
            centered[:, i] * centered[:, j] - covariance[i, j]
            for i in range(nfactors)
            for j in range(i, nfactors)
        ]
    )
    return np.column_stack([pricing, centered, covariance_moments])


def _jacobian(function, theta: np.ndarray) -> np.ndarray:
    baseline = function(theta)
    result = np.empty((len(baseline), len(theta)))
    for index in range(len(theta)):
        step = 1e-6 * max(1.0, abs(theta[index]))
        plus = theta.copy()
        minus = theta.copy()
        plus[index] += step
        minus[index] -= step
        result[:, index] = (function(plus) - function(minus)) / (2 * step)
    return result


@dataclass
class GMMResult:
    factors: list[str]
    step: str
    b: np.ndarray
    lambda_estimate: np.ndarray
    covariance_b: np.ndarray
    covariance_lambda: np.ndarray
    objective: float
    j_statistic: float
    j_pvalue: float
    j_df: int
    hj_distance: float
    hj_pvalue: float
    converged: bool
    nw_lags: int


def estimate_gmm(
    assets: pd.DataFrame,
    factors: pd.DataFrame,
) -> tuple[GMMResult, GMMResult]:
    """Estimate Equation (4) by identity and optimal-weight two-step GMM."""
    returns = assets.to_numpy(float)
    factor_values = factors.to_numpy(float)
    nobs, nassets = returns.shape
    nfactors = factor_values.shape[1]
    factor_mean = factor_values.mean(axis=0)
    centered = factor_values - factor_mean
    factor_covariance = centered.T @ centered / nobs
    return_factor_covariance = returns.T @ centered / nobs
    b_start = np.linalg.pinv(return_factor_covariance) @ returns.mean(axis=0)
    theta_start = _pack(b_start, factor_mean, factor_covariance)
    nw_lags = automatic_nw_lags(nobs)

    def mean_moments(theta):
        return _gmm_moments(theta, returns, factor_values).mean(axis=0)

    first = optimize.least_squares(
        mean_moments,
        theta_start,
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
        max_nfev=20_000,
    )
    theta_first = first.x
    first_moment_values = _gmm_moments(theta_first, returns, factor_values)
    long_run = long_run_covariance(first_moment_values, nw_lags)
    optimal_weight = np.linalg.pinv(long_run, rcond=1e-10)

    def optimal_objective(theta):
        moments = mean_moments(theta)
        return float(moments @ optimal_weight @ moments)

    eigenvalues, eigenvectors = np.linalg.eigh((optimal_weight + optimal_weight.T) / 2)
    positive = eigenvalues > eigenvalues.max() * 1e-10
    square_root_weight = (
        np.sqrt(eigenvalues[positive])[:, None] * eigenvectors[:, positive].T
    )
    second = optimize.least_squares(
        lambda theta: square_root_weight @ mean_moments(theta),
        theta_first,
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
        max_nfev=20_000,
    )

    def summarize(theta, weight, step, converged):
        moments = _gmm_moments(theta, returns, factor_values)
        mean_values = moments.mean(axis=0)
        moment_covariance = long_run_covariance(moments, nw_lags)
        derivative = _jacobian(mean_moments, theta)
        bread = np.linalg.pinv(derivative.T @ weight @ derivative)
        covariance_theta = (
            bread
            @ derivative.T
            @ weight
            @ moment_covariance
            @ weight
            @ derivative
            @ bread
            / nobs
        )
        b, _, covariance = _unpack(theta, nfactors)
        lambda_estimate = covariance @ b

        def lambda_function(candidate):
            candidate_b, _, candidate_covariance = _unpack(candidate, nfactors)
            return candidate_covariance @ candidate_b

        lambda_derivative = _jacobian(lambda_function, theta)
        covariance_lambda = lambda_derivative @ covariance_theta @ lambda_derivative.T
        objective = float(mean_values @ weight @ mean_values)
        j_statistic = nobs * objective
        degrees_freedom = nassets - nfactors

        return GMMResult(
            factors=list(factors.columns),
            step=step,
            b=b,
            lambda_estimate=lambda_estimate,
            covariance_b=covariance_theta[:nfactors, :nfactors],
            covariance_lambda=covariance_lambda,
            objective=objective,
            j_statistic=j_statistic,
            j_pvalue=float(stats.chi2.sf(j_statistic, degrees_freedom)),
            j_df=degrees_freedom,
            # Deliberately omitted until the Hansen-Jagannathan boundary-null
            # inference (a weighted chi-square law, not the GMM J-test) is
            # implemented and independently validated.
            hj_distance=np.nan,
            hj_pvalue=np.nan,
            converged=bool(converged),
            nw_lags=nw_lags,
        )

    identity = np.eye(first_moment_values.shape[1])
    return (
        summarize(theta_first, identity, "one_step_identity", first.success),
        summarize(second.x, optimal_weight, "two_step_optimal", second.success),
    )


def _model_rows(
    name: str,
    fmb: FamaMacBethResult,
    one_step: GMMResult,
    two_step: GMMResult,
) -> list[dict]:
    rows = []
    fmb_nw_pvalues = _pvalues(fmb.lambda_estimate, fmb.covariance_nw)
    fmb_shanken_pvalues = _pvalues(fmb.lambda_estimate, fmb.covariance_shanken)
    for index, factor in enumerate(fmb.factors):
        rows.append(
            {
                "model": name,
                "method": "fama_macbeth",
                "parameter": f"lambda_{factor}",
                "estimate_monthly": fmb.lambda_estimate[index],
                "annualized_percent": fmb.lambda_estimate[index] * 1200,
                "pvalue_nw": fmb_nw_pvalues[index],
                "pvalue_shanken": fmb_shanken_pvalues[index],
            }
        )
    for result in [one_step, two_step]:
        b_pvalues = _pvalues(result.b, result.covariance_b)
        lambda_pvalues = _pvalues(result.lambda_estimate, result.covariance_lambda)
        for index, factor in enumerate(result.factors):
            rows.extend(
                [
                    {
                        "model": name,
                        "method": result.step,
                        "parameter": f"lambda_{factor}",
                        "estimate_monthly": result.lambda_estimate[index],
                        "annualized_percent": result.lambda_estimate[index] * 1200,
                        "pvalue_nw": lambda_pvalues[index],
                        "pvalue_shanken": np.nan,
                    },
                    {
                        "model": name,
                        "method": result.step,
                        "parameter": f"b_{factor}",
                        "estimate_monthly": result.b[index],
                        "annualized_percent": np.nan,
                        "pvalue_nw": b_pvalues[index],
                        "pvalue_shanken": np.nan,
                    },
                ]
            )
    return rows


def _fit_rows(
    name: str,
    fmb: FamaMacBethResult,
    one_step: GMMResult,
    two_step: GMMResult,
) -> dict:
    return {
        "model": name,
        "n_months": np.nan,
        "nw_lags": fmb.nw_lags,
        "fmb_r2_gls_percent": fmb.r2_gls * 100,
        "fmb_r2_ols_percent": fmb.r2_ols * 100,
        "fmb_chi2_nw": fmb.chi2_nw,
        "fmb_chi2_nw_pvalue": fmb.chi2_nw_pvalue,
        "fmb_chi2_shanken": fmb.chi2_shanken,
        "fmb_chi2_shanken_pvalue": fmb.chi2_shanken_pvalue,
        "gmm_one_step_hj_distance": one_step.hj_distance,
        "gmm_hj_pvalue": np.nan,
        "gmm_two_step_j": two_step.j_statistic,
        "gmm_two_step_j_df": two_step.j_df,
        "gmm_two_step_j_pvalue": two_step.j_pvalue,
        "gmm_one_step_converged": one_step.converged,
        "gmm_two_step_converged": two_step.converged,
    }


def _published_comparison(parameter_rows: pd.DataFrame, fit_rows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, published in PUBLISHED_TABLE3.items():
        for field, published_value in published.items():
            if field.startswith(("lambda_", "b_")):
                reconstructed = parameter_rows[
                    parameter_rows.model.eq(model)
                    & parameter_rows.method.eq(
                        "fama_macbeth" if field.startswith("lambda_") else "one_step_identity"
                    )
                    & parameter_rows.parameter.eq(field)
                ].estimate_monthly
                value = reconstructed.iloc[0] if len(reconstructed) else np.nan
            else:
                fit = fit_rows[fit_rows.model.eq(model)].iloc[0]
                mapping = {
                    "r2_gls_percent": "fmb_r2_gls_percent",
                    "r2_ols_percent": "fmb_r2_ols_percent",
                    "chi2_nw": "fmb_chi2_nw",
                    "chi2_shanken": "fmb_chi2_shanken",
                    "hj_distance": "gmm_one_step_hj_distance",
                }
                value = fit[mapping[field]]
            rows.append(
                {
                    "model": model,
                    "statistic": field,
                    "published": published_value,
                    "reconstructed": value,
                    "difference": value - published_value,
                }
            )
    return pd.DataFrame(rows)


def _figure_two(
    assets: pd.DataFrame,
    fitted: np.ndarray,
    output: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    realized = assets.mean().to_numpy() * 12
    fitted_annual = fitted * 12
    lower = min(realized.min(), fitted_annual.min())
    upper = max(realized.max(), fitted_annual.max())
    padding = (upper - lower) * 0.08 or 0.01
    figure, axis = plt.subplots(figsize=(8, 7))
    axis.plot([lower - padding, upper + padding], [lower - padding, upper + padding], "k--")
    axis.scatter(realized, fitted_annual, color="#1f77b4")
    for label, x_value, y_value in zip(TEST_ASSETS, realized, fitted_annual):
        axis.annotate(label, (x_value, y_value), xytext=(4, 4), textcoords="offset points")
    axis.set_xlabel("Realized annualized average excess return")
    axis.set_ylabel("Fitted annualized average excess return")
    axis.set_title("Figure 2 replication: realized versus fitted returns")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def build_asset_pricing_replication(folder=None) -> dict[str, pd.DataFrame]:
    """Build the Phase 1 cross-sectional and equity-volatility outputs."""
    if folder is None:
        sample = load_config()["sample"]
        folder = ROOT / "data/raw/wrds_contracts" / (
            f"{sample['download_start']}_{sample['proposed_wrds_end']}"
        )
    output = REPLICATION_TABLES / "asset_pricing"
    figure_output = REPLICATION_FIGURES / "asset_pricing"
    exploration_output = PROCESSING_EXPLORATION / "results" / "asset_pricing"
    table8_exploration_output = (
        PROCESSING_EXPLORATION / "results" / "equity_volatility"
    )
    diagnostic_output = exploration_output / "time_series_diagnostics"
    output.mkdir(parents=True, exist_ok=True)
    figure_output.mkdir(parents=True, exist_ok=True)
    exploration_output.mkdir(parents=True, exist_ok=True)
    table8_exploration_output.mkdir(parents=True, exist_ok=True)
    diagnostic_output.mkdir(parents=True, exist_ok=True)

    snapshots, _, _ = monthly_snapshots(folder, prefer_actual_notice=True)
    panel = construct_monthly(snapshots, max_quote_age=7)
    panel = panel[panel.date.between(PAPER_START, PAPER_END)]
    panel = apply_paper_windows(panel)
    assets, factors, weights = construct_baseline_returns(panel)
    assets.to_csv(exploration_output / "monthly_test_asset_returns.csv")
    factors.to_csv(output / "monthly_factor_returns.csv")
    factor_summary_table(factors).to_csv(
        output / "three_factor_summary_statistics.csv", index=False
    )
    weights.to_csv(exploration_output / "monthly_portfolio_weights.csv", index=False)
    weight_validation, sorting_validation = construction_validation(panel, weights)
    weight_validation.to_csv(
        exploration_output / "portfolio_weight_validation.csv", index=False
    )
    sorting_validation.to_csv(
        exploration_output / "sorting_validation_summary.csv", index=False
    )

    rank_carry, _ = construct_carry(panel, "rank")
    rank_factors = factors
    rank_combined_expanded = pd.concat([assets, rank_factors], axis=1).dropna()
    rank_assets_expanded = rank_combined_expanded[TEST_ASSETS]
    rank_factors_expanded = rank_combined_expanded[FACTORS]

    equity_path = ROOT / "data/processed/equity_volatility_monthly.parquet"
    if not equity_path.exists():
        from .auxiliary import prepare_equities

        prepare_equities()
    table8_results = table8_equity_volatility_results(
        assets,
        rank_factors,
        rank_carry,
        pd.read_parquet(equity_path),
    )
    table8_output = REPLICATION_TABLES / "equity_volatility"
    table8_output.mkdir(parents=True, exist_ok=True)
    table8_results["panel_a"].to_csv(
        table8_output / "table8_panel_a_time_series.csv", index=False
    )
    table8_results["panel_a_sample"].to_csv(
        table8_exploration_output / "table8_panel_a_common_sample.csv"
    )
    table8_results["panel_b_model"].to_csv(
        table8_output / "table8_panel_b_cross_sectional_model.csv", index=False
    )
    table8_results["panel_b_pricing"].to_csv(
        table8_output / "table8_panel_b_pricing_errors.csv", index=False
    )
    table8_results["panel_b_sample"].to_csv(
        table8_exploration_output / "table8_panel_b_common_sample.csv"
    )
    table8_results["sample_coverage"].to_csv(
        table8_output / "table8_sample_and_country_coverage.csv", index=False
    )

    coverage = pd.concat(
        [coverage_table(assets, "test_asset"), coverage_table(factors, "factor")],
        ignore_index=True,
    )
    coverage.to_csv(
        exploration_output / "portfolio_and_factor_coverage.csv", index=False
    )

    combined = pd.concat([assets, factors], axis=1).dropna()
    common_assets = combined[TEST_ASSETS]
    common_factors = combined[FACTORS]
    combined.to_csv(exploration_output / "common_regression_sample.csv")
    pd.DataFrame(
        [
            {
                "specification": "unconditional_rank_bottom_top_5",
                "start": combined.index.min(),
                "end": combined.index.max(),
                "months": len(combined),
                "calendar_months_paper": 501,
                "nw_lags": automatic_nw_lags(len(combined)),
            }
        ]
    ).to_csv(output / "common_sample_summary.csv", index=False)

    models = {
        "three_factor": ["AVG", "CARRY", "CMOM"],
        "omit_momentum": ["AVG", "CARRY"],
        "omit_carry": ["AVG", "CMOM"],
        "avg_only": ["AVG"],
    }
    parameter_rows = []
    fit_rows = []
    fmb_results = {}
    time_series_frames = []
    pricing_rows = []
    for name, factor_names in models.items():
        selected_factors = common_factors[factor_names]
        fmb = fama_macbeth(common_assets, selected_factors)
        one_step, two_step = estimate_gmm(common_assets, selected_factors)
        fmb_results[name] = fmb
        parameter_rows.extend(_model_rows(name, fmb, one_step, two_step))
        fit = _fit_rows(name, fmb, one_step, two_step)
        fit["n_months"] = len(combined)
        fit_rows.append(fit)
        time_series_frames.append(time_series_regressions(common_assets, selected_factors, name))
        for asset_index, asset in enumerate(TEST_ASSETS):
            pricing_rows.append(
                {
                    "model": name,
                    "asset": asset,
                    "realized_monthly": common_assets[asset].mean(),
                    "fitted_monthly": fmb.fitted[asset_index],
                    "pricing_error_monthly": fmb.pricing_errors[asset_index],
                    "realized_annualized_percent": common_assets[asset].mean() * 1200,
                    "fitted_annualized_percent": fmb.fitted[asset_index] * 1200,
                    "pricing_error_annualized_percent": fmb.pricing_errors[asset_index] * 1200,
                }
            )

    parameter_frame = pd.DataFrame(parameter_rows)
    fit_frame = pd.DataFrame(fit_rows)
    parameter_frame.to_csv(output / "table3_parameter_estimates.csv", index=False)
    fit_frame.to_csv(output / "table3_model_fit_and_tests.csv", index=False)
    pd.concat(time_series_frames, ignore_index=True).to_csv(
        exploration_output / "time_series_betas.csv", index=False
    )
    table6, table6_joint = time_series_table(
        rank_assets_expanded, rank_factors_expanded[FACTORS], "three_factor"
    )
    table7_two_factor, table7_two_factor_joint = time_series_table(
        rank_assets_expanded,
        rank_factors_expanded[["AVG", "CARRY"]],
        "omit_momentum",
    )
    table7_carry_only, table7_carry_only_joint = time_series_table(
        rank_assets_expanded, rank_factors_expanded[["CARRY"]], "carry_only"
    )
    for frame in [table6, table6_joint, table7_two_factor, table7_two_factor_joint,
                  table7_carry_only, table7_carry_only_joint]:
        frame.insert(0, "carry_policy", "unconditional_rank_bottom_top_5")
    table6.to_csv(diagnostic_output / "table6_time_series_three_factor.csv", index=False)
    table6_joint.to_csv(diagnostic_output / "table6_joint_alpha_test.csv", index=False)
    pd.concat([table7_two_factor, table7_carry_only], ignore_index=True).to_csv(
        diagnostic_output / "table7_time_series_restricted_models.csv", index=False
    )
    pd.concat(
        [table7_two_factor_joint, table7_carry_only_joint], ignore_index=True
    ).to_csv(diagnostic_output / "table7_joint_alpha_tests.csv", index=False)
    pricing_frame = pd.DataFrame(pricing_rows)
    pricing_frame.to_csv(output / "realized_fitted_and_pricing_errors.csv", index=False)
    _published_comparison(parameter_frame, fit_frame).to_csv(
        output / "table3_published_vs_reconstructed.csv", index=False
    )

    primary_pricing = pricing_frame[pricing_frame.model.eq("three_factor")]
    primary_mae = primary_pricing.pricing_error_annualized_percent.abs().mean()
    pd.DataFrame(
        [
            {
                "model": "three_factor",
                "annualized_mae_percent": primary_mae,
                "published_figure2_annualized_mae_percent": 1.84,
            }
        ]
    ).to_csv(output / "figure2_fit_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "status": "not_implemented",
                "reason": (
                    "HJ distance and p-value withheld until the nonstandard "
                    "boundary-null weighted-chi-square inference is independently validated"
                ),
                "gmm_j_test_is_not_hj_test": True,
            }
        ]
    ).to_csv(exploration_output / "hj_distance_status.csv", index=False)
    _figure_two(
        common_assets,
        fmb_results["three_factor"].fitted,
        figure_output / "figure2_realized_vs_fitted.png",
    )
    return {
        "assets": assets,
        "factors": factors,
        "common": combined,
        "parameters": parameter_frame,
        "fit": fit_frame,
        "pricing": pricing_frame,
    }
