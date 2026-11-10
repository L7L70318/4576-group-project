"""Null-imposing bootstrap and Monte Carlo size calibration for GMM J-tests."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from project.asset_pricing import FACTORS, TEST_ASSETS, estimate_gmm
from project.extension_analysis import concentration
from project.paths import EXTENSION_FIGURES, EXTENSION_TABLES

STABILITY_TABLES = EXTENSION_TABLES / "01_instability"
TABLES = EXTENSION_TABLES / "03_mechanism"
FIGURES = EXTENSION_FIGURES / "03_mechanism"
SEED = 4576
BOOTSTRAP_DRAWS = 499
MONTE_CARLO_DRAWS = 499
ROLLING_BOOTSTRAP_DRAWS = 199
BLOCK_LENGTHS = [12, 24]
REDUCED_ASSETS = [asset for asset in TEST_ASSETS if asset not in {"P1", "P2"}]
MODELS = {
    "all_12_assets": TEST_ASSETS,
    "exclude_P1_P2": REDUCED_ASSETS,
}


def load_common_sample():
    panel = pd.read_csv(TABLES / "hhi_pricing_moment_panel.csv", parse_dates=["date"])
    assets = panel.pivot(index="date", columns="asset", values="u")
    # The u panel cannot recover returns. Rebuild from the reproducible source.
    _, returns, factors, _ = concentration.extended_inputs()
    common = pd.concat([returns[TEST_ASSETS], factors[FACTORS]], axis=1).dropna()
    assert common.index.equals(assets.index)
    characteristics = pd.read_csv(
        TABLES / "hhi_portfolio_characteristics.csv", parse_dates=["date"]
    )
    hhi = characteristics.pivot(index="date", columns="asset", values="hhi")
    return common, hhi.reindex(common.index)[TEST_ASSETS]


def target_windows(common):
    rolling = pd.read_csv(
        STABILITY_TABLES / "rolling_gmm_j_tests.csv", parse_dates=["start", "end"]
    )
    targets = []
    definitions = [
        ("rejection_ending_2014_02", pd.Timestamp("2014-02-28")),
        ("post_rejection_ending_2024_02", pd.Timestamp("2024-02-29")),
    ]
    for label, end_date in definitions:
        for window in [120, 180]:
            row = rolling[rolling.window_months.eq(window) & rolling.end.eq(end_date)].iloc[0]
            sample = common.loc[row.start : row.end]
            assert len(sample) == window
            targets.append((f"{label}_{window}m", window, sample))
    return targets


def circular_block_indices(size, block, rng):
    starts = rng.integers(0, size, size=int(np.ceil(size / block)))
    return ((starts[:, None] + np.arange(block)) % size).ravel()[:size]


def impose_sdf_null(sample):
    _, fitted = estimate_gmm(sample[TEST_ASSETS], sample[FACTORS])
    factors = sample[FACTORS].to_numpy(float)
    returns = sample[TEST_ASSETS].to_numpy(float)
    centered = factors - factors.mean(axis=0)
    sdf = 1 - centered @ fitted.b
    pricing_errors = (returns * sdf[:, None]).mean(axis=0) / sdf.mean()
    null_returns = returns - pricing_errors
    null_moments = (null_returns * sdf[:, None]).mean(axis=0)
    assert np.max(np.abs(null_moments)) < 1e-12
    return null_returns, factors, fitted, pricing_errors


def fit_draw(returns, factors, asset_names):
    asset_indices = [TEST_ASSETS.index(asset) for asset in asset_names]
    asset_frame = pd.DataFrame(returns[:, asset_indices], columns=asset_names)
    factor_frame = pd.DataFrame(factors, columns=FACTORS)
    _, result = estimate_gmm(asset_frame, factor_frame)
    return result.j_statistic, result.j_pvalue, result.converged


def wilson_interval(successes, draws, alpha=0.05):
    z = stats.norm.ppf(1 - alpha / 2)
    proportion = successes / draws
    denominator = 1 + z**2 / draws
    center = (proportion + z**2 / (2 * draws)) / denominator
    radius = (
        z * np.sqrt(proportion * (1 - proportion) / draws + z**2 / (4 * draws**2)) / denominator
    )
    return center - radius, center + radius


def summarize_size(draws, group_columns):
    rows = []
    for keys, group in draws.groupby(group_columns, sort=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        key_values = dict(zip(group_columns, keys))
        model = key_values["model"]
        degrees_freedom = len(MODELS[model]) - len(FACTORS)
        critical = stats.chi2.ppf(0.95, degrees_freedom)
        rejected = int(group.j_statistic.gt(critical).sum())
        lower, upper = wilson_interval(rejected, len(group))
        observed_j = group.observed_j.iloc[0]
        rows.append(
            {
                **key_values,
                "draws": len(group),
                "j_df": degrees_freedom,
                "chi2_5pct_critical": critical,
                "empirical_5pct_critical": group.j_statistic.quantile(0.95),
                "false_rejections_at_nominal_5pct": rejected,
                "empirical_size": rejected / len(group),
                "size_wilson_95_lower": lower,
                "size_wilson_95_upper": upper,
                "size_distortion_ratio": (rejected / len(group)) / 0.05,
                "observed_j": observed_j,
                "observed_asymptotic_pvalue": group.observed_asymptotic_pvalue.iloc[0],
                "null_calibrated_pvalue": (1 + group.j_statistic.ge(observed_j).sum())
                / (len(group) + 1),
                "convergence_rate": group.converged.mean(),
            }
        )
    return pd.DataFrame(rows)


def null_block_bootstrap(targets):
    rng = np.random.default_rng(SEED)
    rows = []
    diagnostics = []
    for target, window, sample in targets:
        null_returns, factor_values, fitted, pricing_errors = impose_sdf_null(sample)
        diagnostics.append(
            {
                "target": target,
                "window_months": window,
                "maximum_absolute_null_shift_bp": np.abs(pricing_errors).max() * 10_000,
                "mean_absolute_null_shift_bp": np.abs(pricing_errors).mean() * 10_000,
                "fitted_b_AVG": fitted.b[0],
                "fitted_b_CARRY": fitted.b[1],
                "fitted_b_CMOM": fitted.b[2],
            }
        )
        observed = {}
        for model, names in MODELS.items():
            j_statistic, pvalue, _ = fit_draw(
                sample[TEST_ASSETS].to_numpy(float),
                sample[FACTORS].to_numpy(float),
                names,
            )
            observed[model] = (j_statistic, pvalue)
        for block in BLOCK_LENGTHS:
            for draw in range(BOOTSTRAP_DRAWS):
                indices = circular_block_indices(window, block, rng)
                draw_returns = null_returns[indices]
                draw_factors = factor_values[indices]
                for model, names in MODELS.items():
                    j_statistic, pvalue, converged = fit_draw(draw_returns, draw_factors, names)
                    rows.append(
                        {
                            "target": target,
                            "window_months": window,
                            "block_months": block,
                            "draw": draw,
                            "model": model,
                            "j_statistic": j_statistic,
                            "asymptotic_pvalue": pvalue,
                            "converged": converged,
                            "observed_j": observed[model][0],
                            "observed_asymptotic_pvalue": observed[model][1],
                        }
                    )
    draws = pd.DataFrame(rows)
    summary = summarize_size(draws, ["target", "window_months", "block_months", "model"])
    return draws, summary, pd.DataFrame(diagnostics)


def nearest_correlation(matrix):
    symmetric = (matrix + matrix.T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    positive = eigenvectors @ np.diag(np.clip(eigenvalues, 1e-8, None)) @ eigenvectors.T
    scale = np.sqrt(np.diag(positive))
    return positive / np.outer(scale, scale)


def calibrate_monte_carlo_dgp(sample, hhi):
    factors = sample[FACTORS].to_numpy(float)
    centered = factors - factors.mean(axis=0)
    returns = sample[TEST_ASSETS].to_numpy(float)
    design = np.column_stack([np.ones(len(sample)), centered])
    coefficients = np.linalg.lstsq(design, returns, rcond=None)[0]
    slopes = coefficients[1:]
    residuals = returns - design @ coefficients

    hhi_values = hhi.loc[sample.index, TEST_ASSETS].to_numpy(float)
    log_hhi = np.log(hhi_values)
    log_squared = np.log(residuals**2 + np.nanmedian(residuals**2) * 0.01)
    within_hhi = log_hhi - log_hhi.mean(axis=0)
    within_squared = log_squared - log_squared.mean(axis=0)
    variance_gamma = float(np.sum(within_hhi * within_squared) / np.sum(within_hhi**2))
    relative_scale = np.exp(0.5 * variance_gamma * within_hhi)
    standardized = residuals / relative_scale
    sigma = standardized.std(axis=0, ddof=1)
    correlation = nearest_correlation(np.corrcoef(standardized / sigma, rowvar=False))

    pooled = (standardized / sigma).ravel()
    excess_kurtosis = max(float(stats.kurtosis(pooled, fisher=True, bias=False)), 0.01)
    student_df = float(np.clip(4 + 6 / excess_kurtosis, 4.5, 30))

    _, fitted = estimate_gmm(sample[TEST_ASSETS], sample[FACTORS])
    sdf = 1 - centered @ fitted.b
    conditional_without_intercept = centered @ slopes
    null_intercept = -(sdf[:, None] * conditional_without_intercept).mean(axis=0) / sdf.mean()
    return {
        "factors": factors,
        "centered": centered,
        "slopes": slopes,
        "sigma": sigma,
        "correlation": correlation,
        "relative_scale": relative_scale,
        "student_df": student_df,
        "null_intercept": null_intercept,
        "variance_gamma": variance_gamma,
    }


def simulate_returns(dgp, distribution, rng):
    nobs = len(dgp["factors"])
    nassets = len(TEST_ASSETS)
    normal = rng.multivariate_normal(np.zeros(nassets), dgp["correlation"], size=nobs)
    if distribution == "gaussian_homoskedastic":
        innovations = normal
        scale = 1
    elif distribution in {"student_t_homoskedastic", "student_t_hhi_scaled"}:
        degrees = dgp["student_df"]
        chi_squared = rng.chisquare(degrees, size=nobs)
        innovations = normal / np.sqrt(chi_squared[:, None] / degrees)
        innovations *= np.sqrt((degrees - 2) / degrees)
        scale = dgp["relative_scale"] if distribution.endswith("hhi_scaled") else 1
    else:
        raise ValueError(distribution)
    residuals = innovations * dgp["sigma"] * scale
    conditional = dgp["null_intercept"] + dgp["centered"] @ dgp["slopes"]
    return conditional + residuals


def monte_carlo_calibration(targets, hhi):
    rng = np.random.default_rng(SEED + 1)
    rows = []
    diagnostics = []
    for target, window, sample in targets:
        if "rejection_ending_2014_02" not in target:
            continue
        dgp = calibrate_monte_carlo_dgp(sample, hhi)
        observed = {}
        for model, names in MODELS.items():
            observed[model] = fit_draw(
                sample[TEST_ASSETS].to_numpy(float),
                sample[FACTORS].to_numpy(float),
                names,
            )[:2]
        diagnostics.append(
            {
                "target": target,
                "window_months": window,
                "estimated_hhi_log_variance_slope": dgp["variance_gamma"],
                "calibrated_student_df": dgp["student_df"],
                "mean_residual_sigma": dgp["sigma"].mean(),
                "maximum_residual_sigma": dgp["sigma"].max(),
            }
        )
        for distribution in [
            "gaussian_homoskedastic",
            "student_t_homoskedastic",
            "student_t_hhi_scaled",
        ]:
            for draw in range(MONTE_CARLO_DRAWS):
                draw_returns = simulate_returns(dgp, distribution, rng)
                for model, names in MODELS.items():
                    j_statistic, pvalue, converged = fit_draw(draw_returns, dgp["factors"], names)
                    rows.append(
                        {
                            "target": target,
                            "window_months": window,
                            "distribution": distribution,
                            "draw": draw,
                            "model": model,
                            "j_statistic": j_statistic,
                            "asymptotic_pvalue": pvalue,
                            "converged": converged,
                            "observed_j": observed[model][0],
                            "observed_asymptotic_pvalue": observed[model][1],
                        }
                    )
    draws = pd.DataFrame(rows)
    summary = summarize_size(draws, ["target", "window_months", "distribution", "model"])
    return draws, summary, pd.DataFrame(diagnostics)


def rolling_window_bootstrap_calibration(common):
    """Recalibrate every reported rolling window with a local 12-month block null."""
    rng = np.random.default_rng(SEED + 2)
    rolling = pd.read_csv(
        STABILITY_TABLES / "rolling_gmm_j_tests.csv", parse_dates=["start", "end"]
    )
    rows = []
    for window_row in rolling.itertuples(index=False):
        sample = common.loc[window_row.start : window_row.end]
        assert len(sample) == window_row.window_months
        null_returns, factor_values, _, _ = impose_sdf_null(sample)
        observed = {
            model: fit_draw(
                sample[TEST_ASSETS].to_numpy(float),
                sample[FACTORS].to_numpy(float),
                names,
            )
            for model, names in MODELS.items()
        }
        simulated = {model: [] for model in MODELS}
        for _ in range(ROLLING_BOOTSTRAP_DRAWS):
            indices = circular_block_indices(window_row.window_months, 12, rng)
            for model, names in MODELS.items():
                j_statistic, _, _ = fit_draw(null_returns[indices], factor_values[indices], names)
                simulated[model].append(j_statistic)
        for model, values in simulated.items():
            values = np.asarray(values)
            degrees_freedom = len(MODELS[model]) - len(FACTORS)
            chi_critical = stats.chi2.ppf(0.95, degrees_freedom)
            observed_j, observed_pvalue, _ = observed[model]
            rows.append(
                {
                    "window_months": window_row.window_months,
                    "start": window_row.start,
                    "end": window_row.end,
                    "model": model,
                    "draws": ROLLING_BOOTSTRAP_DRAWS,
                    "block_months": 12,
                    "observed_j": observed_j,
                    "asymptotic_pvalue": observed_pvalue,
                    "asymptotic_reject_5pct": observed_pvalue < 0.05,
                    "bootstrap_pvalue": (1 + np.sum(values >= observed_j))
                    / (ROLLING_BOOTSTRAP_DRAWS + 1),
                    "bootstrap_reject_5pct": (
                        (1 + np.sum(values >= observed_j)) / (ROLLING_BOOTSTRAP_DRAWS + 1)
                    )
                    < 0.05,
                    "chi2_5pct_critical": chi_critical,
                    "bootstrap_5pct_critical": np.quantile(values, 0.95),
                    "local_null_size_at_nominal_5pct": np.mean(values > chi_critical),
                }
            )
    results = pd.DataFrame(rows)
    summary = (
        results.groupby(["window_months", "model"])
        .agg(
            windows=("end", "size"),
            asymptotic_rejections=("asymptotic_reject_5pct", "sum"),
            bootstrap_rejections=("bootstrap_reject_5pct", "sum"),
            mean_local_null_size=("local_null_size_at_nominal_5pct", "mean"),
            median_local_null_size=("local_null_size_at_nominal_5pct", "median"),
            mean_chi2_critical=("chi2_5pct_critical", "mean"),
            mean_bootstrap_critical=("bootstrap_5pct_critical", "mean"),
        )
        .reset_index()
    )
    return results, summary


def plot_size(bootstrap_summary, monte_carlo_summary):
    bootstrap_plot = bootstrap_summary.copy()
    bootstrap_plot["method"] = "Block " + bootstrap_plot.block_months.astype(str) + "m"
    monte_plot = monte_carlo_summary.copy()
    monte_plot["method"] = monte_plot.distribution.replace(
        {
            "gaussian_homoskedastic": "Gaussian",
            "student_t_homoskedastic": "Student-t",
            "student_t_hhi_scaled": "Student-t + HHI scale",
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    rejection = bootstrap_plot[bootstrap_plot.target.str.contains("rejection_ending_2014_02")]
    for axis, window in zip(axes, [120, 180]):
        frames = [
            rejection[rejection.window_months.eq(window)][["method", "model", "empirical_size"]],
            monte_plot[monte_plot.window_months.eq(window)][["method", "model", "empirical_size"]],
        ]
        values = pd.concat(frames, ignore_index=True)
        methods = values.method.unique()
        x = np.arange(len(methods))
        width = 0.35
        for offset, (model, label) in enumerate(
            [("all_12_assets", "12 assets"), ("exclude_P1_P2", "Exclude P1/P2")]
        ):
            series = values[values.model.eq(model)].set_index("method").reindex(methods)
            axis.bar(
                x + (offset - 0.5) * width,
                series.empirical_size,
                width,
                label=label,
            )
        axis.axhline(0.05, color="crimson", linestyle="--", linewidth=1)
        axis.set_xticks(x, methods, rotation=35, ha="right")
        axis.set_title(f"{window // 12}-year calibration")
        axis.set_ylabel("Empirical false-rejection rate")
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
    figure.suptitle("Actual size of the nominal 5% two-step GMM J-test")
    figure.tight_layout()
    figure.savefig(FIGURES / "gmm_j_size_calibration.png", dpi=180)
    plt.close(figure)


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    common, hhi = load_common_sample()
    targets = target_windows(common)

    bootstrap_draws, bootstrap_summary, bootstrap_diagnostics = null_block_bootstrap(targets)
    bootstrap_draws.to_csv(TABLES / "null_block_bootstrap_draws.csv", index=False)
    bootstrap_summary.to_csv(TABLES / "null_block_bootstrap_size.csv", index=False)
    bootstrap_diagnostics.to_csv(TABLES / "null_block_bootstrap_diagnostics.csv", index=False)

    monte_draws, monte_summary, monte_diagnostics = monte_carlo_calibration(targets, hhi)
    monte_draws.to_csv(TABLES / "gmm_monte_carlo_draws.csv", index=False)
    monte_summary.to_csv(TABLES / "gmm_monte_carlo_size.csv", index=False)
    monte_diagnostics.to_csv(TABLES / "gmm_monte_carlo_diagnostics.csv", index=False)

    rolling_calibration, rolling_summary = rolling_window_bootstrap_calibration(common)
    rolling_calibration.to_csv(TABLES / "rolling_j_null_bootstrap_calibration.csv", index=False)
    rolling_summary.to_csv(TABLES / "rolling_j_null_bootstrap_summary.csv", index=False)
    plot_size(bootstrap_summary, monte_summary)

    print("\nNull block-bootstrap size:\n", bootstrap_summary.to_string(index=False))
    print("\nMonte Carlo size:\n", monte_summary.to_string(index=False))
    print("\nMonte Carlo DGP diagnostics:\n", monte_diagnostics.to_string(index=False))
    print("\nAll-window rolling calibration:\n", rolling_summary.to_string(index=False))


if __name__ == "__main__":
    main()
