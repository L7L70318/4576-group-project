"""Complete the robust test-asset and simultaneous-inference framework."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from project.asset_pricing import FACTORS, TEST_ASSETS, estimate_gmm
from project.extension_analysis import concentration
from project.paths import EXTENSION_FIGURES, EXTENSION_TABLES

STABILITY_TABLES = EXTENSION_TABLES / "01_instability"
MECHANISM_TABLES = EXTENSION_TABLES / "03_mechanism"
TABLES = EXTENSION_TABLES / "04_robust_framework"
FIGURES = EXTENSION_FIGURES / "04_robust_framework"
SEED = 4675
LOCAL_DRAWS = 199
SIMULTANEOUS_DRAWS = 499
POWER_DRAWS = 499
BLOCK_MONTHS = 12
QUARTILE_ASSETS = ["S1", "S2", "S3", "S4"]
NON_CARRY_ASSETS = [
    "Q1",
    "Q2",
    "Q3",
    "Q4",
    "Q5",
    "Agriculture",
    "Livestock",
    "Metal",
]


def model_data():
    data, assets, factors, _ = concentration.extended_inputs()
    quartile_returns, _ = concentration.full_slope_quartiles(data)
    quartile_names = [*QUARTILE_ASSETS, *NON_CARRY_ASSETS]
    original = pd.concat([assets[TEST_ASSETS], factors[FACTORS]], axis=1).dropna()
    quartile = pd.concat(
        [quartile_returns[QUARTILE_ASSETS], assets[NON_CARRY_ASSETS], factors[FACTORS]],
        axis=1,
    ).dropna()
    common_dates = original.index.intersection(quartile.index)
    original = original.loc[common_dates]
    quartile = quartile.loc[common_dates]
    assert len(original) == 597
    return {
        "original_sign_split": (original, TEST_ASSETS),
        "full_slope_quartiles": (quartile, quartile_names),
    }


def circular_block_indices(size, block, rng):
    starts = rng.integers(0, size, size=int(np.ceil(size / block)))
    return ((starts[:, None] + np.arange(block)) % size).ravel()[:size]


def fit_j(returns, factors, asset_names):
    asset_frame = pd.DataFrame(returns, columns=asset_names)
    factor_frame = pd.DataFrame(factors, columns=FACTORS)
    _, result = estimate_gmm(asset_frame, factor_frame)
    return result.j_statistic, result.j_pvalue


def impose_null(sample, asset_names):
    _, fitted = estimate_gmm(sample[asset_names], sample[FACTORS])
    factors = sample[FACTORS].to_numpy(float)
    returns = sample[asset_names].to_numpy(float)
    centered = factors - factors.mean(axis=0)
    sdf = 1 - centered @ fitted.b
    pricing_errors = (returns * sdf[:, None]).mean(axis=0) / sdf.mean()
    null_returns = returns - pricing_errors
    assert np.abs((null_returns * sdf[:, None]).mean(axis=0)).max() < 1e-12
    return null_returns, factors


def rolling_definitions():
    return pd.read_csv(STABILITY_TABLES / "rolling_gmm_j_tests.csv", parse_dates=["start", "end"])[
        ["window_months", "start", "end"]
    ]


def local_quartile_bootstrap(models):
    rng = np.random.default_rng(SEED)
    quartile, names = models["full_slope_quartiles"]
    rows = []
    for window in rolling_definitions().itertuples(index=False):
        sample = quartile.loc[window.start : window.end]
        null_returns, factors = impose_null(sample, names)
        observed_j, observed_p = fit_j(
            sample[names].to_numpy(float), sample[FACTORS].to_numpy(float), names
        )
        statistics = []
        for _ in range(LOCAL_DRAWS):
            indices = circular_block_indices(len(sample), BLOCK_MONTHS, rng)
            statistic, _ = fit_j(null_returns[indices], factors[indices], names)
            statistics.append(statistic)
        statistics = np.asarray(statistics)
        chi_critical = stats.chi2.ppf(0.95, len(names) - len(FACTORS))
        bootstrap_p = (1 + np.sum(statistics >= observed_j)) / (LOCAL_DRAWS + 1)
        rows.append(
            {
                "window_months": window.window_months,
                "start": window.start,
                "end": window.end,
                "model": "full_slope_quartiles",
                "draws": LOCAL_DRAWS,
                "block_months": BLOCK_MONTHS,
                "observed_j": observed_j,
                "asymptotic_pvalue": observed_p,
                "asymptotic_reject_5pct": observed_p < 0.05,
                "bootstrap_pvalue": bootstrap_p,
                "bootstrap_reject_5pct": bootstrap_p < 0.05,
                "chi2_5pct_critical": chi_critical,
                "bootstrap_5pct_critical": np.quantile(statistics, 0.95),
                "local_null_size_at_nominal_5pct": np.mean(statistics > chi_critical),
            }
        )
    quartile_results = pd.DataFrame(rows)
    original = pd.read_csv(
        MECHANISM_TABLES / "rolling_j_null_bootstrap_calibration.csv",
        parse_dates=["start", "end"],
    )
    original = original[original.model.eq("all_12_assets")].copy()
    original["model"] = "original_sign_split"
    common_columns = quartile_results.columns
    combined = pd.concat(
        [original[common_columns], quartile_results], ignore_index=True
    ).sort_values(["window_months", "end", "model"])
    summary = (
        combined.groupby(["window_months", "model"])
        .agg(
            windows=("end", "size"),
            asymptotic_rejections=("asymptotic_reject_5pct", "sum"),
            bootstrap_rejections=("bootstrap_reject_5pct", "sum"),
            mean_local_null_size=("local_null_size_at_nominal_5pct", "mean"),
            median_bootstrap_pvalue=("bootstrap_pvalue", "median"),
            mean_bootstrap_critical=("bootstrap_5pct_critical", "mean"),
        )
        .reset_index()
    )
    return combined, summary


def longest_run(values):
    longest = current = 0
    for value in values:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def sequence_statistics(j_values, degrees_freedom, region_mask):
    critical = stats.chi2.ppf(0.95, degrees_freedom)
    rejected = np.asarray(j_values) > critical
    excess = np.maximum(np.asarray(j_values) - critical, 0)
    return {
        "max_j": np.max(j_values),
        "rejection_count": rejected.sum(),
        "longest_rejection_run": longest_run(rejected),
        "sum_positive_excess_j": excess.sum(),
        "region_sum_j": np.asarray(j_values)[region_mask].sum(),
        "region_rejection_count": rejected[region_mask].sum(),
    }


def rolling_j_sequence(returns, factors, asset_names, window):
    statistics = []
    for end in range(window, len(returns) + 1, 12):
        statistic, _ = fit_j(returns[end - window : end], factors[end - window : end], asset_names)
        statistics.append(statistic)
    return np.asarray(statistics)


def simultaneous_bootstrap(models):
    rng = np.random.default_rng(SEED + 1)
    definitions = rolling_definitions()
    rows = []
    draw_rows = []
    for model, (sample, names) in models.items():
        null_returns, factors = impose_null(sample, names)
        observed_returns = sample[names].to_numpy(float)
        observed_factors = sample[FACTORS].to_numpy(float)
        for window in [120, 180]:
            dates = definitions[definitions.window_months.eq(window)].end
            region_mask = dates.between("2003-01-01", "2014-12-31").to_numpy()
            observed_j = rolling_j_sequence(observed_returns, observed_factors, names, window)
            observed = sequence_statistics(observed_j, len(names) - len(FACTORS), region_mask)
            simulations = {statistic: [] for statistic in observed}
            for draw in range(SIMULTANEOUS_DRAWS):
                indices = circular_block_indices(len(sample), BLOCK_MONTHS, rng)
                simulated_j = rolling_j_sequence(
                    null_returns[indices], factors[indices], names, window
                )
                values = sequence_statistics(simulated_j, len(names) - len(FACTORS), region_mask)
                for statistic, value in values.items():
                    simulations[statistic].append(value)
                    draw_rows.append(
                        {
                            "model": model,
                            "window_months": window,
                            "draw": draw,
                            "statistic": statistic,
                            "value": value,
                        }
                    )
            for statistic, observed_value in observed.items():
                values = np.asarray(simulations[statistic])
                rows.append(
                    {
                        "model": model,
                        "window_months": window,
                        "statistic": statistic,
                        "observed_value": observed_value,
                        "null_mean": values.mean(),
                        "null_95pct_critical": np.quantile(values, 0.95),
                        "simultaneous_pvalue": (1 + np.sum(values >= observed_value))
                        / (SIMULTANEOUS_DRAWS + 1),
                        "draws": SIMULTANEOUS_DRAWS,
                        "block_months": BLOCK_MONTHS,
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(draw_rows)


def power_calibration(models):
    rng = np.random.default_rng(SEED + 2)
    definitions = rolling_definitions()
    rows = []
    for model, (data, names) in models.items():
        carry_names = TEST_ASSETS[:2] if model == "original_sign_split" else QUARTILE_ASSETS[:2]
        carry_indices = [names.index(name) for name in carry_names]
        for window in [120, 180]:
            definition = definitions[
                definitions.window_months.eq(window)
                & definitions.end.eq(pd.Timestamp("2014-02-28"))
            ].iloc[0]
            sample = data.loc[definition.start : definition.end]
            null_returns, factors = impose_null(sample, names)
            null_statistics = []
            alternative_statistics = {50: [], 100: []}
            for _ in range(POWER_DRAWS):
                indices = circular_block_indices(window, BLOCK_MONTHS, rng)
                draw_returns = null_returns[indices]
                statistic, _ = fit_j(draw_returns, factors[indices], names)
                null_statistics.append(statistic)
                for signal_bp, statistics_list in alternative_statistics.items():
                    alternative = draw_returns.copy()
                    alternative[:, carry_indices[0]] -= signal_bp / 10_000
                    alternative[:, carry_indices[1]] += signal_bp / 10_000
                    statistic, _ = fit_j(alternative, factors[indices], names)
                    statistics_list.append(statistic)
            critical = np.quantile(null_statistics, 0.95)
            for signal_bp, values in alternative_statistics.items():
                rows.append(
                    {
                        "model": model,
                        "window_months": window,
                        "signal_pattern": f"first_carry_minus_{signal_bp}bp_second_plus_{signal_bp}bp",
                        "signal_each_leg_bp": signal_bp,
                        "draws": POWER_DRAWS,
                        "bootstrap_5pct_critical": critical,
                        "empirical_power": np.mean(np.asarray(values) > critical),
                    }
                )
    return pd.DataFrame(rows)


def plot_framework(local_summary, simultaneous):
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    pivot = local_summary.pivot(
        index="window_months", columns="model", values="bootstrap_rejections"
    )
    pivot.plot(kind="bar", ax=axes[0], color=["#d97706", "#355c9a"])
    axes[0].set_ylabel("Bootstrap-calibrated rejected windows")
    axes[0].set_xlabel("Window length (months)")
    axes[0].set_title("Local finite-sample calibration")
    axes[0].legend(["Full-slope quartiles", "Original sign split"])
    axes[0].grid(axis="y", alpha=0.25)

    selected = simultaneous[simultaneous.statistic.eq("longest_rejection_run")].copy()
    x = np.arange(len(selected))
    axes[1].bar(x - 0.18, selected.observed_value, 0.36, label="Observed")
    axes[1].bar(
        x + 0.18,
        selected.null_95pct_critical,
        0.36,
        label="Null 95% critical",
    )
    axes[1].set_xticks(
        x,
        [f"{row.model}\n{row.window_months}m" for row in selected.itertuples(index=False)],
        rotation=20,
        ha="right",
    )
    axes[1].set_ylabel("Consecutive annual-step rejected windows")
    axes[1].set_title("Simultaneous longest-run test")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.25)
    figure.suptitle("Robust commodity SDF testing framework")
    figure.tight_layout()
    figure.savefig(FIGURES / "robust_testing_framework.png", dpi=180)
    plt.close(figure)


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    models = model_data()
    local_results, local_summary = local_quartile_bootstrap(models)
    simultaneous, simultaneous_draws = simultaneous_bootstrap(models)
    power = power_calibration(models)

    local_results.to_csv(TABLES / "robust_framework_local_bootstrap.csv", index=False)
    local_summary.to_csv(TABLES / "robust_framework_local_bootstrap_summary.csv", index=False)
    simultaneous.to_csv(TABLES / "robust_framework_simultaneous_tests.csv", index=False)
    simultaneous_draws.to_csv(TABLES / "robust_framework_simultaneous_draws.csv", index=False)
    power.to_csv(TABLES / "robust_framework_power_calibration.csv", index=False)
    plot_framework(local_summary, simultaneous)

    print("\nLocal bootstrap:\n", local_summary.to_string(index=False))
    print("\nSimultaneous tests:\n", simultaneous.to_string(index=False))
    print("\nPower calibration:\n", power.to_string(index=False))


if __name__ == "__main__":
    main()
