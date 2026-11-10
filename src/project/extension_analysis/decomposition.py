"""Decompose the 2014-to-2015 rolling-J transition and rejection assets."""

from __future__ import annotations

import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize

from project.asset_pricing import (
    FACTORS,
    PAPER_END,
    TEST_ASSETS,
    _gmm_moments,
    _pack,
    _unpack,
    automatic_nw_lags,
    construct_baseline_returns,
    long_run_covariance,
)
from project.paths import EXTENSION_FIGURES, EXTENSION_TABLES, ROOT

STABILITY_TABLES = EXTENSION_TABLES / "01_instability"
TABLES = EXTENSION_TABLES / "02_source"
FIGURES = EXTENSION_FIGURES / "02_source"
COMPONENTS = ["sample", "b", "factor_mean", "factor_covariance", "weight"]
GROUPS = {
    "carry_P": ["P1", "P2", "P3", "P4"],
    "momentum_Q": ["Q1", "Q2", "Q3", "Q4", "Q5"],
    "sectors": ["Agriculture", "Livestock", "Metal"],
}


def extended_common_sample():
    panel = pd.read_parquet(ROOT / "data/processed/commodity_monthly.parquet")
    windows = pd.read_csv(ROOT / "configs/paper_sample_dates.csv", parse_dates=["start", "end"])
    windows.loc[windows.end.eq(PAPER_END), "end"] = panel.date.max()
    data = panel.merge(windows, on="paper_code", validate="many_to_one")
    data = data[data.date.ge(data.start) & data.date.le(data.end)].drop(columns=["start", "end"])
    assets, factors, _ = construct_baseline_returns(data)
    return pd.concat([assets[TEST_ASSETS], factors[FACTORS]], axis=1).dropna()


def exact_gmm(sample: pd.DataFrame):
    returns = sample[TEST_ASSETS].to_numpy(float)
    factors = sample[FACTORS].to_numpy(float)
    nobs, nassets = returns.shape
    nfactors = factors.shape[1]
    factor_mean = factors.mean(axis=0)
    centered = factors - factor_mean
    factor_covariance = centered.T @ centered / nobs
    return_factor_covariance = returns.T @ centered / nobs
    b_start = np.linalg.pinv(return_factor_covariance) @ returns.mean(axis=0)
    theta_start = _pack(b_start, factor_mean, factor_covariance)
    nlags = automatic_nw_lags(nobs)

    def mean_moments(theta):
        return _gmm_moments(theta, returns, factors).mean(axis=0)

    first = optimize.least_squares(
        mean_moments,
        theta_start,
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
        max_nfev=20_000,
    )
    first_moments = _gmm_moments(first.x, returns, factors)
    weight = np.linalg.pinv(long_run_covariance(first_moments, nlags), rcond=1e-10)
    eigenvalues, eigenvectors = np.linalg.eigh((weight + weight.T) / 2)
    keep = eigenvalues > eigenvalues.max() * 1e-10
    square_root_weight = np.sqrt(eigenvalues[keep])[:, None] * eigenvectors[:, keep].T
    second = optimize.least_squares(
        lambda theta: square_root_weight @ mean_moments(theta),
        first.x,
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
        max_nfev=20_000,
    )
    moments = _gmm_moments(second.x, returns, factors)
    mean = moments.mean(axis=0)
    j_statistic = float(nobs * mean @ weight @ mean)
    b, factor_mean, factor_covariance = _unpack(second.x, nfactors)
    return {
        "returns": returns,
        "factors": factors,
        "theta": second.x,
        "b": b,
        "factor_mean": factor_mean,
        "factor_covariance": factor_covariance,
        "weight": weight,
        "moments": moments,
        "j_statistic": j_statistic,
        "nw_lags": nlags,
        "nassets": nassets,
    }


def hybrid_j(old, new, state):
    data_source = new if state["sample"] else old
    b = new["b"] if state["b"] else old["b"]
    mean = new["factor_mean"] if state["factor_mean"] else old["factor_mean"]
    covariance = (
        new["factor_covariance"] if state["factor_covariance"] else old["factor_covariance"]
    )
    weight = new["weight"] if state["weight"] else old["weight"]
    theta = _pack(b, mean, covariance)
    moments = _gmm_moments(theta, data_source["returns"], data_source["factors"])
    average = moments.mean(axis=0)
    return float(len(moments) * average @ weight @ average)


def shapley_decomposition(old, new):
    contributions = dict.fromkeys(COMPONENTS, 0.0)
    permutations = list(itertools.permutations(COMPONENTS))
    cache = {}

    def value(state):
        key = tuple(state[component] for component in COMPONENTS)
        if key not in cache:
            cache[key] = hybrid_j(old, new, state)
        return cache[key]

    for ordering in permutations:
        state = dict.fromkeys(COMPONENTS, False)
        previous = value(state)
        for component in ordering:
            state[component] = True
            current = value(state)
            contributions[component] += current - previous
            previous = current
    for component in COMPONENTS:
        contributions[component] /= len(permutations)
    return (
        contributions,
        value(dict.fromkeys(COMPONENTS, False)),
        value(dict.fromkeys(COMPONENTS, True)),
    )


def transition_windows(data, months):
    old_end = data.index.get_loc(pd.Timestamp("2014-02-28")) + 1
    new_end = data.index.get_loc(pd.Timestamp("2015-02-28")) + 1
    return data.iloc[old_end - months : old_end], data.iloc[new_end - months : new_end]


def replacement_block_rows(old_sample, new_sample, old_model, new_model, months):
    blocks = {
        "dropped": old_sample.loc[old_sample.index.difference(new_sample.index)],
        "overlap": old_sample.loc[old_sample.index.intersection(new_sample.index)],
        "added": new_sample.loc[new_sample.index.difference(old_sample.index)],
    }
    rows = []
    for block_name, block in blocks.items():
        for reference_name, model in [("old_model", old_model), ("new_model", new_model)]:
            values = _gmm_moments(
                model["theta"],
                block[TEST_ASSETS].to_numpy(float),
                block[FACTORS].to_numpy(float),
            )[:, : len(TEST_ASSETS)]
            for asset, error in zip(TEST_ASSETS, values.mean(axis=0)):
                rows.append(
                    {
                        "window_months": months,
                        "block": block_name,
                        "block_start": block.index.min(),
                        "block_end": block.index.max(),
                        "block_months": len(block),
                        "reference_parameters": reference_name,
                        "asset": asset,
                        "mean_pricing_moment_bps": error * 10_000,
                    }
                )
        for factor in FACTORS:
            rows.append(
                {
                    "window_months": months,
                    "block": block_name,
                    "block_start": block.index.min(),
                    "block_end": block.index.max(),
                    "block_months": len(block),
                    "reference_parameters": "raw_factor_return",
                    "asset": factor,
                    "mean_pricing_moment_bps": block[factor].mean() * 10_000,
                }
            )
    return rows


def rejection_asset_rows(data, rolling):
    targets = rolling[
        rolling.window_months.eq(120)
        & rolling.reject_5_percent
        & rolling.end.between(pd.Timestamp("2003-09-30"), pd.Timestamp("2014-02-28"))
    ]
    rows = []
    for target in targets.itertuples(index=False):
        sample = data.loc[target.start : target.end]
        model = exact_gmm(sample)
        pricing = model["moments"][:, : len(TEST_ASSETS)]
        average = pricing.mean(axis=0)
        covariance = long_run_covariance(pricing, model["nw_lags"])
        weight = np.linalg.pinv(covariance, rcond=1e-10)
        weighted = weight @ average
        contributions = len(sample) * average * weighted
        for asset, error, contribution in zip(TEST_ASSETS, average, contributions):
            group = next(name for name, members in GROUPS.items() if asset in members)
            rows.append(
                {
                    "window_start": target.start,
                    "window_end": target.end,
                    "asset": asset,
                    "asset_group": group,
                    "pricing_error_bps": error * 10_000,
                    "abs_pricing_error_bps": abs(error) * 10_000,
                    "pricing_block_quadratic_contribution": contribution,
                }
            )
    return pd.DataFrame(rows)


def plot_rejection_heatmap(values: pd.DataFrame, output: Path):
    table = values.pivot(index="asset", columns="window_end", values="pricing_error_bps")
    table = table.reindex(TEST_ASSETS)
    bound = np.nanmax(np.abs(table.to_numpy()))
    figure, axis = plt.subplots(figsize=(13, 6))
    image = axis.imshow(table, aspect="auto", cmap="RdBu_r", vmin=-bound, vmax=bound)
    axis.set_yticks(np.arange(len(table.index)), table.index)
    axis.set_xticks(
        np.arange(len(table.columns)),
        [date.strftime("%Y-%m") for date in table.columns],
        rotation=45,
        ha="right",
    )
    axis.set_title("Pricing errors in the 2003–2014 rolling rejection region")
    axis.set_xlabel("10-year window end")
    axis.set_ylabel("Test asset")
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("Mean pricing moment (basis points per month)")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main():
    data = extended_common_sample()
    rolling = pd.read_csv(
        STABILITY_TABLES / "rolling_gmm_j_tests.csv", parse_dates=["start", "end"]
    )
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    decomposition_rows = []
    block_rows = []
    for months in [120, 180]:
        old_sample, new_sample = transition_windows(data, months)
        old_model, new_model = exact_gmm(old_sample), exact_gmm(new_sample)
        contributions, old_j, new_j = shapley_decomposition(old_model, new_model)
        for component, contribution in contributions.items():
            decomposition_rows.append(
                {
                    "window_months": months,
                    "old_start": old_sample.index.min(),
                    "old_end": old_sample.index.max(),
                    "new_start": new_sample.index.min(),
                    "new_end": new_sample.index.max(),
                    "old_j_statistic": old_j,
                    "new_j_statistic": new_j,
                    "j_change": new_j - old_j,
                    "component": component,
                    "shapley_j_contribution": contribution,
                    "share_of_absolute_component_changes": abs(contribution)
                    / sum(abs(value) for value in contributions.values()),
                }
            )
        block_rows.extend(
            replacement_block_rows(old_sample, new_sample, old_model, new_model, months)
        )

    decomposition = pd.DataFrame(decomposition_rows)
    decomposition.to_csv(TABLES / "2014_2015_j_decomposition.csv", index=False)
    pd.DataFrame(block_rows).to_csv(TABLES / "2014_2015_replacement_blocks.csv", index=False)

    assets = rejection_asset_rows(data, rolling)
    assets.to_csv(TABLES / "rejection_region_asset_errors.csv", index=False)
    asset_summary = (
        assets.groupby(["asset_group", "asset"])
        .agg(
            rejection_windows=("window_end", "size"),
            mean_error_bps=("pricing_error_bps", "mean"),
            mean_abs_error_bps=("abs_pricing_error_bps", "mean"),
            rms_error_bps=("pricing_error_bps", lambda x: np.sqrt(np.mean(x**2))),
            max_abs_error_bps=("abs_pricing_error_bps", "max"),
            mean_quadratic_contribution=(
                "pricing_block_quadratic_contribution",
                "mean",
            ),
        )
        .reset_index()
        .sort_values("mean_abs_error_bps", ascending=False)
    )
    asset_summary.to_csv(TABLES / "rejection_region_asset_summary.csv", index=False)
    group_summary = (
        assets.groupby(["window_end", "asset_group"])
        .agg(
            group_mae_bps=("abs_pricing_error_bps", "mean"),
            group_rmse_bps=("pricing_error_bps", lambda x: np.sqrt(np.mean(x**2))),
            group_quadratic_contribution=(
                "pricing_block_quadratic_contribution",
                "sum",
            ),
        )
        .reset_index()
        .groupby("asset_group")
        .agg(
            rejection_windows=("window_end", "size"),
            mean_group_mae_bps=("group_mae_bps", "mean"),
            mean_group_rmse_bps=("group_rmse_bps", "mean"),
            mean_group_quadratic_contribution=(
                "group_quadratic_contribution",
                "mean",
            ),
        )
        .reset_index()
        .sort_values("mean_group_rmse_bps", ascending=False)
    )
    group_summary.to_csv(TABLES / "rejection_region_group_summary.csv", index=False)
    plot_rejection_heatmap(assets, FIGURES / "rejection_region_asset_heatmap.png")

    print("J DECOMPOSITION")
    print(decomposition.to_string(index=False))
    print("\nASSET SUMMARY")
    print(asset_summary.to_string(index=False))
    print("\nGROUP SUMMARY")
    print(group_summary.to_string(index=False))


if __name__ == "__main__":
    main()
