"""Audit P1/P2 construction and rerun the SDF model without those test assets."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from project.asset_pricing import (
    FACTORS,
    PAPER_END,
    TEST_ASSETS,
    construct_baseline_returns,
    estimate_gmm,
)
from project.paths import EXTENSION_FIGURES, EXTENSION_TABLES, ROOT

TABLES = EXTENSION_TABLES / "02_source"
FIGURES = EXTENSION_FIGURES / "02_source"
REDUCED_ASSETS = [asset for asset in TEST_ASSETS if asset not in {"P1", "P2"}]
ROLLING_WINDOWS = [120, 180]
ROLLING_STEP = 12


def extended_inputs():
    panel = pd.read_parquet(ROOT / "data/processed/commodity_monthly.parquet")
    windows = pd.read_csv(ROOT / "configs/paper_sample_dates.csv", parse_dates=["start", "end"])
    windows.loc[windows.end.eq(PAPER_END), "end"] = panel.date.max()
    data = panel.merge(windows, on="paper_code", validate="many_to_one")
    data = data[data.date.ge(data.start) & data.date.le(data.end)].drop(columns=["start", "end"])
    assets, factors, weights = construct_baseline_returns(data)
    return data, assets, factors, weights


def audit_construction(data, assets, weights):
    selected = weights[weights.portfolio.isin(["P1", "P2"])].copy()
    details = selected.merge(
        data[
            [
                "date",
                "paper_code",
                "formation_date",
                "slope",
                "excess_return",
                "entry_price",
                "exit_price",
            ]
        ],
        on=["date", "paper_code"],
        how="left",
        validate="many_to_one",
    )
    details["manual_contribution"] = details.weight * details.excess_return

    manual = (
        details.groupby(["date", "portfolio"])
        .manual_contribution.agg(lambda values: values.sum() if values.notna().all() else np.nan)
        .unstack("portfolio")
    )
    return_diff = manual.subtract(assets[["P1", "P2"]]).abs()

    universe = data.copy()
    if "mapping_status" in universe:
        universe = universe[universe.mapping_status.eq("verified")]
    if "selected_futcode" in universe:
        universe = universe[universe.selected_futcode.notna()]
    negative = universe[universe.slope.lt(0)].groupby("date").size().rename("negative_signals")
    monthly = (
        details.groupby(["date", "portfolio"])
        .agg(
            actual_members=("paper_code", "size"),
            weight_sum=("weight", "sum"),
            min_weight=("weight", "min"),
            max_weight=("weight", "max"),
            min_slope=("slope", "min"),
            max_slope=("slope", "max"),
            missing_returns=("excess_return", lambda values: values.isna().sum()),
        )
        .reset_index()
        .merge(negative, on="date", how="left")
    )
    monthly["expected_members"] = np.where(
        monthly.portfolio.eq("P1"),
        np.ceil(monthly.negative_signals / 2),
        np.floor(monthly.negative_signals / 2),
    )
    monthly["member_count_valid"] = monthly.actual_members.eq(monthly.expected_members)
    monthly["all_slopes_negative"] = monthly.max_slope.lt(0)
    monthly["equal_weight_valid"] = np.isclose(
        monthly.min_weight, 1 / monthly.actual_members
    ) & np.isclose(monthly.max_weight, 1 / monthly.actual_members)
    monthly["weight_sum_valid"] = np.isclose(monthly.weight_sum, 1)
    monthly["manual_return_abs_diff"] = monthly.set_index(["date", "portfolio"]).index.map(
        return_diff.stack(future_stack=True)
    )

    memberships = {
        key: set(group.paper_code) for key, group in details.groupby(["date", "portfolio"])
    }
    dates = sorted(details.date.unique())
    overlap = pd.Series(
        [
            len(memberships.get((date, "P1"), set()) & memberships.get((date, "P2"), set()))
            for date in dates
        ],
        index=dates,
    )
    boundary = details.pivot_table(
        index="date", columns="portfolio", values="slope", aggfunc=["min", "max"]
    )

    price_identity = details.exit_price / details.entry_price - 1
    checks = [
        (
            "P1/P2 member slopes are strictly negative",
            int((details.slope >= 0).sum()),
            len(details),
        ),
        ("P1/P2 memberships do not overlap", int((overlap > 0).sum()), len(overlap)),
        (
            "P1 is below P2 in the negative-slope ranking",
            int((boundary[("max", "P1")] > boundary[("min", "P2")] + 1e-12).sum()),
            int(boundary[[("max", "P1"), ("min", "P2")]].dropna().shape[0]),
        ),
        (
            "member counts equal the two-way negative-slope split",
            int((~monthly.member_count_valid).sum()),
            len(monthly),
        ),
        (
            "portfolio weights are equal and sum to one",
            int((~(monthly.equal_weight_valid & monthly.weight_sum_valid)).sum()),
            len(monthly),
        ),
        (
            "portfolio return equals weighted constituent return",
            int((monthly.manual_return_abs_diff.fillna(0) > 1e-12).sum()),
            len(monthly),
        ),
        (
            "constituent return equals exit/entry minus one",
            int(((details.excess_return - price_identity).abs() > 1e-12).sum()),
            int(details.excess_return.notna().sum()),
        ),
        (
            "return month is one month after formation month",
            int(((details.formation_date + pd.offsets.MonthEnd(1)) != details.date).sum()),
            len(details),
        ),
    ]
    summary = pd.DataFrame(checks, columns=["check", "violations", "observations_checked"])
    summary["passed"] = summary.violations.eq(0)

    concentration = []
    for period, start, end in [
        ("full", None, None),
        ("rejection_region_2003_2014", "2003-01-01", "2014-12-31"),
    ]:
        sample = monthly if start is None else monthly[monthly.date.between(start, end)]
        for portfolio, group in sample.groupby("portfolio"):
            concentration.append(
                {
                    "period": period,
                    "portfolio": portfolio,
                    "months": len(group),
                    "mean_members": group.actual_members.mean(),
                    "median_members": group.actual_members.median(),
                    "minimum_members": group.actual_members.min(),
                    "maximum_members": group.actual_members.max(),
                    "months_with_at_most_2_members": group.actual_members.le(2).sum(),
                    "share_months_with_at_most_2_members": group.actual_members.le(2).mean(),
                }
            )

    period_details = details[details.date.between("2003-01-01", "2014-12-31")]
    returns = assets.loc["2003":"2014", ["P1", "P2"]].stack().rename("portfolio_return")
    largest = returns.abs().nlargest(20).index
    outliers = period_details.set_index(["date", "portfolio"]).loc[largest].reset_index()
    outliers = outliers.merge(returns.reset_index(), on=["date", "portfolio"])
    outliers["absolute_portfolio_return"] = outliers.portfolio_return.abs()
    outliers["contribution_share"] = np.divide(
        outliers.manual_contribution,
        outliers.portfolio_return,
        out=np.full(len(outliers), np.nan),
        where=outliers.portfolio_return.ne(0),
    )
    outliers = outliers.sort_values(
        ["absolute_portfolio_return", "date", "portfolio", "manual_contribution"],
        ascending=[False, True, True, False],
    )

    return monthly, summary, pd.DataFrame(concentration), outliers


def result_row(model, sample_definition, data, asset_names):
    _, result = estimate_gmm(data[asset_names], data[FACTORS])
    row = {
        "model": model,
        "sample_definition": sample_definition,
        "observations": len(data),
        "start": data.index.min(),
        "end": data.index.max(),
        "test_assets": len(asset_names),
        "j_statistic": result.j_statistic,
        "j_df": result.j_df,
        "j_pvalue": result.j_pvalue,
        "reject_5_percent": result.j_pvalue < 0.05,
        "nw_lags": result.nw_lags,
        "converged": result.converged,
    }
    for index, factor in enumerate(FACTORS):
        row[f"b_{factor}"] = result.b[index]
        row[f"lambda_{factor}"] = result.lambda_estimate[index]
    return row


def rolling_comparison(common):
    rows = []
    for window in ROLLING_WINDOWS:
        for end in range(window, len(common) + 1, ROLLING_STEP):
            sample = common.iloc[end - window : end]
            for model, names in [("all_12_assets", TEST_ASSETS), ("exclude_P1_P2", REDUCED_ASSETS)]:
                row = result_row(model, "fixed_original_common_months", sample, names)
                row.update({"window_months": window})
                rows.append(row)
    return pd.DataFrame(rows)


def plot_rolling(results):
    figure, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    colors = {"all_12_assets": "#355c9a", "exclude_P1_P2": "#d97706"}
    labels = {"all_12_assets": "Original 12 assets", "exclude_P1_P2": "10 assets: P1/P2 excluded"}
    for axis, window in zip(axes, ROLLING_WINDOWS):
        values = results[results.window_months.eq(window)]
        for model, color in colors.items():
            line = values[values.model.eq(model)]
            axis.plot(
                line.end,
                line.j_pvalue,
                marker="o",
                markersize=3,
                linewidth=1.4,
                color=color,
                label=labels[model],
            )
        axis.axhline(0.05, color="crimson", linestyle="--", linewidth=1)
        axis.set_ylim(0, 1)
        axis.set_ylabel("GMM J-test p-value")
        axis.set_title(f"Rolling {window // 12}-year window; identical months and factors")
        axis.grid(alpha=0.25)
        axis.legend(loc="upper left")
    axes[-1].set_xlabel("Window end month")
    figure.suptitle("Effect of removing P1 and P2 from the test-asset cross-section")
    figure.tight_layout()
    figure.savefig(FIGURES / "p1_p2_exclusion_rolling_j.png", dpi=180)
    plt.close(figure)


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    data, assets, factors, weights = extended_inputs()
    monthly, audit, concentration, outliers = audit_construction(data, assets, weights)
    monthly.to_csv(TABLES / "p1_p2_monthly_construction_audit.csv", index=False)
    audit.to_csv(TABLES / "p1_p2_construction_audit.csv", index=False)
    concentration.to_csv(TABLES / "p1_p2_concentration.csv", index=False)
    outliers.to_csv(TABLES / "p1_p2_outlier_contributions.csv", index=False)

    original_common = pd.concat([assets[TEST_ASSETS], factors[FACTORS]], axis=1).dropna()
    reduced_natural = pd.concat([assets[REDUCED_ASSETS], factors[FACTORS]], axis=1).dropna()
    full_rows = [
        result_row("all_12_assets", "fixed_original_common_months", original_common, TEST_ASSETS),
        result_row(
            "exclude_P1_P2", "fixed_original_common_months", original_common, REDUCED_ASSETS
        ),
        result_row(
            "exclude_P1_P2", "model_specific_common_months", reduced_natural, REDUCED_ASSETS
        ),
    ]
    full = pd.DataFrame(full_rows)
    full.to_csv(TABLES / "p1_p2_exclusion_full_sample.csv", index=False)

    rolling = rolling_comparison(original_common)
    rolling.to_csv(TABLES / "p1_p2_exclusion_rolling_j.csv", index=False)
    rolling_summary = (
        rolling.groupby(["window_months", "model"])
        .agg(
            windows=("j_pvalue", "size"),
            rejected=("reject_5_percent", "sum"),
            rejection_rate=("reject_5_percent", "mean"),
            median_j_pvalue=("j_pvalue", "median"),
        )
        .reset_index()
    )
    rolling_summary.to_csv(TABLES / "p1_p2_exclusion_rolling_summary.csv", index=False)

    assert audit.passed.all(), audit.loc[~audit.passed].to_dict("records")
    print(audit.to_string(index=False))
    print("\nConcentration:\n", concentration.to_string(index=False))
    print("\nFull-sample comparison:\n", full.to_string(index=False))
    print("\nRolling comparison:\n", rolling_summary.to_string(index=False))


if __name__ == "__main__":
    main()
