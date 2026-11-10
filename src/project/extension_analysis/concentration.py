"""Test concentration mechanisms and a diversified full-slope carry factor."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from project.asset_pricing import (
    FACTORS,
    PAPER_END,
    TEST_ASSETS,
    automatic_nw_lags,
    construct_baseline_returns,
    estimate_gmm,
    long_run_covariance,
)
from project.paths import EXTENSION_FIGURES, EXTENSION_TABLES, ROOT

TABLES = EXTENSION_TABLES / "03_mechanism"
FIGURES = EXTENSION_FIGURES / "03_mechanism"
ROLLING_WINDOWS = [120, 180]
ROLLING_STEP = 12
PORTFOLIO_MAP = {
    "sector_agriculture": "Agriculture",
    "sector_livestock": "Livestock",
    "sector_metals": "Metal",
}
SECTORS = ["agriculture", "energy", "livestock", "metals"]


def extended_inputs():
    panel = pd.read_parquet(ROOT / "data/processed/commodity_monthly.parquet")
    windows = pd.read_csv(ROOT / "configs/paper_sample_dates.csv", parse_dates=["start", "end"])
    windows.loc[windows.end.eq(PAPER_END), "end"] = panel.date.max()
    data = panel.merge(windows, on="paper_code", validate="many_to_one")
    data = data[data.date.ge(data.start) & data.date.le(data.end)].drop(columns=["start", "end"])
    assets, factors, weights = construct_baseline_returns(data)
    return data, assets, factors, weights


def test_asset_characteristics(data, weights):
    selected = weights.copy()
    selected["asset"] = selected.portfolio.replace(PORTFOLIO_MAP)
    selected = selected[selected.asset.isin(TEST_ASSETS)]
    details = selected.merge(
        data[["date", "paper_code", "slope", "sector"]],
        on=["date", "paper_code"],
        how="left",
        validate="many_to_one",
    )
    details["weight_squared"] = details.weight**2
    details["weighted_abs_slope"] = details.weight * details.slope.abs()
    details["weighted_slope"] = details.weight * details.slope
    for sector in SECTORS:
        details[f"share_{sector}"] = details.weight * details.sector.eq(sector)
    aggregate = {
        "n_members": ("paper_code", "size"),
        "hhi": ("weight_squared", "sum"),
        "abs_slope": ("weighted_abs_slope", "sum"),
        "mean_slope": ("weighted_slope", "sum"),
    }
    aggregate.update({f"share_{sector}": (f"share_{sector}", "sum") for sector in SECTORS})
    result = details.groupby(["date", "asset"]).agg(**aggregate).reset_index()
    result["family"] = np.select(
        [result.asset.str.startswith("P"), result.asset.str.startswith("Q")],
        ["carry_sort", "momentum_sort"],
        default="sector",
    )
    result["hhi_identity_error"] = (result.hhi - 1 / result.n_members).abs()
    assert result.hhi_identity_error.max() < 1e-12
    return result


def pricing_moment_panel(assets, factors, characteristics):
    common = pd.concat([assets[TEST_ASSETS], factors[FACTORS]], axis=1).dropna()
    _, result = estimate_gmm(common[TEST_ASSETS], common[FACTORS])
    centered = common[FACTORS] - common[FACTORS].mean()
    sdf = 1 - centered.to_numpy(float) @ result.b
    moments = common[TEST_ASSETS].mul(sdf, axis=0)
    panel = moments.stack().rename("u").reset_index()
    panel.columns = ["date", "asset", "u"]
    panel = panel.merge(characteristics, on=["date", "asset"], how="left", validate="one_to_one")
    lagged_volatility = (
        common[TEST_ASSETS]
        .rolling(36, min_periods=24)
        .std()
        .shift(1)
        .stack()
        .rename("lagged_36m_return_volatility")
        .reset_index()
    )
    lagged_volatility.columns = ["date", "asset", "lagged_36m_return_volatility"]
    panel = panel.merge(
        lagged_volatility,
        on=["date", "asset"],
        how="left",
        validate="one_to_one",
    )
    panel["u_bp"] = panel.u * 10_000
    panel["abs_u_bp"] = panel.u.abs() * 10_000
    if panel.hhi.isna().any():
        raise ValueError("Missing HHI for a priced test-asset month")
    return common, result, panel


def two_way_demean(frame, columns):
    values = frame[columns].astype(float)
    asset_mean = values.groupby(frame.asset).transform("mean")
    date_mean = values.groupby(frame.date).transform("mean")
    return values - asset_mean - date_mean + values.mean()


def ols_hac_by_time(frame, outcome, regressors, model, two_way_fe=False):
    columns = [outcome, *regressors]
    sample = frame.dropna(subset=columns).copy()
    if two_way_fe:
        transformed = two_way_demean(sample, columns)
        y = transformed[outcome].to_numpy(float)
        x = transformed[regressors].to_numpy(float)
        names = regressors
    else:
        y = sample[outcome].to_numpy(float)
        x = np.column_stack([np.ones(len(sample)), sample[regressors].to_numpy(float)])
        names = ["intercept", *regressors]
    beta = np.linalg.pinv(x) @ y
    residual = y - x @ beta
    score_rows = pd.DataFrame(x * residual[:, None], index=sample.index)
    score_rows["date"] = sample.date.to_numpy()
    time_scores = score_rows.groupby("date").sum().drop(columns=[], errors="ignore")
    nperiods = len(time_scores)
    nlags = automatic_nw_lags(nperiods)
    meat = nperiods * long_run_covariance(time_scores.to_numpy(float), nlags)
    bread = np.linalg.pinv(x.T @ x)
    covariance = bread @ meat @ bread
    standard_errors = np.sqrt(np.clip(np.diag(covariance), 0, None))
    pvalues = 2 * stats.norm.sf(np.abs(beta / standard_errors))
    rows = []
    for index, name in enumerate(names):
        rows.append(
            {
                "model": model,
                "outcome": outcome,
                "coefficient": name,
                "estimate": beta[index],
                "hac_standard_error": standard_errors[index],
                "hac_pvalue": pvalues[index],
                "observations": len(sample),
                "months": nperiods,
                "portfolios": sample.asset.nunique(),
                "portfolio_and_month_fixed_effects": two_way_fe,
                "nw_lags": nlags,
            }
        )
    return pd.DataFrame(rows)


def hhi_regressions(panel):
    controls = [
        "hhi",
        "abs_slope",
        "share_agriculture",
        "share_energy",
        "share_livestock",
    ]
    volatility_controls = [*controls, "lagged_36m_return_volatility"]
    results = []
    for outcome in ["abs_u_bp", "u_bp"]:
        results.append(
            ols_hac_by_time(
                panel,
                outcome,
                ["hhi"],
                f"{outcome}_pooled",
                two_way_fe=False,
            )
        )
        results.append(
            ols_hac_by_time(
                panel,
                outcome,
                controls,
                f"{outcome}_slope_sector_portfolio_month_FE",
                two_way_fe=True,
            )
        )
        results.append(
            ols_hac_by_time(
                panel,
                outcome,
                volatility_controls,
                f"{outcome}_slope_sector_lagged_vol_portfolio_month_FE",
                two_way_fe=True,
            )
        )
        results.append(
            ols_hac_by_time(
                panel[panel.family.eq("carry_sort")],
                outcome,
                controls,
                f"{outcome}_P1_P4_slope_sector_portfolio_month_FE",
                two_way_fe=True,
            )
        )
    result = pd.concat(results, ignore_index=True)
    result["effect_of_hhi_increase_0_25"] = np.where(
        result.coefficient.eq("hhi"), result.estimate * 0.25, np.nan
    )
    return result


def hac_mean(values):
    values = np.asarray(values, dtype=float)
    nlags = automatic_nw_lags(len(values))
    mean = values.mean()
    variance = float(long_run_covariance(values, nlags)[0, 0] / len(values))
    standard_error = np.sqrt(max(variance, 0))
    return mean, standard_error, nlags


def concentration_group_statistics(panel):
    panel = panel.copy()
    panel["concentration_group"] = np.select(
        [panel.n_members.le(2), panel.n_members.le(4)],
        ["high_N1_2", "medium_N3_4"],
        default="low_N5_plus",
    )
    rows = []
    for family in ["all", "carry_sort", "momentum_sort", "sector"]:
        family_panel = panel if family == "all" else panel[panel.family.eq(family)]
        for group, observations in family_panel.groupby("concentration_group"):
            monthly = observations.groupby("date")[["u_bp", "abs_u_bp"]].mean()
            for outcome in ["u_bp", "abs_u_bp"]:
                mean, standard_error, nlags = hac_mean(monthly[outcome])
                rows.append(
                    {
                        "family": family,
                        "concentration_group": group,
                        "outcome": outcome,
                        "asset_months": len(observations),
                        "months": len(monthly),
                        "mean": mean,
                        "hac_standard_error": standard_error,
                        "ci_95_lower": mean - 1.96 * standard_error,
                        "ci_95_upper": mean + 1.96 * standard_error,
                        "nw_lags": nlags,
                    }
                )
    return pd.DataFrame(rows)


def matched_hhi_comparison(panel):
    feature_names = [
        "abs_slope",
        "share_agriculture",
        "share_energy",
        "share_livestock",
        "share_metals",
    ]
    scales = panel[feature_names].std().replace(0, 1)
    matches = []
    for (date, family), group in panel.groupby(["date", "family"]):
        treated = group[group.n_members.le(2)]
        controls = group[group.n_members.ge(4)]
        if treated.empty or controls.empty:
            continue
        control_features = controls[feature_names].div(scales).to_numpy(float)
        for row in treated.itertuples():
            treated_features = (
                pd.Series({name: getattr(row, name) for name in feature_names}) / scales
            ).to_numpy(float)
            distances = np.sqrt(((control_features - treated_features) ** 2).sum(axis=1))
            match = controls.iloc[int(np.argmin(distances))]
            matches.append(
                {
                    "date": date,
                    "family": family,
                    "high_hhi_asset": row.asset,
                    "low_hhi_asset": match.asset,
                    "high_hhi": row.hhi,
                    "low_hhi": match.hhi,
                    "control_distance": distances.min(),
                    "abs_u_difference_bp": row.abs_u_bp - match.abs_u_bp,
                    "signed_u_difference_bp": row.u_bp - match.u_bp,
                }
            )
    matches = pd.DataFrame(matches)
    rows = []
    for family in ["all", *sorted(matches.family.unique())]:
        sample = matches if family == "all" else matches[matches.family.eq(family)]
        for outcome in ["abs_u_difference_bp", "signed_u_difference_bp"]:
            monthly = sample.groupby("date")[outcome].mean()
            mean, standard_error, nlags = hac_mean(monthly)
            rows.append(
                {
                    "family": family,
                    "outcome": outcome,
                    "matches": len(sample),
                    "months": len(monthly),
                    "mean_high_minus_low": mean,
                    "hac_standard_error": standard_error,
                    "hac_pvalue": 2 * stats.norm.sf(abs(mean / standard_error)),
                    "ci_95_lower": mean - 1.96 * standard_error,
                    "ci_95_upper": mean + 1.96 * standard_error,
                    "nw_lags": nlags,
                }
            )
    return matches, pd.DataFrame(rows)


def full_slope_quartiles(data):
    memberships = []
    for date, month in data.groupby("date", sort=True):
        universe = month
        if "mapping_status" in universe:
            universe = universe[universe.mapping_status.eq("verified")]
        if "selected_futcode" in universe:
            universe = universe[universe.selected_futcode.notna()]
        universe = universe.dropna(subset=["slope"]).sort_values(["slope", "paper_code"])
        if len(universe) < 4:
            continue
        for number, indices in enumerate(np.array_split(np.arange(len(universe)), 4), 1):
            selected = universe.iloc[indices]
            for row in selected.itertuples():
                memberships.append(
                    {
                        "date": date,
                        "portfolio": f"S{number}",
                        "paper_code": row.paper_code,
                        "weight": 1 / len(selected),
                    }
                )
    memberships = pd.DataFrame(memberships)
    joined = memberships.merge(
        data[["date", "paper_code", "excess_return"]],
        on=["date", "paper_code"],
        validate="many_to_one",
    )
    joined["contribution"] = joined.weight * joined.excess_return
    returns = (
        joined.groupby(["date", "portfolio"])
        .contribution.agg(lambda values: values.sum() if values.notna().all() else np.nan)
        .unstack("portfolio")
    )
    return returns, memberships


def gmm_with_asset_errors(experiment, sample, asset_names):
    _, result = estimate_gmm(sample[asset_names], sample[FACTORS])
    centered = sample[FACTORS] - sample[FACTORS].mean()
    sdf = 1 - centered.to_numpy(float) @ result.b
    errors = sample[asset_names].mul(sdf, axis=0).mean() * 10_000
    row = {
        "experiment": experiment,
        "observations": len(sample),
        "start": sample.index.min(),
        "end": sample.index.max(),
        "test_assets": len(asset_names),
        "j_statistic": result.j_statistic,
        "j_df": result.j_df,
        "j_pvalue": result.j_pvalue,
        "reject_5_percent": result.j_pvalue < 0.05,
        "nw_lags": result.nw_lags,
        "converged": result.converged,
        "mean_abs_P1_P2_error_bp": errors.reindex(["P1", "P2"]).abs().mean(),
    }
    for index, factor in enumerate(FACTORS):
        row[f"b_{factor}"] = result.b[index]
        row[f"lambda_{factor}"] = result.lambda_estimate[index]
    asset_rows = pd.DataFrame(
        {
            "experiment": experiment,
            "asset": errors.index,
            "pricing_error_bp": errors.values,
        }
    )
    return row, asset_rows


def diversified_carry_experiment(data, assets, factors):
    quartiles, memberships = full_slope_quartiles(data)
    carry_star = (quartiles.S1 - quartiles.S4).rename("CARRY")
    star_factors = pd.concat([factors[["AVG", "CMOM"]], carry_star], axis=1)[
        ["AVG", "CARRY", "CMOM"]
    ]
    baseline = pd.concat([assets[TEST_ASSETS], factors[FACTORS]], axis=1).dropna()
    star_natural = pd.concat([assets[TEST_ASSETS], star_factors], axis=1).dropna()
    common_dates = baseline.index.intersection(star_natural.index)
    baseline_common = baseline.loc[common_dates]
    star_common = star_natural.loc[common_dates]

    designs = [
        ("A_baseline_C5_natural", baseline, TEST_ASSETS),
        ("B_carry_star_natural", star_natural, TEST_ASSETS),
        ("C_baseline_C5_common_months", baseline_common, TEST_ASSETS),
        ("D_carry_star_common_months", star_common, TEST_ASSETS),
    ]
    full_rows = []
    error_rows = []
    for name, sample, names in designs:
        row, errors = gmm_with_asset_errors(name, sample, names)
        full_rows.append(row)
        error_rows.append(errors)

    subset_definitions = {
        "all_12": TEST_ASSETS,
        "carry_P1_P4": ["P1", "P2", "P3", "P4"],
        "noncarry_Q_and_sectors": [
            "Q1",
            "Q2",
            "Q3",
            "Q4",
            "Q5",
            "Agriculture",
            "Livestock",
            "Metal",
        ],
    }
    subset_rows = []
    for subset, names in subset_definitions.items():
        for factor_name, sample in [
            ("baseline_C5", baseline_common),
            ("carry_star", star_common),
        ]:
            row, _ = gmm_with_asset_errors(f"{subset}_{factor_name}", sample, names)
            row["asset_subset"] = subset
            row["carry_factor"] = factor_name
            subset_rows.append(row)

    rolling_rows = []
    rolling_errors = []
    for window in ROLLING_WINDOWS:
        for end in range(window, len(common_dates) + 1, ROLLING_STEP):
            for name, sample in [
                ("baseline_C5", baseline_common),
                ("carry_star", star_common),
            ]:
                period = sample.iloc[end - window : end]
                row, errors = gmm_with_asset_errors(name, period, TEST_ASSETS)
                row["window_months"] = window
                rolling_rows.append(row)
                errors["window_months"] = window
                errors["start"] = period.index.min()
                errors["end"] = period.index.max()
                rolling_errors.append(errors)

    factor_comparison = pd.concat(
        [
            factors.loc[common_dates, "CARRY"].rename("baseline_C5"),
            carry_star.loc[common_dates].rename("carry_star"),
        ],
        axis=1,
    )
    factor_summary = pd.DataFrame(
        [
            {
                "common_months": len(factor_comparison),
                "start": factor_comparison.index.min(),
                "end": factor_comparison.index.max(),
                "correlation": factor_comparison.corr().iloc[0, 1],
                "baseline_annualized_mean": factor_comparison.baseline_C5.mean() * 12,
                "carry_star_annualized_mean": factor_comparison.carry_star.mean() * 12,
                "baseline_annualized_volatility": factor_comparison.baseline_C5.std() * np.sqrt(12),
                "carry_star_annualized_volatility": factor_comparison.carry_star.std()
                * np.sqrt(12),
                "monthly_rmse": np.sqrt(
                    np.mean((factor_comparison.carry_star - factor_comparison.baseline_C5) ** 2)
                ),
            }
        ]
    )
    return (
        pd.DataFrame(full_rows),
        pd.concat(error_rows, ignore_index=True),
        pd.DataFrame(subset_rows),
        pd.DataFrame(rolling_rows),
        pd.concat(rolling_errors, ignore_index=True),
        factor_summary,
        factor_comparison,
        memberships,
    )


def plot_diversified_carry(rolling):
    figure, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for axis, window in zip(axes, ROLLING_WINDOWS):
        values = rolling[rolling.window_months.eq(window)]
        for experiment, label in [
            ("baseline_C5", "Baseline rank C5"),
            ("carry_star", "Full-slope quartile spread"),
        ]:
            line = values[values.experiment.eq(experiment)]
            axis.plot(
                line.end,
                line.j_pvalue,
                marker="o",
                markersize=3,
                linewidth=1.4,
                label=label,
            )
        axis.axhline(0.05, color="crimson", linestyle="--", linewidth=1)
        axis.set_ylim(0, 1)
        axis.set_ylabel("GMM J-test p-value")
        axis.set_title(f"Rolling {window // 12}-year window; original 12 test assets")
        axis.grid(alpha=0.25)
        axis.legend(loc="upper left")
    axes[-1].set_xlabel("Window end month")
    figure.suptitle("Baseline C5 versus diversified carry factor")
    figure.tight_layout()
    figure.savefig(FIGURES / "diversified_carry_rolling_j.png", dpi=180)
    plt.close(figure)


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    data, assets, factors, weights = extended_inputs()
    characteristics = test_asset_characteristics(data, weights)
    common, _, moment_panel = pricing_moment_panel(assets, factors, characteristics)
    full_regressions = hhi_regressions(moment_panel)
    full_regressions.insert(0, "sample_period", "full_1974_2025")
    rejection_period_regressions = hhi_regressions(
        moment_panel[moment_panel.date.between("2003-01-01", "2014-12-31")]
    )
    rejection_period_regressions.insert(0, "sample_period", "2003_2014")
    regressions = pd.concat([full_regressions, rejection_period_regressions], ignore_index=True)
    groups = concentration_group_statistics(moment_panel)
    matches, match_summary = matched_hhi_comparison(moment_panel)

    characteristics.to_csv(TABLES / "hhi_portfolio_characteristics.csv", index=False)
    moment_panel.to_csv(TABLES / "hhi_pricing_moment_panel.csv", index=False)
    regressions.to_csv(TABLES / "hhi_pricing_moment_regressions.csv", index=False)
    groups.to_csv(TABLES / "hhi_concentration_group_moments.csv", index=False)
    matches.to_csv(TABLES / "hhi_matched_pairs.csv", index=False)
    match_summary.to_csv(TABLES / "hhi_matched_comparison_summary.csv", index=False)

    print("Original common months:", len(common))
    print(
        "\nHHI coefficients:\n",
        regressions[regressions.coefficient.eq("hhi")].to_string(index=False),
    )
    print("\nMatched comparison:\n", match_summary.to_string(index=False))


if __name__ == "__main__":
    main()
