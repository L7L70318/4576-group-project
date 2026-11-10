"""Re-run paper-style tables with full-slope quartiles on two paired samples."""

from __future__ import annotations

import numpy as np
import pandas as pd

from project.asset_pricing import (
    FACTORS,
    PAPER_END,
    PAPER_START,
    TEST_ASSETS,
    _fit_rows,
    _model_rows,
    _pvalues,
    construct_baseline_returns,
    estimate_gmm,
    fama_macbeth,
    time_series_table,
)
from project.extension_analysis import concentration
from project.paths import EXTENSION_TABLES, ROOT

TABLES = EXTENSION_TABLES / "04_robust_framework"
BOOTSTRAP_DRAWS = 499
BLOCK_MONTHS = 12
SEED = 4576
QUARTILE_NAMES = ["S1", "S2", "S3", "S4"]
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
MODEL_FACTORS = {
    "three_factor": ["AVG", "CARRY", "CMOM"],
    "omit_momentum": ["AVG", "CARRY"],
    "omit_carry": ["AVG", "CMOM"],
    "avg_only": ["AVG"],
}


def filtered_panel(extended):
    panel = pd.read_parquet(ROOT / "data/processed/commodity_monthly.parquet")
    windows = pd.read_csv(ROOT / "configs/paper_sample_dates.csv", parse_dates=["start", "end"])
    if extended:
        windows.loc[windows.end.eq(PAPER_END), "end"] = panel.date.max()
    data = panel.merge(windows, on="paper_code", validate="many_to_one")
    data = data[data.date.ge(data.start) & data.date.le(data.end)].drop(columns=["start", "end"])
    if not extended:
        data = data[data.date.between(PAPER_START, PAPER_END)]
    return data


def construct_sample(extended):
    data = filtered_panel(extended)
    original_assets, factors, original_weights = construct_baseline_returns(data)
    quartile_returns, quartile_weights = concentration.full_slope_quartiles(data)
    quartile_assets = pd.concat(
        [quartile_returns[QUARTILE_NAMES], original_assets[NON_CARRY_ASSETS]], axis=1
    )
    quartile_asset_names = [*QUARTILE_NAMES, *NON_CARRY_ASSETS]
    paired = pd.concat(
        [
            original_assets[TEST_ASSETS].add_prefix("original_"),
            quartile_assets[quartile_asset_names].add_prefix("quartile_"),
            factors[FACTORS],
        ],
        axis=1,
    ).dropna()
    original = pd.concat(
        [
            paired[[f"original_{name}" for name in TEST_ASSETS]].rename(
                columns=lambda name: name.removeprefix("original_")
            ),
            paired[FACTORS],
        ],
        axis=1,
    )
    quartile = pd.concat(
        [
            paired[[f"quartile_{name}" for name in quartile_asset_names]].rename(
                columns=lambda name: name.removeprefix("quartile_")
            ),
            paired[FACTORS],
        ],
        axis=1,
    )
    return {
        "data": data,
        "factors": factors,
        "original_assets": original_assets,
        "quartile_assets": quartile_assets,
        "original_weights": original_weights,
        "quartile_weights": quartile_weights,
        "original": original,
        "quartile": quartile,
        "original_names": TEST_ASSETS,
        "quartile_names": quartile_asset_names,
    }


def circular_block_indices(size, rng):
    starts = rng.integers(0, size, size=int(np.ceil(size / BLOCK_MONTHS)))
    return ((starts[:, None] + np.arange(BLOCK_MONTHS)) % size).ravel()[:size]


def impose_gmm_null(assets, factors):
    _, result = estimate_gmm(assets, factors)
    factor_values = factors.to_numpy(float)
    returns = assets.to_numpy(float)
    sdf = 1 - (factor_values - factor_values.mean(axis=0)) @ result.b
    errors = (returns * sdf[:, None]).mean(axis=0) / sdf.mean()
    return returns - errors, factor_values, result


def impose_fmb_null(assets, factors):
    result = fama_macbeth(assets, factors)
    return assets.to_numpy(float) - result.pricing_errors, factors.to_numpy(float), result


def bootstrap_table3(assets, factors, rng):
    names = list(assets.columns)
    null_gmm, factor_values, observed_gmm = impose_gmm_null(assets, factors)
    null_fmb, _, observed_fmb = impose_fmb_null(assets, factors)
    gmm_statistics = []
    nw_statistics = []
    shanken_statistics = []
    for _ in range(BOOTSTRAP_DRAWS):
        indices = circular_block_indices(len(assets), rng)
        draw_factors = pd.DataFrame(factor_values[indices], columns=factors.columns)
        draw_gmm_assets = pd.DataFrame(null_gmm[indices], columns=names)
        draw_fmb_assets = pd.DataFrame(null_fmb[indices], columns=names)
        _, gmm = estimate_gmm(draw_gmm_assets, draw_factors)
        fmb = fama_macbeth(draw_fmb_assets, draw_factors)
        gmm_statistics.append(gmm.j_statistic)
        nw_statistics.append(fmb.chi2_nw)
        shanken_statistics.append(fmb.chi2_shanken)

    def calibrated(observed, values):
        values = np.asarray(values)
        return {
            "pvalue": (1 + np.sum(values >= observed)) / (len(values) + 1),
            "critical_95": np.quantile(values, 0.95),
        }

    gmm = calibrated(observed_gmm.j_statistic, gmm_statistics)
    nw = calibrated(observed_fmb.chi2_nw, nw_statistics)
    shanken = calibrated(observed_fmb.chi2_shanken, shanken_statistics)
    return {
        "gmm_two_step_j_bootstrap_pvalue": gmm["pvalue"],
        "gmm_two_step_j_bootstrap_critical_95": gmm["critical_95"],
        "fmb_chi2_nw_bootstrap_pvalue": nw["pvalue"],
        "fmb_chi2_nw_bootstrap_critical_95": nw["critical_95"],
        "fmb_chi2_shanken_bootstrap_pvalue": shanken["pvalue"],
        "fmb_chi2_shanken_bootstrap_critical_95": shanken["critical_95"],
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_block_months": BLOCK_MONTHS,
    }


def table3_outputs(sample_name, design, frame, asset_names, rng):
    assets = frame[asset_names]
    all_factors = frame[FACTORS]
    parameter_rows = []
    fit_rows = []
    pricing_rows = []
    figure_rows = []
    for model, factor_names in MODEL_FACTORS.items():
        selected = all_factors[factor_names]
        fmb = fama_macbeth(assets, selected)
        one_step, two_step = estimate_gmm(assets, selected)
        parameters = _model_rows(model, fmb, one_step, two_step)
        for row in parameters:
            row.update({"sample": sample_name, "asset_design": design})
        parameter_rows.extend(parameters)
        fit = _fit_rows(model, fmb, one_step, two_step)
        fit.update(
            {
                "sample": sample_name,
                "asset_design": design,
                "n_months": len(frame),
                **bootstrap_table3(assets, selected, rng),
            }
        )
        fit_rows.append(fit)
        for index, asset in enumerate(asset_names):
            pricing_rows.append(
                {
                    "sample": sample_name,
                    "asset_design": design,
                    "model": model,
                    "asset": asset,
                    "realized_monthly": assets[asset].mean(),
                    "fitted_monthly": fmb.fitted[index],
                    "pricing_error_monthly": fmb.pricing_errors[index],
                    "realized_annualized_percent": assets[asset].mean() * 1200,
                    "fitted_annualized_percent": fmb.fitted[index] * 1200,
                    "pricing_error_annualized_percent": fmb.pricing_errors[index] * 1200,
                }
            )
        if model == "three_factor":
            realized = assets.mean().to_numpy()
            figure_rows.append(
                {
                    "sample": sample_name,
                    "asset_design": design,
                    "n_months": len(frame),
                    "annualized_mae_percent": np.abs(fmb.pricing_errors).mean() * 1200,
                    "annualized_max_abs_error_percent": np.abs(fmb.pricing_errors).max() * 1200,
                    "realized_fitted_correlation": np.corrcoef(realized, fmb.fitted)[0, 1],
                }
            )
    return (
        pd.DataFrame(parameter_rows),
        pd.DataFrame(fit_rows),
        pd.DataFrame(pricing_rows),
        pd.DataFrame(figure_rows),
    )


def time_series_bootstrap(assets, factors, model, rng):
    coefficients, joint = time_series_table(assets, factors, model)
    alphas = coefficients.set_index("asset").alpha.reindex(assets.columns)
    null_returns = assets.to_numpy(float) - alphas.to_numpy(float)
    factor_values = factors.to_numpy(float)
    statistics = []
    for _ in range(BOOTSTRAP_DRAWS):
        indices = circular_block_indices(len(assets), rng)
        draw_assets = pd.DataFrame(null_returns[indices], columns=assets.columns)
        draw_factors = pd.DataFrame(factor_values[indices], columns=factors.columns)
        _, draw_joint = time_series_table(draw_assets, draw_factors, model)
        statistics.append(draw_joint.joint_alpha_chi2.iloc[0])
    observed = joint.joint_alpha_chi2.iloc[0]
    joint["bootstrap_pvalue"] = (1 + np.sum(np.asarray(statistics) >= observed)) / (
        BOOTSTRAP_DRAWS + 1
    )
    joint["bootstrap_critical_95"] = np.quantile(statistics, 0.95)
    joint["bootstrap_draws"] = BOOTSTRAP_DRAWS
    return coefficients, joint


def time_series_outputs(sample_name, design, frame, asset_names, rng):
    assets = frame[asset_names]
    specifications = {
        "three_factor": FACTORS,
        "omit_momentum": ["AVG", "CARRY"],
        "carry_only": ["CARRY"],
    }
    coefficients = []
    joints = []
    for model, factor_names in specifications.items():
        table, joint = time_series_bootstrap(assets, frame[factor_names], model, rng)
        table.insert(0, "asset_design", design)
        table.insert(0, "sample", sample_name)
        joint.insert(0, "asset_design", design)
        joint.insert(0, "sample", sample_name)
        coefficients.append(table)
        joints.append(joint)
    return pd.concat(coefficients, ignore_index=True), pd.concat(joints, ignore_index=True)


def descriptive_statistics(sample_name, design, frame, carry_names, hhi):
    rows = []
    for order, asset in enumerate(carry_names, 1):
        values = frame[asset]
        annualized_volatility = values.std(ddof=1) * np.sqrt(12)
        rows.append(
            {
                "sample": sample_name,
                "asset_design": design,
                "order": order,
                "asset": asset,
                "n_months": len(values),
                "annualized_mean_percent": values.mean() * 1200,
                "annualized_volatility_percent": annualized_volatility * 100,
                "annualized_sharpe": values.mean() * 12 / annualized_volatility,
                "monthly_skewness": values.skew(),
                "positive_months_percent": values.gt(0).mean() * 100,
                "mean_hhi": hhi.reindex(frame.index)[asset].mean(),
                "mean_members": (1 / hhi.reindex(frame.index)[asset]).mean(),
            }
        )
    result = pd.DataFrame(rows)
    result["means_monotonically_decreasing"] = (
        result.annualized_mean_percent.is_monotonic_decreasing
    )
    return result


def hhi_frames(sample):
    original = sample["original_weights"].copy()
    original = original[original.portfolio.isin(["P1", "P2", "P3", "P4"])]
    original_hhi = (
        original.assign(weight_squared=original.weight**2)
        .groupby(["date", "portfolio"])
        .weight_squared.sum()
        .unstack("portfolio")
    )
    quartile = sample["quartile_weights"].copy()
    quartile_hhi = (
        quartile.assign(weight_squared=quartile.weight**2)
        .groupby(["date", "portfolio"])
        .weight_squared.sum()
        .unstack("portfolio")
    )
    return original_hhi, quartile_hhi


def table8_panel_b(sample_name, sample, equity, rng):
    delta = equity["delta_vol_equity_g20_proxy"].rename("DELTA_VOL_EQUITY")
    original = sample["original_assets"][["P1", "P2", "P3", "P4"]]
    quartile = sample["quartile_assets"][QUARTILE_NAMES]
    avg = sample["factors"][["AVG"]]
    paired = pd.concat(
        [original.add_prefix("original_"), quartile.add_prefix("quartile_"), avg, delta],
        axis=1,
    ).dropna()
    rows = []
    pricing = []
    for design, prefix, names in [
        ("original_sign_split", "original_", ["P1", "P2", "P3", "P4"]),
        ("full_slope_quartiles", "quartile_", QUARTILE_NAMES),
    ]:
        assets = paired[[f"{prefix}{name}" for name in names]].rename(
            columns={f"{prefix}{name}": name for name in names}
        )
        factors = paired[["AVG", "DELTA_VOL_EQUITY"]]
        fmb = fama_macbeth(assets, factors)
        null_returns = assets.to_numpy(float) - fmb.pricing_errors
        factor_values = factors.to_numpy(float)
        bootstrap_statistics = []
        for _ in range(BOOTSTRAP_DRAWS):
            indices = circular_block_indices(len(paired), rng)
            draw = fama_macbeth(
                pd.DataFrame(null_returns[indices], columns=names),
                pd.DataFrame(factor_values[indices], columns=factors.columns),
            )
            bootstrap_statistics.append(draw.chi2_nw)
        nw_pvalues = _pvalues(fmb.lambda_estimate, fmb.covariance_nw)
        shanken_pvalues = _pvalues(fmb.lambda_estimate, fmb.covariance_shanken)
        rows.append(
            {
                "sample": sample_name,
                "asset_design": design,
                "n_months": len(paired),
                "start": paired.index.min(),
                "end": paired.index.max(),
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
                "pricing_error_chi2_nw_bootstrap_pvalue": (
                    1 + np.sum(np.asarray(bootstrap_statistics) >= fmb.chi2_nw)
                )
                / (BOOTSTRAP_DRAWS + 1),
            }
        )
        for index, asset in enumerate(names):
            pricing.append(
                {
                    "sample": sample_name,
                    "asset_design": design,
                    "asset": asset,
                    "realized_monthly": assets[asset].mean(),
                    "fitted_monthly": fmb.fitted[index],
                    "pricing_error_monthly": fmb.pricing_errors[index],
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(pricing)


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    equity = pd.read_parquet(ROOT / "data/processed/equity_volatility_monthly.parquet")
    equity.index = pd.to_datetime(equity.index)
    outputs = {
        "parameters": [],
        "fit": [],
        "pricing": [],
        "figure2": [],
        "time_series": [],
        "joint_alpha": [],
        "descriptive": [],
        "table8": [],
        "table8_pricing": [],
        "coverage": [],
    }
    for sample_name, extended in [
        ("paper_sample_paired", False),
        ("extended_sample_paired", True),
    ]:
        sample = construct_sample(extended)
        original_hhi, quartile_hhi = hhi_frames(sample)
        for design, key, names, carry_names, hhi in [
            (
                "original_sign_split",
                "original",
                sample["original_names"],
                ["P1", "P2", "P3", "P4"],
                original_hhi,
            ),
            (
                "full_slope_quartiles",
                "quartile",
                sample["quartile_names"],
                QUARTILE_NAMES,
                quartile_hhi,
            ),
        ]:
            frame = sample[key]
            parameters, fit, pricing, figure2 = table3_outputs(
                sample_name, design, frame, names, rng
            )
            time_series, joint = time_series_outputs(sample_name, design, frame, names, rng)
            outputs["parameters"].append(parameters)
            outputs["fit"].append(fit)
            outputs["pricing"].append(pricing)
            outputs["figure2"].append(figure2)
            outputs["time_series"].append(time_series)
            outputs["joint_alpha"].append(joint)
            outputs["descriptive"].append(
                descriptive_statistics(sample_name, design, frame, carry_names, hhi)
            )
            outputs["coverage"].append(
                pd.DataFrame(
                    [
                        {
                            "sample": sample_name,
                            "asset_design": design,
                            "start": frame.index.min(),
                            "end": frame.index.max(),
                            "months": len(frame),
                            "test_assets": len(names),
                        }
                    ]
                )
            )
        table8, table8_pricing = table8_panel_b(sample_name, sample, equity, rng)
        outputs["table8"].append(table8)
        outputs["table8_pricing"].append(table8_pricing)

    filenames = {
        "parameters": "table3_parameter_estimates.csv",
        "fit": "table3_model_fit_and_tests.csv",
        "pricing": "table3_pricing_errors.csv",
        "figure2": "figure2_fit_summary.csv",
        "time_series": "table6_7_time_series_coefficients.csv",
        "joint_alpha": "table6_7_joint_alpha_tests.csv",
        "descriptive": "carry_portfolio_descriptive_statistics.csv",
        "table8": "table8_panel_b_model.csv",
        "table8_pricing": "table8_panel_b_pricing_errors.csv",
        "coverage": "paired_sample_coverage.csv",
    }
    for key, filename in filenames.items():
        pd.concat(outputs[key], ignore_index=True).to_csv(TABLES / filename, index=False)

    print(pd.concat(outputs["coverage"], ignore_index=True).to_string(index=False))
    print("\nTable 3:\n", pd.concat(outputs["fit"], ignore_index=True).to_string(index=False))
    print(
        "\nJoint alphas:\n",
        pd.concat(outputs["joint_alpha"], ignore_index=True).to_string(index=False),
    )
    print(
        "\nTable 8 Panel B:\n",
        pd.concat(outputs["table8"], ignore_index=True).to_string(index=False),
    )


if __name__ == "__main__":
    main()
