"""High/low-VIX state robustness for original and full-slope test assets."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from project.asset_pricing import FACTORS, estimate_gmm
from project.extension_analysis import table_robustness
from project.paths import EXTENSION_TABLES, ROOT

TABLES = EXTENSION_TABLES / "05_vix_robustness"
BOOTSTRAP_DRAWS = 499
BLOCK_MONTHS = 12
SEED = 4576


def vix_series():
    vix = pd.read_csv(
        ROOT / "data/interim/vix_monthly.csv",
        parse_dates=["formation_month", "observation_date"],
    )
    vix["return_month"] = vix.formation_month + pd.offsets.MonthEnd(1)
    return vix.set_index("return_month")[["formation_month", "observation_date", "VIX"]]


def circular_block_indices(size, rng):
    starts = rng.integers(0, size, size=int(np.ceil(size / BLOCK_MONTHS)))
    return ((starts[:, None] + np.arange(BLOCK_MONTHS)) % size).ravel()[:size]


def state_fit(returns, factors, asset_names):
    asset_frame = pd.DataFrame(returns, columns=asset_names)
    factor_frame = pd.DataFrame(factors, columns=FACTORS)
    _, result = estimate_gmm(asset_frame, factor_frame)
    centered = factor_frame - factor_frame.mean()
    sdf = 1 - centered.to_numpy(float) @ result.b
    errors = asset_frame.mul(sdf, axis=0).mean() * 10_000
    return result, errors


def null_shift(returns, factors, result):
    centered = factors - factors.mean(axis=0)
    sdf = 1 - centered @ result.b
    errors = (returns * sdf[:, None]).mean(axis=0) / sdf.mean()
    return returns - errors


def bootstrap_states(frame, asset_names, high, rng):
    returns = frame[asset_names].to_numpy(float)
    factors = frame[FACTORS].to_numpy(float)
    state_masks = {"high": high, "low": ~high}
    observed = {}
    null_returns = returns.copy()
    for state, mask in state_masks.items():
        result, errors = state_fit(returns[mask], factors[mask], asset_names)
        observed[state] = (result, errors)
        null_returns[mask] = null_shift(returns[mask], factors[mask], result)

    bootstrap = {"high": [], "low": []}
    state_months = {"high": [], "low": []}
    for _ in range(BOOTSTRAP_DRAWS):
        indices = circular_block_indices(len(frame), rng)
        draw_high = high[indices]
        for state, mask in {"high": draw_high, "low": ~draw_high}.items():
            result, _ = state_fit(null_returns[indices][mask], factors[indices][mask], asset_names)
            bootstrap[state].append(result.j_statistic)
            state_months[state].append(int(mask.sum()))
    return observed, bootstrap, state_months


def carry_hhi(sample, design, dates):
    original_hhi, quartile_hhi = table_robustness.hhi_frames(sample)
    hhi = original_hhi if design == "original_sign_split" else quartile_hhi
    return hhi.reindex(dates).mean(axis=1)


def run_design(
    sample_name,
    threshold_rule,
    threshold,
    design,
    frame,
    asset_names,
    carry_hhi_values,
    rng,
):
    high = frame.VIX.gt(threshold).to_numpy()
    observed, bootstrap, state_months = bootstrap_states(frame, asset_names, high, rng)
    rows = []
    pricing_rows = []
    for state, mask in {"high": high, "low": ~high}.items():
        result, errors = observed[state]
        values = np.asarray(bootstrap[state])
        rows.append(
            {
                "sample": sample_name,
                "threshold_rule": threshold_rule,
                "vix_threshold": threshold,
                "asset_design": design,
                "state": state,
                "months": int(mask.sum()),
                "start": frame.index[mask].min(),
                "end": frame.index[mask].max(),
                "mean_vix": frame.VIX.to_numpy()[mask].mean(),
                "mean_carry_hhi": carry_hhi_values.to_numpy()[mask].mean(),
                "j_statistic": result.j_statistic,
                "j_df": result.j_df,
                "asymptotic_pvalue": result.j_pvalue,
                "asymptotic_reject_5pct": result.j_pvalue < 0.05,
                "bootstrap_pvalue": (1 + np.sum(values >= result.j_statistic))
                / (BOOTSTRAP_DRAWS + 1),
                "bootstrap_reject_5pct": (
                    (1 + np.sum(values >= result.j_statistic)) / (BOOTSTRAP_DRAWS + 1)
                )
                < 0.05,
                "bootstrap_critical_95": np.quantile(values, 0.95),
                "bootstrap_null_size_at_chi2_5pct": np.mean(
                    values > stats.chi2.ppf(0.95, result.j_df)
                ),
                "bootstrap_state_months_mean": np.mean(state_months[state]),
                "bootstrap_draws": BOOTSTRAP_DRAWS,
                "block_months": BLOCK_MONTHS,
            }
        )
        for asset, error in errors.items():
            pricing_rows.append(
                {
                    "sample": sample_name,
                    "threshold_rule": threshold_rule,
                    "vix_threshold": threshold,
                    "asset_design": design,
                    "state": state,
                    "asset": asset,
                    "pricing_error_bp": error,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(pricing_rows)


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    vix = vix_series()
    samples = {
        "paper_sample_paired": table_robustness.construct_sample(False),
        "extended_sample_paired": table_robustness.construct_sample(True),
    }

    aligned = {}
    for sample_name, sample in samples.items():
        aligned[sample_name] = {}
        for design, key, names in [
            ("original_sign_split", "original", sample["original_names"]),
            ("full_slope_quartiles", "quartile", sample["quartile_names"]),
        ]:
            frame = sample[key].join(vix, how="inner").dropna()
            aligned[sample_name][design] = (frame, names)

    paper_vix = aligned["paper_sample_paired"]["original_sign_split"][0].VIX
    fixed_threshold = float(paper_vix.median())
    rng = np.random.default_rng(SEED)
    results = []
    pricing = []
    coverage = []
    for sample_name, sample_designs in aligned.items():
        sample_median = float(sample_designs["original_sign_split"][0].VIX.median())
        thresholds = [("fixed_paper_median", fixed_threshold)]
        if sample_name == "extended_sample_paired":
            thresholds.append(("sample_specific_median", sample_median))
        for design, (frame, names) in sample_designs.items():
            hhi = carry_hhi(samples[sample_name], design, frame.index)
            coverage.append(
                {
                    "sample": sample_name,
                    "asset_design": design,
                    "months": len(frame),
                    "start": frame.index.min(),
                    "end": frame.index.max(),
                    "vix_median": sample_median,
                    "fixed_paper_vix_median": fixed_threshold,
                }
            )
            for threshold_rule, threshold in thresholds:
                state_results, state_pricing = run_design(
                    sample_name,
                    threshold_rule,
                    threshold,
                    design,
                    frame,
                    names,
                    hhi,
                    rng,
                )
                results.append(state_results)
                pricing.append(state_pricing)

    pd.concat(results, ignore_index=True).to_csv(TABLES / "vix_state_j_tests.csv", index=False)
    pd.concat(pricing, ignore_index=True).to_csv(
        TABLES / "vix_state_pricing_errors.csv", index=False
    )
    pd.DataFrame(coverage).to_csv(TABLES / "vix_state_sample_coverage.csv", index=False)
    print(pd.concat(results, ignore_index=True).to_string(index=False))


if __name__ == "__main__":
    main()
