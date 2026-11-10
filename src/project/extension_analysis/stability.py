"""Rolling GMM J-tests and unknown-break tests for pricing moments."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from project.asset_pricing import (
    FACTORS,
    PAPER_END,
    TEST_ASSETS,
    construct_baseline_returns,
    estimate_gmm,
    long_run_covariance,
)
from project.paths import EXTENSION_FIGURES, EXTENSION_TABLES, ROOT

TABLES = EXTENSION_TABLES / "01_instability"
FIGURES = EXTENSION_FIGURES / "01_instability"
ROLLING_WINDOWS = [120, 180]
ROLLING_STEP = 12
NW_LAGS = 3
BREAK_TRIM = 0.15
BREAK_GRID_STEP = 12
BOOTSTRAP_DRAWS = 999
BOOTSTRAP_BLOCK = 12
MAX_BREAKS = 3
MIN_SEGMENT = 60
SEED = 4576


def extended_common_sample() -> pd.DataFrame:
    panel = pd.read_parquet(ROOT / "data/processed/commodity_monthly.parquet")
    windows = pd.read_csv(ROOT / "configs/paper_sample_dates.csv", parse_dates=["start", "end"])
    windows.loc[windows.end.eq(PAPER_END), "end"] = panel.date.max()
    data = panel.merge(windows, on="paper_code", validate="many_to_one")
    data = data[data.date.ge(data.start) & data.date.le(data.end)].drop(columns=["start", "end"])
    assets, factors, _ = construct_baseline_returns(data)
    return pd.concat([assets[TEST_ASSETS], factors[FACTORS]], axis=1).dropna()


def rolling_j_tests(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for window in ROLLING_WINDOWS:
        for end in range(window, len(data) + 1, ROLLING_STEP):
            sample = data.iloc[end - window : end]
            _, result = estimate_gmm(sample[TEST_ASSETS], sample[FACTORS])
            rows.append(
                {
                    "window_months": window,
                    "start": sample.index.min(),
                    "end": sample.index.max(),
                    "observations": len(sample),
                    "j_statistic": result.j_statistic,
                    "j_df": result.j_df,
                    "j_pvalue": result.j_pvalue,
                    "reject_5_percent": result.j_pvalue < 0.05,
                    "nw_lags": result.nw_lags,
                }
            )
    return pd.DataFrame(rows)


def plot_rolling_tests(results: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for axis, window in zip(axes, ROLLING_WINDOWS):
        values = results[results.window_months.eq(window)]
        axis.plot(values.end, values.j_pvalue, marker="o", markersize=3, linewidth=1.5)
        axis.axhline(0.05, color="crimson", linestyle="--", linewidth=1, label="5% level")
        axis.fill_between(
            values.end,
            0,
            0.05,
            color="crimson",
            alpha=0.08,
            label="rejection region",
        )
        axis.set_ylim(0, 1)
        axis.set_ylabel("Two-step GMM J-test p-value")
        axis.set_title(f"Rolling {window // 12}-year window ({window} complete months)")
        axis.grid(alpha=0.25)
        axis.legend(loc="upper left")
    axes[-1].set_xlabel("Window end month")
    figure.suptitle("Time stability of the original three-factor commodity model")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def pricing_moments(data: pd.DataFrame):
    _, result = estimate_gmm(data[TEST_ASSETS], data[FACTORS])
    returns = data[TEST_ASSETS].to_numpy(float)
    factors = data[FACTORS].to_numpy(float)
    centered = factors - factors.mean(axis=0)
    sdf = 1 - centered @ result.b
    moments = returns * sdf[:, None]
    return moments, result


def break_candidates(nobs: int) -> np.ndarray:
    lower = max(MIN_SEGMENT, int(np.ceil(BREAK_TRIM * nobs)))
    upper = min(nobs - MIN_SEGMENT, int(np.floor((1 - BREAK_TRIM) * nobs)))
    return np.arange(lower, upper + 1, BREAK_GRID_STEP)


def sup_wald(values: np.ndarray, candidates: np.ndarray):
    rows = []
    for split in candidates:
        left, right = values[:split], values[split:]
        difference = left.mean(axis=0) - right.mean(axis=0)
        covariance = long_run_covariance(left, NW_LAGS) / len(left) + long_run_covariance(
            right, NW_LAGS
        ) / len(right)
        rank = np.linalg.matrix_rank(covariance, tol=1e-10)
        statistic = float(difference @ np.linalg.pinv(covariance, rcond=1e-10) @ difference)
        rows.append((split, statistic, rank))
    return max(rows, key=lambda row: row[1]), rows


def circular_indices(size: int, block: int, rng: np.random.Generator) -> np.ndarray:
    starts = rng.integers(0, size, size=int(np.ceil(size / block)))
    return ((starts[:, None] + np.arange(block)) % size).ravel()[:size]


def bootstrap_sup_wald(values: np.ndarray, candidates: np.ndarray):
    rng = np.random.default_rng(SEED)
    centered = values - values.mean(axis=0)
    statistics = np.empty(BOOTSTRAP_DRAWS)
    for draw in range(BOOTSTRAP_DRAWS):
        indices = circular_indices(len(values), BOOTSTRAP_BLOCK, rng)
        statistics[draw] = sup_wald(centered[indices], candidates)[0][1]
    return statistics


def segment_cost_setup(values: np.ndarray):
    covariance = np.cov(values, rowvar=False, ddof=0)
    weight = np.linalg.pinv(covariance, rcond=1e-10, hermitian=True)
    cumulative = np.vstack([np.zeros(values.shape[1]), np.cumsum(values, axis=0)])
    quadratic = np.r_[
        0.0,
        np.cumsum(np.einsum("ti,ij,tj->t", values, weight, values)),
    ]

    def cost(start: int, end: int) -> float:
        count = end - start
        mean = (cumulative[end] - cumulative[start]) / count
        return float(quadratic[end] - quadratic[start] - count * mean @ weight @ mean)

    return cost


def bai_perron(values: np.ndarray, dates: pd.DatetimeIndex) -> pd.DataFrame:
    nobs, moment_count = values.shape
    cost = segment_cost_setup(values)
    rows = []
    for break_count in range(MAX_BREAKS + 1):
        segment_count = break_count + 1
        objective = np.full((segment_count + 1, nobs + 1), np.inf)
        previous = np.full((segment_count + 1, nobs + 1), -1, dtype=int)
        objective[0, 0] = 0.0
        for segment in range(1, segment_count + 1):
            for end in range(segment * MIN_SEGMENT, nobs + 1):
                lower = (segment - 1) * MIN_SEGMENT
                upper = end - MIN_SEGMENT
                for start in range(lower, upper + 1):
                    candidate = objective[segment - 1, start] + cost(start, end)
                    if candidate < objective[segment, end]:
                        objective[segment, end] = candidate
                        previous[segment, end] = start
        ssr = objective[segment_count, nobs]
        cuts = []
        end = nobs
        for segment in range(segment_count, 1, -1):
            cut = previous[segment, end]
            cuts.append(cut)
            end = cut
        cuts.reverse()
        parameter_count = segment_count * moment_count
        bic = nobs * moment_count * np.log(ssr / (nobs * moment_count)) + parameter_count * np.log(
            nobs
        )
        rows.append(
            {
                "break_count": break_count,
                "segment_count": segment_count,
                "minimum_segment_months": MIN_SEGMENT,
                "weighted_ssr": ssr,
                "bic": bic,
                "break_indices": "|".join(str(cut) for cut in cuts),
                "last_month_before_break": "|".join(
                    dates[cut - 1].date().isoformat() for cut in cuts
                ),
                "first_month_after_break": "|".join(dates[cut].date().isoformat() for cut in cuts),
            }
        )
    result = pd.DataFrame(rows)
    result["bic_selected"] = result.bic.eq(result.bic.min())
    return result


def main():
    data = extended_common_sample()
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    rolling = rolling_j_tests(data)
    rolling.to_csv(TABLES / "rolling_gmm_j_tests.csv", index=False)
    plot_rolling_tests(rolling, FIGURES / "rolling_gmm_j_tests.png")

    moments, full_result = pricing_moments(data)
    candidates = break_candidates(len(data))
    observed, candidate_rows = sup_wald(moments, candidates)
    bootstrap = bootstrap_sup_wald(moments, candidates)
    pvalue = (1 + np.sum(bootstrap >= observed[1])) / (BOOTSTRAP_DRAWS + 1)
    critical_95 = float(np.quantile(bootstrap, 0.95))
    candidate_table = pd.DataFrame(
        [
            {
                "split_index": split,
                "last_month_before_break": data.index[split - 1],
                "first_month_after_break": data.index[split],
                "wald_statistic": statistic,
                "covariance_rank": rank,
            }
            for split, statistic, rank in candidate_rows
        ]
    )
    candidate_table.to_csv(TABLES / "sup_wald_candidates.csv", index=False)
    pd.DataFrame({"bootstrap_sup_wald": bootstrap}).to_csv(
        TABLES / "sup_wald_bootstrap.csv", index=False
    )

    bp = bai_perron(moments, data.index)
    bp.to_csv(TABLES / "bai_perron_break_selection.csv", index=False)
    selected = bp.loc[bp.bic.idxmin()]
    summary = {
        "sample_months": len(data),
        "sample_start": data.index.min().date().isoformat(),
        "sample_end": data.index.max().date().isoformat(),
        "full_sample_gmm_j_pvalue": float(full_result.j_pvalue),
        "sup_wald_statistic": float(observed[1]),
        "sup_wald_covariance_rank": int(observed[2]),
        "sup_wald_last_month_before_break": data.index[observed[0] - 1].date().isoformat(),
        "sup_wald_first_month_after_break": data.index[observed[0]].date().isoformat(),
        "sup_wald_bootstrap_pvalue": float(pvalue),
        "sup_wald_bootstrap_95_critical": critical_95,
        "sup_wald_candidate_step_complete_months": int(BREAK_GRID_STEP),
        "sup_wald_trim_fraction": float(BREAK_TRIM),
        "bootstrap_draws": int(BOOTSTRAP_DRAWS),
        "bootstrap_block_months": int(BOOTSTRAP_BLOCK),
        "bai_perron_selected_break_count": int(selected.break_count),
        "bai_perron_selected_first_months": selected.first_month_after_break,
        "bai_perron_minimum_segment_months": int(MIN_SEGMENT),
        "inference_caveat": (
            "Break tests use pricing moments from the fixed full-sample SDF; "
            "the sup-Wald block bootstrap imposes a constant mean but does not "
            "re-estimate SDF nuisance parameters in each draw."
        ),
    }
    (TABLES / "break_test_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print("\nROLLING REJECTION COUNTS")
    print(rolling.groupby("window_months").reject_5_percent.agg(["sum", "count", "mean"]))
    print("\nBAI-PERRON")
    print(bp.to_string(index=False))


if __name__ == "__main__":
    main()
