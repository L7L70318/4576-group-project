"""Diagnostics that explain Carry gaps without tuning the strategy."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data import construct_monthly, load_reference, monthly_snapshots
from .paths import ROOT, load_config

PAPER_START = pd.Timestamp("1970-01-31")
PAPER_END = pd.Timestamp("2011-09-30")
PUBLISHED_C5_MEAN = 16.34
PUBLISHED_C5_LONG_MEAN = 14.14
PUBLISHED_C5_SHORT_MEAN = 2.20


def apply_paper_windows(panel: pd.DataFrame) -> pd.DataFrame:
    """Apply the commodity-specific start/end dates in Internet-I."""
    windows = pd.read_csv(
        ROOT / "configs/paper_sample_dates.csv", parse_dates=["start", "end"]
    )
    result = panel.merge(windows, on="paper_code", how="inner", validate="many_to_one")
    in_window = result.date.ge(result.start) & result.date.le(result.end)
    return result[in_window].drop(columns=["start", "end"])


def construct_carry(panel: pd.DataFrame, policy: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construct C1-C5 and their legs under a named diagnostic policy."""
    if policy not in {"strict_sign", "rank"}:
        raise ValueError(policy)
    returns = []
    memberships = []
    for date, group in panel.groupby("date", sort=True):
        universe = group
        if "mapping_status" in universe:
            universe = universe[universe.mapping_status.eq("verified")]
        if "selected_futcode" in universe:
            universe = universe[universe.selected_futcode.notna()]
        universe = universe.dropna(subset=["slope"]).sort_values(["slope", "paper_code"])
        negative = universe[universe.slope.lt(0)]
        positive = universe[universe.slope.gt(0)]
        row = {"date": date, "n_signals": len(universe),
               "n_negative": len(negative), "n_positive": len(positive)}
        for k in range(1, 6):
            if policy == "strict_sign":
                long = negative.head(k)
                short = positive.tail(k)
                valid_formation = len(long) == len(short) == k
            else:
                long = universe.head(k)
                short = universe.tail(k)
                valid_formation = len(universe) >= 2 * k
            complete_returns = (
                valid_formation
                and long.excess_return.notna().all()
                and short.excess_return.notna().all()
            )
            if complete_returns:
                long_return = long.excess_return.mean()
                short_return = -short.excess_return.mean()
                row[f"C{k}_long"] = long_return
                row[f"C{k}_short"] = short_return
                row[f"C{k}"] = long_return + short_return
            else:
                row[f"C{k}_long"] = np.nan
                row[f"C{k}_short"] = np.nan
                row[f"C{k}"] = np.nan
            memberships.append({
                "date": date,
                "policy": policy,
                "k": k,
                "valid_formation": valid_formation,
                "complete_returns": complete_returns,
                "long_members": "|".join(long.paper_code) if valid_formation else "",
                "short_members": "|".join(short.paper_code) if valid_formation else "",
                "long_boundary_slope": long.slope.max() if valid_formation else np.nan,
                "short_boundary_slope": short.slope.min() if valid_formation else np.nan,
            })
        returns.append(row)
    return pd.DataFrame(returns).set_index("date"), pd.DataFrame(memberships)


def common_sample_statistics(returns: pd.DataFrame, policy: str) -> pd.DataFrame:
    """Recalculate C1-C5 on the intersection of all five valid-return months."""
    strategies = [f"C{k}" for k in range(1, 6)]
    common = returns[strategies].dropna()
    rows = []
    for strategy in strategies:
        x = common[strategy]
        annual_sd = x.std(ddof=1) * np.sqrt(12)
        rows.append({
            "policy": policy,
            "portfolio": strategy,
            "common_n": len(x),
            "start": x.index.min() if len(x) else pd.NaT,
            "end": x.index.max() if len(x) else pd.NaT,
            "annualized_mean_percent": x.mean() * 1200,
            "annualized_sd_percent": annual_sd * 100,
            "annualized_sharpe": x.mean() * 12 / annual_sd,
            "monthly_skewness": x.skew(),
            "positive_months_percent": x.gt(0).mean() * 100,
        })
    return pd.DataFrame(rows)


def period_statistics(returns: pd.DataFrame, policy: str) -> pd.DataFrame:
    periods = [
        ("1970-1989", "1970-01-31", "1989-12-31"),
        ("1990-1999", "1990-01-31", "1999-12-31"),
        ("2000-2011", "2000-01-31", "2011-09-30"),
    ]
    rows = []
    for period, start, end in periods:
        period_frame = returns.loc[start:end]
        for column, leg in [("C5", "total"), ("C5_long", "long"), ("C5_short", "short")]:
            x = period_frame[column].dropna()
            rows.append({
                "policy": policy,
                "period": period,
                "leg": leg,
                "calendar_months": len(pd.date_range(start, end, freq="ME")),
                "valid_months": len(x),
                "coverage_percent": len(x) / len(pd.date_range(start, end, freq="ME")) * 100,
                "annualized_mean_percent": x.mean() * 1200 if len(x) else np.nan,
                "monthly_mean_percent": x.mean() * 100 if len(x) else np.nan,
                "cumulative_return_percent": ((1 + x).prod() - 1) * 100 if len(x) else np.nan,
            })
    return pd.DataFrame(rows)


def comparison_metrics(left: pd.Series, right: pd.Series, label: str) -> dict:
    paired = pd.concat([left.rename("left"), right.rename("right")], axis=1).dropna()
    error = paired.left - paired.right
    return {
        "comparison": label,
        "n_common": len(paired),
        "start": paired.index.min() if len(paired) else pd.NaT,
        "end": paired.index.max() if len(paired) else pd.NaT,
        "left_annualized_mean_percent": paired.left.mean() * 1200,
        "right_annualized_mean_percent": paired.right.mean() * 1200,
        "annualized_mean_difference_pp": error.mean() * 1200,
        "monthly_correlation": paired.left.corr(paired.right) if len(paired) >= 2 else np.nan,
        "mae_bp": error.abs().mean() * 10_000,
        "rmse_bp": np.sqrt((error**2).mean()) * 10_000,
        "left_cumulative_return_percent": ((1 + paired.left).prod() - 1) * 100,
        "right_cumulative_return_percent": ((1 + paired.right).prod() - 1) * 100,
    }


def leg_summary(returns_by_case: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Summarize C5 long and short legs on each case's own valid C5 months."""
    rows = []
    for case, returns in returns_by_case.items():
        complete = returns.dropna(subset=["C5"])
        for column, leg in [("C5_long", "long"), ("C5_short", "short"), ("C5", "total")]:
            x = complete[column]
            rows.append(
                {
                    "case": case,
                    "leg": leg,
                    "n": len(x),
                    "annualized_mean_percent": x.mean() * 1200,
                    "cumulative_return_percent": ((1 + x).prod() - 1) * 100,
                }
            )
    for leg, mean in [
        ("long", PUBLISHED_C5_LONG_MEAN),
        ("short", PUBLISHED_C5_SHORT_MEAN),
        ("total", PUBLISHED_C5_MEAN),
    ]:
        rows.append(
            {
                "case": "published_table_9",
                "leg": leg,
                "n": 501,
                "annualized_mean_percent": mean,
                "cumulative_return_percent": np.nan,
            }
        )
    return pd.DataFrame(rows)


def membership_overlap(
    reconstructed: pd.DataFrame,
    author: pd.DataFrame,
    comparison: str,
) -> pd.DataFrame:
    """Compare C5 member sets without treating ordering as a disagreement."""
    paired = reconstructed.merge(author, on=["date", "k"], suffixes=("_left", "_right"))
    paired = paired[paired.k.eq(5) & paired.complete_returns_left & paired.complete_returns_right]
    rows = []
    for row in paired.itertuples():
        left_long = set(filter(None, row.long_members_left.split("|")))
        right_long = set(filter(None, row.long_members_right.split("|")))
        left_short = set(filter(None, row.short_members_left.split("|")))
        right_short = set(filter(None, row.short_members_right.split("|")))

        def jaccard(left, right):
            return len(left & right) / len(left | right) if left | right else np.nan

        rows.append(
            {
                "comparison": comparison,
                "date": row.date,
                "long_identical": left_long == right_long,
                "short_identical": left_short == right_short,
                "long_jaccard": jaccard(left_long, right_long),
                "short_jaccard": jaccard(left_short, right_short),
                "reconstructed_long": "|".join(sorted(left_long)),
                "author_long": "|".join(sorted(right_long)),
                "reconstructed_short": "|".join(sorted(left_short)),
                "author_short": "|".join(sorted(right_short)),
            }
        )
    return pd.DataFrame(rows)


def member_contract_audit(
    panel: pd.DataFrame,
    memberships: pd.DataFrame,
    sample: str,
    policy: str,
) -> pd.DataFrame:
    """Attach selected signal and holding contracts to valid C5 memberships."""
    selected = memberships[
        memberships.k.eq(5) & memberships.complete_returns
    ][["date", "long_members", "short_members"]]
    rows = []
    for row in selected.itertuples():
        for side, values in [("long", row.long_members), ("short", row.short_members)]:
            rows.extend(
                {"date": row.date, "paper_code": code, "side": side}
                for code in values.split("|")
                if code
            )
    members = pd.DataFrame(rows)
    if members.empty:
        return members
    audit_columns = [
        "date",
        "formation_date",
        "paper_code",
        "slope",
        "front_futcode",
        "second_futcode",
        "selected_futcode",
        "selected_contract_rank",
        "notice_source",
        "expiry",
        "excess_return",
    ]
    audit_columns = [column for column in audit_columns if column in panel]
    result = members.merge(
        panel[audit_columns], on=["date", "paper_code"], validate="one_to_one"
    )
    result.insert(0, "policy", policy)
    result.insert(0, "sample", sample)
    return result


def _author_reference_comparison(reconstructed: dict[str, pd.DataFrame], out: Path):
    """Use both plausible author slope labels; do not select a winner silently."""
    try:
        reference = load_reference()
    except (FileNotFoundError, StopIteration):
        pd.DataFrame([{
            "status": "unavailable",
            "required_file": "data/raw/paper_reference/FuturesReturn_to_post.xls",
            "public_archive": "mnsc.2017.2840-sm_data.zip",
            "publisher_url": (
                "https://pubsonline.informs.org/doi/suppl/10.1287/"
                "mnsc.2017.2840/suppl_file/mnsc.2017.2840-sm_data.zip"
            ),
            "retrieval_result": (
                "local copy was removed per cleanup manifest; publisher returned HTTP 403; "
                "no Git, Trash, snapshot, or Wayback copy found"
            ),
            "consequence": "author monthly factor correlation, errors, and membership overlap not identified",
        }]).to_csv(out / "author_reference_status.csv", index=False)
        return

    reference = reference.copy()
    reference["mapping_status"] = "verified"
    reference["selected_futcode"] = 1
    reference["formation_date"] = reference.date - pd.offsets.MonthEnd(1)
    outputs = []
    author_memberships = []
    overlap_outputs = []
    for timing in ["slope_same_row", "slope_lagged_one_row"]:
        author = reference.copy()
        if timing == "slope_lagged_one_row":
            signals = author.pivot(index="date", columns="paper_code", values="slope")
            signals = signals.reindex(pd.date_range(signals.index.min(), signals.index.max(), freq="ME")).shift(1)
            author = author.drop(columns="slope").merge(
                signals.stack().rename("slope").reset_index(),
                on=["date", "paper_code"], how="left")
        for policy in ["strict_sign", "rank"]:
            author_returns, membership = construct_carry(author, policy)
            membership["author_slope_timing"] = timing
            author_memberships.append(membership)
            key = f"paper_windows_{policy}"
            outputs.append(comparison_metrics(
                reconstructed[key]["C5"], author_returns["C5"],
                f"{key}_vs_author_{timing}"))
            reconstructed_membership = pd.read_csv(
                out / "carry_memberships.csv", parse_dates=["date"]
            )
            reconstructed_membership = reconstructed_membership[
                reconstructed_membership["sample"].eq("paper_windows")
                & reconstructed_membership["policy"].eq(policy)
            ]
            overlap_outputs.append(
                membership_overlap(
                    reconstructed_membership,
                    membership,
                    f"{key}_vs_author_{timing}",
                )
            )
    pd.DataFrame(outputs).to_csv(out / "author_c5_common_month_metrics.csv", index=False)
    pd.concat(author_memberships, ignore_index=True).to_csv(
        out / "author_carry_memberships.csv", index=False)
    pd.concat(overlap_outputs, ignore_index=True).to_csv(
        out / "author_c5_membership_overlap.csv", index=False
    )


def build_discrepancy_audit(folder=None):
    if folder is None:
        sample = load_config()["sample"]
        folder = ROOT / "data/raw/wrds_contracts" / (
            f"{sample['download_start']}_{sample['proposed_wrds_end']}")
    out = ROOT / "results/replication/carry_discrepancy"
    out.mkdir(parents=True, exist_ok=True)

    snapshots, _, _ = monthly_snapshots(folder, prefer_actual_notice=True)
    full = construct_monthly(snapshots, max_quote_age=7)
    paper_range = full[full.date.between(PAPER_START, PAPER_END)]
    paper_windows = apply_paper_windows(paper_range)

    panels = {"full_wrds": paper_range, "paper_windows": paper_windows}
    returns_by_case = {}
    memberships = []
    contract_audits = []
    common_stats = []
    periods = []
    for sample_name, panel in panels.items():
        for policy in ["strict_sign", "rank"]:
            key = f"{sample_name}_{policy}"
            returns, membership = construct_carry(panel, policy)
            returns_by_case[key] = returns
            returns.to_csv(out / f"{key}_returns.csv")
            membership["sample"] = sample_name
            memberships.append(membership)
            contract_audits.append(
                member_contract_audit(panel, membership, sample_name, policy)
            )
            stats = common_sample_statistics(returns, policy)
            stats["sample"] = sample_name
            common_stats.append(stats)
            period = period_statistics(returns, policy)
            period["sample"] = sample_name
            periods.append(period)

    pd.concat(memberships, ignore_index=True).to_csv(out / "carry_memberships.csv", index=False)
    pd.concat(contract_audits, ignore_index=True).to_csv(
        out / "c5_member_contract_audit.csv", index=False
    )
    pd.concat(common_stats, ignore_index=True).to_csv(out / "c1_c5_common_sample.csv", index=False)
    pd.concat(periods, ignore_index=True).to_csv(out / "c5_period_and_leg_decomposition.csv", index=False)
    leg_summary(returns_by_case).to_csv(out / "c5_leg_summary.csv", index=False)

    calendar = pd.date_range(PAPER_START, PAPER_END, freq="ME")
    missing_rows = []
    for case, returns in returns_by_case.items():
        missing = returns["C5"].reindex(calendar).isna()
        missing_rows.extend(
            {"case": case, "date": date} for date in calendar[missing]
        )
    pd.DataFrame(missing_rows).to_csv(out / "c5_missing_months.csv", index=False)

    comparisons = []
    for policy in ["strict_sign", "rank"]:
        comparisons.append(comparison_metrics(
            returns_by_case[f"full_wrds_{policy}"]["C5"],
            returns_by_case[f"paper_windows_{policy}"]["C5"],
            f"full_wrds_vs_paper_windows_{policy}"))
    no_barley, _ = construct_carry(
        paper_windows[paper_windows.paper_code.ne("LBR")], "rank")
    comparisons.append(comparison_metrics(
        returns_by_case["paper_windows_rank"]["C5"], no_barley["C5"],
        "paper_windows_rank_with_vs_without_barley"))
    pd.DataFrame(comparisons).to_csv(out / "sample_and_barley_effects.csv", index=False)

    accounting = []
    for case in ["full_wrds_rank", "paper_windows_rank"]:
        rank = returns_by_case[case]["C5"].dropna()
        missing_months = 501 - len(rank)
        implied_missing_annual_mean = (
            (PUBLISHED_C5_MEAN / 1200 * 501 - rank.mean() * len(rank))
            / missing_months * 1200
            if missing_months
            else np.nan
        )
        accounting.append(
            {
                "case": case,
                "published_months": 501,
                "reconstructed_months": len(rank),
                "missing_months": missing_months,
                "published_annualized_mean_percent": PUBLISHED_C5_MEAN,
                "reconstructed_annualized_mean_percent": rank.mean() * 1200,
                "gap_percentage_points": PUBLISHED_C5_MEAN - rank.mean() * 1200,
                "implied_missing_months_annualized_mean_percent": implied_missing_annual_mean,
            }
        )
    pd.DataFrame(accounting).to_csv(out / "coverage_gap_accounting.csv", index=False)

    _author_reference_comparison(returns_by_case, out)
    return returns_by_case
