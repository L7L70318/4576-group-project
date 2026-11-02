"""Reproducible diagnostics for the paper's carry-factor data construction."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import construct_monthly, momentum_signal, monthly_snapshots
from .paths import PROCESSING_EXPLORATION, ROOT, load_config
from .portfolios import make_portfolios

PAPER_START = pd.Timestamp("1970-01-31")
PAPER_END = pd.Timestamp("2011-09-30")
EXTENSION_START = pd.Timestamp("2012-01-31")
EXTENSION_END = pd.Timestamp("2025-12-31")


def carry_availability(panel: pd.DataFrame) -> pd.DataFrame:
    """Count formation-time signals and explain every unavailable strict C5.

    ``panel.date`` labels the realized return month, whereas
    ``panel.formation_date`` and ``panel.slope`` refer to the preceding
    formation month.  Output retains both labels to make that timing explicit.
    """
    rows = []
    for return_date, group in panel.groupby("date", sort=True):
        universe = group
        if "mapping_status" in universe:
            universe = universe[universe.mapping_status.eq("verified")]
        mapped = len(universe)
        holding = universe
        if "selected_futcode" in holding:
            holding = holding[holding.selected_futcode.notna()]
        signals = holding[holding.slope.notna()]
        backwardation = int(signals.slope.lt(0).sum())
        contango = int(signals.slope.gt(0).sum())
        flat = int(signals.slope.eq(0).sum())
        strict = backwardation >= 5 and contango >= 5
        if strict:
            reason = "available"
        elif len(holding) < 10:
            reason = "insufficient_holding_contract_coverage"
        elif len(signals) < 10:
            reason = "missing_curve_signal"
        elif backwardation < 5 and contango < 5:
            reason = "both_sides_below_five_due_to_flat_or_imbalance"
        elif backwardation < 5:
            reason = "market_imbalance_fewer_than_five_backwardated"
        else:
            reason = "market_imbalance_fewer_than_five_contangoed"
        formation = group.formation_date.dropna()
        rows.append({
            "formation_date": formation.iloc[0] if len(formation) else return_date - pd.offsets.MonthEnd(1),
            "return_date": return_date,
            "mapped_commodities": mapped,
            "holding_eligible": len(holding),
            "valid_carry_signals": len(signals),
            "backwardation": backwardation,
            "contango": contango,
            "flat": flat,
            "strict_c5_available": strict,
            "failure_reason": reason,
        })
    return pd.DataFrame(rows).sort_values("return_date")


def return_statistics(returns: pd.DataFrame) -> pd.DataFrame:
    """Paper-matched statistics: arithmetic monthly returns, annualized."""
    aliases = {"C5": "CARRY", "M5": "CMOM"}
    columns = [f"C{k}" for k in range(1, 6)] + [f"M{k}" for k in range(1, 6)] + ["AVG"]
    rows = []
    for name in columns:
        if name not in returns:
            continue
        x = returns[name].dropna()
        annual_sd = x.std(ddof=1) * np.sqrt(12)
        rows.append({
            "portfolio": name,
            "factor_alias": aliases.get(name, name),
            "n": x.size,
            "start": x.index.min() if x.size else pd.NaT,
            "end": x.index.max() if x.size else pd.NaT,
            "annualized_mean_percent": x.mean() * 1200 if x.size else np.nan,
            "annualized_sd_percent": annual_sd * 100 if x.size else np.nan,
            "annualized_sharpe": x.mean() * 12 / annual_sd if annual_sd > 0 else np.nan,
            "monthly_skewness": x.skew() if x.size else np.nan,
            "positive_months_percent": x.gt(0).mean() * 100 if x.size else np.nan,
        })
    return pd.DataFrame(rows)


def compare_carry_policies(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare the primary rank rule with transparent sign-based diagnostics."""
    rows = []
    for date, group in panel.groupby("date", sort=True):
        universe = group
        if "mapping_status" in universe:
            universe = universe[universe.mapping_status.eq("verified")]
        if "selected_futcode" in universe:
            universe = universe[universe.selected_futcode.notna()]
        universe = universe.dropna(subset=["slope"]).sort_values(["slope", "paper_code"])
        negative = universe[universe.slope.lt(0)].head(5)
        positive = universe[universe.slope.gt(0)].tail(5)

        def portfolio_return(long, short):
            positions = pd.concat([long.assign(weight=0.2), short.assign(weight=-0.2)])
            if positions.empty or positions.excess_return.isna().any():
                return np.nan
            return (positions.weight * positions.excess_return).sum()

        strict = portfolio_return(negative, positive) if len(negative) == len(positive) == 5 else np.nan
        cash_slots = portfolio_return(negative, positive)
        extremes = (
            portfolio_return(universe.head(5), universe.tail(5))
            if len(universe) >= 10 else np.nan
        )
        rows.append({
            "date": date,
            "n_signals": len(universe),
            "n_negative": universe.slope.lt(0).sum(),
            "n_positive": universe.slope.gt(0).sum(),
            "strict_sign_c5": strict,
            "signed_cash_slots_c5": cash_slots,
            "unconditional_extremes_c5": extremes,
        })
    monthly = pd.DataFrame(rows).set_index("date")
    summary = []
    for policy in ["strict_sign_c5", "signed_cash_slots_c5", "unconditional_extremes_c5"]:
        returns = monthly[policy].dropna()
        annual_sd = returns.std(ddof=1) * np.sqrt(12)
        summary.append({
            "policy": policy,
            "n": len(returns),
            "start": returns.index.min() if len(returns) else pd.NaT,
            "end": returns.index.max() if len(returns) else pd.NaT,
            "annualized_mean_percent": returns.mean() * 1200,
            "annualized_sd_percent": annual_sd * 100,
            "annualized_sharpe": returns.mean() * 12 / annual_sd,
            "monthly_skewness": returns.skew(),
            "positive_months_percent": returns.gt(0).mean() * 100,
        })
    return monthly, pd.DataFrame(summary)


def compare_panels(old: pd.DataFrame, revised: pd.DataFrame) -> pd.DataFrame:
    """Row-level old/new selection and return comparison."""
    fields = ["paper_code", "date", "formation_date", "selected_futcode",
              "notice_cutoff", "notice_source", "excess_return"]
    joined = old[fields].merge(
        revised[fields], on=["paper_code", "date", "formation_date"], how="outer",
        suffixes=("_old", "_revised"), validate="one_to_one")
    old_code = joined.selected_futcode_old.fillna(-1)
    new_code = joined.selected_futcode_revised.fillna(-1)
    joined["selection_changed"] = old_code.ne(new_code)
    joined["return_difference_bp"] = (
        joined.excess_return_revised - joined.excess_return_old) * 10_000
    return joined


def build_audit(folder=None):
    """Rebuild old/revised panels and write all carry-audit artifacts."""
    if folder is None:
        settings = load_config()["sample"]
        folder = ROOT / "data/raw/wrds_contracts" / (
            f"{settings['download_start']}_{settings['proposed_wrds_end']}")
    out = PROCESSING_EXPLORATION / "results" / "carry_audit"
    out.mkdir(parents=True, exist_ok=True)

    revised_snapshots, _, revised_meta = monthly_snapshots(folder, prefer_actual_notice=True)
    old_snapshots, _, old_meta = monthly_snapshots(folder, prefer_actual_notice=False)
    revised = construct_monthly(revised_snapshots, max_quote_age=7)
    old = construct_monthly(old_snapshots, max_quote_age=7)
    revised = revised.merge(momentum_signal(revised), on=["paper_code", "date"], how="left")
    old = old.merge(momentum_signal(old), on=["paper_code", "date"], how="left")

    availability = carry_availability(revised)
    availability.to_csv(out / "c5_monthly_availability.csv", index=False)
    availability.assign(period=np.select(
        [availability.return_date.between(PAPER_START, PAPER_END),
         availability.return_date.between(EXTENSION_START, EXTENSION_END)],
        ["paper_1970_2011", "extension_2012_2025"], default="outside")) \
        .groupby(["period", "failure_reason"], dropna=False).agg(
            months=("return_date", "size"),
            strict_c5_months=("strict_c5_available", "sum"),
            min_signals=("valid_carry_signals", "min"),
            min_backwardation=("backwardation", "min"),
            min_contango=("contango", "min")) \
        .reset_index().to_csv(out / "c5_availability_summary.csv", index=False)

    old_returns, _ = make_portfolios(old)
    revised_returns, revised_weights = make_portfolios(revised)
    old_returns.loc[PAPER_START:PAPER_END].to_csv(out / "factor_returns_old_fnd_paper_sample.csv")
    revised_returns.loc[PAPER_START:PAPER_END].to_csv(out / "factor_returns_revised_paper_sample.csv")
    revised_weights.to_csv(out / "portfolio_weights_revised.csv", index=False)
    return_statistics(old_returns.loc[PAPER_START:PAPER_END]).to_csv(
        out / "table_1_2_statistics_old_fnd.csv", index=False)
    return_statistics(revised_returns.loc[PAPER_START:PAPER_END]).to_csv(
        out / "table_1_2_statistics_revised.csv", index=False)
    return_statistics(revised_returns.loc[EXTENSION_START:EXTENSION_END]).to_csv(
        out / "table_1_2_statistics_extension.csv", index=False)
    policy_monthly, policy_summary = compare_carry_policies(
        revised[revised.date.between(PAPER_START, PAPER_END)])
    policy_monthly.to_csv(out / "carry_policy_monthly_comparison.csv")
    policy_summary.to_csv(out / "carry_policy_summary.csv", index=False)

    changes = compare_panels(old, revised)
    changes.to_csv(out / "fnd_selection_return_comparison.csv", index=False)
    source_coverage = revised_meta.groupby(["paper_code", "notice_source"]).size().rename("contracts").reset_index()
    source_coverage.to_csv(out / "fnd_source_coverage.csv", index=False)
    old_meta.groupby(["paper_code", "notice_source"]).size().rename("contracts").reset_index().to_csv(
        out / "fnd_source_coverage_old.csv", index=False)
    return revised, revised_returns, availability, changes
