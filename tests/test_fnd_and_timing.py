import numpy as np
import pandas as pd

from project.carry_audit import carry_availability, compare_carry_policies
from project.data import construct_monthly, prepare_metadata
from project.portfolios import make_portfolios


def _mapping():
    return pd.DataFrame([
        {"contrcode": 1, "paper_code": "C", "commodity": "Corn",
         "sector": "agriculture", "priority": 1, "status": "verified", "note": ""},
        {"contrcode": 2, "paper_code": "FC", "commodity": "Feeder cattle",
         "sector": "livestock", "priority": 1, "status": "verified", "note": ""},
    ])


def _meta(contrcode, futcode, contrdate, fnd=None, ltd="2020-03-20"):
    return {
        "contrcode": contrcode, "futcode": futcode, "contrdate": contrdate,
        "lasttrddate": ltd, "expirationdate": None, "firstnoticedate": fnd,
        "startdate": "2019-01-01",
    }


def test_actual_fnd_and_cash_ltd_override_convention():
    raw = pd.DataFrame([
        _meta(1, 11, "0320", fnd="2020-02-12"),
        _meta(1, 12, "0520", fnd=None, ltd="2020-05-20"),
        _meta(2, 21, "0320", fnd=None, ltd="2020-03-26"),
    ])
    result = prepare_metadata(raw, _mapping(), prefer_actual_notice=True).set_index("futcode")
    assert result.loc[11, "notice_cutoff"] == pd.Timestamp("2020-02-12")
    assert result.loc[11, "notice_source"] == "vendor_firstnoticedate"
    assert result.loc[12, "notice_source"] == "table_I_convention_US_federal_calendar_proxy"
    assert result.loc[21, "notice_cutoff"] == pd.Timestamp("2020-03-26")
    assert result.loc[21, "notice_source"] == "vendor_lasttrddate_cash_settled"


def test_holding_return_uses_same_contract_and_safe_later_rank():
    formation = pd.Timestamp("2020-01-31")
    realization = pd.Timestamp("2020-02-29")
    base = {
        "contrcode": 1, "paper_code": "C", "commodity": "Corn",
        "sector": "agriculture", "priority": 1, "status": "verified",
        "quote_age_days": 0, "firstnoticedate": pd.NaT,
        "notice_source": "test", "notice_is_exact": True, "open_interest": 1.0,
    }
    rows = []
    # Rank 2 is unsafe before the intended exit, so rank 3 must be selected.
    contracts = [
        (101, "2020-02-01", 90.0, "2020-02-10"),
        (102, "2020-03-01", 100.0, "2020-02-20"),
        (103, "2020-04-01", 110.0, "2020-03-15"),
    ]
    for futcode, maturity, price, cutoff in contracts:
        rows.append({**base, "futcode": futcode, "date": formation,
                     "quote_date": formation, "maturity": pd.Timestamp(maturity),
                     "expiry": pd.Timestamp(maturity) + pd.offsets.MonthEnd(0),
                     "settlement": price, "notice_cutoff": pd.Timestamp(cutoff)})
    rows.append({**base, "futcode": 103, "date": realization,
                 "quote_date": realization, "maturity": pd.Timestamp("2020-04-01"),
                 "expiry": pd.Timestamp("2020-04-30"), "settlement": 121.0,
                 "notice_cutoff": pd.Timestamp("2020-03-15")})
    panel = construct_monthly(pd.DataFrame(rows), max_quote_age=7)
    selected = panel.loc[panel.date.eq(realization)].iloc[0]
    assert selected.selected_futcode == 103
    assert selected.selected_contract_rank == 3
    assert np.isclose(selected.excess_return, 0.10)


def test_carry_weights_use_formation_slope_for_same_rows_return():
    date = pd.Timestamp("2020-02-29")
    rows = []
    for i in range(10):
        rows.append({
            "paper_code": f"X{i}", "date": date,
            "formation_date": pd.Timestamp("2020-01-31"),
            "slope": (-10 + i) / 100 if i < 5 else (i - 4) / 100,
            "excess_return": 0.01 * (i + 1), "mapping_status": "verified",
            "selected_futcode": i + 1, "sector": "test",
        })
    panel = pd.DataFrame(rows)
    _, weights = make_portfolios(panel)
    c5 = weights[weights.portfolio.eq("C5")].set_index("paper_code").weight
    assert set(c5[c5.gt(0)].index) == {f"X{i}" for i in range(5)}
    assert set(c5[c5.lt(0)].index) == {f"X{i}" for i in range(5, 10)}
    assert weights.date.nunique() == 1 and weights.date.iloc[0] == date


def test_availability_distinguishes_market_sign_imbalance():
    date = pd.Timestamp("2020-02-29")
    panel = pd.DataFrame({
        "paper_code": [f"X{i}" for i in range(10)],
        "date": date,
        "formation_date": pd.Timestamp("2020-01-31"),
        "slope": [-0.1] * 4 + [0.1] * 6,
        "mapping_status": "verified",
        "selected_futcode": range(10),
    })
    result = carry_availability(panel).iloc[0]
    assert not result.strict_c5_available
    assert result.failure_reason == "market_imbalance_fewer_than_five_backwardated"


def test_carry_policy_alternatives_do_not_replace_strict_baseline():
    date = pd.Timestamp("2020-02-29")
    panel = pd.DataFrame({
        "paper_code": [f"X{i}" for i in range(10)],
        "date": date,
        "slope": [-0.1] * 4 + [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
        "excess_return": np.arange(10) / 100,
        "mapping_status": "verified",
        "selected_futcode": range(10),
    })
    monthly, _ = compare_carry_policies(panel)
    row = monthly.iloc[0]
    assert pd.isna(row.strict_sign_c5)
    assert pd.notna(row.signed_cash_slots_c5)
    assert pd.notna(row.unconditional_extremes_c5)
