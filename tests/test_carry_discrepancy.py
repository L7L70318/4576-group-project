import pandas as pd

from project.carry_discrepancy import (
    comparison_metrics,
    construct_carry,
    membership_overlap,
)


def _panel(slopes, returns, date="2000-01-31"):
    codes = [f"X{i}" for i in range(len(slopes))]
    return pd.DataFrame(
        {
            "date": pd.Timestamp(date),
            "paper_code": codes,
            "slope": slopes,
            "excess_return": returns,
            "mapping_status": "verified",
            "selected_futcode": range(len(slopes)),
        }
    )


def test_rank_and_strict_are_preserved_as_distinct_diagnostics():
    panel = _panel(
        [-0.5, -0.4, -0.3, -0.2, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        [0.01] * 10,
    )
    strict, _ = construct_carry(panel, "strict_sign")
    rank, _ = construct_carry(panel, "rank")
    assert pd.isna(strict.iloc[0].C5)
    assert rank.iloc[0].C5 == 0


def test_comparison_metrics_only_uses_exact_common_months():
    index = pd.to_datetime(["2000-01-31", "2000-02-29", "2000-03-31"])
    left = pd.Series([0.01, 0.02, None], index=index)
    right = pd.Series([0.00, None, 0.03], index=index)
    result = comparison_metrics(left, right, "test")
    assert result["n_common"] == 1
    assert result["annualized_mean_difference_pp"] == 12
    assert result["mae_bp"] == 100


def test_membership_overlap_is_set_based():
    left = pd.DataFrame(
        {
            "date": [pd.Timestamp("2000-01-31")],
            "k": [5],
            "complete_returns": [True],
            "long_members": ["A|B"],
            "short_members": ["C|D"],
        }
    )
    right = left.assign(long_members="B|A", short_members="D|E")
    result = membership_overlap(left, right, "test").iloc[0]
    assert result.long_identical
    assert not result.short_identical
    assert result.short_jaccard == 1 / 3
