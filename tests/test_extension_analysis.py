import numpy as np
import pandas as pd

from project.extension_analysis import table_robustness, vix_states


def test_full_slope_paper_sample_is_paired_and_diversified():
    sample = table_robustness.construct_sample(False)
    assert len(sample["original"]) == len(sample["quartile"]) == 427
    assert sample["original"].index.equals(sample["quartile"].index)

    original_hhi, quartile_hhi = table_robustness.hhi_frames(sample)
    paired_dates = sample["original"].index
    original_values = original_hhi.reindex(paired_dates).to_numpy()
    quartile_values = quartile_hhi.reindex(paired_dates).to_numpy()
    assert np.nanmean(quartile_values) < np.nanmean(original_values)
    assert np.isfinite(quartile_values).all()


def test_vix_uses_formation_month_for_next_month_return():
    vix = vix_states.vix_series().reset_index()
    expected = vix["formation_month"] + pd.offsets.MonthEnd(1)
    assert expected.equals(vix["return_month"])
    assert vix["VIX"].notna().all()
