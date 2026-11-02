# Carry and global equity-volatility innovations

## Scope

This output reconstructs Table 8 of Bakshi, Gao, and Rossi (2019), using the
repository's unconditional bottom-five/top-five rank Carry definition. The
global equity-volatility variable is a transparent proxy constructed from 19
G20 country indexes available through WRDS Datastream; it is not the authors'
original series.

For each country, daily local-index returns are computed without filling gaps.
Returns following gaps longer than seven calendar days are set to missing.
Daily global equity volatility is the equally weighted mean absolute return
across countries available that day. The monthly series is the mean of daily
global volatility within the month, and its innovation is the first monthly
difference:

$$
\Delta VOL_t^{equity}=VOL_t^{equity}-VOL_{t-1}^{equity}.
$$

## Panel A: time-series relation

The regression is

$$
Y_t=\alpha+\beta\Delta VOL_t^{equity}+\epsilon_t,
$$

where \(Y_t\) is alternatively rank CARRY, its long/backwardated leg, or its
short/contangoed leg. Newey-West inference uses the automatic lag rule, which
selects five lags for the 447-month sample from 1974-02 through 2011-09.

| Dependent return | Intercept | Slope on equity-vol innovation | NW p-value | Adjusted R-squared |
|---|---:|---:|---:|---:|
| CARRY | 0.01084 | -2.7249 | 0.0061 | 1.27% |
| Long/backwardated leg | 0.00798 | -3.4054 | 0.0346 | 2.58% |
| Short/contangoed leg | 0.00286 | 0.6806 | 0.5651 | -0.11% |

The sign and leg decomposition reproduce the paper's economic conclusion.
Carry performs poorly when global equity volatility rises, and the relation is
concentrated in the long/backwardated leg. The paper reports slopes of -1.938
for CARRY, -2.585 for the long leg, and 0.647 for the short leg; the
reconstructed magnitudes are directionally and economically comparable.

## Panel B: pricing the Carry portfolios

The second exercise uses AVG and \(\Delta VOL_t^{equity}\) to price P1-P4 in a
no-intercept Fama-MacBeth cross-sectional regression. The balanced sample has
442 months from 1973-02 through 2011-09.

| Statistic | Reconstructed |
|---|---:|
| AVG risk premium, monthly | 0.00467 |
| Equity-volatility innovation risk premium, monthly | -0.00431 |
| Equity-volatility NW p-value | 0.00013 |
| Equity-volatility Shanken p-value | 0.0432 |
| OLS uncentered R-squared | 90.78% |
| GLS uncentered R-squared | 93.79% |
| Pricing-error chi-square, NW p-value | 0.0620 |
| Pricing-error chi-square, Shanken p-value | 0.4565 |

The negative and statistically significant risk premium supports the paper's
interpretation that investors require compensation for holding Carry exposures
that perform poorly when global equity volatility rises. The Shanken
pricing-error test does not reject the two-factor specification; the
Newey-West test is borderline at the 5% level.

## Country coverage and limitations

The country composition expands over time. On the Panel A months, the average
number of available country returns per day averages 12.1 and ranges from 3.7
to 17.9 at the monthly-average level. Early sample coverage is therefore much
thinner than coverage after 1990. The proxy also uses representative local
indexes, including the DAX performance index, and should not be described as
an exact reconstruction of the authors' G20 equity series.

The main outputs are under `results/replication/tables/equity_volatility/`:

- `table8_panel_a_time_series.csv`
- `table8_panel_b_cross_sectional_model.csv`
- `table8_panel_b_pricing_errors.csv`
- `table8_sample_and_country_coverage.csv`

The exact common samples are retained locally under
`processing_exploration/results/equity_volatility/` and are excluded from the
checkpoint submission.

Rebuild the outputs from the repository root with:

```bash
python run_all.py
```

Use `python run_all.py --results-only --offline` to rebuild only these results
from existing processed inputs.
