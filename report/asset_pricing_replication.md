# Cross-sectional asset-pricing replication

## Scope and Carry definition

The primary specification uses unconditional rank Carry: each month it is long
the five commodities with the lowest slopes and short the five with the highest
slopes. This produces substantially closer sample coverage to the paper's 501
reported months than a literal requirement of five negative and five positive
slopes. The rule is disclosed as `unconditional_rank_bottom_top_5`; the results
do not claim that the authors' unobserved implementation has been recovered.

The 12 test assets are P1-P4, Q1-Q5, Agriculture, Livestock, and Metal. Energy
is excluded as in Section 5.1 because of its shorter history. Portfolios use
formation-month information, equal weights, monthly rebalancing, and the
subsequent same-contract return. Missing constituent returns are never filled
with zero or reweighted after the fact.

## Regression sample

The balanced rank-C5 sample contains 427 months from 1974-08 through 2011-09,
compared with the paper's 501 months. Missing early WRDS history, unavailable
original Propane, and vendor-specific contract coverage account for much of the
remaining difference.

## Estimation

The second-pass Fama-MacBeth regression has no intercept, following the paper.
Newey-West inference uses a Bartlett kernel and the automatic lag rule
`floor(4(T/100)^(2/9))`, selecting five lags. Shanken inference incorporates
first-pass residual covariance and factor-estimation uncertainty.

OLS and GLS cross-sectional R-squared values are uncentered. GMM implements the
paper's asset-pricing, factor-mean, and factor-covariance moments. HJ distance is
not reported because its boundary-null weighted-chi-square inference has not
been independently validated; the ordinary two-step GMM J-test remains
separately labelled.

## Reconstructed Table 3

### Three-factor risk premia

| Monthly risk premium | Published | Reconstructed | Annualized reconstructed |
|---|---:|---:|---:|
| AVG | 0.0050 | 0.00346 | 4.15% |
| CARRY | 0.0180 | 0.01545 | 18.53% |
| CMOM | 0.0120 | 0.01502 | 18.02% |

### Model fit

| Statistic | Published | Reconstructed |
|---|---:|---:|
| GLS uncentered R-squared | 96.3% | 98.42% |
| OLS uncentered R-squared | 93.9% | 91.91% |
| Pricing-error chi-square, NW p-value | 0.22 | 0.454 |
| Pricing-error chi-square, Shanken p-value | 0.25 | 0.531 |
| Two-step GMM J-test p-value | 0.167 | 0.586 |
| Figure 2 annualized MAE | 1.84% | 1.80% |

The three-factor model is not rejected and fits the cross section substantially
better than its nested alternatives.

| Model | OLS R-squared | GLS R-squared | Two-step J p-value |
|---|---:|---:|---:|
| AVG + CARRY + CMOM | 91.91% | 98.42% | 0.586 |
| AVG + CARRY | 75.56% | 96.06% | 0.007 |
| AVG + CMOM | 76.22% | 94.86% | 0.013 |
| AVG only | 42.60% | 85.41% | <0.001 |

This recovers the paper's core cross-sectional conclusion: both Carry and
Momentum add material explanatory power, while the complete three-factor model
is not rejected.

## Figure 2

The realized-versus-fitted figure has an annualized mean absolute error of
1.80%, close to the published 1.84%. The figure is stored separately from the
underlying tables under `results/replication/figures/asset_pricing/`.

## Optional time-series diagnostics

Table 6/7-style time-series alpha tests are not part of the project's specified
outputs. They are retained locally under
`processing_exploration/results/asset_pricing/time_series_diagnostics/` and are
excluded from Git and the formal replication map.

## Outputs and reproduction

Primary tables are under `results/replication/tables/asset_pricing/` and the
figure is under `results/replication/figures/asset_pricing/`. Rebuild the full
Phase 1 submission from the repository root with:

```bash
python run_all.py
```

Use `python run_all.py --offline` to reuse existing WRDS and public-data
snapshots without opening a network connection.
