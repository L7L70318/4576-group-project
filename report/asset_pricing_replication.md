# Phase 2 baseline asset-pricing replication

## Scope and specification

This phase uses the audited WRDS reconstruction without tuning factor returns. All outputs are separate from the data and Carry audits under `results/replication/asset_pricing/`.

The primary specification follows the paper's explicit written rule: CARRY is C5, long the five lowest negative slopes and short the five highest positive slopes. This report calls that implementation `strict-sign` to distinguish it from the unconditional rank diagnostic; `strict-sign` is our label, not terminology used by the authors. The paper nevertheless reports 501 observations for C5, so the authors' treatment of months with fewer than five commodities on either sign remains unverified. Commodity-specific Internet-I sample windows are applied before portfolio formation.

The 12 test assets are ordered as P1–P4, Q1–Q5, Agriculture, Livestock, and Metal. Energy is excluded as in Section 5.1 because of its shorter history. Portfolios use formation-month information, equal weights, monthly rebalancing, and subsequent-month same-contract returns. Missing constituent returns make the portfolio return missing; they are never replaced by zero or followed by ex-post reweighting.

## Portfolio coverage and regression sample

Individual portfolio coverage ranges from 447 to 463 months for the test assets. Factor coverage is 455 months for AVG, 315 for strict-sign CARRY, and 442 for CMOM. Requiring all 12 assets and all three factors produces a balanced sample of **312 months from February 1975 through September 2011**, compared with the paper's 501 months.

The 186 unavailable strict-sign C5 months decompose exactly into 37 months with no WRDS monthly panel (1970-01 through 1973-01), 12 months with fewer than ten valid signals (1973-02 through 1974-01), 132 months with fewer than five backwardated commodities, and five months with a valid formation portfolio but a missing selected-member return (1989-05, 1999-12, 2003-11, 2008-04, and 2008-07). The final common regression loses three additional C5-valid months: 1984-10, 1997-09, and 2002-11, because one or more P/Q/category/AVG/CMOM returns are missing.

Formation and weighting validation is exported at the month-portfolio level. It checks equal absolute weights, required net weights, negative/positive Carry sides, monotonic P1–P4 slope boundaries, and monotonic Q1–Q5 momentum boundaries.

## Estimation methodology

First-pass time-series regressions include a constant and the applicable factors. Newey–West covariance estimates use a Bartlett kernel and the automatic lag rule `floor(4(T/100)^(2/9))`, giving five lags for the 312-month common sample.

The second-pass regression is the paper's no-intercept regression of average returns on estimated betas. Newey–West inference is computed from the monthly cross-sectional risk-premium series. The Shanken covariance uses the first-pass residual long-run covariance, the factor covariance, and the multiplier `1 + lambda' Sigma_f^{-1} lambda`.

OLS uncentered R-squared uses unweighted pricing errors. GLS uncentered R-squared follows Lewellen, Nagel, and Shanken's Prescription 3 and weights pricing errors by the inverse first-pass residual covariance matrix.

GMM implements Equation (4): test-asset pricing moments, factor-mean moments, and the unique factor-covariance moments. With three factors this gives 21 moments and 12 parameters. Both identity-weight one-step GMM and optimal-weight two-step GMM are reported. The two-step Hansen J-test has 9 degrees of freedom in the three-factor model.

HJ distance is not implemented in the reported reconstruction. Although its inverse-second-moment distance definition is identifiable, its zero-distance null is a boundary problem with a nonstandard weighted-chi-square distribution. That inference has not yet been independently validated here. Both the distance and p-value are therefore blank, and the ordinary GMM J-test is kept in its own column rather than being relabeled as HJ inference.

## Reconstructed Table 3

### Three-factor risk premia

| Monthly risk premium | Published | Reconstructed | Annualized reconstructed | NW p-value | Shanken p-value |
|---|---:|---:|---:|---:|---:|
| AVG | 0.0050 | 0.00239 | 2.86% | 0.267 | 0.230 |
| CARRY | 0.0180 | 0.01215 | 14.58% | 0.002 | 0.002 |
| CMOM | 0.0120 | 0.01025 | 12.30% | 0.004 | 0.011 |

The one-step GMM estimates of lambda are 0.00238, 0.01214, and 0.01024 and are essentially identical to the two-pass values, as in the paper. The corresponding SDF loadings are 1.580, 3.046, and 1.843; Carry and Momentum loadings are significant at the 5% level, whereas AVG is not.

### Model fit

| Statistic | Published | Reconstructed |
|---|---:|---:|
| GLS uncentered R-squared | 96.3% | 95.64% |
| OLS uncentered R-squared | 93.9% | 73.32% |
| Pricing-error chi-square, NW | 11.91 (p=0.22) | 19.77 (p=0.019) |
| Pricing-error chi-square, Shanken | 11.40 (p=0.25) | 18.65 (p=0.028) |
| HJ distance | 0.006 | Not reported; inference not yet validated |
| Two-step GMM J-test | paper p=0.167 | 18.04, df=9 (p=0.035) |
| Figure 2 annualized MAE | 1.84% | 2.53% |

The reconstructed three-factor model therefore fits substantially better than its nested alternatives, but unlike the published result it is rejected at the 5% level on both the pricing-error test and the two-step J-test. The very high GLS R-squared coexists with a lower OLS R-squared because the largest errors occur in portfolios that receive less GLS weight.

## Restricted models

| Model | FMB risk premia, monthly | OLS R-squared | GLS R-squared | Two-step J p-value |
|---|---|---:|---:|---:|
| Three factors | AVG 0.00239; CARRY 0.01215; CMOM 0.01025 | 73.32% | 95.64% | 0.035 |
| Omit Momentum | AVG 0.00249; CARRY 0.01358 | 60.52% | 92.49% | 0.006 |
| Omit Carry | AVG 0.00256; CMOM 0.01107 | 56.03% | 93.67% | 0.004 |
| AVG only | AVG 0.00277 | 32.05% | 86.59% | <0.001 |

The ordering agrees qualitatively with the paper: including both Carry and Momentum provides the best cross-sectional fit. The available WRDS sample does not support the paper's stronger conclusion that the complete three-factor model is not rejected.

## Rank-C5 expanded-sample diagnostic

At the user's request, the complete analysis was rerun after removing the
`strict-sign` availability requirement. This diagnostic forms C5 from the five
lowest and five highest slopes every month with at least ten valid signals,
whether or not each side has the sign stated in the paper. It is stored
separately under `results/replication/asset_pricing/rank_carry_expanded/` and
does not overwrite or replace the written-rule specification above.

The balanced sample increases from 312 to **427 months**, spanning August 1974
through September 2011. The expanded-sample results are:

| Statistic | Strict-sign sample | Rank-C5 expanded sample | Published |
|---|---:|---:|---:|
| Months | 312 | 427 | 501 |
| AVG risk premium, annualized | 2.86% | 4.15% | 6.00% |
| CARRY risk premium, annualized | 14.58% | 18.53% | 21.60% |
| CMOM risk premium, annualized | 12.30% | 18.02% | 14.40% |
| OLS uncentered R-squared | 73.32% | 91.91% | 93.9% |
| GLS uncentered R-squared | 95.64% | 98.42% | 96.3% |
| Pricing-error chi-square, NW p-value | 0.019 | 0.454 | 0.22 |
| Pricing-error chi-square, Shanken p-value | 0.028 | 0.531 | 0.25 |
| Two-step GMM J-test p-value | 0.035 | 0.586 | 0.167 |
| Figure 2 annualized MAE | 2.53% | 1.80% | 1.84% |

Under this expanded sample, the three-factor model is not rejected by either
pricing-error test or by the two-step GMM J-test. In contrast, all three nested
models remain rejected: the J-test p-values are 0.007 for the model omitting
Momentum, 0.013 for the model omitting Carry, and below 0.001 for AVG only.
The qualitative core result of the paper is therefore recovered under this
availability policy.

This comparison is evidence of **sample-selection sensitivity**, not evidence
that the unconditional rank rule is the authors' verified implementation. On
the 312 common months, rank C5 and strict-sign C5 are identical; all changes
above come from admitting 115 additional months. The paper's prose requires
negative slopes on the long side and positive slopes on the short side, while
its reported 501 C5 observations are difficult to reconcile with dropping
months that lack five commodities of each sign. Until the authors' original
monthly data or missing-month policy is recovered, both specifications must be
retained and neither result should be selected merely because it is closer to
the published fit.

## Main discrepancies and interpretation

The balanced sample is selected heavily by strict C5 availability. For example, Agriculture averages 3.65% annualized on its own available sample but -2.35% on the 312 common months. This selection effect, missing 1970–1974 coverage, absent original Propane, and vendor-specific curve rankings materially change the cross section even when construction rules are held fixed.

On the exact 312 strict-sign regression months, the rank and strict CARRY series and all three estimated risk premia are numerically identical. This is expected: whenever strict C5 exists, the five most negative and five most positive slopes are also the unconditional bottom and top five. The alternative rule only changes which additional months are admitted, so it cannot change a genuinely common-sample result.

The results should be described as a WRDS-based reconstruction over 1975-02 to 2011-09, not as an exact 1970-01 to 2011-09 replication.

## Reproduction and outputs

Run:

```bash
PYTHONPATH=src python scripts/07_replicate_asset_pricing.py
PYTHONPATH=src pytest -q
```

Key outputs include monthly asset and factor returns, portfolio weights and validation, the balanced regression panel, first-pass betas with Newey–West inference, Fama–MacBeth and GMM estimates, model tests, published comparisons, strict/rank sensitivity, and `figure2_realized_vs_fitted.png`. The expanded rank-C5 rerun and its figure are under `results/replication/asset_pricing/rank_carry_expanded/`.
