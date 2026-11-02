# Phase 1 presentation outline

## Research question

Can average commodity returns, Carry, and Momentum explain the cross section of
commodity futures returns, and is Carry compensation linked to global equity
volatility risk?

## Data and construction

- WRDS commodity futures contracts, 1970–2011 paper comparison window.
- Same-contract monthly returns with first-notice/last-trade safety checks.
- Rank Carry: long bottom-five slopes, short top-five slopes.
- Momentum: long five winners, short five losers using prior six-month returns.
- Nineteen-country equity-volatility proxy from local daily index returns.

## Three-factor results

- Balanced sample: 427 months, 1974-08 to 2011-09.
- OLS cross-sectional R-squared: 91.91%.
- GLS cross-sectional R-squared: 98.42%.
- Two-step GMM J-test p-value: 0.586.
- Figure 2 MAE: 1.80%, versus 1.84% published.

## Economic source of Carry

- Carry slope on equity-volatility innovation: -2.725, NW p-value 0.006.
- Long/backwardated leg slope: -3.405, p-value 0.035.
- Short/contangoed leg slope: 0.681, p-value 0.565.
- Equity-volatility risk price is negative and significant in P1-P4 pricing.

## Interpretation

The reconstructed three-factor model fits the commodity cross section and is
not rejected. Carry performs poorly when global equity volatility rises, with
the exposure concentrated in its long leg, consistent with compensation for
bad-state risk.

## Limitations

- WRDS rather than the authors' CME files.
- Missing original Propane and shorter early history.
- Representative 19-country proxy rather than the authors' equity series.
- Thin early country-index coverage and vendor-specific contract histories.
