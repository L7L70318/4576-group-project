# IEOR 4576 — Commodity Risk Replication and Robustness Extension

Replication and extension of Bakshi, Gao, and Rossi (2019), “Understanding the
Sources of Risk Underlying the Cross Section of Commodity Returns.” The project
first reconstructs the paper's AVG, CARRY, and CMOM model, then studies why its
rolling cross-sectional pricing tests become unstable in some periods.

## Team

| Name | UNI |
|---|---|
| Xinyu WU | xw3186 |
| Zhitao QU | zq2272 |
| Anvesha Churi | asc2287 |

## Research design

### Replication

The replication reconstructs factor returns, Table 3 and Figure 2 asset-pricing
results, and the Table 8 relation between Carry and global equity-volatility
innovations. Rank Carry is long the five lowest-slope commodities and short the
five highest-slope commodities.

### Extension: robust rolling SDF tests

The extension follows one connected sequence:

1. Identify instability with 120- and 180-month rolling GMM J-tests and unknown
   pricing-moment break tests.
2. Locate the rejection source using asset-level moments, leave-asset-out tests,
   and a construction audit of the sign-split backwardation portfolios P1/P2.
3. Diagnose concentration and finite-sample distortion using portfolio HHI,
   matched comparisons, null-imposing block bootstraps, and Monte Carlo size
   calibration.
4. Retain the paper-comparable two-step GMM estimator but use bootstrap-calibrated
   local and simultaneous inference. As a diversified test-asset robustness
   design, replace sign-split P1–P4 with four full-cross-section slope quartiles
   S1–S4 while leaving AVG, CARRY, CMOM, Q1–Q5, and sector portfolios unchanged.
5. Use high/low VIX states as a final robustness check, without adding VIX to the
   SDF or presuming a VIX-dependent mechanism.

## Main findings

The replication matches the paper closely: on the 427-month balanced sample,
the three-factor model has OLS and GLS cross-sectional R-squared values of
91.91% and 98.42%, and Figure 2 annualized MAE is 1.80%.

Conventional rolling inference is much less stable: the original assets reject
in 21/40 ten-year and 15/35 fifteen-year windows. P1/P2 account for most of the
2003–2014 episode. Their construction passes all implementation checks, but
their sign-conditioned cells are often highly concentrated. HHI predicts larger
absolute pricing moments, not persistent signed errors.

The nominal 5% two-step GMM J-test is severely oversized in short samples. A
null-imposing bootstrap reduces original-asset rejection counts to 5/40 and
8/35. With full-slope quartiles, conventional counts are 2/40 and 2/35 and
bootstrap-calibrated counts are 0/40 and 0/35. The diversified design also
preserves the paper-style cross-sectional and time-series conclusions. VIX-state
results are threshold-sensitive for the original assets but uniformly
non-rejecting for the full-slope design.

The extension therefore does not claim that concentration mechanically causes
mispricing or that the original SDF is universally valid. It shows that the
strongest apparent failures are amplified by concentrated test assets and
oversized asymptotic inference.

## Installation

Use Python 3.10–3.13 from the repository root:

```bash
python -m venv .venv
python -m pip install -e ".[dev]"
```

Raw WRDS snapshots and credentials are not committed. If acquisition is
required, the pipeline prompts for WRDS credentials when no standard libpq
password-file entry is available; passwords are never echoed or stored.

## Reproduction

Build the complete replication and extension:

```bash
python run_all.py
```

Using cached data only:

```bash
python run_all.py --results-only --offline
```

Build either component independently:

```bash
python run_all.py --results-only --offline --replication-only
python run_all.py --results-only --offline --extension-only
```

The four files in `scripts/` are thin ordered entry points. Reusable logic is
under `src/project/`; reported results never depend on notebooks.

## Repository layout

- `configs/`: sample dates, universe mappings, estimation parameters, and seed.
- `data/raw/`: untouched downloads; not committed.
- `data/interim/`: cleaned and aligned inputs, including monthly VIX; not committed.
- `data/processed/`: final analysis-ready panels; not committed.
- `src/project/`: importable acquisition, portfolio, estimation, evaluation, and
  extension code.
- `src/project/extension_analysis/`: ordered stability, mechanism, inference,
  table-robustness, and VIX-state modules.
- `scripts/`: thin entry points for acquisition, dataset construction,
  replication, and extension.
- `notebooks/`: exploration only.
- `results/replication/`: generated paper-replication tables and figures.
- `results/extension/`: generated extension tables and figures, organized by the
  five research steps.
- `processing_exploration/`: ignored diagnostics, superseded experiments, and
  the recoverable legacy output layout.
- `tests/`: timing, construction, regression, and inference sanity checks.
- `report/`: methodology and replication documentation.

## Results map

| Output | Primary result |
|---|---|
| Replication Table 3 | `results/replication/tables/asset_pricing/table3_model_fit_and_tests.csv` |
| Replication Figure 2 | `results/replication/figures/asset_pricing/figure2_realized_vs_fitted.png` |
| Replication Table 8 | `results/replication/tables/equity_volatility/table8_panel_b_cross_sectional_model.csv` |
| Rolling instability | `results/extension/tables/01_instability/rolling_gmm_j_tests.csv` |
| P1/P2 source diagnostic | `results/extension/tables/02_source/p1_p2_exclusion_rolling_summary.csv` |
| HHI mechanism | `results/extension/tables/03_mechanism/hhi_pricing_moment_regressions.csv` |
| J-test size calibration | `results/extension/tables/03_mechanism/null_block_bootstrap_size.csv` |
| Robust simultaneous tests | `results/extension/tables/04_robust_framework/robust_framework_simultaneous_tests.csv` |
| Paper-table robustness | `results/extension/tables/04_robust_framework/table3_model_fit_and_tests.csv` |
| VIX-state robustness | `results/extension/tables/05_vix_robustness/vix_state_j_tests.csv` |

See `results/extension/README.md` for the complete extension interpretation and
`report/replication_map.csv` for the paper-result mapping.

## Limitations

The reconstruction uses WRDS rather than the authors' CME files. Propane is
unavailable, early country-index coverage is thin, and some exchange-calendar
and global equity-volatility inputs require documented proxies. Rolling windows
overlap heavily, so they are not independent events; the extension addresses
this with full-sequence simultaneous bootstrap statistics. The HHI evidence is
observational and should not be interpreted causally. Full-slope quartiles are
a robustness design, not a literal replacement for the paper's economically
defined sign-split portfolios.
