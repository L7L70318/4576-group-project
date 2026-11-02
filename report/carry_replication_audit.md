# Commodity Carry replication audit

## Executive assessment

The revised pipeline is materially safer at the contract-selection level, but the current CARRY factor is **not yet reliable enough for subsequent GMM or Fama–MacBeth tests**. Neither strict-sign nor unconditional rank is designated as the official replication. The paper's text supports strict signs, while its reported 501 observations are inconsistent with the months that strict signs can form in the WRDS reconstruction.

The original rank diagnostic has 447 observations and a 13.42% annualized mean, leaving a 2.92 percentage-point gap to the published 16.34%. Applying every commodity-specific sample window in Internet-I lowers this mean to 12.95%; sample-window alignment therefore increases rather than closes the gap. The strongest current evidence points to the long backwardation leg and to missing/different historical source data, not to FND or Barley.

The audit was run against 5,017,880 WRDS daily observations for 30 mapped `contrcode` series. Generated data and results are ignored by Git.

## 1. Missing commodities and mapping audit

| Target | WRDS evidence | Paper sample | Decision |
|---|---|---:|---|
| Western Barley | `contrcode=3160`, `WESTERN BARLEY`, CAD, metric tonne, ticker AB; prices 1983-02-28–2012-08-15 | 1994-07–2003-05 | Accept and mark verified |
| Propane | Earliest named candidate is `contrcode=3847`, Mont Belvieu LDH Propane (OPIS) Swap Pit; prices start 2007-09-25 | 1987-09–2006-10 | Reject |
| Other propane candidates | Argus/OPIS swaps or electronic/index contracts beginning 2008–2015 | 1987-09–2006-10 | Reject; not the original physical contract |

The existing mapped `contrcode` values each resolve to one contract description and one currency. BO and RR intentionally contain old/new series rather than duplicate identities. Detailed evidence is in `wrds_missing_commodity_candidates.csv` and `wrds_existing_mapping_audit.csv`.

## 2. First notice day audit

The former implementation replaced every vendor FND with an Internet-I approximation. The revision applies this hierarchy:

1. contract-specific WRDS `firstnoticedate` for physically delivered contracts;
2. WRDS `lasttrddate` for cash-settled Feeder Cattle and Lean Hogs;
3. Internet-I calendar convention when the relevant vendor date is missing.

The metadata contain 4,216 contracts with vendor FND, 602 cash-settled contracts with vendor LTD, and 5,866 convention fallbacks. Among 15,076 monthly panel rows, the selected-row source labels are 4,965 vendor FND, 857 cash LTD, 9,126 convention fallback, and 128 unresolved/no selection.

Only three monthly selections change relative to the convention-only implementation, two in the paper sample:

| Commodity | Return month(s) | Effect |
|---|---|---|
| Live cattle | 2010-07 and 2010-08 | Actual FND for October 2010 contract was 2010-04-10, already unsafe; selection moves to December 2010 contract. Return changes by -37.01bp and +114.74bp. |
| Palladium | 2016-03 | Selection moves from May to June 2016; return changes by +0.91bp. |

Paper-period C5 annualized mean changes only from 10.073% to 10.065%. FND is therefore important for correctness but does not explain the replication gap.

## 3. Signal and return timing

The verified semantics are:

- `formation_date`: month-end `t` at which the curve is observed;
- `slope`: `log(F1/F0)` observed at `formation_date`;
- `date`: month-end `t+1`, labeling the realized holding return;
- `excess_return`: price change of the same `selected_futcode` between `t` and `t+1`;
- portfolio weights stored under `date=t+1` are selected using the row's formation-time slope.

No extra `.shift(1)` is appropriate for the independent WRDS panel. Targeted regression tests fail if Carry membership is taken from a different row or if entry/exit contract IDs differ.

## 4. Strict C5 availability

For the paper period available in WRDS:

- 323 calendar months pass the formation-time sign-count test;
- 129 months have fewer than five backwardated commodities;
- 12 early months have insufficient holding-contract coverage;
- after requiring complete returns for selected members, C5 has 318 observations.

For 2012–2025:

- 154 months form strict C5;
- 14 months have fewer than five backwardated commodities;
- the strict factor has 154 complete return observations.

Every failure and its counts are recorded in `c5_monthly_availability.csv`.

## 5. Portfolio-rule diagnostic

The paper text requires five negative-slope longs and five positive-slope shorts, but Table 1 reports 501 observations. Three transparent policies were compared without changing the baseline silently:

| Policy | N | Annual mean | Annual SD | Sharpe |
|---|---:|---:|---:|---:|
| Strict signs | 318 | 10.06% | 20.59% | 0.49 |
| Strict signs, unfilled slots held as cash | 459 | 12.44% | 20.42% | 0.61 |
| Lowest/highest five regardless of sign | 447 | 13.42% | 20.69% | 0.65 |
| Published C5 | 501 | 16.34% | 22.26% | 0.73 |

On the 318 common valid strict-sign months, all three previously implemented policies are identical. Their full-sample differences arise entirely from whether additional months are admitted. Even the unconditional-extremes diagnostic does not close the published gap, so changing the sorting rule alone is not a valid fix.

## 6. Table 1 and Table 2 comparison

| Portfolio | Reconstructed N | Reconstructed mean | Published N | Published mean |
|---|---:|---:|---:|---:|
| C1 | 459 | 12.06% | 501 | 9.31% |
| C2 | 449 | 11.26% | 501 | 10.08% |
| C3 | 417 | 10.28% | 501 | 12.14% |
| C4 | 369 | 8.87% | 501 | 14.27% |
| C5/CARRY | 318 | 10.06% | 501 | 16.34% |
| M5/CMOM | 442 | 14.79% | 501 | 16.11% |
| AVG | 455 | 5.66% | 501 | 6.27% |

The declining observation count from C1 to C5 is an availability artifact. On exactly the same months, it disappears:

| Diagnostic sample | Common N | C1 | C2 | C3 | C4 | C5 |
|---|---:|---:|---:|---:|---:|---:|
| Strict signs, full WRDS | 318 | -0.71% | 3.49% | 4.99% | 9.00% | 10.06% |
| Rank, full WRDS | 447 | 8.12% | 8.58% | 11.66% | 13.63% | 13.42% |
| Strict signs, paper windows | 315 | -2.29% | 3.72% | 4.88% | 9.51% | 9.67% |
| Rank, paper windows | 447 | 6.33% | 7.71% | 10.44% | 13.25% | 12.95% |

Thus the reconstructed portfolios broadly rise from C1 to C5 once sample availability is held fixed. The mismatch is not a simple reversal of the paper's sorting pattern.

The publisher's official supplemental page now confirms that the public archive is named `mnsc.2017.2840-sm_data.zip`; this is a more precise target than guessing the internal TSV URL. The project cleanup manifest shows that the 384,494-byte `FuturesReturn_to_post.xls` previously existed locally but was removed as a supposed duplicate. It is absent from Git, Trash, local snapshots, and surviving derived files. Re-downloading the ZIP is currently blocked by the publisher's HTTP 403 protection in this environment, and no archived mirror was found. Consequently, exact author-versus-reconstruction monthly correlation, MAE/RMSE, cumulative comparison, and membership overlap remain pending. The audit code will calculate all four automatically once `FuturesReturn_to_post.xls` from that ZIP is restored under `data/raw/paper_reference/`.

## 7. Sample-window, leg, and period diagnosis

Applying Internet-I's commodity-specific start and end dates changes rank C5 from 13.42% to 12.95% on the same 447 months (correlation 0.986, MAE 38.5 bp/month, RMSE 101.3 bp/month). Barley itself changes the paper-window result by only +0.09 percentage points. These are real vendor/sample effects, but neither explains the 2.92-point shortfall.

Table 9 supplies the decisive leg benchmark:

| C5 leg | Published mean | Rank, full WRDS | Difference, reconstructed minus published |
|---|---:|---:|---:|
| Long backwardation | 14.14% | 9.61% | -4.53 pp |
| Short contango | 2.20% | 3.81% | +1.61 pp |
| Total | 16.34% | 13.42% | -2.92 pp |

The lower total return is therefore a **long-leg problem**. The reconstructed short leg is stronger than published and offsets about 1.61 points of the 4.53-point long-leg deficit. Under paper-specific windows the same conclusion holds: long 9.51%, short 3.45%, total 12.95%.

Rank C5 by subperiod is:

| Period | Valid/calendar months | Coverage | Total mean | Long mean | Short mean |
|---|---:|---:|---:|---:|---:|
| 1970–1989 | 190/240 | 79.2% | 13.94% | 11.55% | 2.39% |
| 1990–1999 | 119/120 | 99.2% | 4.71% | -0.69% | 5.40% |
| 2000–2011 | 138/141 | 97.9% | 20.21% | 15.83% | 4.38% |

The 54 missing paper months comprise all months from 1970-01 through 1974-01 plus 1989-05, 1999-12, 2003-11, 2008-04, and 2008-07. If the 54 unobserved months alone were to reconcile the published unconditional mean, they would need an annualized mean of 40.52%. This arithmetic shows that coverage could matter materially, but it is not proof that coverage alone explains the gap.

## 8. Evidence classification

Evidence for coverage/vendor differences:

1. WRDS cannot construct rank C5 in 54 of the paper's 501 months, concentrated before February 1974.
2. Original physical Propane is absent; the later OPIS/Argus products are invalid substitutes.
3. Paper-window alignment and Barley have measured, small effects and do not close the gap.
4. Aggregate slope means and contango frequencies differ most for Western Barley, Palladium, Copper, RBOB, and Soybean Oil, so vendor curves can change boundary rankings.

Evidence about implementation:

1. FND correction changes only two paper-period Live Cattle selections and about 0.01 percentage points of strict C5; it is not the source of the gap.
2. The code uses end-of-month front/next-maturity prices for the signal and the subsequent month's same-contract holding return, matching the paper's stated timing.
3. The common-sample C1–C5 pattern is increasing, so the earlier declining pattern was caused by changing month availability rather than the return formula.
4. Exact member-level implementation verification is still unresolved because the official author ZIP is blocked. Contract mapping, maturity choice, and signal timing cannot be fully separated until those signals are available.

## 9. Remaining discrepancy hierarchy

1. WRDS begins many core series in 1973–1979, whereas the authors' CME data include returns from 1970. The reconstructed factor cannot have the paper's 501 observations.
2. Original physical Propane is absent. Later OPIS/Argus instruments are not valid substitutes.
3. Strict sign counts fail in many contango-dominated months, especially 2007–2010.
4. Individual WRDS slope means and contango frequencies are generally close to Internet-II, but some series differ materially, especially Western Barley, Palladium, Copper, RBOB, and Soybean Oil.
5. Exact FND changes very few selections and is not the main numerical cause.

No strategy parameter has been tuned to match the published mean. Strict-sign and rank remain diagnostics, not competing claims of an official replication.

## 10. Reproduction

With the project installed and WRDS credentials stored in `.pgpass`:

```bash
python run_all.py --data-only --offline
python scripts/05_audit_carry.py
python scripts/06_diagnose_carry_gap.py
pytest -q
```

The audit outputs are written to `results/replication/carry_audit/`. The most important files are:

- `wrds_missing_commodity_candidates.csv`
- `c5_monthly_availability.csv`
- `fnd_selection_return_comparison.csv`
- `carry_policy_summary.csv`
- `published_vs_revised_table_1_2.csv`
- `c5_old_vs_revised_cumulative.png`
- `carry_policy_cumulative.png`

The additional discrepancy outputs are under `results/replication/carry_discrepancy/`, especially `c1_c5_common_sample.csv`, `sample_and_barley_effects.csv`, `c5_leg_summary.csv`, `c5_period_and_leg_decomposition.csv`, `c5_missing_months.csv`, and `coverage_gap_accounting.csv`.
