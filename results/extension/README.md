# Extension results: robust commodity SDF testing

The extension asks whether the paper's three-factor commodity SDF is stable
through time and whether apparent rolling failures reflect broad economic
mispricing or fragile test assets and finite-sample inference.

## Research sequence

1. **Identify instability.** Re-estimate the original SDF in 120- and
   180-month rolling windows and test for unknown breaks in pricing moments.
2. **Locate the source.** Decompose the 2003–2014 rejection episode and audit
   the original backwardation portfolios P1/P2.
3. **Diagnose the mechanism.** Relate portfolio HHI to pricing-moment
   volatility and calibrate the two-step GMM J-test under an imposed null.
4. **Use a robust testing framework.** Retain the original estimator for
   comparability, replace asymptotic pointwise inference with local and
   simultaneous block bootstraps, and use full-slope quartiles S1–S4 as a
   diversified robustness design.
5. **Check VIX states.** Re-estimate the unchanged three-factor SDF separately
   in high- and low-VIX months; VIX is a state robustness check, not a new SDF
   factor.

## Main findings

- Conventional rolling J-tests reject in 21/40 ten-year and 15/35 fifteen-year
  windows, with a cluster ending in 2003–2014. Sup-Wald and Bai–Perron tests do
  not support one permanent break.
- P1/P2 dominate this episode. Removing them lowers conventional rejection
  counts to 5/40 and 6/35. Their construction is correct, but their
  sign-conditioned cells are frequently small and concentrated.
- HHI predicts larger absolute pricing moments, including after controls, but
  not persistent signed moments. This supports a concentration-to-noise
  mechanism rather than a causal claim of systematic mispricing.
- The nominal 5% asymptotic J-test is severely oversized: null rejection rates
  average 25.7% at 120 months and 16.9% at 180 months for the original assets.
  Null-imposing bootstrap calibration reduces their rolling rejections to 5/40
  and 8/35.
- Full-slope quartiles reduce conventional rejections to 2/40 and 2/35 and
  bootstrap-calibrated rejections to 0/40 and 0/35. Matched simulations do not
  indicate lower power than the original design.
- Paper-style tables remain stable under the diversified design. In the paired
  paper sample, three-factor OLS R-squared rises from 91.91% to 93.61%, GLS
  R-squared from 98.42% to 99.77%, and annualized MAE falls from 1.80% to 1.45%.
- VIX-state conclusions are threshold-sensitive for the original assets. The
  full-slope assets are not rejected in either state under any reported sample
  or cutoff, so the exercise does not establish a separate VIX-dependent SDF.

The defensible conclusion is not that the original model is uniformly valid.
Rather, its strongest rolling failures are amplified by concentrated test
assets and oversized asymptotic inference. A diversified slope sort combined
with null-imposing block-bootstrap inference produces a materially more stable
assessment.

## Results map

| Question | Primary table or figure |
|---|---|
| When does the original model reject? | `tables/01_instability/rolling_gmm_j_tests.csv` |
| Is there an unknown permanent break? | `tables/01_instability/break_test_summary.json` |
| Which assets drive the episode? | `tables/02_source/rejection_region_asset_summary.csv` |
| Does excluding P1/P2 change rejection rates? | `tables/02_source/p1_p2_exclusion_rolling_summary.csv` |
| Is the P1/P2 construction correct? | `tables/02_source/p1_p2_construction_audit.csv` |
| Does concentration predict noisy moments? | `tables/03_mechanism/hhi_pricing_moment_regressions.csv` |
| How oversized is the J-test? | `tables/03_mechanism/null_block_bootstrap_size.csv` |
| What remains after local calibration? | `tables/03_mechanism/rolling_j_null_bootstrap_summary.csv` |
| Are overlapping rejection sequences jointly unusual? | `tables/04_robust_framework/robust_framework_simultaneous_tests.csv` |
| Are full-slope results caused by low power? | `tables/04_robust_framework/robust_framework_power_calibration.csv` |
| Are paper-style tables robust to S1–S4? | `tables/04_robust_framework/table3_model_fit_and_tests.csv` |
| Do high- and low-VIX states reject? | `tables/05_vix_robustness/vix_state_j_tests.csv` |

All monetary pricing errors are monthly unless explicitly annualized. Bootstrap
p-values use null-imposing circular calendar blocks; exact draw counts and
block lengths are stored in the corresponding tables.
