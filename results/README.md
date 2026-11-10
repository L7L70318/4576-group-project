# Generated results

Everything below `replication/` and `extension/` is generated and can be
rebuilt with:

```bash
python run_all.py
```

With cached data, `python run_all.py --results-only --offline` rebuilds all
reported outputs. Use `--replication-only` or `--extension-only` to rebuild one
part.

The directory follows the submission layout:

```text
replication/
├── tables/
│   ├── asset_pricing/
│   └── equity_volatility/
└── figures/
    └── asset_pricing/
extension/
├── tables/
│   ├── 01_instability/
│   ├── 02_source/
│   ├── 03_mechanism/
│   ├── 04_robust_framework/
│   └── 05_vix_robustness/
└── figures/
    ├── 01_instability/
    ├── 02_source/
    ├── 03_mechanism/
    └── 04_robust_framework/
```

Replication results use unconditional bottom-five/top-five rank Carry. The
extension retains that construction as its benchmark, diagnoses instability in
the original sign-split test assets, and evaluates diversified full-slope
quartiles with finite-sample bootstrap inference. See `extension/README.md` for
the result map and interpretation.

Intermediate experiments and superseded output layouts are kept in the
Git-ignored local `processing_exploration/` directory.

Interpretation and limitations are documented under `report/`.
