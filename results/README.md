# Generated results

Everything below `replication/` is generated and can be rebuilt with:

```bash
python run_all.py
```

With cached data, `python run_all.py --results-only --offline` rebuilds only
the reported Phase 1 outputs.

The directory follows the submission layout:

```text
replication/
├── tables/
│   ├── asset_pricing/
│   └── equity_volatility/
└── figures/
    └── asset_pricing/
```

Primary reported results use unconditional bottom-five/top-five rank Carry.
Intermediate samples, weights, audits, and optional diagnostics are written to
the Git-ignored local `processing_exploration/` directory.

Interpretation and limitations are documented under `report/`.
