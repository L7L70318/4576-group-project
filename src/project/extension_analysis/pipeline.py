"""Ordered entry point for the complete model-stability extension."""

from . import (
    concentration,
    decomposition,
    portfolio_audit,
    robust_framework,
    size_calibration,
    stability,
    table_robustness,
    vix_states,
)


def run_extension():
    """Rebuild all reported extension tables and figures in dependency order."""
    stages = [
        ("1/8 rolling stability and break tests", stability.main),
        ("2/8 P1/P2 source decomposition", decomposition.main),
        ("3/8 P1/P2 construction audit", portfolio_audit.main),
        ("4/8 concentration mechanism", concentration.main),
        ("5/8 finite-sample size calibration", size_calibration.main),
        ("6/8 robust testing framework", robust_framework.main),
        ("7/8 paper-table robustness", table_robustness.main),
        ("8/8 VIX-state robustness", vix_states.main),
    ]
    for label, stage in stages:
        print(f"\n[extension] {label}")
        stage()
