"""Pricing estimation entry points."""

def replicate():
    """Rebuild the checkpoint's required Phase 1 outputs."""
    from .asset_pricing import build_asset_pricing_replication

    return build_asset_pricing_replication()

def extend():
    print("Extension work is outside the Phase 1 submission scope.")
