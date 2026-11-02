"""Pricing estimation entry points."""

def replicate():
    from .asset_pricing import build_asset_pricing_replication

    return build_asset_pricing_replication()

def extend():
    raise NotImplementedError("Extensions are scheduled after Phase 1.")
