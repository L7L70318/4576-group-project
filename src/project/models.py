"""Pricing estimation entry points."""


def replicate():
    """Rebuild the paper-replication outputs."""
    from .asset_pricing import build_asset_pricing_replication

    return build_asset_pricing_replication()


def extend():
    """Rebuild the model-stability extension."""
    from .extension_analysis import run_extension

    return run_extension()
