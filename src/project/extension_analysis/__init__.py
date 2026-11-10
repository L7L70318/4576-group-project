"""Model-stability extension and finite-sample robustness analysis."""


def run_extension():
    """Import and run the pipeline lazily to keep submodules independent."""
    from .pipeline import run_extension as _run_extension

    return _run_extension()


__all__ = ["run_extension"]
