"""Rebuild the data-stage artifacts without fitting pricing models."""
from .acquisition import connect_wrds, download_futures
from .auxiliary import download_equities, download_public, prepare_equities, prepare_public
from .data import build_dataset, reference_path
from .diagnostics import run_diagnostics
from .paths import ROOT, ensure_directories
from .portfolios import make_portfolios


def prepare_data(download=False):
    ensure_directories()
    if download:
        conn=connect_wrds()
        try:
            download_futures(conn)
            download_equities(conn)
        finally:
            conn.close()
        download_public(1986,2025)
    p=build_dataset()
    f,w=make_portfolios(p)
    f.to_parquet(ROOT/'data/processed/portfolios.parquet')
    w.to_parquet(ROOT/'data/processed/weights.parquet',index=False)
    if reference_path() is not None:
        run_diagnostics(p,'main')
        from .diagnostics import overlap_validation

        overlap_validation(p)
    else:
        print("Optional author benchmark unavailable; skipping overlap diagnostics.")
    prepare_public()
    prepare_equities()
