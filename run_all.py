"""Build the paper replication and model-stability extension."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setup", action="store_true", help="Only create data/results folders")
    parser.add_argument(
        "--data-only", action="store_true", help="Run implemented data preparation only"
    )
    parser.add_argument(
        "--results-only",
        action="store_true",
        help="Rebuild replication and extension results from existing data",
    )
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument(
        "--replication-only", action="store_true", help="Build only the paper replication"
    )
    scope.add_argument(
        "--extension-only", action="store_true", help="Build only the model-stability extension"
    )
    parser.add_argument(
        "--offline", action="store_true", help="Reuse cached raw data without network access"
    )
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT / "src"))
    from project.paths import ensure_directories

    ensure_directories()
    if args.setup:
        print("Project folders ready. Use the ordered scripts or --data-only.")
        return
    if args.data_only:
        from project.preparation import prepare_data

        prepare_data(download=not args.offline)
        return
    if not args.results_only and not args.extension_only:
        from project.preparation import prepare_data

        prepare_data(download=not args.offline)
    from project.models import extend, replicate

    if not args.extension_only:
        replicate()
    if not args.replication_only:
        extend()


if __name__ == "__main__":
    main()
