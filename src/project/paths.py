"""Project-relative paths for an editable source checkout."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "config.yaml"
REPLICATION_TABLES = ROOT / "results" / "replication" / "tables"
REPLICATION_FIGURES = ROOT / "results" / "replication" / "figures"
PROCESSING_EXPLORATION = ROOT / "processing_exploration"


def load_config():
    import yaml
    with CONFIG.open(encoding='utf-8') as stream:
        return yaml.safe_load(stream)


GENERATED_DIRS = (
    "data/raw", "data/interim", "data/processed",
    "results/replication/tables", "results/replication/figures",
    "results/extension/tables", "results/extension/figures",
)

def ensure_directories():
    """Create ignored data/output folders in a fresh clone."""
    for directory in GENERATED_DIRS:
        (ROOT / directory).mkdir(parents=True, exist_ok=True)
