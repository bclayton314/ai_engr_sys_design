from datetime import datetime, UTC
from pathlib import Path


def utc_timestamp() -> str:
    """
    Return a compact UTC timestamp for run IDs.
    """
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: str | Path) -> Path:
    """
    Create a directory if it does not exist and return it as a Path.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p