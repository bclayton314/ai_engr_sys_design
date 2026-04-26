from datetime import datetime, timezone
from pathlib import Path


def utc_timestamp() -> str:
    """
    Return a compact UTC timestamp for run IDs.
    """
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")


def ensure_dir(path: str | Path) -> Path:
    """
    Create a directory if it does not exist and return it as a Path.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p