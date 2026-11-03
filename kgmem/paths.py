"""Locations of the example checkpoints in a source checkout."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def checkpoint_path(name: str) -> str:
    return str(REPO_ROOT / "checkpoints" / name)


def ensure_parent(path):
    """Create a file path's parent directory and return the path unchanged."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return path
