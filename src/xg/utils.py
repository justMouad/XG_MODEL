"""Small shared helpers: IO that works with or without pyarrow, seeding, git hash."""
from __future__ import annotations

import json
import random
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from .logging_utils import get_logger

log = get_logger(__name__)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _has_pyarrow() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        return False


def table_path(base: Path | str) -> Path:
    """Return the path (with extension) of a table stored under ``base`` (no extension)."""
    base = Path(base)
    for ext in (".parquet", ".csv.gz"):
        if base.with_suffix(ext).exists():
            return base.with_suffix(ext)
    return base.with_suffix(".parquet" if _has_pyarrow() else ".csv.gz")


def save_table(df: pd.DataFrame, base: Path | str) -> Path:
    base = Path(base)
    base.parent.mkdir(parents=True, exist_ok=True)
    if _has_pyarrow():
        out = base.with_suffix(".parquet")
        df.to_parquet(out, index=False)
    else:
        out = base.with_suffix(".csv.gz")
        df.to_csv(out, index=False)
    log.info("Saved %s rows x %s cols -> %s", len(df), df.shape[1], out)
    return out


def load_table(base: Path | str) -> pd.DataFrame:
    p = table_path(base)
    if not p.exists():
        raise FileNotFoundError(f"Table not found: {p} (run the previous pipeline step first)")
    return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "nogit"


def json_dump(obj, path: Path | str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=_json_default)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
