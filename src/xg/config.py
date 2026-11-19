"""Configuration management: YAML file + dotted-key overrides, with project paths."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "config.yaml"


class Config(dict):
    """Dict with attribute-style access, dotted ``get_path`` and a stable hash."""

    def __getattr__(self, item: str) -> Any:
        try:
            v = self[item]
        except KeyError as e:
            raise AttributeError(item) from e
        return Config(v) if isinstance(v, dict) else v

    def get_path(self, dotted: str, default: Any = None) -> Any:
        cur: Any = self
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur

    def set_path(self, dotted: str, value: Any) -> None:
        cur: Any = self
        parts = dotted.split(".")
        for part in parts[:-1]:
            cur = cur.setdefault(part, {})
        cur[parts[-1]] = value

    def hash(self) -> str:
        blob = json.dumps(self, sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:10]

    # -- paths ----------------------------------------------------------------
    def path(self, key: str) -> Path:
        p = Path(self.project[key])
        return p if p.is_absolute() else PROJECT_ROOT / p


def load_config(path: str | os.PathLike | None = None, overrides: dict[str, Any] | None = None) -> Config:
    path = Path(path) if path else DEFAULT_CONFIG
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    cfg = Config(copy.deepcopy(raw))
    for k, v in (overrides or {}).items():
        cfg.set_path(k, v)
    return cfg


def resolve(cfg: Config, p: str | os.PathLike) -> Path:
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p
