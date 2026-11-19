"""Dashboard data access: PostgreSQL (or SQLite) when available, otherwise the CSV exports (also what Power BI reads)."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from ..config import PROJECT_ROOT, load_config

TABLES = ["fact_shots", "fact_minutes", "dim_match", "model_comparison", "model_cv_metrics", "model_curves", "feature_importance",
          "model_pdp", "model_shot_explanations", "model_runs"]


def export_dir() -> Path:
    cfg = load_config()
    return cfg.path("reports_dir") / "exports"


def load_all(source: str = "auto") -> tuple[dict[str, pd.DataFrame], str]:
    """Return (tables, source_used). ``source``: auto | database | csv."""
    cfg = load_config()
    if source in ("auto", "database") and (os.environ.get("DATABASE_URL") or source == "database"):
        try:
            from ..db.postgres import get_engine, read_table
            eng = get_engine(cfg)
            out = {}
            for t in TABLES:
                try:
                    out[t] = read_table(eng, t)
                except Exception:
                    pass
            if "fact_shots" in out:
                return out, f"database ({eng.dialect.name})"
        except Exception:
            if source == "database":
                raise
    d = export_dir()
    out = {t: pd.read_csv(d / f"{t}.csv") for t in TABLES if (d / f"{t}.csv").exists()}
    if "fact_shots" not in out:
        raise FileNotFoundError(f"No exported data in {d}. Run: python -m xg.cli run-all  (or --synthetic for a demo)")
    return out, f"csv exports ({d})"


def latest_run_dir() -> Path | None:
    p = load_config().path("models_dir") / "latest.json"
    if not p.exists():
        return None
    return Path(json.loads(p.read_text())["path"]).parent
