"""PostgreSQL (or SQLite fallback) analytical storage via SQLAlchemy."""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from ..config import Config, resolve
from ..logging_utils import get_logger

log = get_logger(__name__)

VIEWS_SQL = [
    """CREATE VIEW v_player_summary AS
       SELECT player, team, shots, goals, xg, goals_minus_xg, z_finishing, finishing_label, xg_per90 FROM analytics_players""",
    """CREATE VIEW v_team_summary AS
       SELECT team, shots_for, goals_for, xg_for, xg_against, xg_diff, attacking_efficiency FROM analytics_teams""",
]
INDEXES = {"fact_shots": ["match_id", "player_id", "team", "competition_name", "season_name"], "analytics_players": ["player_id"], "analytics_teams": ["team"]}


def get_engine(cfg: Config):
    """Engine for PostgreSQL (env DATABASE_URL > config); falls back to a local SQLite file if unreachable."""
    from sqlalchemy import create_engine, text
    url = os.environ.get("DATABASE_URL") or cfg.database.get("url")
    try:
        eng = create_engine(url, pool_pre_ping=True)
        with eng.connect() as c:
            c.execute(text("SELECT 1"))
        log.info("Connected to %s", eng.url.render_as_string(hide_password=True))
        return eng
    except Exception as e:
        fb = resolve(cfg, cfg.database.get("fallback_sqlite", "data/xg.sqlite"))
        fb.parent.mkdir(parents=True, exist_ok=True)
        log.warning("Database %s unreachable (%s) - falling back to SQLite %s", (url or "").split("@")[-1], type(e).__name__, fb)
        return create_engine(f"sqlite:///{fb}")


def load_tables(cfg: Config, tables: dict[str, pd.DataFrame]) -> None:
    from sqlalchemy import text
    eng = get_engine(cfg)
    with eng.begin() as con:
        for v in ("v_player_summary", "v_team_summary"):
            con.execute(text(f"DROP VIEW IF EXISTS {v}"))
    for name, df in tables.items():
        try:
            df.to_sql(name, eng, if_exists="replace", index=False, chunksize=5000, method="multi" if eng.dialect.name != "sqlite" else None)
            log.info("Loaded %-28s %8s rows", name, len(df))
        except Exception as e:
            log.error("Failed loading %s: %s", name, e)
            raise
    with eng.begin() as con:
        for tbl, cols in INDEXES.items():
            if tbl in tables:
                for c in cols:
                    if c in tables[tbl].columns:
                        con.execute(text(f'CREATE INDEX IF NOT EXISTS ix_{tbl}_{c} ON "{tbl}" ("{c}")'))
        for v in VIEWS_SQL:
            try:
                con.execute(text(v))
            except Exception as e:
                log.warning("View not created: %s", e)


def read_table(engine, name: str, where: str | None = None) -> pd.DataFrame:
    q = f'SELECT * FROM "{name}"' + (f" WHERE {where}" if where else "")
    return pd.read_sql(q, engine)


def load_exports_to_db(cfg: Config, export_dir: Path | None = None) -> None:
    export_dir = export_dir or cfg.path("reports_dir") / "exports"
    tables = {p.stem: pd.read_csv(p) for p in sorted(export_dir.glob("*.csv"))}
    if not tables:
        raise FileNotFoundError(f"No exported tables in {export_dir}; run `python -m xg.cli analytics` first")
    load_tables(cfg, tables)
