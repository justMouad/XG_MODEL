"""Pipeline glue: each function is one CLI step and communicates through files on disk."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from .analytics.export import export_tables
from .analytics.football import run_analytics
from .config import Config
from .data.ingestion import ingest
from .data.splits import train_test_split_matches
from .data.synthetic import make_synthetic
from .eda.explore import run_eda
from .explain.interpret import run_interpretation
from .features.engineering import build_features
from .logging_utils import get_logger
from .models.train import train_all
from .utils import load_table, save_table

log = get_logger(__name__)


def step_ingest(cfg: Config, synthetic: bool = False, n_synth_matches: int = 150) -> None:
    if synthetic:
        log.warning("SYNTHETIC DATA: pipeline demo only. Metrics have no football meaning.")
        tables = make_synthetic(n_synth_matches, seed=cfg.project["seed"])
        for k, df in tables.items():
            save_table(df, cfg.path("data_dir") / "raw" / k)
    else:
        ingest(cfg)


def step_features(cfg: Config) -> pd.DataFrame:
    raw = load_table(cfg.path("data_dir") / "raw" / "shots_raw")
    feats = build_features(raw, cfg)
    save_table(feats, cfg.path("data_dir") / "processed" / "shots_features")
    return feats


def step_eda(cfg: Config) -> dict:
    df = load_table(cfg.path("data_dir") / "processed" / "shots_features")
    return run_eda(df, cfg.path("reports_dir") / "eda")


def step_train(cfg: Config, models: list[str] | None = None) -> dict:
    df = load_table(cfg.path("data_dir") / "processed" / "shots_features")
    res = train_all(cfg, df, models)
    # keep reports in sync: copy the headline plots next to the run
    return res


def latest_run_dir(cfg: Config) -> Path:
    latest = cfg.path("models_dir") / "latest.json"
    if not latest.exists():
        raise FileNotFoundError("No trained model found - run the `train` step first")
    return Path(json.loads(latest.read_text())["path"]).parent


def step_explain(cfg: Config) -> None:
    run_dir = latest_run_dir(cfg)
    bundle = joblib.load(run_dir / "model.joblib")
    df = load_table(cfg.path("data_dir") / "processed" / "shots_features").sort_values(["match_date", "match_id", "shot_id"]).reset_index(drop=True)
    _, te = train_test_split_matches(df, cfg.split["test_size"], cfg.split["strategy"], cfg.project["seed"])
    run_interpretation(bundle, df.iloc[te].reset_index(drop=True), run_dir / "explain", cfg)


def step_analytics(cfg: Config) -> dict[str, pd.DataFrame]:
    run_dir = latest_run_dir(cfg)
    proc, raw = cfg.path("data_dir") / "processed", cfg.path("data_dir") / "raw"
    feats = load_table(proc / "shots_features")
    xg = load_table(proc / "shots_xg")
    df = feats.merge(xg[["shot_id", "xg", "is_test_shot"]], on="shot_id", how="inner")
    if len(df) != len(xg):
        log.warning("Feature table and xG table differ in length (%s vs %s)", len(df), len(xg))
    try:
        minutes = load_table(raw / "minutes")
    except FileNotFoundError:
        minutes = None
    try:
        matches = load_table(raw / "matches")
    except FileNotFoundError:
        matches = None
    analytics = run_analytics(df, minutes, matches)
    for k, v in analytics.items():
        save_table(v, proc / "analytics" / k)
    return export_tables(df, matches, analytics, run_dir, cfg.path("reports_dir") / "exports", minutes)


def step_load_db(cfg: Config) -> None:
    from .db.postgres import load_exports_to_db
    load_exports_to_db(cfg)
