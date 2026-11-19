"""Inference: predict xG for new shots.

Accepts either
  * a DataFrame in the *raw shot schema* (same columns as ``shots_raw``, minimal: x, y + any optional context), or
  * a list/JSON of StatsBomb event dicts (type == Shot), parsed with the same code as training.
Missing optional columns are filled with NaN/"Missing" and handled by the training-time imputers.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ..config import PROJECT_ROOT, Config
from ..features.engineering import build_features
from ..logging_utils import get_logger

log = get_logger(__name__)

REQUIRED = ["x", "y"]


class XGPredictor:
    def __init__(self, bundle: dict):
        self.bundle = bundle
        self.model = bundle["model"]
        self.numeric, self.categorical = bundle["numeric"], bundle["categorical"]
        self.feature_cfg = bundle.get("feature_config", {})

    # -- construction -----------------------------------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path | None = None, models_dir: str | Path | None = None) -> "XGPredictor":
        if path is None:
            root = Path(models_dir) if models_dir else PROJECT_ROOT / "models"
            latest = root / "latest.json"
            if not latest.exists():
                raise FileNotFoundError(f"No trained model: {latest} missing. Run `python -m xg.cli train` first.")
            path = json.loads(latest.read_text())["path"]
        if not Path(path).exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        bundle = joblib.load(path)
        log.info("Loaded model %s (%s, run %s)", bundle["name"], Path(path).name, bundle.get("run_id"))
        return cls(bundle)

    # -- preparation --------------------------------------------------------------------------------------
    def _prepare(self, shots: pd.DataFrame) -> pd.DataFrame:
        shots = shots.copy()
        missing = [c for c in REQUIRED if c not in shots.columns]
        if missing:
            raise ValueError(f"Input is missing required columns: {missing}")
        bad = shots[(shots["x"] < 0) | (shots["x"] > 120) | (shots["y"] < 0) | (shots["y"] > 80)]
        if len(bad):
            raise ValueError(f"{len(bad)} shots have coordinates outside the 120x80 StatsBomb pitch")
        for c in ("match_id", "match_date", "player_id", "shot_id", "is_goal"):
            if c not in shots.columns:
                shots[c] = np.nan if c != "shot_id" else np.arange(len(shots))
        for c in ("minute", "second"):
            if c not in shots.columns:
                shots[c] = np.nan
        cfg = Config({"features": {"use_freeze_frame": self.feature_cfg.get("use_freeze_frame", True),
                                   "use_player_history": False}})
        feats = build_features(shots, cfg)
        for c in self.numeric:
            if c not in feats.columns:
                feats[c] = np.nan
        for c in self.categorical:
            if c not in feats.columns:
                feats[c] = "Missing"
        return feats

    # -- API ------------------------------------------------------------------------------------------------
    def predict(self, shots: pd.DataFrame) -> pd.Series:
        feats = self._prepare(shots)
        p = self.model.predict_proba(feats[self.numeric + self.categorical])[:, 1]
        return pd.Series(p, index=feats.index, name="xg")

    def predict_events(self, events: list[dict], match: dict | None = None) -> pd.DataFrame:
        """Predict from raw StatsBomb events. ``match`` is the matches.json entry (optional but recommended)."""
        from ..data.ingestion import parse_match_events
        match = match or {"match_id": 0, "match_date": None, "home_team": {"home_team_name": "home"}, "away_team": {"away_team_name": "away"},
                          "competition": {}, "season": {}}
        rows = pd.DataFrame(parse_match_events(events, match))
        if rows.empty:
            return rows
        rows = rows[rows["shot_type"] != "Penalty"].reset_index(drop=True)
        rows["xg"] = self.predict(rows).values
        return rows[["shot_id", "player", "team", "minute", "x", "y", "xg", "is_goal"]]

    def explain(self, shots: pd.DataFrame, index: int = 0, top_k: int = 8) -> pd.DataFrame:
        from ..explain.interpret import explain_shot
        feats = self._prepare(shots)
        return explain_shot(self.bundle, feats.iloc[[index]], top_k)


def predict_file(model_path: str | None, input_path: str, output_path: str | None) -> pd.DataFrame:
    pred = XGPredictor.load(model_path)
    p = Path(input_path)
    if p.suffix == ".json":
        data = json.loads(p.read_text())
        df = pd.DataFrame(data) if isinstance(data, list) and data and "x" in data[0] else None
        if df is None:
            res = pred.predict_events(data)
            out = res
        else:
            df["xg"] = pred.predict(df).values
            out = df
    else:
        df = pd.read_csv(p)
        df["xg"] = pred.predict(df).values
        out = df
    if output_path:
        out.to_csv(output_path, index=False)
        log.info("Wrote %s", output_path)
    return out
