"""Football-specific feature engineering on the raw shot table.

StatsBomb pitch: 120 x 80 yards, attacking direction is always towards x = 120, goal mouth spans
y in [36, 44] (8 yards). All shots are therefore already normalised to attack left -> right.

Leakage policy
--------------
* Columns in ``LEAKY_COLUMNS`` (outcome, end location, StatsBomb's own xG, deflected, ...) are never features.
* ``score_diff`` / possession counters are computed in ingestion from events strictly *before* the shot.
* Optional player history (``add_player_history``) only uses matches that finished before the current match
  (strict ``match_date``/``match_id`` ordering, shifted by one match), so no shot ever sees its own match.
"""
from __future__ import annotations

import json
from typing import Iterable

import numpy as np
import pandas as pd

from ..config import Config
from ..logging_utils import get_logger

log = get_logger(__name__)

GOAL_X, GOAL_Y_LOW, GOAL_Y_HIGH, GOAL_Y_MID = 120.0, 36.0, 44.0, 40.0
FALLBACK_CONVERSION = 0.10

#: Columns that encode (or are only known after) the shot result. Never used as model inputs.
LEAKY_COLUMNS = ["outcome", "is_goal", "statsbomb_xg", "end_x", "end_y", "end_z", "deflected", "redirect"]

NUMERIC_BASE = [
    "x", "y", "distance", "angle", "log_distance", "abs_y_offset", "minute_exact", "period",
    "is_home", "score_diff", "under_pressure", "first_time", "aerial_won", "follows_dribble",
    "one_on_one", "open_goal", "prev_event_gap", "carry_length", "poss_events_before",
    "poss_passes_before", "poss_duration",
    "kp_exists", "kp_length", "kp_angle", "kp_cross", "kp_cut_back", "kp_switch", "kp_through_ball",
    "kp_start_x", "kp_start_y", "kp_end_x", "kp_end_y", "kp_progress", "kp_end_distance",
]
NUMERIC_FREEZE = [
    "ff_available", "n_defenders_in_cone", "n_attackers_in_cone", "nearest_defender_dist", "n_defenders_within_3",
    "n_defenders_ahead", "gk_available", "gk_dist_goal", "gk_dist_shooter", "gk_off_line", "gk_lateral_offset",
    "gk_in_cone", "n_players_visible",
]
NUMERIC_HISTORY = ["player_prior_shots", "player_prior_conv_shrunk", "player_prior_conv_vs_league"]
CATEGORICAL_BASE = ["body_part", "technique", "shot_type", "play_pattern", "position_group", "prev_event_type",
                    "kp_height", "kp_type", "kp_technique"]
CATEGORICAL_CONTEXT = ["competition_name"]


# --------------------------------------------------------------------------- geometry
def shot_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.hypot(GOAL_X - x, GOAL_Y_MID - y)


def shot_angle(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Visible goal-mouth angle (radians) between the two posts, seen from (x, y)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    v1x, v1y = GOAL_X - x, GOAL_Y_LOW - y
    v2x, v2y = GOAL_X - x, GOAL_Y_HIGH - y
    dot = v1x * v2x + v1y * v2y
    n1, n2 = np.hypot(v1x, v1y), np.hypot(v2x, v2y)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.clip(dot / (n1 * n2), -1.0, 1.0)
    return np.arccos(cos)


def _in_triangle(px, py, ax, ay, bx, by, cx, cy) -> bool:
    d1 = (px - bx) * (ay - by) - (ax - bx) * (py - by)
    d2 = (px - cx) * (by - cy) - (bx - cx) * (py - cy)
    d3 = (px - ax) * (cy - ay) - (cx - ax) * (py - ay)
    neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (neg and pos)


_FF_NAN = {k: np.nan for k in NUMERIC_FREEZE}
_FF_NAN.update({"ff_available": 0, "gk_available": 0, "gk_in_cone": 0})


def freeze_frame_features(x: float, y: float, ff_json: str | None) -> dict:
    """Defender / goalkeeper features from the StatsBomb shot freeze frame."""
    if not ff_json or x is None or y is None or x != x or y != y:
        return dict(_FF_NAN)
    try:
        ff = json.loads(ff_json)
    except (TypeError, ValueError):
        return dict(_FF_NAN)
    if not ff:
        return dict(_FF_NAN)

    n_def_cone = n_att_cone = n_def_3 = n_def_ahead = 0
    nearest = np.inf
    gk = None
    for p in ff:
        loc = p.get("location")
        if not loc:
            continue
        px, py = loc[0], loc[1]
        is_team = bool(p.get("teammate"))
        pos = (p.get("position") or {}).get("name", "")
        if not is_team and pos == "Goalkeeper":
            gk = (px, py)
            continue
        in_cone = _in_triangle(px, py, x, y, GOAL_X, GOAL_Y_LOW, GOAL_X, GOAL_Y_HIGH)
        if is_team:
            n_att_cone += int(in_cone)
        else:
            d = float(np.hypot(px - x, py - y))
            nearest = min(nearest, d)
            n_def_cone += int(in_cone)
            n_def_3 += int(d <= 3.0)
            n_def_ahead += int(px > x)

    out = {
        "ff_available": 1, "n_players_visible": len(ff), "n_defenders_in_cone": n_def_cone,
        "n_attackers_in_cone": n_att_cone, "nearest_defender_dist": nearest if np.isfinite(nearest) else np.nan,
        "n_defenders_within_3": n_def_3, "n_defenders_ahead": n_def_ahead,
        "gk_available": 0, "gk_dist_goal": np.nan, "gk_dist_shooter": np.nan, "gk_off_line": np.nan,
        "gk_lateral_offset": np.nan, "gk_in_cone": 0,
    }
    if gk is not None:
        gx, gy = gk
        out.update({
            "gk_available": 1,
            "gk_dist_goal": float(np.hypot(GOAL_X - gx, GOAL_Y_MID - gy)),
            "gk_dist_shooter": float(np.hypot(gx - x, gy - y)),
            "gk_off_line": float(GOAL_X - gx),
            "gk_in_cone": int(_in_triangle(gx, gy, x, y, GOAL_X, GOAL_Y_LOW, GOAL_X, GOAL_Y_HIGH)),
        })
        # signed perpendicular distance of GK from the shooter->goal-centre line
        vx, vy = GOAL_X - x, GOAL_Y_MID - y
        norm = np.hypot(vx, vy)
        if norm > 0:
            out["gk_lateral_offset"] = float(((gx - x) * vy - (gy - y) * vx) / norm)
    return out


def position_group(pos: str | None) -> str:
    if not isinstance(pos, str):
        return "Unknown"
    p = pos.lower()
    if "goalkeeper" in p:
        return "Goalkeeper"
    if "back" in p or "defen" in p:
        return "Defender"
    if "forward" in p or "wing" in p or "striker" in p:  # wing-backs are caught by the "back" test above
        return "Forward"
    if "midfield" in p:
        return "Midfielder"
    return "Other"


# --------------------------------------------------------------------------- main builder
def build_features(raw: pd.DataFrame, cfg: Config | None = None) -> pd.DataFrame:
    """Return the raw table with engineered features appended (no rows dropped)."""
    use_ff = True if cfg is None else cfg.features.get("use_freeze_frame", True)
    df = raw.copy()
    # optional raw columns may be absent (e.g. at inference time): treat them as unknown, never as a silent zero
    raw_numeric = [c for c in NUMERIC_BASE if c not in ("distance", "angle", "log_distance", "abs_y_offset", "minute_exact", "kp_progress", "kp_end_distance")]
    for c in raw_numeric + ["minute", "second"]:
        if c not in df.columns:
            df[c] = np.nan
    for c in CATEGORICAL_BASE + CATEGORICAL_CONTEXT + ["position"]:
        if c not in df.columns and c != "position_group":
            df[c] = np.nan
    if "freeze_frame_json" not in df.columns:
        df["freeze_frame_json"] = None

    for c in ("x", "y"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["x", "y"]).reset_index(drop=True)
    if len(df) < before:
        log.warning("Dropped %s shots without coordinates", before - len(df))

    df["distance"] = shot_distance(df["x"].values, df["y"].values)
    df["angle"] = shot_angle(df["x"].values, df["y"].values)
    df["log_distance"] = np.log1p(df["distance"])
    df["abs_y_offset"] = (df["y"] - GOAL_Y_MID).abs()
    df["minute_exact"] = df["minute"] + df["second"].fillna(0) / 60.0  # NaN stays NaN (unknown != minute 0)
    df["position_group"] = df["position"].map(position_group)

    # assist geometry
    for c in ("kp_start_x", "kp_start_y", "kp_end_x", "kp_end_y"):
        df[c] = pd.to_numeric(df.get(c), errors="coerce")
    d_start = np.hypot(GOAL_X - df["kp_start_x"], GOAL_Y_MID - df["kp_start_y"])
    d_end = np.hypot(GOAL_X - df["kp_end_x"], GOAL_Y_MID - df["kp_end_y"])
    df["kp_progress"] = d_start - d_end
    df["kp_end_distance"] = d_end

    if use_ff:
        ff = [freeze_frame_features(x, y, j) for x, y, j in zip(df["x"], df["y"], df["freeze_frame_json"])]
        df = pd.concat([df, pd.DataFrame(ff, index=df.index)], axis=1)
    else:
        for c in NUMERIC_FREEZE:
            df[c] = np.nan

    # rare-category hygiene for prev_event_type is handled by the one-hot min_frequency in preprocessing
    for c in CATEGORICAL_BASE + CATEGORICAL_CONTEXT:
        if c in df.columns:
            df[c] = df[c].astype("object").where(df[c].notna(), "Missing")

    if cfg is not None and cfg.features.get("use_player_history", False):
        df = add_player_history(df, k=float(cfg.features.get("history_shrinkage_shots", 40)))
    return df


def add_player_history(df: pd.DataFrame, k: float = 40.0) -> pd.DataFrame:
    """Empirical-Bayes shrunk player conversion using ONLY strictly earlier matches.

    Matches are ordered by (match_date, match_id); statistics for a match are the cumulative totals of all
    *previous* matches (exclusive cumulative sum), so no shot ever contributes to its own features.
    """
    df = df.copy()
    order = df[["match_id", "match_date"]].drop_duplicates().sort_values(["match_date", "match_id"]).reset_index(drop=True)
    order["match_rank"] = np.arange(len(order))
    df = df.merge(order[["match_id", "match_rank"]], on="match_id", how="left")

    pm = df.groupby(["player_id", "match_rank"]).agg(shots=("is_goal", "size"), goals=("is_goal", "sum")).reset_index()
    pm = pm.sort_values(["player_id", "match_rank"])
    pm["prior_shots"] = pm.groupby("player_id")["shots"].cumsum() - pm["shots"]
    pm["prior_goals"] = pm.groupby("player_id")["goals"].cumsum() - pm["goals"]

    league = df.groupby("match_rank").agg(shots=("is_goal", "size"), goals=("is_goal", "sum")).sort_index()
    league["cum_shots"] = league["shots"].cumsum() - league["shots"]
    league["cum_goals"] = league["goals"].cumsum() - league["goals"]
    league["league_rate"] = np.where(league["cum_shots"] > 0, league["cum_goals"] / league["cum_shots"].clip(lower=1), np.nan)
    pm = pm.merge(league[["league_rate"]], left_on="match_rank", right_index=True, how="left")
    # When no earlier match exists the league rate is unknown: use a FIXED prior (never a dataset-wide mean, which would
    # leak future outcomes). 0.10 is a typical shot conversion rate; it only affects the very first match(es).
    base = pm["league_rate"].fillna(FALLBACK_CONVERSION)
    pm["player_prior_shots"] = pm["prior_shots"]
    pm["player_prior_conv_shrunk"] = (pm["prior_goals"] + k * base) / (pm["prior_shots"] + k)
    pm["player_prior_conv_vs_league"] = pm["player_prior_conv_shrunk"] / base.clip(lower=1e-3)
    df = df.merge(pm[["player_id", "match_rank", "player_prior_shots", "player_prior_conv_shrunk", "player_prior_conv_vs_league"]],
                  on=["player_id", "match_rank"], how="left")
    return df.drop(columns=["match_rank"])


# --------------------------------------------------------------------------- feature lists
def feature_lists(cfg: Config | None = None) -> tuple[list[str], list[str]]:
    """Return (numeric_features, categorical_features) according to the config."""
    num = list(NUMERIC_BASE)
    cat = list(CATEGORICAL_BASE)
    if cfg is None or cfg.features.get("use_freeze_frame", True):
        num += NUMERIC_FREEZE
    if cfg is not None and cfg.features.get("use_player_history", False):
        num += NUMERIC_HISTORY
    if cfg is not None and cfg.features.get("use_competition_context", False):
        cat += CATEGORICAL_CONTEXT
    forbidden = set(LEAKY_COLUMNS) & set(num + cat)
    assert not forbidden, f"Leaky columns in feature list: {forbidden}"
    return num, cat
