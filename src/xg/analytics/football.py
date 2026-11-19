"""Football analytics on top of out-of-fold xG.

Shot quality  = how good the chances a player/team takes are        -> xG per shot, xG per 90.
Finishing     = how much a player converts relative to that quality -> Goals - xG (with uncertainty).

Statistical care
----------------
* A shot is a Bernoulli(p) trial, so Var(G) = sum p(1-p) and  z = (G - xG) / sqrt(sum p(1-p)).
  |z| < 1.64 means the over/under-performance is within what luck alone commonly produces.
* ``finishing_skill_shrunk`` = (G + k) / (xG + k): an empirical-Bayes style ratio pulled to 1 for small samples.
* ``split_half_reliability`` measures whether Goals-minus-xG is persistent (player skill) or mostly noise.
* StatsBomb Open Data has no post-shot xG (PSxG); with a pre-shot model, "finishing" is a blend of shot placement/power
  skill, residual context the model misses, and luck. It is a ranking signal, not proof of skill.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ..logging_utils import get_logger

log = get_logger(__name__)


def add_finishing_stats(g: pd.DataFrame, k: float = 10.0, z_thr: float = 1.0) -> pd.DataFrame:
    g = g.copy()
    g["goals_minus_xg"] = g["goals"] - g["xg"]
    g["g_per_xg"] = g["goals"] / g["xg"].replace(0, np.nan)
    g["xg_per_shot"] = g["xg"] / g["shots"]
    g["conversion"] = g["goals"] / g["shots"]
    g["z_finishing"] = g["goals_minus_xg"] / np.sqrt(g["xg_var"].clip(lower=1e-9))
    g["p_value"] = 2 * (1 - stats.norm.cdf(g["z_finishing"].abs()))
    g["finishing_skill_shrunk"] = (g["goals"] + k) / (g["xg"] + k)
    g["finishing_label"] = np.select([g["z_finishing"] >= z_thr, g["z_finishing"] <= -z_thr], ["Overperforming", "Underperforming"], "In line")
    return g


def _agg(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    d = df.assign(_var=df["xg"] * (1 - df["xg"]))
    out = d.groupby(by, dropna=False).agg(shots=("xg", "size"), goals=("is_goal", "sum"), xg=("xg", "sum"), xg_var=("_var", "sum"),
                                          matches=("match_id", "nunique")).reset_index()
    return out


def player_table(df: pd.DataFrame, minutes: pd.DataFrame | None, min_shots: int = 1, by_season: bool = False) -> pd.DataFrame:
    by = ["player_id", "player"] + (["competition_name", "season_name"] if by_season else [])
    out = _agg(df, by)
    team = df.groupby(by + ["team"]).size().reset_index(name="n").sort_values("n", ascending=False).drop_duplicates(by)[by + ["team"]]
    out = out.merge(team, on=by, how="left")
    if minutes is not None and len(minutes):
        mm = minutes.merge(df[["match_id", "competition_name", "season_name"]].drop_duplicates("match_id"), on="match_id", how="left")
        mby = ["player_id"] + (["competition_name", "season_name"] if by_season else [])
        mm = mm.groupby(mby)["minutes"].sum().reset_index()
        out = out.merge(mm, on=mby, how="left")
        for c in ("shots", "goals", "xg"):
            out[f"{c}_per90"] = out[c] / out["minutes"].replace(0, np.nan) * 90
    else:
        out["minutes"] = np.nan
        out["xg_per90"] = out["goals_per90"] = out["shots_per90"] = np.nan
    out = add_finishing_stats(out)
    return out[out["shots"] >= min_shots].sort_values("xg", ascending=False).reset_index(drop=True)


def team_table(df: pd.DataFrame, by_season: bool = False) -> pd.DataFrame:
    extra = ["competition_name", "season_name"] if by_season else []
    att = _agg(df, ["team"] + extra).rename(columns={"shots": "shots_for", "goals": "goals_for", "xg": "xg_for", "xg_var": "xg_var_for"})
    dfn = df.assign(team_def=df["opponent"])
    dfe = _agg(dfn.drop(columns=["team"]).rename(columns={"team_def": "team"}), ["team"] + extra)
    dfe = dfe.rename(columns={"shots": "shots_against", "goals": "goals_against", "xg": "xg_against"})[["team"] + extra + ["shots_against", "goals_against", "xg_against"]]
    t = att.merge(dfe, on=["team"] + extra, how="outer").fillna(0)
    t["xg_diff"] = t["xg_for"] - t["xg_against"]
    t["goal_diff"] = t["goals_for"] - t["goals_against"]
    t["xg_per_shot"] = t["xg_for"] / t["shots_for"].replace(0, np.nan)
    t["conversion"] = t["goals_for"] / t["shots_for"].replace(0, np.nan)
    t["xg_per_match"] = t["xg_for"] / t["matches"].replace(0, np.nan)
    t["xga_per_match"] = t["xg_against"] / t["matches"].replace(0, np.nan)
    t["shots_per_match"] = t["shots_for"] / t["matches"].replace(0, np.nan)
    t["goals_minus_xg"] = t["goals_for"] - t["xg_for"]
    t["attacking_efficiency"] = t["goals_for"] / t["xg_for"].replace(0, np.nan)   # finishing efficiency vs chance quality
    t["z_finishing"] = t["goals_minus_xg"] / np.sqrt(t["xg_var_for"].clip(lower=1e-9))
    t["luck_adjusted_goal_diff"] = t["xg_diff"]  # xG difference as the luck-neutral performance estimate
    return t.sort_values("xg_for", ascending=False).reset_index(drop=True)


def shot_quality_distribution(df: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
    bins = [0, 0.03, 0.06, 0.1, 0.2, 0.35, 0.5, 1.0]
    labels = ["<3%", "3-6%", "6-10%", "10-20%", "20-35%", "35-50%", ">50%"]
    d = df.assign(xg_band=pd.cut(df["xg"], bins=bins, labels=labels, include_lowest=True))
    keys = ["xg_band"] + ([by] if by else [])
    out = d.groupby(keys, observed=True).agg(shots=("xg", "size"), goals=("is_goal", "sum"), xg=("xg", "sum")).reset_index()
    out["observed_rate"] = out["goals"] / out["shots"]
    out["mean_xg"] = out["xg"] / out["shots"]
    return out


def competition_season_table(df: pd.DataFrame) -> pd.DataFrame:
    out = _agg(df, ["competition_name", "season_name"])
    out["xg_per_shot"] = out["xg"] / out["shots"]
    out["conversion"] = out["goals"] / out["shots"]
    out["goals_per_match"] = out["goals"] / out["matches"]
    out["xg_per_match"] = out["xg"] / out["matches"]
    out["g_per_xg"] = out["goals"] / out["xg"]
    return out.sort_values(["competition_name", "season_name"]).reset_index(drop=True)


def split_half_reliability(df: pd.DataFrame, min_shots_half: int = 20, seed: int = 42) -> dict:
    """Is finishing (goals - xG per shot) persistent? Correlate odd-/even-match halves for each player.
    Spearman rho near 0 => mostly luck/noise; clearly positive => a repeatable component exists."""
    rng = np.random.default_rng(seed)
    matches = df["match_id"].unique()
    half = dict(zip(matches, rng.integers(0, 2, len(matches))))
    d = df.assign(half=df["match_id"].map(half))
    g = d.groupby(["player_id", "half"]).agg(shots=("xg", "size"), res=("is_goal", "sum"), xg=("xg", "sum")).reset_index()
    g["res_per_shot"] = (g["res"] - g["xg"]) / g["shots"]
    w = g.pivot(index="player_id", columns="half", values=["shots", "res_per_shot"])
    w = w.dropna()
    w = w[(w[("shots", 0)] >= min_shots_half) & (w[("shots", 1)] >= min_shots_half)]
    if len(w) < 10:
        return {"n_players": int(len(w)), "spearman_rho": None, "note": "too few players with enough shots in both halves"}
    rho, p = stats.spearmanr(w[("res_per_shot", 0)], w[("res_per_shot", 1)])
    return {"n_players": int(len(w)), "spearman_rho": float(rho), "p_value": float(p), "min_shots_per_half": min_shots_half}


def overunder_performers(players: pd.DataFrame, min_shots: int = 30, top: int = 15) -> dict[str, pd.DataFrame]:
    p = players[players["shots"] >= min_shots]
    cols = ["player", "team", "shots", "goals", "xg", "goals_minus_xg", "z_finishing", "finishing_skill_shrunk", "finishing_label"]
    return {"over": p.sort_values("z_finishing", ascending=False).head(top)[cols], "under": p.sort_values("z_finishing").head(top)[cols]}


def run_analytics(df: pd.DataFrame, minutes: pd.DataFrame | None, matches: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """Compute every analytics table from a shots frame that already contains an out-of-fold ``xg`` column."""
    assert "xg" in df.columns and df["xg"].notna().all(), "xg column required with no NaN"
    out = {
        "players": player_table(df, minutes),
        "players_by_season": player_table(df, minutes, by_season=True),
        "teams": team_table(df),
        "teams_by_season": team_table(df, by_season=True),
        "shot_quality": shot_quality_distribution(df),
        "shot_quality_by_competition": shot_quality_distribution(df, "competition_name"),
        "competitions": competition_season_table(df),
    }
    rel = split_half_reliability(df)
    out["finishing_reliability"] = pd.DataFrame([rel])
    log.info("Finishing split-half reliability: %s", rel)
    return out
