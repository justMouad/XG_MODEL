"""Exploratory data analysis: structure, missingness, balance, relationships with scoring probability."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..features.engineering import GOAL_X, GOAL_Y_HIGH, GOAL_Y_LOW
from ..logging_utils import get_logger
from ..utils import json_dump

log = get_logger(__name__)


def draw_half_pitch(ax, color="#444"):
    """Attacking half (x 60..120) of a StatsBomb pitch, goal on the right."""
    ax.set_xlim(60, 121)
    ax.set_ylim(80, 0)
    kw = dict(color=color, lw=1)
    ax.plot([60, 120, 120, 60, 60], [0, 0, 80, 80, 0], **kw)
    ax.plot([102, 120, 120, 102, 102], [18, 18, 62, 62, 18], **kw)       # penalty area
    ax.plot([114, 120, 120, 114, 114], [30, 30, 50, 50, 30], **kw)       # six-yard box
    ax.plot([GOAL_X, GOAL_X], [GOAL_Y_LOW, GOAL_Y_HIGH], color="red", lw=3)
    ax.add_patch(plt.Circle((60, 40), 10, fill=False, **kw))
    ax.set_aspect("equal")
    ax.axis("off")


def goal_rate_by(df: pd.DataFrame, col: str, bins=None, min_n: int = 30) -> pd.DataFrame:
    s = pd.cut(df[col], bins) if bins is not None else df[col]
    g = df.groupby(s, observed=True)["is_goal"].agg(["size", "sum", "mean"]).rename(columns={"size": "shots", "sum": "goals", "mean": "goal_rate"})
    return g[g["shots"] >= min_n].reset_index()


def run_eda(df: pd.DataFrame, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict = {
        "n_shots": int(len(df)), "n_goals": int(df.is_goal.sum()), "goal_rate": float(df.is_goal.mean()),
        "imbalance_ratio_neg_to_pos": float((1 - df.is_goal.mean()) / max(df.is_goal.mean(), 1e-9)),
        "n_matches": int(df.match_id.nunique()), "n_players": int(df.player_id.nunique()), "n_teams": int(df.team.nunique()),
        "date_range": [str(df.match_date.min()), str(df.match_date.max())],
        "freeze_frame_coverage": float(df["freeze_frame_json"].notna().mean()) if "freeze_frame_json" in df else None,
        "key_pass_coverage": float(df["kp_exists"].mean()) if "kp_exists" in df else None,
        "statsbomb_xg_coverage": float(df["statsbomb_xg"].notna().mean()) if "statsbomb_xg" in df else None,
    }
    miss = df.isna().mean().sort_values(ascending=False)
    miss[miss > 0].rename("missing_fraction").to_csv(out_dir / "missingness.csv")
    desc = df.select_dtypes("number").describe().T
    desc.to_csv(out_dir / "numeric_summary.csv")
    for col in ["body_part", "technique", "shot_type", "play_pattern", "position_group", "prev_event_type", "competition_name"]:
        if col in df:
            goal_rate_by(df, col, min_n=1).to_csv(out_dir / f"goal_rate_by_{col}.csv", index=False)

    # plots ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    d = goal_rate_by(df, "distance", bins=np.arange(0, 56, 3))
    axes[0, 0].plot([i.mid for i in d["distance"]], d["goal_rate"], marker="o")
    axes[0, 0].set(title="Goal rate vs distance to goal centre", xlabel="yards", ylabel="goal rate")
    a = goal_rate_by(df, "angle", bins=np.linspace(0, np.pi / 2, 12))
    axes[0, 1].plot([np.degrees(i.mid) for i in a["angle"]], a["goal_rate"], marker="o", color="C1")
    axes[0, 1].set(title="Goal rate vs visible goal angle", xlabel="degrees")
    bp = goal_rate_by(df, "body_part", min_n=1)
    axes[0, 2].bar(bp["body_part"], bp["goal_rate"], color="C2")
    axes[0, 2].set(title="Goal rate by body part")
    pp = goal_rate_by(df, "play_pattern", min_n=20)
    axes[1, 0].barh(pp["play_pattern"], pp["goal_rate"], color="C3")
    axes[1, 0].set(title="Goal rate by play pattern")
    hb = axes[1, 1]
    draw_half_pitch(hb)
    h = hb.hexbin(df["x"], df["y"], gridsize=24, extent=(60, 120, 0, 80), cmap="viridis", mincnt=3, bins="log", alpha=.9)
    hb.set_title("Shot density (log)")
    hexg = axes[1, 2]
    draw_half_pitch(hexg)
    hh = hexg.hexbin(df["x"], df["y"], C=df["is_goal"], reduce_C_function=np.mean, gridsize=18, extent=(60, 120, 0, 80), cmap="magma", mincnt=8)
    hexg.set_title("Goal rate by location (hex, n>=8)")
    fig.colorbar(hh, ax=hexg, fraction=0.04)
    for ax in axes.ravel()[:5]:
        ax.tick_params(axis="x", rotation=45, labelsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "eda_overview.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    top = miss[miss > 0].head(25)
    ax.barh(top.index[::-1], top.values[::-1])
    ax.set(title="Missing fraction (top 25 columns)")
    fig.tight_layout()
    fig.savefig(out_dir / "missingness.png", dpi=130)
    plt.close(fig)

    # simple leakage sanity check: univariate AUC of every numeric feature
    from sklearn.metrics import roc_auc_score
    aucs = {}
    for c in df.select_dtypes("number").columns:
        v = df[c]
        if c in ("is_goal",) or v.nunique() < 2 or v.isna().all():
            continue
        try:
            aucs[c] = float(roc_auc_score(df["is_goal"], v.fillna(v.median())))
        except ValueError:
            pass
    pd.Series(aucs, name="univariate_auc").sort_values(key=lambda s: (s - .5).abs(), ascending=False).to_csv(out_dir / "univariate_auc.csv")
    json_dump(summary, out_dir / "eda_summary.json")
    log.info("EDA written to %s", out_dir)
    return summary
