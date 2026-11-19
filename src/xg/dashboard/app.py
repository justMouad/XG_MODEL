"""Streamlit dashboard.   Run:  streamlit run src/xg/dashboard/app.py

Every player/team table is recomputed live from the filtered shots, so all visuals respond to the sidebar filters.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # allow `streamlit run` without installing the package

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from xg.analytics.football import add_finishing_stats, player_table, shot_quality_distribution, team_table
from xg.dashboard import figures as F
from xg.dashboard.data_access import latest_run_dir, load_all

st.set_page_config(page_title="xG Analytics Platform", page_icon="⚽", layout="wide")


@st.cache_data(show_spinner="Loading data ...")
def _load(source: str):
    return load_all(source)


# ----------------------------------------------------------------------------- sidebar
st.sidebar.title("⚽ xG Platform")
source = st.sidebar.selectbox("Data source", ["auto", "csv", "database"], help="auto = PostgreSQL if DATABASE_URL is set, else CSV exports")
try:
    T, used = _load(source)
except Exception as e:
    st.error(f"Could not load data: {e}")
    st.stop()
st.sidebar.caption(f"Source: {used}")

shots = T["fact_shots"].copy()
shots["match_date"] = pd.to_datetime(shots["match_date"], errors="coerce")
minutes_all = T.get("fact_minutes")

comp_opts = sorted(shots["competition_name"].dropna().unique())
comps = st.sidebar.multiselect("Competition", comp_opts, default=comp_opts)
f1 = shots[shots["competition_name"].isin(comps)]
season_opts = sorted(f1["season_name"].dropna().unique())
seasons = st.sidebar.multiselect("Season", season_opts, default=season_opts)
f2 = f1[f1["season_name"].isin(seasons)]
team_opts = sorted(f2["team"].dropna().unique())
teams = st.sidebar.multiselect("Team", team_opts, default=[])
stypes = st.sidebar.multiselect("Shot type", sorted(f2["shot_type"].dropna().unique()), default=[])
bparts = st.sidebar.multiselect("Body part", sorted(f2["body_part"].dropna().unique()), default=[])
patterns = st.sidebar.multiselect("Play pattern", sorted(f2["play_pattern"].dropna().unique()), default=[])
split_view = st.sidebar.radio("Shots to include", ["All shots (out-of-fold xG)", "Held-out test period only"], index=0)

fs = f2
if teams:
    fs = fs[fs["team"].isin(teams)]
if stypes:
    fs = fs[fs["shot_type"].isin(stypes)]
if bparts:
    fs = fs[fs["body_part"].isin(bparts)]
if patterns:
    fs = fs[fs["play_pattern"].isin(patterns)]
if split_view.startswith("Held-out") and "is_test_shot" in fs:
    fs = fs[fs["is_test_shot"] == 1]
players_opts = sorted(fs["player"].dropna().unique())
if fs.empty:
    st.warning("No shots match the current filters.")
    st.stop()

mins = None
if minutes_all is not None:
    mins = minutes_all[minutes_all["match_id"].isin(fs["match_id"].unique())]

tabs = st.tabs(["Overview", "Player Analytics", "Team Analytics", "Shot Analysis", "Model Analytics"])
run = T["model_runs"].iloc[-1] if "model_runs" in T and len(T["model_runs"]) else None

# ----------------------------------------------------------------------------- Overview
with tabs[0]:
    st.header("Overview")
    c = st.columns(6)
    c[0].metric("Total shots", f"{len(fs):,}")
    c[1].metric("Total goals", f"{int(fs.is_goal.sum()):,}")
    c[2].metric("Total xG", f"{fs.xg.sum():,.1f}")
    c[3].metric("Avg xG / shot", f"{fs.xg.mean():.3f}")
    c[4].metric("Conversion", f"{100 * fs.is_goal.mean():.1f}%")
    c[5].metric("Goals − xG", f"{fs.is_goal.sum() - fs.xg.sum():+.1f}")
    if run is not None and "model_comparison" in T:
        mc = T["model_comparison"]
        best = mc[mc["model"] == run["best_model"]].iloc[0]
        st.subheader(f"Model performance — best model: {run['best_model']}  (calibration: {run['calibration']})")
        m = st.columns(6)
        m[0].metric("Test ROC-AUC", f"{best['roc_auc']:.3f}")
        m[1].metric("Test PR-AUC", f"{best['pr_auc']:.3f}")
        m[2].metric("Test log loss", f"{best['log_loss']:.4f}")
        m[3].metric("Test Brier", f"{best['brier_score']:.4f}")
        m[4].metric("Test ECE", f"{best['ece']:.4f}")
        m[5].metric("Test xG / goals", f"{best['xg_over_goals']:.3f}")
        st.caption(f"Run {run['run_id']} · trained on matches up to {run['train_until']}, tested from {run['test_from']}. "
                   "xG shown in this dashboard is out-of-fold: every shot is scored by a model that never saw its match.")
    left, right = st.columns(2)
    cs = fs.groupby(["competition_name", "season_name"]).agg(goals=("is_goal", "sum"), xg=("xg", "sum")).reset_index()
    cs["label"] = cs["competition_name"] + " " + cs["season_name"].astype(str)
    fig = go.Figure([go.Bar(x=cs["label"], y=cs["goals"], name="Goals"), go.Bar(x=cs["label"], y=cs["xg"], name="xG")])
    fig.update_layout(barmode="group", title="Goals vs xG by competition-season", height=380)
    left.plotly_chart(fig, use_container_width=True)
    sq = shot_quality_distribution(fs)
    fig = go.Figure([go.Bar(x=sq["xg_band"].astype(str), y=sq["shots"], name="Shots"),
                     go.Scatter(x=sq["xg_band"].astype(str), y=sq["observed_rate"], name="Observed goal rate", yaxis="y2", mode="lines+markers"),
                     go.Scatter(x=sq["xg_band"].astype(str), y=sq["mean_xg"], name="Mean xG", yaxis="y2", mode="lines+markers")])
    fig.update_layout(title="Shot-quality distribution (bars) and calibration by band (lines)", yaxis2=dict(overlaying="y", side="right", tickformat=".0%"), height=380)
    right.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------------- Players
with tabs[1]:
    st.header("Player Analytics")
    min_shots = st.slider("Minimum shots", 1, 100, 20, key="ps")
    pt = player_table(fs, mins, min_shots)
    if pt.empty:
        st.info("No players meet the minimum-shots threshold.")
    else:
        l, r = st.columns(2)
        fig = px.scatter(pt, x="xg", y="goals", color="z_finishing", color_continuous_scale="RdBu", color_continuous_midpoint=0,
                         hover_data=["player", "team", "shots", "goals_minus_xg"], title="Goals vs xG (colour = finishing z-score)")
        mx = float(max(pt.xg.max(), pt.goals.max()))
        fig.add_trace(go.Scatter(x=[0, mx], y=[0, mx], mode="lines", line=dict(dash="dash", color="grey"), showlegend=False))
        l.plotly_chart(fig, use_container_width=True)
        top = pd.concat([pt.nlargest(10, "goals_minus_xg"), pt.nsmallest(10, "goals_minus_xg")]).drop_duplicates("player_id")
        fig = px.bar(top.sort_values("goals_minus_xg"), x="goals_minus_xg", y="player", orientation="h", color="finishing_label",
                     title="Biggest over- / under-performers (Goals − xG)",
                     color_discrete_map={"Overperforming": "#2ca02c", "Underperforming": "#d62728", "In line": "#999"})
        r.plotly_chart(fig, use_container_width=True)
        st.caption("z-score = (G − xG)/√Σp(1−p). |z|<1.64 is consistent with luck. Shot quality (xG/shot) and finishing (G − xG) are separate columns; "
                   "see the README for the limits of inferring finishing skill without post-shot xG.")
        show = ["player", "team", "shots", "goals", "xg", "goals_minus_xg", "z_finishing", "finishing_skill_shrunk", "finishing_label", "xg_per_shot",
                "conversion", "minutes", "xg_per90", "goals_per90", "shots_per90"]
        st.dataframe(pt[show].round(3), use_container_width=True, height=380)
        st.subheader("Player comparison & shot maps")
        pick = st.multiselect("Select 1–3 players", sorted(pt["player"]), default=list(pt["player"].head(2)), max_selections=3)
        if pick:
            st.dataframe(pt[pt.player.isin(pick)][show].set_index("player").T.round(3), use_container_width=True)
            cols = st.columns(len(pick))
            for col, name in zip(cols, pick):
                col.plotly_chart(F.shot_map(fs[fs.player == name], title=name), use_container_width=True)

# ----------------------------------------------------------------------------- Teams
with tabs[2]:
    st.header("Team Analytics")
    tt = team_table(fs)
    l, r = st.columns(2)
    fig = px.bar(tt.sort_values("xg_for").tail(25), x="xg_for", y="team", orientation="h", title="Team xG ranking (for)")
    l.plotly_chart(fig, use_container_width=True)
    fig = px.scatter(tt, x="xg_for", y="goals_for", text="team", color="attacking_efficiency", color_continuous_scale="RdYlGn", color_continuous_midpoint=1,
                     title="Goals vs xG (colour = attacking efficiency G/xG)")
    mx = float(max(tt.xg_for.max(), tt.goals_for.max()))
    fig.add_trace(go.Scatter(x=[0, mx], y=[0, mx], mode="lines", line=dict(dash="dash"), showlegend=False))
    fig.update_traces(textposition="top center", selector=dict(mode="markers+text"))
    r.plotly_chart(fig, use_container_width=True)
    st.caption("xG against is only available for teams whose opponents' shots are within the current filter (it is exact when no team filter is set).")
    l, r = st.columns(2)
    fig = px.scatter(tt, x="xga_per_match", y="xg_per_match", text="team", title="Attack vs defence (per match)")
    fig.update_yaxes(title="xG for / match")
    fig.update_xaxes(title="xG against / match", autorange="reversed")
    l.plotly_chart(fig, use_container_width=True)
    dist = fs.assign(dist_band=pd.cut(fs["distance"], [0, 6, 12, 18, 25, 35, 100]))
    dd = dist.groupby([dist["team"], "dist_band"], observed=True).size().reset_index(name="shots")
    keep = tt.head(8)["team"]
    fig = px.bar(dd[dd.team.isin(keep)], x="team", y="shots", color=dd["dist_band"].astype(str), title="Shot-distance distribution (top-8 xG teams)")
    r.plotly_chart(fig, use_container_width=True)
    st.dataframe(tt[["team", "matches", "shots_for", "goals_for", "xg_for", "xg_against", "xg_diff", "xg_per_shot", "conversion",
                     "attacking_efficiency", "z_finishing", "xg_per_match", "xga_per_match"]].round(3), use_container_width=True, height=400)

# ----------------------------------------------------------------------------- Shot analysis
with tabs[3]:
    st.header("Shot Analysis")
    st.caption("Use the sidebar to filter by competition, season, team, shot type, body part and play pattern. Select players below.")
    sel_players = st.multiselect("Player(s)", players_opts, default=[])
    d = fs[fs["player"].isin(sel_players)] if sel_players else fs
    xr = st.slider("xG range", 0.0, 1.0, (0.0, 1.0), 0.01)
    d = d[(d.xg >= xr[0]) & (d.xg <= xr[1])]
    half = st.toggle("Attacking half only", value=True)
    c1, c2 = st.columns([3, 2])
    c1.plotly_chart(F.shot_map(d, half), use_container_width=True)
    c2.plotly_chart(F.xg_heatmap(d, half), use_container_width=True)
    st.write(f"**{len(d):,} shots · {int(d.is_goal.sum())} goals · {d.xg.sum():.1f} xG**")
    st.dataframe(d.sort_values("xg", ascending=False).head(200)[["match_date", "player", "team", "opponent", "minute", "shot_type", "body_part", "play_pattern",
                                                                   "distance", "angle", "xg", "is_goal"]].round(3), use_container_width=True, height=320)

# ----------------------------------------------------------------------------- Model analytics
with tabs[4]:
    st.header("Model Analytics")
    if "model_comparison" not in T:
        st.info("Run the training step to populate model analytics.")
    else:
        mc = T["model_comparison"]
        cols = ["model", "log_loss", "brier_score", "roc_auc", "pr_auc", "ece", "calib_slope", "calib_intercept", "xg_over_goals", "precision", "recall", "f1", "cv_select", "search", "seconds"]
        st.subheader("Model comparison (chronological hold-out test, calibrated)")
        st.dataframe(mc[[c for c in cols if c in mc.columns]].sort_values("log_loss").round(4), use_container_width=True)
        st.caption("Primary criterion: log loss and Brier score (proper scoring rules) plus calibration (ECE, slope≈1, intercept≈0). "
                   "Precision/recall use a threshold fitted on training out-of-fold predictions. "
                   "'StatsBomb xG (benchmark)' is the provider's xG; 'Constant' is the no-skill baseline.")
        if "model_cv_metrics" in T:
            cv = T["model_cv_metrics"]
            fig = px.bar(cv, x="model", y="log_loss", color="variant", barmode="group", title="Cross-validated log loss on TRAIN: raw vs calibrated")
            st.plotly_chart(fig, use_container_width=True)
        if "model_curves" in T:
            cur = T["model_curves"]
            a, b, c3 = st.columns(3)
            a.plotly_chart(F.curve_fig(cur, "roc"), use_container_width=True)
            b.plotly_chart(F.curve_fig(cur, "pr"), use_container_width=True)
            c3.plotly_chart(F.calibration_fig(cur), use_container_width=True)
        if "feature_importance" in T and len(T["feature_importance"]):
            fi = T["feature_importance"]
            meth = st.selectbox("Importance method", sorted(fi["method"].unique()), index=0)
            top = fi[fi.method == meth].nlargest(25, "importance").sort_values("importance")
            st.plotly_chart(px.bar(top, x="importance", y="feature", orientation="h", title=f"Feature importance — {meth}", height=600), use_container_width=True)
        if "model_pdp" in T:
            pdp = T["model_pdp"]
            f = st.selectbox("Partial dependence feature", sorted(pdp["feature"].unique()))
            g = pdp[pdp.feature == f]
            numeric_axis = pd.to_numeric(g["value"], errors="coerce").notna().all()
            fig = (px.line(g.assign(value=pd.to_numeric(g["value"])), x="value", y="avg_xg", markers=True) if numeric_axis
                   else px.bar(g, x="value", y="avg_xg"))
            fig.update_layout(title=f"Partial dependence: {f}")
            st.plotly_chart(fig, use_container_width=True)
        rd = latest_run_dir()
        if rd is not None:
            for img, cap in [("shap_summary.png", "SHAP summary (beeswarm)"), ("shap_bar.png", "SHAP mean |value|")]:
                p = rd / "explain" / img
                if p.exists():
                    st.image(str(p), caption=cap)
            if not (rd / "explain" / "shap_summary.png").exists():
                st.info("SHAP plots not found — install `shap` and run `python -m xg.cli explain`.")
        if "model_shot_explanations" in T:
            st.subheader("Individual shot explanations")
            st.dataframe(T["model_shot_explanations"].round(4), use_container_width=True)
