"""Plotly figure builders for the dashboard (kept free of Streamlit so they can be reused/tested)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

LINE = dict(color="#8a8f98", width=1)


def pitch_shapes(half: bool = True) -> list[dict]:
    """StatsBomb 120x80 pitch, attacking left->right, y axis reversed by the figure layout."""
    x0 = 60 if half else 0
    s = [dict(type="rect", x0=x0, y0=0, x1=120, y1=80, line=LINE),
         dict(type="rect", x0=102, y0=18, x1=120, y1=62, line=LINE), dict(type="rect", x0=114, y0=30, x1=120, y1=50, line=LINE),
         dict(type="line", x0=120, y0=36, x1=120, y1=44, line=dict(color="#d62728", width=5)),
         dict(type="circle", x0=60 - 10, y0=30, x1=60 + 10, y1=50, line=LINE)]
    if not half:
        s += [dict(type="line", x0=60, y0=0, x1=60, y1=80, line=LINE), dict(type="rect", x0=0, y0=18, x1=18, y1=62, line=LINE),
              dict(type="rect", x0=0, y0=30, x1=6, y1=50, line=LINE)]
    return s


def _pitch_layout(fig: go.Figure, half: bool, title: str, height: int = 560) -> go.Figure:
    fig.update_layout(shapes=pitch_shapes(half), title=title, height=height, plot_bgcolor="#f4f7f4", margin=dict(l=10, r=10, t=50, b=10),
                      xaxis=dict(range=[58 if half else -2, 122], showgrid=False, zeroline=False, visible=False),
                      yaxis=dict(range=[82, -2], showgrid=False, zeroline=False, visible=False, scaleanchor="x", scaleratio=1),
                      legend=dict(orientation="h", y=-0.02))
    return fig


def shot_map(df: pd.DataFrame, half: bool = True, title: str = "Shot map (marker size = xG)", max_points: int = 4000) -> go.Figure:
    d = df.sample(max_points, random_state=0) if len(df) > max_points else df
    fig = go.Figure()
    for goal, name, color, sym in [(0, "No goal", "#6c8ebf", "circle"), (1, "Goal", "#d62728", "star")]:
        s = d[d["is_goal"] == goal]
        fig.add_trace(go.Scatter(
            x=s["x"], y=s["y"], mode="markers", name=name,
            marker=dict(size=np.clip(s["xg"] * 60 + 5, 5, 40), color=color, symbol=sym, opacity=0.7 if goal == 0 else 0.95, line=dict(width=0.5, color="white")),
            customdata=np.stack([s["player"].astype(str), s["team"].astype(str), s["minute"].fillna(-1), s["xg"], s["body_part"].astype(str),
                                 s["play_pattern"].astype(str), s["match_date"].astype(str)], axis=-1),
            hovertemplate="<b>%{customdata[0]}</b> (%{customdata[1]})<br>min %{customdata[2]} | %{customdata[6]}<br>xG %{customdata[3]:.3f}"
                          "<br>%{customdata[4]} | %{customdata[5]}<extra></extra>"))
    return _pitch_layout(fig, half, title)


def xg_heatmap(df: pd.DataFrame, half: bool = True, title: str = "Average xG by location") -> go.Figure:
    xr, yr = (60, 120) if half else (0, 120)
    fig = go.Figure(go.Histogram2d(x=df["x"], y=df["y"], z=df["xg"], histfunc="avg", xbins=dict(start=xr, end=yr, size=4),
                                   ybins=dict(start=0, end=80, size=4), colorscale="YlOrRd", colorbar=dict(title="avg xG"), zauto=True))
    return _pitch_layout(fig, half, title)


def calibration_fig(curves: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="black"), name="perfect"))
    for m, g in curves[curves["curve"] == "calibration"].groupby("model"):
        fig.add_trace(go.Scatter(x=g["x"], y=g["y"], mode="lines+markers", name=m))
    fig.update_layout(title="Calibration (held-out test)", xaxis_title="Mean predicted xG", yaxis_title="Observed goal rate", height=450)
    return fig


def curve_fig(curves: pd.DataFrame, kind: str) -> go.Figure:
    fig = go.Figure()
    if kind == "roc":
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="black"), name="chance"))
    for m, g in curves[curves["curve"] == kind].groupby("model"):
        fig.add_trace(go.Scatter(x=g["x"], y=g["y"], mode="lines", name=m))
    fig.update_layout(title="ROC curves" if kind == "roc" else "Precision-Recall curves", height=450,
                      xaxis_title="False positive rate" if kind == "roc" else "Recall", yaxis_title="True positive rate" if kind == "roc" else "Precision")
    return fig
