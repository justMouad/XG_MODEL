"""Static (matplotlib) evaluation plots saved to disk."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _save(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_roc(curve_data: dict, metrics: dict, path):
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, c in curve_data.items():
        ax.plot(c["roc"]["fpr"], c["roc"]["tpr"], label=f"{name} (AUC={metrics[name]['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curves (held-out test)")
    ax.legend(fontsize=7)
    _save(fig, path)


def plot_pr(curve_data: dict, metrics: dict, path):
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, c in curve_data.items():
        ax.plot(c["pr"]["recall"], c["pr"]["precision"], label=f"{name} (AP={metrics[name]['pr_auc']:.3f})")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall curves (held-out test)")
    ax.legend(fontsize=7)
    _save(fig, path)


def plot_calibration(curve_data: dict, path, title="Calibration (held-out test)"):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    for name, c in curve_data.items():
        ax.plot(c["calibration"]["mean_pred"], c["calibration"]["frac_pos"], marker="o", ms=3, label=name)
    ax.set(xlabel="Mean predicted xG", ylabel="Observed goal rate", title=title)
    ax.legend(fontsize=7)
    _save(fig, path)


def plot_model_comparison(df: pd.DataFrame, path):
    metrics = ["log_loss", "brier_score", "roc_auc", "pr_auc"]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    d = df.sort_values("log_loss")
    for ax, m in zip(axes, metrics):
        ax.barh(d["model"], d[m], color="#2b7bba")
        ax.set_title(m + (" (lower=better)" if m in ("log_loss", "brier_score") else " (higher=better)"), fontsize=9)
        lo, hi = d[m].min(), d[m].max()
        pad = (hi - lo) * 0.3 + 1e-4
        ax.set_xlim(lo - pad, hi + pad)
        ax.tick_params(labelsize=7)
    _save(fig, path)
