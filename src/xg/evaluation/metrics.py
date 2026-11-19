"""Probabilistic evaluation of xG models. Calibration is first-class, not an afterthought."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score, log_loss, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)

EPS = 1e-6


def calibration_table(y, p, n_bins: int = 10, strategy: str = "quantile") -> pd.DataFrame:
    y, p = np.asarray(y, float), np.asarray(p, float)
    if strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    else:
        edges = np.linspace(0, 1, n_bins + 1)
    if len(edges) < 2:  # constant predictions -> a single bin
        edges = np.array([p.min() - 1e-9, p.max() + 1e-9])
    idx = np.clip(np.digitize(p, edges[1:-1], right=True), 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.sum() == 0:
            continue
        rows.append({"bin": b, "n": int(m.sum()), "mean_pred": p[m].mean(), "frac_pos": y[m].mean(), "goals": y[m].sum(), "xg": p[m].sum()})
    return pd.DataFrame(rows)


def expected_calibration_error(y, p, n_bins: int = 10) -> float:
    t = calibration_table(y, p, n_bins)
    return float((t["n"] * (t["mean_pred"] - t["frac_pos"]).abs()).sum() / t["n"].sum())


def calibration_slope_intercept(y, p) -> tuple[float, float]:
    """Logistic recalibration: logit(P(y=1)) = a + b * logit(p). Perfect calibration: a=0, b=1."""
    z = logit(np.clip(p, EPS, 1 - EPS)).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(z, y)
    return float(lr.intercept_[0]), float(lr.coef_[0, 0])


def best_f1_threshold(y, p) -> float:
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    return float(thr[int(np.nanargmax(f1))]) if len(thr) else 0.5


def evaluate(y, p, threshold: float | None = None) -> dict[str, float]:
    """Full metric set. ``threshold`` for precision/recall should come from TRAINING data (never from y_test)."""
    y = np.asarray(y).astype(int)
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    thr = 0.5 if threshold is None else threshold
    pred = (p >= thr).astype(int)
    a, b = calibration_slope_intercept(y, p) if 0 < y.sum() < len(y) else (np.nan, np.nan)
    base = y.mean()
    ll = log_loss(y, p, labels=[0, 1])
    ll0 = log_loss(y, np.full_like(p, base), labels=[0, 1])
    return {
        "n": int(len(y)), "goals": int(y.sum()), "xg_total": float(p.sum()), "xg_over_goals": float(p.sum() / max(y.sum(), 1)),
        "roc_auc": float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else np.nan,
        "pr_auc": float(average_precision_score(y, p)) if y.sum() else np.nan,
        "log_loss": float(ll), "brier_score": float(brier_score_loss(y, p)),
        "skill_vs_prevalence_logloss": float(1 - ll / ll0), "ece": expected_calibration_error(y, p),
        "calib_intercept": a, "calib_slope": b, "threshold": float(thr),
        "precision": float(precision_score(y, pred, zero_division=0)), "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
    }


def curves(y, p, n_bins: int = 10, max_points: int = 400) -> dict:
    """Curve data (JSON-serialisable) for ROC, PR and calibration; downsampled for the dashboard."""
    fpr, tpr, _ = roc_curve(y, p)
    prec, rec, _ = precision_recall_curve(y, p)

    def ds(a, b):
        if len(a) <= max_points:
            return a.tolist(), b.tolist()
        i = np.linspace(0, len(a) - 1, max_points).astype(int)
        return a[i].tolist(), b[i].tolist()

    f, t = ds(fpr, tpr)
    r, pr = ds(rec, prec)
    cal = calibration_table(y, p, n_bins)
    return {"roc": {"fpr": f, "tpr": t}, "pr": {"recall": r, "precision": pr},
            "calibration": {"mean_pred": cal["mean_pred"].tolist(), "frac_pos": cal["frac_pos"].tolist(), "n": cal["n"].tolist()}}
