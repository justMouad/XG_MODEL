"""Model interpretation: SHAP, feature importance, partial dependence and single-shot explanations.

SHAP is used when installed; otherwise model-agnostic fallbacks (grouped permutation importance, baseline
ablation) are used so the pipeline never hard-fails on a missing optional dependency.
SHAP values explain the *raw* model output; the calibration map is monotone so rankings/directions are unchanged.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from ..features.preprocessing import feature_names
from ..logging_utils import get_logger

log = get_logger(__name__)

TREE_MODELS = {"random_forest", "extra_trees", "xgboost", "lightgbm", "catboost"}


def _parts(bundle: dict):
    model = bundle["model"]
    pipe = model.pipeline_
    pre, est = pipe.named_steps["pre"], pipe.named_steps["model"]
    return model, pre, est, feature_names(pre, bundle["numeric"])


# --------------------------------------------------------------------------- importance
def builtin_importance(bundle: dict) -> pd.DataFrame:
    _, _, est, names = _parts(bundle)
    if hasattr(est, "feature_importances_"):
        imp = np.asarray(est.feature_importances_, float)
        kind = "impurity/gain"
    elif hasattr(est, "coef_"):
        imp = np.abs(np.ravel(est.coef_))
        kind = "|standardised coefficient|"
    else:
        return pd.DataFrame(columns=["feature", "importance", "kind"])
    n = min(len(names), len(imp))
    return pd.DataFrame({"feature": names[:n], "importance": imp[:n], "kind": kind}).sort_values("importance", ascending=False)


def permutation_importance(bundle: dict, X: pd.DataFrame, y: np.ndarray, n_repeats: int = 3, seed: int = 42) -> pd.DataFrame:
    """Increase in log loss when one *raw* feature (all its one-hot columns together) is shuffled. Held-out data only."""
    model = bundle["model"]
    cols = bundle["numeric"] + bundle["categorical"]
    rng = np.random.default_rng(seed)
    base = log_loss(y, np.clip(model.predict_proba(X[cols])[:, 1], 1e-6, 1 - 1e-6), labels=[0, 1])
    rows = []
    for c in cols:
        deltas = []
        for _ in range(n_repeats):
            Xp = X[cols].copy()
            Xp[c] = rng.permutation(Xp[c].values)
            deltas.append(log_loss(y, np.clip(model.predict_proba(Xp)[:, 1], 1e-6, 1 - 1e-6), labels=[0, 1]) - base)
        rows.append({"feature": c, "importance": float(np.mean(deltas)), "std": float(np.std(deltas))})
    return pd.DataFrame(rows).sort_values("importance", ascending=False)


# --------------------------------------------------------------------------- SHAP
def shap_available() -> bool:
    try:
        import shap  # noqa: F401
        return True
    except ImportError:
        return False


def compute_shap(bundle: dict, X: pd.DataFrame, n_sample: int = 2000, n_background: int = 200, seed: int = 42):
    """Return (shap_values[n, p], transformed_X[n, p], feature_names, sample_index) or None if shap is missing."""
    if not shap_available():
        log.warning("shap not installed - skipping SHAP (pip install shap). Permutation importance is still produced.")
        return None
    import shap
    _, pre, est, names = _parts(bundle)
    cols = bundle["numeric"] + bundle["categorical"]
    samp = X[cols].sample(min(n_sample, len(X)), random_state=seed)
    Z = pre.transform(samp)
    if bundle["name"] in TREE_MODELS:
        exp = shap.TreeExplainer(est)
        sv = exp.shap_values(Z)
        if isinstance(sv, list):          # sklearn forests: [class0, class1]
            sv = sv[1]
        sv = np.asarray(sv)
        if sv.ndim == 3:                  # (n, p, classes)
            sv = sv[:, :, 1]
    elif bundle["name"] == "logistic_regression":
        bg = pre.transform(X[cols].sample(min(n_background, len(X)), random_state=seed))
        sv = shap.LinearExplainer(est, bg).shap_values(Z)
    else:  # svm / neural net: model-agnostic and slow -> small sample
        Z = Z[:300]
        samp = samp.iloc[:300]
        bg = shap.kmeans(pre.transform(X[cols].sample(min(n_background, len(X)), random_state=seed)), 20)
        sv = shap.KernelExplainer(lambda a: est.predict_proba(a)[:, 1], bg).shap_values(Z, nsamples=100)
    return np.asarray(sv), Z, names, samp.index


def save_shap_outputs(bundle: dict, X: pd.DataFrame, out_dir: Path, n_sample: int, n_background: int) -> pd.DataFrame | None:
    res = compute_shap(bundle, X, n_sample, n_background)
    if res is None:
        return None
    import shap
    sv, Z, names, idx = res
    imp = pd.DataFrame({"feature": names, "mean_abs_shap": np.abs(sv).mean(0)}).sort_values("mean_abs_shap", ascending=False)
    imp.to_csv(out_dir / "shap_importance.csv", index=False)
    np.savez_compressed(out_dir / "shap_values_sample.npz", values=sv, data=Z, names=np.array(names), index=np.asarray(idx))
    plt.figure()
    shap.summary_plot(sv, Z, feature_names=names, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(out_dir / "shap_summary.png", dpi=140)
    plt.close("all")
    plt.figure()
    shap.summary_plot(sv, Z, feature_names=names, show=False, plot_type="bar", max_display=20)
    plt.tight_layout()
    plt.savefig(out_dir / "shap_bar.png", dpi=140)
    plt.close("all")
    return imp


# --------------------------------------------------------------------------- partial dependence
def partial_dependence(bundle: dict, X: pd.DataFrame, feature: str, grid_points: int = 20, sample: int = 3000, seed: int = 42) -> pd.DataFrame:
    """Average predicted xG as ``feature`` is forced to each grid value (numeric) or each category (categorical)."""
    cols = bundle["numeric"] + bundle["categorical"]
    Xs = X[cols].sample(min(sample, len(X)), random_state=seed).copy()
    model = bundle["model"]
    if feature in bundle["numeric"]:
        v = Xs[feature].dropna()
        grid = np.unique(np.quantile(v, np.linspace(0.02, 0.98, grid_points)))
    else:
        grid = Xs[feature].value_counts().index[:12].tolist()
    rows = []
    for g in grid:
        Xg = Xs.copy()
        Xg[feature] = g
        p = model.predict_proba(Xg)[:, 1]
        rows.append({"feature": feature, "value": g, "avg_xg": float(p.mean())})
    return pd.DataFrame(rows)


def save_pdp_plots(bundle: dict, X: pd.DataFrame, features: list[str], path: Path) -> pd.DataFrame:
    feats = [f for f in features if f in bundle["numeric"] + bundle["categorical"]]
    pdps = pd.concat([partial_dependence(bundle, X, f) for f in feats], ignore_index=True)
    n = len(feats)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.6 * rows), squeeze=False)
    for ax, f in zip(axes.ravel(), feats):
        d = pdps[pdps.feature == f]
        if f in bundle["numeric"]:
            ax.plot(d["value"].astype(float), d["avg_xg"], marker="o", ms=3)
        else:
            ax.bar(d["value"].astype(str), d["avg_xg"])
            ax.tick_params(axis="x", rotation=60, labelsize=7)
        ax.set(title=f"PDP: {f}", ylabel="avg xG")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return pdps


# --------------------------------------------------------------------------- single-shot explanation
def explain_shot(bundle: dict, shot: pd.DataFrame, top_k: int = 8) -> pd.DataFrame:
    """Per-feature contribution for ONE shot (one-row DataFrame in model-feature format).

    With SHAP: exact/approximate Shapley values on the raw model output (log-odds for boosters).
    Without SHAP: change in predicted xG when the feature is replaced by its reference (median/mode) value.
    """
    cols = bundle["numeric"] + bundle["categorical"]
    x = shot[cols].iloc[[0]]
    model, pre, est, names = _parts(bundle)
    if shap_available() and bundle["name"] in TREE_MODELS:
        import shap
        sv = shap.TreeExplainer(est).shap_values(pre.transform(x))
        sv = sv[1] if isinstance(sv, list) else np.asarray(sv)
        if sv.ndim == 3:
            sv = sv[:, :, 1]
        out = pd.DataFrame({"feature": names, "contribution": sv[0]})
        out["method"] = "SHAP (raw-model output)"
        # roll one-hot columns back to their raw feature for readability
        out["raw_feature"] = out["feature"].map(lambda n: next((c for c in sorted(cols, key=len, reverse=True) if n == c or n.startswith(c + "_")), n))
        out = out.groupby("raw_feature", as_index=False).agg(contribution=("contribution", "sum"), method=("method", "first"))
    else:
        ref = bundle["reference"]
        p0 = float(model.predict_proba(x)[0, 1])
        rows = []
        for c in cols:
            xr = x.copy()
            xr[c] = ref[c]
            rows.append({"raw_feature": c, "contribution": p0 - float(model.predict_proba(xr)[0, 1]), "method": "ablation vs reference (probability scale)"})
        out = pd.DataFrame(rows)
    out["value"] = out["raw_feature"].map(lambda c: x.iloc[0][c] if c in x.columns else np.nan)
    out = out.reindex(out["contribution"].abs().sort_values(ascending=False).index).head(top_k)
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------- orchestration
def run_interpretation(bundle: dict, test_df: pd.DataFrame, out_dir: Path, cfg) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = bundle["numeric"] + bundle["categorical"]
    X, y = test_df[cols], test_df["is_goal"].values
    outputs: dict = {}
    b = builtin_importance(bundle)
    b.to_csv(out_dir / "feature_importance_builtin.csv", index=False)
    pi = permutation_importance(bundle, X, y)
    pi.to_csv(out_dir / "feature_importance_permutation.csv", index=False)
    outputs["permutation"] = pi
    sh = save_shap_outputs(bundle, X, out_dir, cfg.explain["shap_sample"], cfg.explain["shap_background"])
    outputs["shap"] = sh
    pdp = save_pdp_plots(bundle, X, cfg.explain["pdp_features"], out_dir / "pdp.png")
    pdp.to_csv(out_dir / "pdp.csv", index=False)
    # a few individual explanations: highest-xG goal, lowest-xG goal, a high-xG miss
    xg = bundle["model"].predict_proba(X)[:, 1]
    picks = {}
    goals = np.where(y == 1)[0]
    misses = np.where(y == 0)[0]
    if len(goals):
        picks["lowest_xg_goal"] = goals[np.argmin(xg[goals])]
        picks["highest_xg_goal"] = goals[np.argmax(xg[goals])]
    if len(misses):
        picks["highest_xg_miss"] = misses[np.argmax(xg[misses])]
    rows = []
    for tag, i in picks.items():
        e = explain_shot(bundle, test_df.iloc[[i]])
        e.insert(0, "case", tag)
        e.insert(1, "shot_id", test_df.iloc[i]["shot_id"])
        e.insert(2, "xg", xg[i])
        rows.append(e)
    if rows:
        pd.concat(rows).to_csv(out_dir / "individual_explanations.csv", index=False)
    return outputs
