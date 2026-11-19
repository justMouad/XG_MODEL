import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from xg.analytics.football import add_finishing_stats, player_table, run_analytics, split_half_reliability, team_table
from xg.config import Config, load_config
from xg.data.synthetic import make_synthetic
from xg.features.engineering import build_features, feature_lists
from xg.features.preprocessing import Winsorizer, build_preprocessor, feature_names
from xg.inference.predictor import XGPredictor
from xg.models.registry import CalibratedXG, Calibrator, build_pipeline

CFG = Config({"features": {"use_freeze_frame": True, "use_player_history": False, "use_competition_context": False}})


def _data(n=120):
    raw = make_synthetic(n, seed=1)["shots_raw"]
    return build_features(raw, CFG)


def test_winsorizer_uses_training_quantiles_only():
    w = Winsorizer(0.0, 0.9).fit(np.arange(100, dtype=float).reshape(-1, 1))
    assert w.transform(np.array([[1000.0]]))[0, 0] == w.hi_[0] < 100


def test_preprocessor_handles_missing_and_unseen_categories():
    df = _data(40)
    num, cat = feature_lists(CFG)
    pre = build_preprocessor(num, cat).fit(df[num + cat])
    names = feature_names(pre, num)
    Z = pre.transform(df[num + cat].head(5).assign(body_part="Totally New"))
    assert Z.shape[1] == len(names) and not np.isnan(Z).any()


def test_calibrator_methods_are_monotone():
    rng = np.random.default_rng(0)
    p = rng.random(2000) * 0.4
    y = (rng.random(2000) < p * 0.5).astype(int)           # over-predicting model
    for method in ("sigmoid", "isotonic"):
        c = Calibrator(method).fit(p, y)
        out = c.transform(np.linspace(0.01, 0.4, 30))
        assert np.all(np.diff(out) >= -1e-9)
        assert c.transform(p).mean() < p.mean()             # calibration pulls probabilities down


def test_calibrated_pipeline_fits_predicts_in_unit_interval():
    df = _data(120)
    num, cat = feature_lists(CFG)
    pipe = build_pipeline("logistic_regression", {"C": 1.0}, num, cat, n_jobs=1)
    m = CalibratedXG(pipe, "sigmoid", n_splits=3).fit(df[num + cat], df["is_goal"].values, df["match_id"].values)
    p = m.predict_proba(df[num + cat])[:, 1]
    assert p.min() > 0 and p.max() < 1
    assert abs(p.sum() / df["is_goal"].sum() - 1) < 0.25   # roughly calibrated in-sample


def test_inference_predictor_roundtrip_and_validation():
    df = _data(120)
    num, cat = feature_lists(CFG)
    m = CalibratedXG(build_pipeline("random_forest", {"n_estimators": 50, "min_samples_leaf": 10}, num, cat, n_jobs=1), "sigmoid", 3).fit(
        df[num + cat], df["is_goal"].values, df["match_id"].values)
    pred = XGPredictor({"model": m, "numeric": num, "categorical": cat, "name": "random_forest", "feature_config": {"use_freeze_frame": True},
                        "reference": {**{c: float(df[c].median()) for c in num}, **{c: df[c].mode().iloc[0] for c in cat}}})
    near = pred.predict(pd.DataFrame({"x": [114.0], "y": [40.0]})).iloc[0]
    far = pred.predict(pd.DataFrame({"x": [70.0], "y": [10.0]})).iloc[0]
    assert 0 < far < near < 1                                     # sanity: closer & central shots are better chances
    try:
        pred.predict(pd.DataFrame({"x": [121.0], "y": [40.0]}))
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    try:
        pred.predict(pd.DataFrame({"x": [100.0]}))
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    ex = pred.explain(pd.DataFrame({"x": [105.0], "y": [38.0]}), 0, 5)
    assert len(ex) == 5 and {"raw_feature", "contribution"} <= set(ex.columns)


def test_finishing_stats_z_score():
    g = pd.DataFrame({"shots": [100], "goals": [20], "xg": [10.0], "xg_var": [9.0]})
    out = add_finishing_stats(g)
    assert abs(out.loc[0, "z_finishing"] - 10 / 3) < 1e-9 and out.loc[0, "finishing_label"] == "Overperforming"
    assert 1 < out.loc[0, "finishing_skill_shrunk"] < out.loc[0, "g_per_xg"]     # shrinkage towards 1


def test_team_and_player_tables_conserve_totals():
    df = _data(80)
    df["xg"] = 0.1
    t = team_table(df)
    assert abs(t["xg_for"].sum() - df["xg"].sum()) < 1e-6 and abs(t["xg_against"].sum() - df["xg"].sum()) < 1e-6
    p = player_table(df, None)
    assert p["shots"].sum() == len(df) and p["goals"].sum() == df["is_goal"].sum()
    out = run_analytics(df, make_synthetic(80, seed=1)["minutes"])
    assert {"players", "teams", "competitions", "shot_quality", "finishing_reliability"} <= set(out)
    assert out["players"]["xg_per90"].notna().any()


def test_split_half_reliability_detects_no_skill():
    rng = np.random.default_rng(0)
    n = 20000
    d = pd.DataFrame({"match_id": rng.integers(0, 600, n), "player_id": rng.integers(0, 60, n)})
    d["xg"] = rng.beta(2, 14, n)
    d["is_goal"] = (rng.random(n) < d["xg"]).astype(int)      # pure luck around xG
    r = split_half_reliability(d, 20)
    assert r["spearman_rho"] is not None and abs(r["spearman_rho"]) < 0.4
