import numpy as np
import pandas as pd

from xg.data.splits import cv_splits, train_test_split_matches
from xg.evaluation.metrics import best_f1_threshold, calibration_slope_intercept, calibration_table, evaluate


def _df(n_matches=50, per=10, seed=0):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({"match_id": np.repeat(np.arange(n_matches), per), "match_date": np.repeat(pd.date_range("2020-01-01", periods=n_matches).astype(str), per)})
    d["is_goal"] = rng.integers(0, 2, len(d))
    return d


def test_time_split_is_chronological_and_match_disjoint():
    d = _df()
    tr, te = train_test_split_matches(d, 0.2, "time")
    assert set(d.match_id.iloc[tr]).isdisjoint(d.match_id.iloc[te])
    assert d.match_date.iloc[tr].max() < d.match_date.iloc[te].min()
    assert abs(d.match_id.iloc[te].nunique() - 10) <= 1


def test_group_split_match_disjoint():
    d = _df()
    tr, te = train_test_split_matches(d, 0.2, "group", seed=1)
    assert set(d.match_id.iloc[tr]).isdisjoint(d.match_id.iloc[te])


def test_cv_folds_never_split_a_match():
    d = _df()
    for strat in ("group_kfold", "blocked_time"):
        folds = cv_splits(d.match_id.values, d.match_date.values, 5, strat)
        assert len(folds) == 5
        for tr, va in folds:
            assert set(d.match_id.iloc[tr]).isdisjoint(d.match_id.iloc[va])
        assert sorted(np.concatenate([va for _, va in folds])) == list(range(len(d)))   # folds partition the data


def test_metrics_on_well_calibrated_probabilities():
    rng = np.random.default_rng(0)
    p = rng.beta(1, 9, 20000)
    y = (rng.random(20000) < p).astype(int)
    m = evaluate(y, p)
    assert m["roc_auc"] > 0.7 and abs(m["xg_over_goals"] - 1) < 0.05 and m["ece"] < 0.01
    a, b = calibration_slope_intercept(y, p)
    assert abs(b - 1) < 0.1 and abs(a) < 0.1
    # overconfident predictions must have slope < 1
    p_bad = np.clip(p * 3, 0, 0.99)
    assert calibration_slope_intercept(y, p_bad)[1] < b or evaluate(y, p_bad)["log_loss"] > m["log_loss"]


def test_constant_predictions_do_not_crash():
    y = np.array([0, 0, 1, 0, 1, 0, 0, 0, 0, 0])
    t = calibration_table(y, np.full(10, 0.2))
    assert len(t) == 1
    assert 0 <= best_f1_threshold(y, np.linspace(0.01, 0.9, 10)) <= 1
    assert evaluate(y, np.full(10, 0.2))["brier_score"] > 0
