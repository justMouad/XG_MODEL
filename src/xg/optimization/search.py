"""Hyperparameter optimisation: Optuna (TPE) when installed, seeded random search otherwise.

Objective: mean grouped-CV log loss of the *raw* (uncalibrated) pipeline on the TRAINING set only.
The held-out test set is never touched here.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import log_loss

from ..logging_utils import get_logger
from ..models.registry import build_pipeline, default_params

log = get_logger(__name__)


class RandomTrial:
    """Minimal Optuna-``Trial`` look-alike used by the random-search fallback."""

    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.params: dict[str, Any] = {}

    def suggest_float(self, name, low, high, log=False, **_):
        v = float(np.exp(self.rng.uniform(np.log(low), np.log(high)))) if log else float(self.rng.uniform(low, high))
        self.params[name] = v
        return v

    def suggest_int(self, name, low, high, log=False, **_):
        v = int(round(np.exp(self.rng.uniform(np.log(low), np.log(high))))) if log else int(self.rng.integers(low, high + 1))
        self.params[name] = v
        return v

    def suggest_categorical(self, name, choices):
        v = choices[int(self.rng.integers(len(choices)))]
        self.params[name] = v
        return v


def search_space(name: str, t) -> dict[str, Any]:
    if name == "logistic_regression":
        return {"C": t.suggest_float("C", 1e-3, 1e2, log=True)}
    if name in ("random_forest", "extra_trees"):
        return {"n_estimators": t.suggest_int("n_estimators", 150, 500), "max_depth": t.suggest_categorical("max_depth", [None, 6, 10, 16]),
                "min_samples_leaf": t.suggest_int("min_samples_leaf", 5, 80, log=True),
                "max_features": t.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5])}
    if name == "xgboost":
        return {"n_estimators": t.suggest_int("n_estimators", 100, 600), "learning_rate": t.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "max_depth": t.suggest_int("max_depth", 2, 7), "min_child_weight": t.suggest_float("min_child_weight", 1, 30, log=True),
                "subsample": t.suggest_float("subsample", 0.5, 1.0), "colsample_bytree": t.suggest_float("colsample_bytree", 0.4, 1.0),
                "reg_lambda": t.suggest_float("reg_lambda", 1e-2, 50, log=True), "gamma": t.suggest_float("gamma", 0.0, 5.0)}
    if name == "lightgbm":
        return {"n_estimators": t.suggest_int("n_estimators", 100, 600), "learning_rate": t.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "num_leaves": t.suggest_int("num_leaves", 4, 64, log=True), "min_child_samples": t.suggest_int("min_child_samples", 10, 150, log=True),
                "subsample": t.suggest_float("subsample", 0.5, 1.0), "subsample_freq": 1,
                "colsample_bytree": t.suggest_float("colsample_bytree", 0.4, 1.0), "reg_lambda": t.suggest_float("reg_lambda", 1e-2, 50, log=True)}
    if name == "catboost":
        return {"iterations": t.suggest_int("iterations", 150, 700), "learning_rate": t.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "depth": t.suggest_int("depth", 3, 8), "l2_leaf_reg": t.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True)}
    if name == "svm":
        return {"C": t.suggest_float("C", 0.1, 10, log=True), "gamma": t.suggest_float("gamma", 1e-3, 1e-1, log=True)}
    if name == "neural_network":
        return {"hidden": t.suggest_categorical("hidden", ["32", "64-32", "128-64"]), "alpha": t.suggest_float("alpha", 1e-5, 1e-1, log=True),
                "learning_rate_init": t.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True)}
    raise ValueError(name)


def _cv_logloss(pipe_factory, X, y, folds) -> float:
    losses = []
    for tr, va in folds:
        m = pipe_factory().fit(X.iloc[tr], y[tr])
        p = np.clip(m.predict_proba(X.iloc[va])[:, 1], 1e-6, 1 - 1e-6)
        losses.append(log_loss(y[va], p, labels=[0, 1]))
    return float(np.mean(losses))


def optimise(name: str, X: pd.DataFrame, y: np.ndarray, folds, numeric, categorical, cfg) -> tuple[dict[str, Any], pd.DataFrame, str]:
    """Return (best_params, trials_dataframe, backend)."""
    seed = cfg.project["seed"]
    tcfg = cfg.training
    opt = cfg.optimization
    n_trials = int(opt.get("per_model_trials", {}).get(name, opt["n_trials"]))
    timeout = float(opt.get("timeout_per_model", 900))

    def factory_for(params):
        return lambda: build_pipeline(name, params, numeric, categorical, seed, tcfg["n_jobs"], tcfg.get("class_weight"),
                                      tcfg.get("svm_max_train_samples", 20000), tuple(cfg.features.get("winsor_quantiles", [0.001, 0.999])))

    if not opt.get("enabled", True) or n_trials <= 0:
        return default_params(name), pd.DataFrame(), "defaults"

    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            return _cv_logloss(factory_for(search_space(name, trial)), X, y, folds)

        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed),
                                    study_name=f"xg_{name}")
        study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False, catch=(ValueError,))
        trials = study.trials_dataframe(attrs=("number", "value", "params", "duration"))
        best = search_space(name, optuna.trial.FixedTrial(study.best_params))
        return best, trials, "optuna"
    except ImportError:
        log.warning("Optuna not installed - using seeded random search for %s", name)

    rng = np.random.default_rng(seed)
    t0, rows, best_v, best_p = time.time(), [], np.inf, None
    for i in range(n_trials):
        trial = RandomTrial(rng)
        params = search_space(name, trial)
        try:
            v = _cv_logloss(factory_for(params), X, y, folds)
        except ValueError as e:
            log.warning("trial %s failed: %s", i, e)
            continue
        rows.append({"number": i, "value": v, **{f"params_{k}": val for k, val in trial.params.items()}})
        if v < best_v:
            best_v, best_p = v, params
        if time.time() - t0 > timeout:
            break
    return best_p or default_params(name), pd.DataFrame(rows), "random_search"
