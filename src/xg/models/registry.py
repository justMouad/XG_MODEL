"""Model zoo: sklearn pipelines (preprocessing + estimator) for every algorithm, plus the calibration wrapper.

Boosting libraries are optional imports: a missing library is logged and the model is skipped.
"""
from __future__ import annotations

import importlib
from typing import Any, Callable

import numpy as np
from scipy.special import expit, logit
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from ..features.preprocessing import build_preprocessor
from ..logging_utils import get_logger

log = get_logger(__name__)

SCALED_MODELS = {"logistic_regression", "svm", "neural_network"}
ALL_MODELS = ["logistic_regression", "random_forest", "extra_trees", "xgboost", "lightgbm", "catboost", "svm", "neural_network"]
_LIB = {"xgboost": "xgboost", "lightgbm": "lightgbm", "catboost": "catboost"}


def available(name: str) -> bool:
    if name in _LIB:
        try:
            importlib.import_module(_LIB[name])
        except ImportError:
            return False
    return name in ALL_MODELS


class SVCProba(BaseEstimator, ClassifierMixin):
    """RBF-SVM exposing ``predict_proba`` as a sigmoid of the margin (Platt-style; the downstream calibrator
    learns the proper slope/intercept). Trains on a random subsample capped at ``max_train_samples`` because
    kernel SVMs scale ~O(n^2)."""

    def __init__(self, C: float = 1.0, gamma: str | float = "scale", max_train_samples: int = 20000,
                 class_weight=None, random_state: int = 42):
        self.C = C
        self.gamma = gamma
        self.max_train_samples = max_train_samples
        self.class_weight = class_weight
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        if len(y) > self.max_train_samples:
            idx = np.random.default_rng(self.random_state).choice(len(y), self.max_train_samples, replace=False)
            X, y = X[idx], y[idx]
        self.svc_ = SVC(C=self.C, gamma=self.gamma, kernel="rbf", class_weight=self.class_weight).fit(X, y)
        self.classes_ = self.svc_.classes_
        return self

    def decision_function(self, X):
        return self.svc_.decision_function(np.asarray(X))

    def predict_proba(self, X):
        p = expit(2.0 * self.decision_function(X))
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def make_estimator(name: str, params: dict[str, Any], seed: int, n_jobs: int, class_weight=None, svm_cap: int = 20000):
    p = dict(params)
    if name == "logistic_regression":
        return LogisticRegression(max_iter=2000, class_weight=class_weight, **p)
    if name == "random_forest":
        return RandomForestClassifier(random_state=seed, n_jobs=n_jobs, class_weight=class_weight, **p)
    if name == "extra_trees":
        return ExtraTreesClassifier(random_state=seed, n_jobs=n_jobs, class_weight=class_weight, **p)
    if name == "svm":
        return SVCProba(random_state=seed, max_train_samples=svm_cap, class_weight=class_weight, **p)
    if name == "neural_network":
        hidden = p.pop("hidden", "64")
        layers = tuple(int(v) for v in str(hidden).split("-"))
        return MLPClassifier(hidden_layer_sizes=layers, max_iter=300, early_stopping=True, n_iter_no_change=15,
                             random_state=seed, **p)
    if name == "xgboost":
        from xgboost import XGBClassifier
        spw = 1.0
        return XGBClassifier(objective="binary:logistic", eval_metric="logloss", tree_method="hist", random_state=seed,
                             n_jobs=n_jobs, verbosity=0, scale_pos_weight=spw, **p)
    if name == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(objective="binary", random_state=seed, n_jobs=n_jobs, verbose=-1, **p)
    if name == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(loss_function="Logloss", random_seed=seed, verbose=0, allow_writing_files=False,
                                  thread_count=n_jobs if n_jobs > 0 else None, **p)
    raise ValueError(f"Unknown model {name}")


def default_params(name: str) -> dict[str, Any]:
    return {
        "logistic_regression": {"C": 1.0},
        "random_forest": {"n_estimators": 300, "min_samples_leaf": 20, "max_features": "sqrt"},
        "extra_trees": {"n_estimators": 300, "min_samples_leaf": 20, "max_features": "sqrt"},
        "xgboost": {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 4, "subsample": 0.8, "colsample_bytree": 0.8},
        "lightgbm": {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 15, "min_child_samples": 30, "subsample": 0.8},
        "catboost": {"iterations": 400, "learning_rate": 0.05, "depth": 5},
        "svm": {"C": 1.0, "gamma": "scale"},
        "neural_network": {"hidden": "64-32", "alpha": 1e-3, "learning_rate_init": 1e-3},
    }[name]


def build_pipeline(name: str, params: dict[str, Any], numeric: list[str], categorical: list[str], seed: int = 42,
                   n_jobs: int = -1, class_weight=None, svm_cap: int = 20000, winsor=(0.001, 0.999)) -> Pipeline:
    pre = build_preprocessor(numeric, categorical, scale=name in SCALED_MODELS, winsor=winsor)
    est = make_estimator(name, params, seed, n_jobs, class_weight, svm_cap)
    return Pipeline([("pre", pre), ("model", est)])


# --------------------------------------------------------------------------- calibration
class Calibrator:
    """Maps raw probabilities to calibrated ones. ``sigmoid`` = Platt scaling on the logit of the raw
    probability; ``isotonic`` = monotone non-parametric; ``none`` = identity."""

    def __init__(self, method: str = "sigmoid"):
        self.method = method

    def fit(self, p: np.ndarray, y: np.ndarray) -> "Calibrator":
        p = np.clip(p, 1e-6, 1 - 1e-6)
        if self.method == "sigmoid":
            self.m_ = LogisticRegression(C=1e6, max_iter=1000).fit(logit(p).reshape(-1, 1), y)
        elif self.method == "isotonic":
            self.m_ = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4).fit(p, y)
        elif self.method == "none":
            self.m_ = None
        else:
            raise ValueError(self.method)
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
        if self.method == "sigmoid":
            return self.m_.predict_proba(logit(p).reshape(-1, 1))[:, 1]
        if self.method == "isotonic":
            return self.m_.predict(p)
        return p


class CalibratedXG:
    """Fitted pipeline + probability calibrator. ``predict_proba`` returns calibrated P(goal).

    ``fit`` learns the calibrator on *cross-fitted* (out-of-fold, grouped by match) predictions so the
    calibration map is never estimated on predictions the base model made for its own training rows.
    """

    def __init__(self, pipeline: Pipeline, method: str = "sigmoid", n_splits: int = 5):
        self.pipeline = pipeline
        self.method = method
        self.n_splits = n_splits

    def fit(self, X, y, groups=None):
        from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
        y = np.asarray(y)
        if self.method != "none":
            if groups is not None:
                cv = list(GroupKFold(n_splits=min(self.n_splits, len(np.unique(groups)))).split(X, y, groups))
            else:
                cv = StratifiedKFold(self.n_splits, shuffle=True, random_state=0)
            oof = cross_val_predict(clone(self.pipeline), X, y, cv=cv, method="predict_proba")[:, 1]
            self.calibrator_ = Calibrator(self.method).fit(oof, y)
        else:
            self.calibrator_ = Calibrator("none")
        self.pipeline_ = clone(self.pipeline).fit(X, y)
        self.classes_ = np.array([0, 1])
        return self

    def predict_raw(self, X) -> np.ndarray:
        return self.pipeline_.predict_proba(X)[:, 1]

    def predict_proba(self, X) -> np.ndarray:
        p = self.calibrator_.transform(self.predict_raw(X))
        return np.column_stack([1 - p, p])

    def predict(self, X, threshold: float = 0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)
