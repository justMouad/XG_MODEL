"""Leakage-safe preprocessing: everything that learns from data lives inside an sklearn Pipeline so it is
re-fitted on the training portion of every CV fold."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip numeric columns to quantiles learned on the training data (outlier control)."""

    def __init__(self, lower: float = 0.001, upper: float = 0.999):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lo_ = np.nanquantile(X, self.lower, axis=0)
        self.hi_ = np.nanquantile(X, self.upper, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features)


def build_preprocessor(numeric: list[str], categorical: list[str], scale: bool = False,
                       winsor: tuple[float, float] = (0.001, 0.999)) -> ColumnTransformer:
    """Numeric: median-impute (+ missing indicators) -> winsorise -> (scale). Categorical: one-hot with rare-level pooling."""
    num_steps = [("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),]
    # the imputer appends indicator columns after the numeric ones; winsorise/scale operate on all (indicators are 0/1)
    num_steps.append(("winsor", Winsorizer(*winsor)))
    if scale:
        num_steps.append(("scale", StandardScaler()))
    cat = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.005, sparse_output=False)),
    ])
    return ColumnTransformer([("num", Pipeline(num_steps), numeric), ("cat", cat, categorical)],
                             remainder="drop", verbose_feature_names_out=False)


def feature_names(pre: ColumnTransformer, numeric: list[str]) -> list[str]:
    """Readable output feature names (handles the imputer's missing-indicator columns)."""
    num_pipe = pre.named_transformers_["num"]
    imp = num_pipe.named_steps["impute"]
    names = list(numeric)
    if imp.indicator_ is not None:
        names += [f"{numeric[i]}__missing" for i in imp.indicator_.features_]
    cat_pipe = pre.named_transformers_["cat"]
    cat_cols = pre.transformers_[1][2]
    names += list(cat_pipe.named_steps["onehot"].get_feature_names_out(cat_cols))
    return names
