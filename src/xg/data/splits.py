"""Match-level and time-aware splitting. A match (and hence all of its shots) never straddles train/test or folds."""
from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit


def train_test_split_matches(df: pd.DataFrame, test_size: float = 0.2, strategy: str = "time", seed: int = 42):
    """Return (train_idx, test_idx) as positional index arrays.

    strategy="time":  the chronologically last ``test_size`` fraction of *matches* is the test set.
    strategy="group": random matches.
    """
    if strategy == "time":
        m = df[["match_id", "match_date"]].drop_duplicates().sort_values(["match_date", "match_id"])
        n_test = max(1, int(round(len(m) * test_size)))
        test_matches = set(m["match_id"].iloc[-n_test:])
        mask = df["match_id"].isin(test_matches).values
        return np.where(~mask)[0], np.where(mask)[0]
    if strategy == "group":
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        tr, te = next(gss.split(df, groups=df["match_id"]))
        return tr, te
    raise ValueError(f"Unknown split strategy: {strategy}")


def cv_splits(groups: pd.Series | np.ndarray, dates: pd.Series | None = None, n_splits: int = 5,
              strategy: str = "group_kfold") -> list[tuple[np.ndarray, np.ndarray]]:
    """Cross-validation folds grouped by match.

    group_kfold:   sklearn GroupKFold (deterministic, balanced by shots).
    blocked_time:  matches sorted chronologically and cut into ``n_splits`` contiguous blocks; each block is a
                   validation fold (train on the rest). Measures robustness to period shifts.
    """
    groups = np.asarray(groups)
    n_groups = len(np.unique(groups))
    n_splits = max(2, min(n_splits, n_groups))
    if strategy == "group_kfold":
        return list(GroupKFold(n_splits=n_splits).split(np.zeros(len(groups)), groups=groups))
    if strategy == "blocked_time":
        if dates is None:
            raise ValueError("blocked_time needs dates")
        m = (pd.DataFrame({"g": groups, "d": np.asarray(dates)}).drop_duplicates("g").sort_values(["d", "g"]))
        blocks = np.array_split(m["g"].values, n_splits)
        out = []
        for b in blocks:
            val = np.isin(groups, b)
            out.append((np.where(~val)[0], np.where(val)[0]))
        return out
    raise ValueError(f"Unknown cv strategy: {strategy}")
