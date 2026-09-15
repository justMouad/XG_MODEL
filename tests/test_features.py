import json
import math

import numpy as np
import pandas as pd

from xg.config import Config
from xg.features.engineering import (LEAKY_COLUMNS, add_player_history, build_features, feature_lists, freeze_frame_features,
                                     position_group, shot_angle, shot_distance)


def test_distance_and_angle_geometry():
    d = shot_distance(np.array([108.0]), np.array([40.0]))[0]
    a = shot_angle(np.array([108.0]), np.array([40.0]))[0]
    assert math.isclose(d, 12.0)
    assert math.isclose(a, 2 * math.atan(4 / 12), rel_tol=1e-9)
    # a shot from the byline well outside the posts sees almost no goal mouth; closer central shots see a larger angle
    assert shot_angle(np.array([119.9]), np.array([20.0]))[0] < 0.1
    assert shot_angle(np.array([110.0]), np.array([40.0]))[0] > shot_angle(np.array([100.0]), np.array([40.0]))[0]


def test_symmetry_in_y():
    a = shot_angle(np.array([105.0, 105.0]), np.array([30.0, 50.0]))
    assert math.isclose(a[0], a[1])


def _ff(players):
    return json.dumps(players)


def test_freeze_frame_features():
    ff = _ff([
        {"location": [119.0, 40.0], "position": {"name": "Goalkeeper"}, "teammate": False},
        {"location": [114.0, 40.0], "position": {"name": "Center Back"}, "teammate": False},   # inside the cone
        {"location": [110.0, 60.0], "position": {"name": "Right Back"}, "teammate": False},    # outside the cone
        {"location": [112.0, 41.0], "position": {"name": "Center Forward"}, "teammate": True},  # teammate in cone
    ])
    f = freeze_frame_features(108.0, 40.0, ff)
    assert f["ff_available"] == 1 and f["gk_available"] == 1
    assert f["n_defenders_in_cone"] == 1 and f["n_attackers_in_cone"] == 1
    assert f["gk_in_cone"] == 1 and math.isclose(f["gk_dist_goal"], 1.0)
    assert math.isclose(f["nearest_defender_dist"], 6.0)
    assert f["n_defenders_ahead"] == 2


def test_freeze_frame_missing_is_nan_not_zero():
    f = freeze_frame_features(100.0, 40.0, None)
    assert f["ff_available"] == 0 and math.isnan(f["n_defenders_in_cone"])


def test_position_group():
    assert position_group("Left Wing Back") == "Defender"
    assert position_group("Right Wing") == "Forward"
    assert position_group("Center Attacking Midfield") == "Midfielder"
    assert position_group(None) == "Unknown"


def test_no_leaky_column_in_feature_lists():
    cfg = Config({"features": {"use_freeze_frame": True, "use_player_history": True, "use_competition_context": True}})
    num, cat = feature_lists(cfg)
    assert not set(LEAKY_COLUMNS) & set(num + cat)
    assert "kp_goal_assist" not in num + cat


def test_build_features_tolerates_minimal_input():
    out = build_features(pd.DataFrame({"x": [100.0], "y": [35.0]}))
    assert {"distance", "angle", "position_group"} <= set(out.columns)
    assert math.isnan(out.loc[0, "minute_exact"])


def _hist_frame(goal_flip=False):
    rows = []
    for m, date in enumerate(["2020-01-01", "2020-01-08", "2020-01-15"]):
        for i in range(4):
            rows.append({"match_id": m, "match_date": date, "player_id": 7, "is_goal": int((i + m) % 2 == 0) if not (goal_flip and m == 1) else 1})
    return pd.DataFrame(rows)


def test_player_history_never_uses_own_or_future_match():
    base = add_player_history(_hist_frame())
    first = base[base.match_id == 0]
    assert (first["player_prior_shots"] == 0).all()              # nothing before the first match
    assert (base[base.match_id == 1]["player_prior_shots"] == 4).all()
    assert (base[base.match_id == 2]["player_prior_shots"] == 8).all()
    flipped = add_player_history(_hist_frame(goal_flip=True))
    # changing the outcomes of match 1 must NOT change match 0 or match 1 features, only match 2
    for m in (0, 1):
        a = base[base.match_id == m]["player_prior_conv_shrunk"].values
        b = flipped[flipped.match_id == m]["player_prior_conv_shrunk"].values
        assert np.allclose(a, b)
    assert not np.allclose(base[base.match_id == 2]["player_prior_conv_shrunk"].values, flipped[flipped.match_id == 2]["player_prior_conv_shrunk"].values)
