"""Synthetic data in the *same schema* as ``ingestion.parse_match_events`` output.

ONLY for unit tests, CI and offline smoke tests of the pipeline. Any number computed on it is meaningless
as football evidence. Real results require the real StatsBomb Open Data (``python -m xg.cli ingest``).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..features.engineering import GOAL_X, GOAL_Y_MID, shot_angle, shot_distance

BODY = ["Right Foot", "Left Foot", "Head", "Other"]
TECH = ["Normal", "Volley", "Half Volley", "Lob", "Backheel", "Overhead Kick"]
PATTERNS = ["Regular Play", "From Corner", "From Free Kick", "From Throw In", "From Counter", "From Keeper", "From Goal Kick"]
POSITIONS = ["Center Forward", "Right Wing", "Left Wing", "Center Attacking Midfield", "Center Back", "Right Back", "Left Defensive Midfield"]
PREV = ["Pass", "Carry", "Ball Receipt*", "Dribble", "Clearance", "Duel"]


def make_synthetic(n_matches: int = 120, seed: int = 0, shots_per_match: int = 24):
    rng = np.random.default_rng(seed)
    comps = [(43, "FIFA World Cup", 3, "2018"), (11, "La Liga", 4, "2018/2019"), (2, "Premier League", 44, "2003/2004")]
    teams = [f"Team {chr(65 + i)}" for i in range(16)]
    n_players = 220
    skill = rng.normal(0.0, 0.25, n_players)  # latent finishing skill (log-odds)
    player_team = rng.integers(0, len(teams), n_players)
    player_pos = rng.choice(POSITIONS, n_players, p=[.25, .15, .12, .15, .1, .1, .13])

    base_date = pd.Timestamp("2018-06-01")
    rows, matches, minutes = [], [], []
    for mi in range(n_matches):
        cid, cname, sid, sname = comps[mi % len(comps)]
        h, a = rng.choice(len(teams), 2, replace=False)
        date = (base_date + pd.Timedelta(days=int(mi * 2.5))).strftime("%Y-%m-%d")
        matches.append({"match_id": 10_000 + mi, "match_date": date, "kick_off": "20:00:00.000", "competition_id": cid,
                        "competition_name": cname, "season_id": sid, "season_name": sname, "home_team": teams[h],
                        "away_team": teams[a], "home_score": 0, "away_score": 0, "stage": "Regular Season", "match_end_minute": 95.0})
        score = {h: 0, a: 0}
        n = rng.poisson(shots_per_match)
        for si in range(n):
            t = h if rng.random() < 0.5 else a
            opp = a if t == h else h
            pool = np.where(player_team == t)[0]
            if len(pool) == 0:
                pool = np.arange(n_players)
            p = int(rng.choice(pool))
            x = float(np.clip(120 - rng.gamma(2.6, 6.0), 60, 119.9))
            y = float(np.clip(rng.normal(40, 11 + (120 - x) * 0.25), 0.5, 79.5))
            body = rng.choice(BODY, p=[.5, .3, .17, .03])
            tech = "Normal" if rng.random() < .85 else rng.choice(TECH[1:])
            pat = rng.choice(PATTERNS, p=[.6, .1, .07, .04, .08, .06, .05])
            stype = "Open Play" if pat not in ("From Free Kick",) else rng.choice(["Free Kick", "Open Play"])
            pressure = int(rng.random() < 0.35)
            first = int(rng.random() < 0.3)
            minute = int(rng.integers(0, 95))
            n_def = int(rng.integers(0, 4))
            gk_x = 120 - rng.uniform(0.5, 6)
            gk_y = 40 + rng.normal(0, 1.5)
            ff = [{"location": [gk_x, gk_y], "player": {"id": 1, "name": "GK"}, "position": {"name": "Goalkeeper"}, "teammate": False}]
            for _ in range(n_def):
                ff.append({"location": [float(np.clip(x + rng.uniform(-2, 8), 40, 119)), float(40 + rng.normal(0, 6))],
                           "player": {"id": 2, "name": "D"}, "position": {"name": "Center Back"}, "teammate": False})
            ff.append({"location": [x - rng.uniform(0, 5), y + rng.normal(0, 5)], "player": {"id": 3, "name": "A"},
                       "position": {"name": "Center Forward"}, "teammate": True})
            d = float(shot_distance(np.array([x]), np.array([y]))[0])
            ang = float(shot_angle(np.array([x]), np.array([y]))[0])
            logit = (-1.7 - 0.09 * d + 2.2 * ang - 0.9 * (body == "Head") - 0.45 * pressure + 0.25 * first
                     - 0.28 * n_def - 1.2 * (stype == "Free Kick") + skill[p] + 0.3 * (pat == "From Counter"))
            prob = 1 / (1 + np.exp(-logit))
            goal = int(rng.random() < prob)
            score[t] += goal
            kp = rng.random() < 0.75
            rows.append({
                "shot_id": f"s{mi}_{si}", "match_id": 10_000 + mi, "match_date": date, "competition_id": cid, "competition_name": cname,
                "season_id": sid, "season_name": sname, "team": teams[t], "opponent": teams[opp], "is_home": int(t == h),
                "player_id": 1000 + p, "player": f"Player {p}", "position": player_pos[p], "period": 1 if minute < 45 else 2,
                "minute": minute, "second": int(rng.integers(0, 60)), "possession": si, "play_pattern": pat, "x": x, "y": y,
                "under_pressure": pressure, "shot_type": stype, "body_part": body, "technique": tech, "first_time": first,
                "aerial_won": int(body == "Head" and rng.random() < .3), "follows_dribble": int(rng.random() < .05),
                "one_on_one": int(rng.random() < .04), "open_goal": int(rng.random() < .01), "deflected": int(rng.random() < .05),
                "redirect": 0, "outcome": "Goal" if goal else rng.choice(["Saved", "Off T", "Blocked", "Wayward", "Post"]),
                "is_goal": goal, "statsbomb_xg": float(np.clip(prob * np.exp(rng.normal(0, .15)), 0.005, 0.95)),
                "end_x": 120.0, "end_y": float(40 + rng.normal(0, 3)), "end_z": 1.0,
                "freeze_frame_json": json.dumps(ff) if rng.random() < 0.9 else None,
                "prev_event_type": rng.choice(PREV), "prev_event_gap": float(rng.exponential(2.0)),
                "carry_length": float(rng.exponential(4)) if rng.random() < .3 else None,
                "poss_events_before": int(rng.integers(0, 15)), "poss_passes_before": int(rng.integers(0, 8)),
                "poss_duration": float(rng.exponential(8)), "score_diff": score[t] - goal - score[opp],
                "kp_exists": int(kp), "kp_height": rng.choice(["Ground Pass", "High Pass", "Low Pass"]) if kp else None,
                "kp_type": rng.choice(["Corner", "Free Kick", "Recovery"]) if kp and rng.random() < .3 else None,
                "kp_technique": "Through Ball" if kp and rng.random() < .1 else None, "kp_body_part": "Right Foot" if kp else None,
                "kp_length": float(rng.gamma(3, 5)) if kp else None, "kp_angle": float(rng.uniform(-3, 3)) if kp else None,
                "kp_cross": int(kp and rng.random() < .2), "kp_cut_back": int(kp and rng.random() < .05),
                "kp_switch": 0, "kp_through_ball": int(kp and rng.random() < .1),
                "kp_start_x": float(np.clip(x - rng.uniform(0, 25), 20, 120)) if kp else None,
                "kp_start_y": float(rng.uniform(0, 80)) if kp else None,
                "kp_end_x": x if kp else None, "kp_end_y": y if kp else None,
            })
        matches[-1]["home_score"], matches[-1]["away_score"] = score[h], score[a]
        for t in (h, a):
            for p in np.where(player_team == t)[0][:14]:
                minutes.append({"match_id": 10_000 + mi, "team": teams[t], "player_id": 1000 + int(p), "player": f"Player {p}",
                                "minutes": float(rng.choice([90.0, 90.0, 75.0, 45.0, 20.0]))})
    return {"shots_raw": pd.DataFrame(rows), "matches": pd.DataFrame(matches), "minutes": pd.DataFrame(minutes)}
