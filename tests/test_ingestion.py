from xg.data.ingestion import _clock_minutes, parse_match_events, parse_minutes

MATCH = {"match_id": 1, "match_date": "2020-01-01", "home_team": {"home_team_name": "A"}, "away_team": {"away_team_name": "B"},
         "competition": {"competition_id": 1, "competition_name": "Liga"}, "season": {"season_id": 1, "season_name": "19/20"}}


def ev(i, typ, team="A", period=1, minute=1, ts="00:01:00.000", poss=1, **extra):
    d = {"id": f"e{i}", "index": i, "period": period, "minute": minute, "second": 0, "timestamp": ts, "type": {"name": typ},
         "possession": poss, "team": {"name": team}, "play_pattern": {"name": "Regular Play"}, "location": [100.0, 40.0],
         "player": {"id": 9, "name": "P"}, "position": {"name": "Center Forward"}}
    d.update(extra)
    return d


def shot(i, outcome="Saved", team="A", **kw):
    return ev(i, "Shot", team=team, shot={"outcome": {"name": outcome}, "body_part": {"name": "Right Foot"}, "technique": {"name": "Normal"},
                                           "type": {"name": "Open Play"}, "end_location": [120, 40, 1], "statsbomb_xg": 0.3, **kw.pop("shot_extra", {})}, **kw)


def test_basic_shot_parsing_and_labels():
    rows = parse_match_events([shot(1, "Goal")], MATCH)
    assert len(rows) == 1 and rows[0]["is_goal"] == 1 and rows[0]["opponent"] == "B" and rows[0]["is_home"] == 1


def test_score_diff_is_pre_shot_and_updates_after_goal():
    events = [shot(1, "Goal"), shot(2, "Saved"), shot(3, "Goal", team="B"), shot(4, "Saved", team="B")]
    rows = parse_match_events(events, MATCH)
    assert [r["score_diff"] for r in rows] == [0, 1, -1, 0]          # a shot never sees its own goal


def test_shootout_excluded_by_default():
    rows = parse_match_events([shot(1, period=5), shot(2, period=2)], MATCH)
    assert len(rows) == 1 and rows[0]["period"] == 2


def test_key_pass_context_and_no_goal_assist_leak():
    kp = ev(1, "Pass", pass_={}, **{"pass": {"length": 20.0, "angle": 0.1, "height": {"name": "Ground Pass"}, "cross": True, "goal_assist": True,
                                                "end_location": [105.0, 38.0]}})
    s = shot(2, "Goal", shot_extra={"key_pass_id": "e1"})
    row = parse_match_events([kp, s], MATCH)[0]
    assert row["kp_exists"] == 1 and row["kp_cross"] == 1 and row["kp_height"] == "Ground Pass"
    assert not any("goal_assist" in k for k in row)


def test_carry_before_shot_and_prev_event():
    carry = ev(1, "Carry", carry={"end_location": [100.0, 44.0]}, location=[100.0, 40.0] if False else [97.0, 40.0])
    rows = parse_match_events([carry, shot(2)], MATCH)
    assert rows[0]["prev_event_type"] == "Carry" and abs(rows[0]["carry_length"] - 5.0) < 1e-9


def test_possession_counters():
    events = [ev(1, "Pass"), ev(2, "Pass"), shot(3)]
    r = parse_match_events(events, MATCH)[0]
    assert r["poss_passes_before"] == 2 and r["poss_events_before"] == 2


def test_minutes_parsing():
    assert abs(_clock_minutes("45:30") - 45.5) < 1e-9
    lu = [{"team_name": "A", "lineup": [{"player_id": 1, "player_name": "X", "positions": [{"from": "00:00", "to": "60:00"}, {"from": "60:00", "to": None}]}]}]
    out = parse_minutes(lu, MATCH, 95.0)
    assert out[0]["minutes"] == 95.0
