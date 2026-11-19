"""Data ingestion for StatsBomb Open Data.

Reads the open-data repository layout (``competitions.json``, ``matches/<comp>/<season>.json``,
``events/<match>.json``, ``lineups/<match>.json``) from a local clone and, optionally, downloads
missing files from GitHub into a cache directory.

Output tables (one row per shot / match / player-match):
    raw/shots_raw, raw/matches, raw/minutes

Fields are taken **only** from what the StatsBomb event schema documents. Post-shot information
(outcome, end location, saved/deflected flags, StatsBomb's own xG, ``goal_assist``) is stored so it can
be used for labelling and benchmarking, and is explicitly excluded from model features in
``xg.features.engineering.LEAKY_COLUMNS``.
"""
from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from ..config import Config, resolve
from ..logging_utils import get_logger
from ..utils import save_table

log = get_logger(__name__)

GITHUB_RAW = "https://raw.githubusercontent.com/statsbomb/open-data/master/data/"
NON_PLAY_EVENTS = {"Starting XI", "Half Start", "Half End", "Substitution", "Tactical Shift", "Referee Ball-Drop",
                   "Injury Stoppage", "Player Off", "Player On", "Bad Behaviour", "Camera on", "Camera off"}


# --------------------------------------------------------------------------- source
class StatsBombSource:
    """Resolve StatsBomb JSON files from a local clone, a cache, or GitHub."""

    def __init__(self, open_data_dir: str | Path, cache_dir: str | Path, download: bool = True, retries: int = 3):
        self.root = Path(open_data_dir)
        self.cache = Path(cache_dir)
        self.download = download
        self.retries = retries

    def get_json(self, rel: str) -> Any:
        for base in (self.root, self.cache):
            p = base / rel
            if p.exists():
                with open(p, "r", encoding="utf-8") as fh:
                    return json.load(fh)
        if not self.download:
            raise FileNotFoundError(f"{rel} not found in {self.root} or {self.cache} and download is disabled")
        return self._download(rel)

    def _download(self, rel: str) -> Any:
        url = GITHUB_RAW + rel
        last: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                with urllib.request.urlopen(url, timeout=60) as r:
                    payload = r.read()
                dest = self.cache / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(payload)
                return json.loads(payload)
            except Exception as e:  # network / 404
                last = e
                log.warning("Download failed (%s/%s) %s: %s", attempt, self.retries, url, e)
                time.sleep(1.5 * attempt)
        raise RuntimeError(f"Could not obtain {rel}") from last


# --------------------------------------------------------------------------- helpers
def _name(d: dict | None) -> str | None:
    return d.get("name") if isinstance(d, dict) else None


def _ts_seconds(ts: str | None) -> float:
    """'00:12:34.567' -> seconds."""
    if not ts:
        return float("nan")
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def _clock_minutes(s: str | None) -> float | None:
    """'67:32' -> 67.53 minutes."""
    if not s:
        return None
    parts = s.split(":")
    return int(parts[0]) + int(parts[1]) / 60.0


# --------------------------------------------------------------------------- event parsing
def parse_match_events(events: list[dict], match: dict, include_shootouts: bool = False) -> list[dict]:
    """Extract one record per shot (with possession/preceding-event/assist context) from a match.

    Pure function (no IO) so it can be unit-tested with hand-made events.
    """
    events = sorted(events, key=lambda e: e.get("index", 0))
    by_id = {e["id"]: e for e in events if "id" in e}
    home = match["home_team"]["home_team_name"]
    away = match["away_team"]["away_team_name"]
    comp = match["competition"]
    season = match["season"]

    goals = {home: 0, away: 0}
    poss_stats: dict[int, dict] = {}
    rows: list[dict] = []
    prev_ev: dict | None = None
    last_pass_in_poss: dict[int, dict] = {}

    for e in events:
        etype = _name(e.get("type"))
        period = e.get("period", 1)
        poss = e.get("possession")
        team = _name(e.get("team"))

        if poss is not None:
            st = poss_stats.setdefault(poss, {"start": _ts_seconds(e.get("timestamp")), "events": 0, "passes": 0})
        else:
            st = {"start": float("nan"), "events": 0, "passes": 0}

        if etype == "Shot" and (include_shootouts or period != 5):
            sh = e.get("shot", {})
            loc = e.get("location") or [None, None]
            end = sh.get("end_location") or [None, None, None]
            outcome = _name(sh.get("outcome"))

            # key pass (assist) context
            kp = by_id.get(sh.get("key_pass_id")) if sh.get("key_pass_id") else None
            if kp is None and poss in last_pass_in_poss:  # fallback: last pass in same possession within 8 s
                cand = last_pass_in_poss[poss]
                if _ts_seconds(e.get("timestamp")) - _ts_seconds(cand.get("timestamp")) <= 8 and cand.get("period") == period:
                    kp = cand
            kpp = (kp or {}).get("pass", {}) if kp else {}
            kp_loc = (kp or {}).get("location") or [None, None]
            kp_end = kpp.get("end_location") or [None, None]

            # carry immediately before the shot
            carry_len = None
            if prev_ev is not None and _name(prev_ev.get("type")) == "Carry":
                c = prev_ev.get("carry", {}).get("end_location")
                l0 = prev_ev.get("location")
                if c and l0:
                    carry_len = ((c[0] - l0[0]) ** 2 + (c[1] - l0[1]) ** 2) ** 0.5

            opp = away if team == home else home
            t_now = _ts_seconds(e.get("timestamp"))
            rows.append({
                "shot_id": e["id"], "match_id": match["match_id"], "match_date": match.get("match_date"),
                "competition_id": comp.get("competition_id"), "competition_name": comp.get("competition_name"),
                "season_id": season.get("season_id"), "season_name": season.get("season_name"),
                "team": team, "opponent": opp, "is_home": int(team == home),
                "player_id": (e.get("player") or {}).get("id"), "player": _name(e.get("player")),
                "position": _name(e.get("position")),
                "period": period, "minute": e.get("minute"), "second": e.get("second"),
                "possession": poss, "play_pattern": _name(e.get("play_pattern")),
                "x": loc[0], "y": loc[1],
                "under_pressure": int(bool(e.get("under_pressure", False))),
                "shot_type": _name(sh.get("type")), "body_part": _name(sh.get("body_part")),
                "technique": _name(sh.get("technique")),
                "first_time": int(bool(sh.get("first_time", False))),
                "aerial_won": int(bool(sh.get("aerial_won", False))),
                "follows_dribble": int(bool(sh.get("follows_dribble", False))),
                "one_on_one": int(bool(sh.get("one_on_one", False))),
                "open_goal": int(bool(sh.get("open_goal", False))),
                "deflected": int(bool(sh.get("deflected", False))),
                "redirect": int(bool(sh.get("redirect", False))),
                # --- label & post-shot info (never used as features)
                "outcome": outcome, "is_goal": int(outcome == "Goal"),
                "statsbomb_xg": sh.get("statsbomb_xg"),
                "end_x": end[0], "end_y": end[1], "end_z": end[2] if len(end) > 2 else None,
                # --- freeze frame kept raw; converted into numeric features in feature engineering
                "freeze_frame_json": json.dumps(sh.get("freeze_frame")) if sh.get("freeze_frame") else None,
                # --- preceding context
                "prev_event_type": _name(prev_ev.get("type")) if prev_ev else None,
                "prev_event_gap": (t_now - _ts_seconds(prev_ev.get("timestamp"))) if prev_ev and prev_ev.get("period") == period else None,
                "carry_length": carry_len,
                "poss_events_before": st["events"], "poss_passes_before": st["passes"],
                "poss_duration": (t_now - st["start"]) if st["start"] == st["start"] else None,
                "score_diff": goals.get(team, 0) - goals.get(opp, 0),
                # --- assisting (key) pass; NOTE: pass.goal_assist is deliberately NOT extracted (it encodes the label)
                "kp_exists": int(kp is not None),
                "kp_height": _name(kpp.get("height")), "kp_type": _name(kpp.get("type")),
                "kp_technique": _name(kpp.get("technique")), "kp_body_part": _name(kpp.get("body_part")),
                "kp_length": kpp.get("length"), "kp_angle": kpp.get("angle"),
                "kp_cross": int(bool(kpp.get("cross", False))), "kp_cut_back": int(bool(kpp.get("cut_back", False))),
                "kp_switch": int(bool(kpp.get("switch", False))), "kp_through_ball": int(bool(kpp.get("through_ball", False))),
                "kp_start_x": kp_loc[0], "kp_start_y": kp_loc[1], "kp_end_x": kp_end[0], "kp_end_y": kp_end[1],
            })

        # --- state updates AFTER processing the event (so a shot never sees its own outcome)
        if etype == "Shot" and _name(e.get("shot", {}).get("outcome")) == "Goal" and period != 5:
            goals[team] = goals.get(team, 0) + 1
        elif etype == "Own Goal For":
            goals[team] = goals.get(team, 0) + 1
        if etype not in NON_PLAY_EVENTS and poss is not None:
            st["events"] += 1
            if etype == "Pass":
                st["passes"] += 1
                last_pass_in_poss[poss] = e
        if etype not in NON_PLAY_EVENTS:
            prev_ev = e
    return rows


def parse_minutes(lineups: list[dict], match: dict, match_end_minute: float) -> list[dict]:
    """Minutes played per player from StatsBomb lineups (position from/to clock strings)."""
    out = []
    for team in lineups:
        for p in team.get("lineup", []):
            minutes = 0.0
            for pos in p.get("positions", []):
                a = _clock_minutes(pos.get("from")) or 0.0
                b = _clock_minutes(pos.get("to"))
                b = match_end_minute if b is None else b
                minutes += max(b - a, 0.0)
            out.append({"match_id": match["match_id"], "team": team["team_name"], "player_id": p["player_id"],
                        "player": p["player_name"], "minutes": round(minutes, 2)})
    return out


# --------------------------------------------------------------------------- orchestration
def list_matches(src: StatsBombSource, cfg: Config) -> list[dict]:
    comps = src.get_json("competitions.json")
    wanted = cfg.data.get("competitions")
    gender = cfg.data.get("gender")
    selected = []
    for c in comps:
        if gender and c.get("competition_gender") != gender:
            continue
        if wanted and not any(w["competition"] == c["competition_name"] and str(w["season"]) == str(c["season_name"]) for w in wanted):
            continue
        selected.append(c)
    log.info("%s competition-seasons selected", len(selected))
    matches: list[dict] = []
    for c in selected:
        try:
            matches.extend(src.get_json(f"matches/{c['competition_id']}/{c['season_id']}.json"))
        except Exception as e:
            log.error("Skipping competition %s/%s: %s", c["competition_id"], c["season_id"], e)
    matches.sort(key=lambda m: (m.get("match_date", ""), m["match_id"]))
    mx = cfg.data.get("max_matches")
    if mx:
        matches = matches[: int(mx)]
    return matches


def ingest(cfg: Config) -> dict[str, pd.DataFrame]:
    """Run the full ingestion and persist raw tables."""
    d = cfg.data
    src = StatsBombSource(resolve(cfg, d["open_data_dir"]), resolve(cfg, d["cache_dir"]), d.get("download_if_missing", True))
    matches = list_matches(src, cfg)
    if not matches:
        raise RuntimeError("No matches selected - check data.competitions / open_data_dir in the config")
    log.info("Ingesting %s matches", len(matches))

    shot_rows: list[dict] = []
    minute_rows: list[dict] = []
    match_rows: list[dict] = []
    failed = 0
    for i, m in enumerate(matches, 1):
        try:
            events = src.get_json(f"events/{m['match_id']}.json")
        except Exception as e:
            failed += 1
            log.error("Events missing for match %s: %s", m["match_id"], e)
            continue
        shot_rows.extend(parse_match_events(events, m, include_shootouts=d.get("include_shootouts", False)))
        end_min = max([e.get("minute", 0) for e in events if e.get("period", 1) <= 4] or [90]) + 1.0
        try:
            minute_rows.extend(parse_minutes(src.get_json(f"lineups/{m['match_id']}.json"), m, end_min))
        except Exception as e:
            log.warning("Lineups unavailable for match %s: %s", m["match_id"], e)
        match_rows.append({
            "match_id": m["match_id"], "match_date": m.get("match_date"), "kick_off": m.get("kick_off"),
            "competition_id": m["competition"]["competition_id"], "competition_name": m["competition"]["competition_name"],
            "season_id": m["season"]["season_id"], "season_name": m["season"]["season_name"],
            "home_team": m["home_team"]["home_team_name"], "away_team": m["away_team"]["away_team_name"],
            "home_score": m.get("home_score"), "away_score": m.get("away_score"),
            "stage": _name(m.get("competition_stage")), "match_end_minute": end_min,
        })
        if i % 25 == 0:
            log.info("... %s/%s matches", i, len(matches))
    if failed:
        log.warning("%s matches failed to load", failed)

    shots = pd.DataFrame(shot_rows)
    if not d.get("include_penalties", False):
        n0 = len(shots)
        shots = shots[shots["shot_type"] != "Penalty"].reset_index(drop=True)
        log.info("Dropped %s penalty shots (data.include_penalties=false)", n0 - len(shots))
    tables = {"shots_raw": shots, "matches": pd.DataFrame(match_rows), "minutes": pd.DataFrame(minute_rows)}
    raw_dir = cfg.path("data_dir") / "raw"
    for k, df in tables.items():
        save_table(df, raw_dir / k)
    log.info("Ingestion done: %s shots, %s goals", len(shots), int(shots["is_goal"].sum()))
    return tables
