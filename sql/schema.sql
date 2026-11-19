-- Reference schema (PostgreSQL). The loader (`python -m xg.cli load-db`) creates equivalent tables with pandas/SQLAlchemy;
-- this file documents the model and provides keys/indexes/views for manual or Power BI/Superset use.
CREATE TABLE IF NOT EXISTS dim_match (
    match_id BIGINT PRIMARY KEY, match_date DATE, kick_off TEXT, competition_id INT, competition_name TEXT, season_id INT, season_name TEXT,
    home_team TEXT, away_team TEXT, home_score INT, away_score INT, stage TEXT, match_end_minute DOUBLE PRECISION);

CREATE TABLE IF NOT EXISTS fact_shots (
    shot_id TEXT PRIMARY KEY, match_id BIGINT REFERENCES dim_match(match_id), match_date DATE, competition_name TEXT, season_name TEXT,
    team TEXT, opponent TEXT, is_home INT, player_id BIGINT, player TEXT, position TEXT, position_group TEXT, period INT, minute INT, second INT,
    play_pattern TEXT, x DOUBLE PRECISION, y DOUBLE PRECISION, shot_type TEXT, body_part TEXT, technique TEXT, under_pressure INT,
    first_time INT, distance DOUBLE PRECISION, angle DOUBLE PRECISION, n_defenders_in_cone DOUBLE PRECISION, gk_dist_goal DOUBLE PRECISION,
    prev_event_type TEXT, score_diff INT, outcome TEXT, is_goal INT, statsbomb_xg DOUBLE PRECISION,
    xg DOUBLE PRECISION,           -- calibrated, OUT-OF-FOLD xG from this project's best model
    is_test_shot INT);             -- 1 = match belongs to the chronological hold-out test set

CREATE INDEX IF NOT EXISTS ix_fact_shots_player ON fact_shots(player_id);
CREATE INDEX IF NOT EXISTS ix_fact_shots_team ON fact_shots(team);
CREATE INDEX IF NOT EXISTS ix_fact_shots_comp ON fact_shots(competition_name, season_name);

-- analytics_* , model_* tables are (re)created by the loader from reports/exports/*.csv
