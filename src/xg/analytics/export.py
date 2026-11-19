"""Assemble BI-ready tables (CSV) from the pipeline outputs. These feed PostgreSQL, the Streamlit dashboard and Power BI."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..logging_utils import get_logger

log = get_logger(__name__)

FACT_COLUMNS = ["shot_id", "match_id", "match_date", "competition_id", "competition_name", "season_id", "season_name", "team", "opponent",
                "is_home", "player_id", "player", "position", "position_group", "period", "minute", "second", "play_pattern", "x", "y",
                "shot_type", "body_part", "technique", "under_pressure", "first_time", "distance", "angle", "n_defenders_in_cone",
                "gk_dist_goal", "prev_event_type", "score_diff", "outcome", "is_goal", "statsbomb_xg", "xg", "is_test_shot"]


def export_tables(df: pd.DataFrame, matches: pd.DataFrame | None, analytics: dict[str, pd.DataFrame], run_dir: Path, out_dir: Path,
                  minutes: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    tables: dict[str, pd.DataFrame] = {}
    tables["fact_shots"] = df[[c for c in FACT_COLUMNS if c in df.columns]].copy()
    if matches is not None:
        tables["dim_match"] = matches
    if minutes is not None and len(minutes):
        tables["fact_minutes"] = minutes
    for k, v in analytics.items():
        tables[f"analytics_{k}"] = v

    manifest = json.loads((run_dir / "manifest.json").read_text())
    run_id = manifest["run_id"]
    comp = pd.read_csv(run_dir / "model_comparison.csv")
    comp.insert(0, "run_id", run_id)
    tables["model_comparison"] = comp
    tables["model_cv_metrics"] = pd.read_parquet(p) if (p := run_dir / "cv_metrics.parquet").exists() else pd.read_csv(run_dir / "cv_metrics.csv.gz")

    curves = json.loads((run_dir / "curves.json").read_text())
    rows = []
    for model, c in curves.items():
        for kind, (xk, yk) in {"roc": ("fpr", "tpr"), "pr": ("recall", "precision"), "calibration": ("mean_pred", "frac_pos")}.items():
            for i, (a, b) in enumerate(zip(c[kind][xk], c[kind][yk])):
                rows.append({"run_id": run_id, "model": model, "curve": kind, "point": i, "x": a, "y": b,
                             "n": c[kind]["n"][i] if kind == "calibration" else None})
    tables["model_curves"] = pd.DataFrame(rows)

    imps = []
    ex = run_dir / "explain"
    for fname, method, col in [("feature_importance_builtin.csv", "builtin", "importance"), ("feature_importance_permutation.csv", "permutation_logloss", "importance"),
                               ("shap_importance.csv", "mean_abs_shap", "mean_abs_shap")]:
        if (ex / fname).exists():
            d = pd.read_csv(ex / fname)[["feature", col]].rename(columns={col: "importance"})
            d.insert(0, "method", method)
            d.insert(0, "run_id", run_id)
            imps.append(d)
    tables["feature_importance"] = pd.concat(imps, ignore_index=True) if imps else pd.DataFrame(columns=["run_id", "method", "feature", "importance"])
    if (ex / "pdp.csv").exists():
        tables["model_pdp"] = pd.read_csv(ex / "pdp.csv").assign(run_id=run_id)
    if (ex / "individual_explanations.csv").exists():
        tables["model_shot_explanations"] = pd.read_csv(ex / "individual_explanations.csv").assign(run_id=run_id)
    tables["model_runs"] = pd.DataFrame([{
        "run_id": run_id, "created": manifest["created"], "best_model": manifest["best_model"], "git": manifest["git"],
        "config_hash": manifest["config_hash"], "calibration": manifest["calibration"], "n_shots": manifest["n_shots"],
        "n_train": manifest["n_train"], "n_test": manifest["n_test"], "train_until": manifest["train_until"], "test_from": manifest["test_from"],
        "oof_log_loss": manifest["oof_all_shots_metrics"]["log_loss"], "oof_roc_auc": manifest["oof_all_shots_metrics"]["roc_auc"],
        "oof_brier": manifest["oof_all_shots_metrics"]["brier_score"], "oof_xg_over_goals": manifest["oof_all_shots_metrics"]["xg_over_goals"],
        "oof_ece": manifest["oof_all_shots_metrics"]["ece"]}])
    for name, t in tables.items():
        t.to_csv(out_dir / f"{name}.csv", index=False)
    log.info("Exported %s tables to %s", len(tables), out_dir)
    return tables
