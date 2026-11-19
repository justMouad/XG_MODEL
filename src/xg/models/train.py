"""Training orchestration: split -> optimise -> cross-validated selection -> held-out evaluation -> versioned artefacts.

Scientific protocol
-------------------
1. Chronological hold-out of whole matches = TEST (never used for tuning, calibration or model selection).
2. Inside TRAIN, grouped-by-match K-fold CV drives Optuna and the model selection.
3. Probability calibration is cross-fitted on out-of-fold predictions (see ``CalibratedXG``).
4. The F1-optimal decision threshold (only for precision/recall reporting) is learned on TRAIN out-of-fold predictions.
5. StatsBomb's own ``shot.statsbomb_xg`` is reported as an external benchmark, never used as a feature.
6. xG for the analytics layer is produced out-of-fold over *all* shots, so no shot is scored by a model that saw it.
"""
from __future__ import annotations

import datetime as dt
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

from ..config import Config
from ..data.splits import cv_splits, train_test_split_matches
from ..evaluation import metrics as M
from ..evaluation import plots as P
from ..features.engineering import feature_lists
from ..logging_utils import get_logger
from ..optimization.search import optimise
from ..utils import git_hash, json_dump, save_table, seed_everything
from .registry import ALL_MODELS, CalibratedXG, Calibrator, available, build_pipeline

log = get_logger(__name__)


def _pipe(cfg: Config, name: str, params: dict, numeric, categorical):
    t = cfg.training
    return build_pipeline(name, params, numeric, categorical, cfg.project["seed"], t["n_jobs"], t.get("class_weight"),
                          t.get("svm_max_train_samples", 20000), tuple(cfg.features.get("winsor_quantiles", [0.001, 0.999])))


def oof_predictions(pipe, X, y, folds, method: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (raw_oof, cross-fitted-calibrated_oof). Calibrator for fold k is learned on the other folds' OOF preds."""
    raw = np.full(len(y), np.nan)
    for tr, va in folds:
        raw[va] = clone(pipe).fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[va])[:, 1]
    cal = np.full(len(y), np.nan)
    for tr, va in folds:
        cal[va] = Calibrator(method).fit(raw[tr], y[tr]).transform(raw[va])
    return raw, cal


def out_of_fold_xg(df: pd.DataFrame, name: str, params: dict, cfg: Config, numeric, categorical, n_splits: int = 5) -> np.ndarray:
    """Calibrated xG for every shot from models that never saw that shot's match (grouped K-fold over ALL data)."""
    X, y = df[numeric + categorical], df["is_goal"].values
    xg = np.full(len(df), np.nan)
    folds = cv_splits(df["match_id"].values, df["match_date"].values, n_splits, "group_kfold")
    for k, (tr, va) in enumerate(folds):
        pipe = _pipe(cfg, name, params, numeric, categorical)
        m = CalibratedXG(pipe, cfg.training["calibration"], n_splits=3).fit(X.iloc[tr], y[tr], df["match_id"].values[tr])
        xg[va] = m.predict_proba(X.iloc[va])[:, 1]
        log.info("OOF xG fold %s/%s done", k + 1, len(folds))
    return xg


def reference_values(df: pd.DataFrame, numeric: list[str], categorical: list[str]) -> dict:
    """Typical value per feature (median / mode) used as the baseline for local ablation explanations."""
    ref = {c: float(df[c].median()) for c in numeric}
    ref.update({c: df[c].mode().iloc[0] for c in categorical})
    return ref


def _env_versions() -> dict:
    import importlib
    out = {"python": platform.python_version()}
    for lib in ["numpy", "pandas", "sklearn", "xgboost", "lightgbm", "catboost", "optuna", "shap"]:
        try:
            out[lib] = importlib.import_module(lib).__version__
        except Exception:
            out[lib] = None
    return out


def train_all(cfg: Config, df: pd.DataFrame, models: list[str] | None = None) -> dict:
    seed = cfg.project["seed"]
    seed_everything(seed)
    t = cfg.training
    numeric, categorical = feature_lists(cfg)
    missing = [c for c in numeric + categorical if c not in df.columns]
    if missing:
        raise KeyError(f"Feature columns missing from the feature table: {missing}")

    df = df.sort_values(["match_date", "match_id", "shot_id"]).reset_index(drop=True)
    run_id = f"{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}_{git_hash()}_{cfg.hash()}"
    run_dir = cfg.path("models_dir") / run_id
    (run_dir / "models").mkdir(parents=True, exist_ok=True)
    log.info("=== Run %s | %s shots | %s goals (%.2f%%)", run_id, len(df), int(df.is_goal.sum()), 100 * df.is_goal.mean())

    tr_idx, te_idx = train_test_split_matches(df, cfg.split["test_size"], cfg.split["strategy"], seed)
    train, test = df.iloc[tr_idx].reset_index(drop=True), df.iloc[te_idx].reset_index(drop=True)
    assert set(train.match_id).isdisjoint(set(test.match_id)), "match leakage between train and test"
    if cfg.split["strategy"] == "time":
        assert train.match_date.max() <= test.match_date.min(), "temporal leakage between train and test"
    log.info("Train: %s shots / %s matches (to %s) | Test: %s shots / %s matches (from %s)", len(train), train.match_id.nunique(),
             train.match_date.max(), len(test), test.match_id.nunique(), test.match_date.min())

    Xtr, ytr, Xte, yte = train[numeric + categorical], train["is_goal"].values, test[numeric + categorical], test["is_goal"].values
    folds = cv_splits(train["match_id"].values, train["match_date"].values, cfg.split["n_splits"], cfg.split.get("cv_strategy", "group_kfold"))
    method = t["calibration"]
    select_by = t.get("select_by", "log_loss")

    wanted = models or t["models"]
    results, cv_rows, test_rows, curve_data, preds_test, best_params_all = [], [], [], {}, {}, {}
    fitted: dict[str, CalibratedXG] = {}

    for name in wanted:
        if name not in ALL_MODELS or not available(name):
            log.warning("Skipping %s (unknown or library not installed)", name)
            continue
        t0 = time.time()
        try:
            log.info("--- %s: hyper-parameter search", name)
            best, trials, backend = optimise(name, Xtr, ytr, folds, numeric, categorical, cfg)
            best_params_all[name] = best
            if len(trials):
                save_table(trials, run_dir / "optuna" / f"trials_{name}")
            pipe = _pipe(cfg, name, best, numeric, categorical)
            raw_oof, cal_oof = oof_predictions(pipe, Xtr, ytr, folds, method)
            thr = M.best_f1_threshold(ytr, cal_oof)
            cv_raw, cv_cal = M.evaluate(ytr, raw_oof, thr), M.evaluate(ytr, cal_oof, thr)
            cv_rows.append({"model": name, "variant": "raw", **cv_raw})
            cv_rows.append({"model": name, "variant": "calibrated", **cv_cal})

            model = CalibratedXG(pipe, method, n_splits=cfg.split["n_splits"]).fit(Xtr, ytr, train["match_id"].values)
            p_raw, p_cal = model.predict_raw(Xte), model.predict_proba(Xte)[:, 1]
            test_rows.append({"model": name, "variant": "raw", **M.evaluate(yte, p_raw, thr)})
            test_rows.append({"model": name, "variant": "calibrated", **M.evaluate(yte, p_cal, thr)})
            curve_data[name] = M.curves(yte, p_cal)
            preds_test[name] = p_cal
            fitted[name] = model
            joblib.dump({"model": model, "numeric": numeric, "categorical": categorical, "params": best, "name": name, "threshold": thr},
                        run_dir / "models" / f"{name}.joblib")
            results.append({"model": name, "cv_select": cv_cal[select_by], "search": backend, "seconds": round(time.time() - t0, 1)})
            log.info("%s | CV(cal) logloss=%.4f brier=%.4f auc=%.4f | TEST logloss=%.4f auc=%.4f | %.0fs", name, cv_cal["log_loss"],
                     cv_cal["brier_score"], cv_cal["roc_auc"], test_rows[-1]["log_loss"], test_rows[-1]["roc_auc"], time.time() - t0)
        except Exception as e:  # keep the run alive if one model fails
            log.exception("Model %s failed: %s", name, e)

    if not results:
        raise RuntimeError("No model trained successfully")

    cv_df, test_df, sel = pd.DataFrame(cv_rows), pd.DataFrame(test_rows), pd.DataFrame(results)
    higher_better = select_by in ("roc_auc", "pr_auc")
    best_name = sel.sort_values("cv_select", ascending=not higher_better).iloc[0]["model"]
    log.info("Best model by CV %s (train only): %s", select_by, best_name)

    # external benchmark: StatsBomb xG on the same test shots
    bench = None
    if "statsbomb_xg" in test and test["statsbomb_xg"].notna().mean() > 0.5:
        ok = test["statsbomb_xg"].notna().values
        bench = {"model": "StatsBomb xG (benchmark)", "variant": "external", **M.evaluate(yte[ok], test.loc[ok, "statsbomb_xg"].values)}
        curve_data["StatsBomb xG (benchmark)"] = M.curves(yte[ok], test.loc[ok, "statsbomb_xg"].values)
    prevalence = {"model": "Constant (train prevalence)", "variant": "baseline",
                  **M.evaluate(yte, np.full(len(yte), ytr.mean()))}

    comp = test_df[test_df.variant == "calibrated"].copy()
    comp = comp.merge(sel[["model", "cv_select", "search", "seconds"]], on="model")
    comp["is_best_cv"] = comp["model"] == best_name
    extra = pd.DataFrame([r for r in (bench, prevalence) if r])
    comp_all = pd.concat([comp, extra], ignore_index=True)

    save_table(cv_df, run_dir / "cv_metrics")
    comp_all.to_csv(run_dir / "model_comparison.csv", index=False)
    test_df.to_csv(run_dir / "test_metrics_raw_vs_calibrated.csv", index=False)
    json_dump(curve_data, run_dir / "curves.json")
    json_dump(best_params_all, run_dir / "best_params.json")
    pd.DataFrame({"shot_id": test.shot_id, "y": yte, **{f"xg_{k}": v for k, v in preds_test.items()}}).to_csv(run_dir / "test_predictions.csv", index=False)

    plot_metrics = {r["model"]: r for _, r in comp_all.iterrows() if r["model"] in curve_data}
    P.plot_roc(curve_data, plot_metrics, run_dir / "roc.png")
    P.plot_pr(curve_data, plot_metrics, run_dir / "pr.png")
    P.plot_calibration(curve_data, run_dir / "calibration.png")
    P.plot_model_comparison(comp, run_dir / "model_comparison.png")

    # production model (optionally refit on all data) -------------------------------------------------
    best_params = best_params_all[best_name]
    final = fitted[best_name]
    if t.get("refit_on_all_data", True):
        log.info("Refitting %s on all %s shots for production", best_name, len(df))
        final = CalibratedXG(_pipe(cfg, best_name, best_params, numeric, categorical), method, n_splits=cfg.split["n_splits"]).fit(
            df[numeric + categorical], df["is_goal"].values, df["match_id"].values)
    thr = float(cv_df[(cv_df.model == best_name) & (cv_df.variant == "calibrated")]["threshold"].iloc[0])
    bundle = {"model": final, "numeric": numeric, "categorical": categorical, "name": best_name, "params": best_params,
              "threshold": thr, "run_id": run_id, "calibration": method, "reference": reference_values(df, numeric, categorical),
              "feature_config": {k: cfg.features.get(k) for k in ("use_freeze_frame", "use_player_history", "use_competition_context")}}
    joblib.dump(bundle, run_dir / "model.joblib")

    # out-of-fold xG for all shots (analytics layer) ----------------------------------------------------
    df["xg"] = out_of_fold_xg(df, best_name, best_params, cfg, numeric, categorical, n_splits=cfg.split["n_splits"])
    df["is_test_shot"] = df["match_id"].isin(set(test.match_id)).astype(int)
    save_table(df[["shot_id", "match_id", "xg", "is_test_shot"]], cfg.path("data_dir") / "processed" / "shots_xg")
    oof_m = M.evaluate(df["is_goal"].values, df["xg"].values, thr)
    log.info("OOF xG over all shots: logloss=%.4f auc=%.4f xG/goals=%.3f", oof_m["log_loss"], oof_m["roc_auc"], oof_m["xg_over_goals"])

    manifest = {
        "run_id": run_id, "created": dt.datetime.now().isoformat(timespec="seconds"), "git": git_hash(), "config_hash": cfg.hash(),
        "seed": seed, "best_model": best_name, "selection_metric": select_by, "calibration": method, "n_shots": len(df),
        "n_train": len(train), "n_test": len(test), "train_until": str(train.match_date.max()), "test_from": str(test.match_date.min()),
        "features": {"numeric": numeric, "categorical": categorical}, "best_params": best_params_all,
        "oof_all_shots_metrics": oof_m, "test_metrics_best": comp[comp.model == best_name].iloc[0].to_dict(),
        "benchmark_statsbomb": bench, "environment": _env_versions(), "config": dict(cfg),
        "shot_id_fingerprint": int(pd.util.hash_pandas_object(df["shot_id"], index=False).sum() % (10 ** 12)),
    }
    json_dump(manifest, run_dir / "manifest.json")
    _register(cfg, run_id, manifest)
    return {"run_dir": run_dir, "run_id": run_id, "best": best_name, "comparison": comp_all, "cv": cv_df, "df_with_xg": df,
            "numeric": numeric, "categorical": categorical, "train_idx": tr_idx, "test_idx": te_idx}


def _register(cfg: Config, run_id: str, manifest: dict) -> None:
    """Append-only model registry + ``latest.json`` pointer (model versioning)."""
    import json
    root = cfg.path("models_dir")
    reg_path = root / "registry.json"
    reg = json.loads(reg_path.read_text()) if reg_path.exists() else []
    reg.append({"run_id": run_id, "created": manifest["created"], "best_model": manifest["best_model"],
                "test_log_loss": manifest["test_metrics_best"].get("log_loss"), "test_roc_auc": manifest["test_metrics_best"].get("roc_auc"),
                "git": manifest["git"], "config_hash": manifest["config_hash"]})
    json_dump(reg, reg_path)
    json_dump({"run_id": run_id, "path": str(root / run_id / "model.joblib")}, root / "latest.json")
