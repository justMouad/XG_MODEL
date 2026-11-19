"""Command line interface:  python -m xg.cli <command> [options]"""
from __future__ import annotations

import argparse
import sys

import yaml

from . import pipeline as pl
from .config import load_config
from .logging_utils import setup_logging


def _parse_overrides(items: list[str] | None) -> dict:
    out = {}
    for it in items or []:
        k, _, v = it.partition("=")
        out[k] = yaml.safe_load(v)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="xg", description="xG prediction & analytics platform")
    ap.add_argument("command", choices=["ingest", "features", "eda", "train", "explain", "analytics", "load-db", "predict", "run-all"])
    ap.add_argument("--config", default=None, help="path to YAML config")
    ap.add_argument("--set", dest="overrides", action="append", metavar="key.path=value", help="override config values")
    ap.add_argument("--synthetic", action="store_true", help="use synthetic data (pipeline demo only)")
    ap.add_argument("--n-synth-matches", type=int, default=150)
    ap.add_argument("--models", nargs="*", default=None, help="subset of models to train")
    ap.add_argument("--input", help="predict: CSV (raw shot schema) or JSON (StatsBomb events)")
    ap.add_argument("--output", help="predict: output CSV")
    ap.add_argument("--model-path", default=None, help="predict: explicit model.joblib (default: models/latest.json)")
    ap.add_argument("--skip-db", action="store_true", help="run-all: do not load PostgreSQL")
    a = ap.parse_args(argv)

    cfg = load_config(a.config, _parse_overrides(a.overrides))
    setup_logging(cfg.project.get("log_level", "INFO"))
    try:
        if a.command == "ingest":
            pl.step_ingest(cfg, a.synthetic, a.n_synth_matches)
        elif a.command == "features":
            pl.step_features(cfg)
        elif a.command == "eda":
            pl.step_eda(cfg)
        elif a.command == "train":
            pl.step_train(cfg, a.models)
        elif a.command == "explain":
            pl.step_explain(cfg)
        elif a.command == "analytics":
            pl.step_analytics(cfg)
        elif a.command == "load-db":
            pl.step_load_db(cfg)
        elif a.command == "predict":
            from .inference.predictor import predict_file
            if not a.input:
                ap.error("predict requires --input")
            out = predict_file(a.model_path, a.input, a.output)
            print(out.head(20).to_string())
        elif a.command == "run-all":
            pl.step_ingest(cfg, a.synthetic, a.n_synth_matches)
            pl.step_features(cfg)
            pl.step_eda(cfg)
            pl.step_train(cfg, a.models)
            pl.step_explain(cfg)
            pl.step_analytics(cfg)
            if not a.skip_db:
                pl.step_load_db(cfg)
    except (FileNotFoundError, RuntimeError, KeyError, ValueError) as e:
        logging_exc = __import__("logging").getLogger("xg.cli")
        logging_exc.error("%s: %s", type(e).__name__, e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
