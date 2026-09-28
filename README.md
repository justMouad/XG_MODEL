# ⚽ xG Platform — Expected Goals modelling, analytics and BI on StatsBomb Open Data

End-to-end, production-oriented project: ingestion → leakage-safe feature engineering → eight tuned and
**calibrated** models → interpretation → football analytics → PostgreSQL → interactive dashboard.

> **Honest status of this delivery**
> The project was written and tested in a sandbox with **no internet and no StatsBomb data**, and without
> `xgboost / lightgbm / catboost / optuna / shap / sqlalchemy / streamlit / plotly`.
> * **Verified there:** 28 unit tests pass; the *entire* pipeline (ingest-schema → features → EDA → training of
>   logistic regression, random forest, extra trees, SVM, MLP with seeded random search → calibration → evaluation →
>   OOF xG → explanation fallbacks → analytics → exports → inference) runs end-to-end on **synthetic data in the
>   StatsBomb schema** (`--synthetic`, demo only — its numbers mean nothing about football).
> * **Written but NOT executed here:** the XGBoost / LightGBM / CatBoost branches, Optuna (TPE) search, SHAP,
>   the PostgreSQL/SQLAlchemy loader and the Streamlit/Plotly dashboard, and parsing of the *real* StatsBomb files
>   (done against the documented schema and hand-built fixtures). Expect to fix small issues on first real run.
> * **No real metrics are reported in this README.** Run the pipeline to obtain them; they land in
>   `models/<run_id>/` and in the dashboard.

---------------------------------------------------------------------------------------------------

## 1. Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .

bash scripts/download_open_data.sh        # git clone of statsbomb/open-data  (or let the code download on demand)
docker compose up -d                       # PostgreSQL 16 (optional; SQLite fallback otherwise)

python -m xg.cli run-all                   # ingest → features → EDA → train → explain → analytics → load-db
streamlit run src/xg/dashboard/app.py      # dashboard on http://localhost:8501
```

Faster / partial runs:

```bash
# one competition-season, small tuning budget
python -m xg.cli run-all \
  --set data.competitions='[{"competition":"FIFA World Cup","season":"2018"}]' \
  --set optimization.n_trials=10

python -m xg.cli ingest | features | eda | train | explain | analytics | load-db      # individual steps
python -m xg.cli train --models logistic_regression lightgbm                           # subset of models
python -m xg.cli run-all --synthetic --skip-db                                         # offline demo, fake data
python -m xg.cli predict --input new_shots.csv --output xg.csv                          # inference
make test      # or: python scripts/run_tests.py (no pytest needed)
```

Config lives in `config/config.yaml`; any key can be overridden with `--set dotted.key=value`.

## 2. Inspecting the data first (what the code relies on)

The code was built from the documented StatsBomb Open Data event schema and does not assume absent fields:

| Source | Used for |
|---|---|
| `competitions.json`, `matches/<c>/<s>.json` | competition/season/date/teams, chronological split |
| `events/<match>.json` – `Shot` events | location, body part, technique, shot type, play pattern, `under_pressure`, `first_time`, `one_on_one`, `open_goal`, `aerial_won`, `follows_dribble`, **`freeze_frame`** (defenders/GK/teammates positions), `key_pass_id`, label (`outcome`) |
| other events in the same match | possession context (events/passes before the shot, possession duration), immediately preceding event, preceding carry, assisting pass (height, length, angle, cross, cut-back, through-ball, switch, start/end), running score |
| `lineups/<match>.json` | minutes played → per-90 metrics |

Coverage varies by competition (e.g. freeze frames, `statsbomb_xg`). Missingness is reported by `xg eda`
(`reports/eda/missingness.csv`) and is handled explicitly: missing ≠ 0 (median imputation **plus** missing-indicator
columns, and `ff_available` / `gk_available` flags).

Not available in the open data and therefore **not** modelled: player-tracking speed, shot power, post-shot
placement xG (PSxG), minutes since substitution, team/league strength ratings.

## 3. Architecture

```
config/config.yaml        all parameters (seed, paths, split, models, tuning, DB)
src/xg/
  config.py logging_utils.py utils.py cli.py pipeline.py
  data/         ingestion.py   StatsBomb JSON → shots_raw / matches / minutes (local clone, cache, or GitHub download)
                splits.py      chronological match hold-out, GroupKFold / blocked-time CV (never split a match)
                synthetic.py   schema-faithful fake data for tests / offline demo ONLY
  features/     engineering.py geometry, freeze-frame, assist, possession, score state, optional player history
                preprocessing.py  imputer + winsoriser + scaler + one-hot, all inside sklearn Pipelines
  eda/          explore.py     structure, missingness, balance, goal-rate plots, pitch density maps
  models/       registry.py    8 estimators, SVM wrapper, Calibrator + CalibratedXG
                train.py       orchestration, selection, versioned artefacts, OOF xG
  optimization/ search.py      Optuna TPE (random-search fallback), grouped-CV log-loss objective
  evaluation/   metrics.py plots.py   ROC/PR AUC, log loss, Brier, ECE, calibration slope/intercept, precision/recall
  explain/      interpret.py   SHAP, permutation + built-in importance, PDP, per-shot explanations
  inference/    predictor.py   XGPredictor: CSV / DataFrame / raw StatsBomb events → xG (+ explanations)
  analytics/    football.py    team & player analytics, finishing-vs-shot-quality statistics
                export.py      BI-ready tables (CSV) used by DB, dashboard and Power BI
  db/           postgres.py    SQLAlchemy loader (PostgreSQL, SQLite fallback), indexes, views
  dashboard/    app.py figures.py data_access.py     Streamlit + Plotly
sql/schema.sql  docker-compose.yml  Makefile  tests/  scripts/
```

Data flow (files on disk between steps, so every step is re-runnable):

```
open-data/ ─ingest→ data/raw/{shots_raw,matches,minutes}
          ─features→ data/processed/shots_features
          ─train→ models/<run_id>/{model.joblib, manifest.json, metrics, curves, plots, optuna/}  + models/latest.json
                  data/processed/shots_xg   (out-of-fold xG for every shot)
          ─explain→ models/<run_id>/explain/{shap_*, importance, pdp, individual_explanations}
          ─analytics→ data/processed/analytics/* + reports/exports/*.csv
          ─load-db→ PostgreSQL (fact_shots, dim_match, analytics_*, model_*)   ─→ Streamlit / Power BI / Superset
```

## 4. Methodology and leakage control

**Target / scope.** `P(goal | pre-shot information)` for non-penalty shots (penalties and shoot-outs excluded by default: a
different, near-constant process; configurable). Own goals are not shots.

**Features (≈ 60).** Location, distance, visible-goal angle, |y| offset; body part, technique, shot type, play pattern,
position group; pressure, first-time, aerial, follows-dribble, one-on-one, open goal; minute/period, home/away, score
difference *before* the shot; possession length/passes, time since previous event, previous event type, carry length;
assisting pass (height, type, technique, length, angle, cross, cut-back, through-ball, switch, progression towards goal);
freeze frame: defenders in the shot cone, attackers in the cone, nearest defender, defenders within 3 yd, defenders
ahead of the ball, goalkeeper distance to goal / shooter, off-line depth, lateral offset, GK-in-cone.

**Leakage safeguards (all enforced in code or tests).**
1. `outcome`, `end_location`, `statsbomb_xg`, `deflected`, `redirect` are excluded by an assertion in `feature_lists`.
   `pass.goal_assist` is **never extracted** – it is `True` only when the shot scored (label leak).
2. Score state and possession counters are updated *after* each event is processed.
3. Train/test split is **chronological by whole matches** (last 20 %). Tuning uses `GroupKFold` by `match_id` on the training
   set only; no match is ever in two folds. (`split.cv_strategy: blocked_time` available.)
4. All learned preprocessing (imputation, winsorisation, scaling, one-hot) sits **inside** the CV pipelines.
5. Probability calibration is **cross-fitted** (calibrator learned on out-of-fold predictions); the F1 threshold used for
   precision/recall comes from training OOF predictions, never from the test set.
6. Model selection uses cross-validated calibrated log loss on **train only**. The test set is touched once for reporting.
7. Optional player history (`features.use_player_history`) uses strictly earlier matches (unit-tested: perturbing a match's
   outcomes cannot change its own or earlier features; the first-match prior is a fixed constant). **Off by default**: a
   player-skill feature inside xG makes "finishing vs xG" analysis circular.
8. Analytics use **out-of-fold xG**, so no shot is scored by a model trained on its own match.
   (This OOF is grouped, not chronological; it is for descriptive analytics, not forecasting claims.)
9. StatsBomb's own xG is only an *external benchmark* in the comparison table.

**Class imbalance (~10 % goals).** No SMOTE / undersampling: they distort probabilities, and xG *is* a probability.
Imbalance is handled with proper scoring rules (log loss/Brier) + calibration; `training.class_weight: balanced` is available
and is automatically followed by calibration. Precision/recall are reported but are secondary.

**Outliers.** Winsorisation at the 0.1 / 99.9 % training quantiles; coordinates outside the pitch raise errors at inference.

**Models.** Logistic regression (baseline), random forest, extra trees, XGBoost, LightGBM, CatBoost, RBF-SVM
(sigmoid-margin probabilities, subsampled to `svm_max_train_samples` because kernel SVMs scale ~O(n²)), neural network
(`sklearn.MLPClassifier`; no PyTorch dependency). Optuna TPE minimises grouped-CV log loss per model.

**Selection & reporting.** Best model = lowest calibrated cross-validated log loss on train (configurable via
`training.select_by`). The comparison table reports ROC-AUC, PR-AUC, log loss, Brier, ECE, calibration slope/intercept,
Σxg/goals, precision/recall/F1, for raw and calibrated variants, vs. StatsBomb xG and a prevalence baseline.
With ~10⁵ shots, differences between top models are often within noise — look at calibration and the CV spread before
declaring a winner.

## 5. Interpretation

`python -m xg.cli explain` writes to `models/<run>/explain/`: built-in importance, **grouped permutation importance**
(Δ log loss on held-out data, one-hot columns shuffled together), SHAP (TreeExplainer / LinearExplainer / Kernel) summary +
bar plots and sample values, **partial dependence** for configurable features, and **individual explanations** (lowest-xG goal,
highest-xG goal, highest-xG miss). `XGPredictor.explain()` explains any new shot (SHAP for tree models, otherwise
ablation-against-median, which is cruder for correlated features such as distance/angle/x).

## 6. Football analytics (`xg.analytics.football`)

Team: shots, goals, xG for/against, xG difference, xG/match, attacking efficiency (G/xG), z-score. Player: xG, goals, G−xG,
xG/shot (**shot quality**), conversion, per-90 (from lineup minutes), **finishing** = G−xG with
z = (G−xG)/√Σp(1−p), empirical-Bayes shrunk G/xG, over/under label. Shot-quality distributions, competition/season tables,
and a **split-half reliability** test (is finishing persistent or noise?).

*Interpretation caveat:* without post-shot xG, "finishing skill" is confounded with shot placement/power, context the
model cannot see, and luck. Treat large |z| on many shots as a signal worth scouting, not as proof.

## 7. Dashboard

Streamlit (`src/xg/dashboard/app.py`), Plotly pitches. Sidebar filters: competition, season, team, shot type, body part, play
pattern, all-shots vs hold-out only. Tabs: **Overview** (KPIs, model KPIs, goals vs xG, quality bands) · **Player Analytics**
(ranking, G vs xG scatter, over/underperformers, comparison, shot maps) · **Team Analytics** · **Shot Analysis** (interactive
pitch map sized by xG, xG heat-map, player filter) · **Model Analytics** (comparison, ROC/PR/calibration, importance, PDP, SHAP,
individual explanations). Data come from PostgreSQL when `DATABASE_URL` is set, else `reports/exports/*.csv`.

**Power BI / Superset:** connect to the PostgreSQL tables (`fact_shots`, `dim_match`, `analytics_*`, `model_*`) or import
`reports/exports/*.csv`. `fact_shots` has one row per shot with the OOF `xg`; relate it to `dim_match` on `match_id`.

## 8. Reproducibility, versioning, operations

* Seeds everywhere (`project.seed`); every run writes `models/<run_id>/manifest.json` (config + hash, git hash, library
  versions, data fingerprint, metrics, features, hyper-parameters); `models/registry.json` is an append-only index and
  `models/latest.json` the production pointer. `run_id = timestamp_githash_confighash`.
* `model.joblib` is a bundle (`CalibratedXG`, feature lists, threshold, reference values). With
  `training.refit_on_all_data: true` the production model is refit on all shots after the evaluation run
  (per-model, train-only fits are kept in `models/<run>/models/`).
* Structured logging, explicit error messages on missing steps/files, failure isolation (one model failing does not abort a run).

## 9. Inference

```python
from xg.inference.predictor import XGPredictor
import pandas as pd
p = XGPredictor.load()                                     # models/latest.json
p.predict(pd.DataFrame({"x":[108,85],"y":[40,20],"body_part":["Right Foot","Head"]}))   # Series of xG
p.predict_events(events_json_list, match_json)             # raw StatsBomb events → xG per shot
p.explain(pd.DataFrame({"x":[108],"y":[40]}))              # top contributing features
```
Only `x, y` (StatsBomb coordinates, attacking towards x=120) are required; absent optional fields are treated as unknown.
Quality drops without freeze-frame/assist context since the model was trained with it.

## 10. Known limitations

Open data covers selected competitions (not a random sample of football) → apply to other leagues with caution
(`features.use_competition_context: false` by default for transferability); xG differences between models are small relative to
sampling noise; no PSxG/tracking data; grouped (not forward-chaining) CV inside the training window; MLP/SVM are included for
completeness and are not expected to beat gradient boosting or even logistic regression on this data.
