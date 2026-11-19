PY ?= python
all: ingest features train analytics db
ingest:    ; $(PY) -m xg.cli ingest
features:  ; $(PY) -m xg.cli features
eda:       ; $(PY) -m xg.cli eda
train:     ; $(PY) -m xg.cli train
explain:   ; $(PY) -m xg.cli explain
analytics: ; $(PY) -m xg.cli analytics
db:        ; $(PY) -m xg.cli load-db
dashboard: ; streamlit run src/xg/dashboard/app.py
test:      ; $(PY) -m pytest -q
demo:      ; $(PY) -m xg.cli run-all --synthetic
