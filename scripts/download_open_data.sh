#!/usr/bin/env bash
# Fetch StatsBomb Open Data (events, lineups, matches, competitions). ~1+ GB for the full repository.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -d open-data ]; then git -C open-data pull --ff-only; else git clone --depth 1 https://github.com/statsbomb/open-data.git open-data; fi
echo "Done. Config key data.open_data_dir already points to open-data/data"
