#!/usr/bin/env bash
set -euo pipefail

uv run python src/training/dataset.py

dvc add data/bank_marketing_data_processed.csv
dvc push

git add data/bank_marketing_data_processed.csv.dvc
git commit -m "chore: update processed dataset in dvc" || true
