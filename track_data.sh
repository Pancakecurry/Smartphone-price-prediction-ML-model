#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "Starting Data Version Control (DVC) Pipeline Tracking..."

# 1. Track the Raw Ingestion Output
echo "Tracking raw dataset..."
dvc add data/01_raw/raw_smartphones.csv

# 2. Track the Processed ML-Ready Parquet Output
echo "Tracking processed parquet dataset..."
dvc add data/03_processed/clean_smartphones.parquet

# 3. Stage the DVC mapping files to Git
echo "Staging DVC map files to Git..."
git add data/01_raw/.gitignore data/01_raw/raw_smartphones.csv.dvc
git add data/03_processed/.gitignore data/03_processed/clean_smartphones.parquet.dvc

# 4. Finalize the Commit
echo "Committing DVC maps to project history..."
git commit -m "chore: initialize data version tracking"

echo "🎉 Data Versioning Complete! The massive datasets are now mapped via DVC locally."
