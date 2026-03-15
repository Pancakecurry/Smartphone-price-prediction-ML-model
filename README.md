# Smartphone Price Prediction Dashboard - Data Engineering

## Architecture Overview
This phase establishes a highly optimized, modular pipeline utilizing **Polars**, **BeautifulSoup/Requests**, and **DVC**.

1. **Extraction (`src/extract`)**: Components derive from `BaseScraper`, enforcing clear OOP principles and handling network transients natively.
2. **Transformation (`src/transform`)**: Features are vectorized and engineered strictly via Polars, keeping time complexity `O(N)` while achieving high-speed data manipulation (casting types, stripping characters, and regex extractions).
3. **Quality & Validation**: Error handling traces are comprehensively piped through `src/logger.py`, maintaining rotating logs. Unittest definitions covering mock extraction networks and transformer logic reside in `tests/`.

## Setup Instructions

Once in the directory (`cd smartphone_price_prediction`), run the automated setup script to build your environment:

```bash
# Initialize and install all dependencies (DVC, Venv, Git)
./setup_env.sh

# Activate your newly created environment
source venv/bin/activate
```