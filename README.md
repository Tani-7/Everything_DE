# everything-data-mentorship

## purpose
analysis of mentorship cohort data to understand demographics, motivations, and predictors of graduation.

## structure
- `schema.sql` - database schema
- `etl.py` - main ETL script
- `load_raw.py` - helper to load CSV to staging
- `api.py` - fastapi endpoints for cleaned data
- `dashboard/` - streamlit dashboard
- `notebooks/` - eda and modeling notebooks
- `data/` - raw and cleaned datasets (NOT committed to git)

## quickstart (local, sqlite)
1. create and activate venv:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   pip install -r requirements.txt
