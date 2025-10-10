# src/models/train_models.py
"""
Train and save all models.

Usage:
    python -m src.models.train_models
"""

import logging
from pathlib import Path

from src.config import settings
from src.data_utils import load_clean_data
from src.stats_models import train_all_models

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

MODEL_DIR = Path(settings.model_dir)
MODEL_DIR.mkdir(exist_ok=True, parents=True)


def main():
    df = load_clean_data()
    if df.empty:
        raise SystemExit("No data found. Run ingest & etl first.")

    logging.info("Training models... This may take a minute.")
    registry, metrics = train_all_models(df, overwrite=True)

    logging.info("Trained models and wrote registry.json to %s", MODEL_DIR)
    for name, meta in metrics.items():
        logging.info("%s metrics: %s", name, meta["metrics"])


if __name__ == "__main__":
    main()
