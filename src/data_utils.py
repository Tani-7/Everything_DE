import os
import logging
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Load environment variables
load_dotenv()
DB_URI = os.getenv("DB_URI")


def get_engine():
    """Create a SQLAlchemy engine from DB_URI in .env"""
    if not DB_URI:
        raise ValueError("DB_URI not set in .env")
    return create_engine(DB_URI)


def load_participants() -> pd.DataFrame:
    """
    Load cleaned participants joined with track and country info.
    Returns: DataFrame with participants + track + country
    """
    engine = get_engine()
    query = """
    SELECT p.*, t.track_name, c.country_name
    FROM participants p
    LEFT JOIN tracks t ON p.track_id = t.id
    LEFT JOIN countries c ON p.country_id = c.id
    """
    logging.info("Connecting to DB and loading participants data...")
    df = pd.read_sql(query, engine)
    logging.info("Loaded %d rows", len(df))

    # Ensure graduation_status is boolean
    if "graduation_status" in df.columns:
        df["graduation_status"] = df["graduation_status"].astype(bool)

    return df


def load_and_clean_data() -> pd.DataFrame:
    """Alias for load_participants (kept for compatibility)."""
    return load_participants()


if __name__ == "__main__":
    df = load_and_clean_data()
    print(df.head())
