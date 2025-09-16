import os
import logging
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

load_dotenv()
DB_URI = os.getenv("DB_URI")


def get_engine():
    if not DB_URI:
        raise ValueError("DB_URI not set in .env")
    return create_engine(DB_URI)

#Load cleaned participants joined with track and country info.
def load_participants() -> pd.DataFrame:
    engine = get_engine()
    query = """
    SELECT p.*, t.track_name, c.country_name
    FROM participants p
    LEFT JOIN tracks t ON p.track_id = t.id
    LEFT JOIN countries c ON p.country_id = c.id
    """
    logging.info("Connecting to DB and loading data...")
    df = pd.read_sql(query, engine)
    logging.info("Loaded %d rows", len(df))
    df["graduation_status"] = df["graduation_status"].astype(bool)
    return df

 #Aliasing load_participants for clarity.
def load_and_clean_data() -> pd.DataFrame:
    return load_participants()


if __name__ == "__main__":
    df = load_and_clean_data()
    print(df.head())
