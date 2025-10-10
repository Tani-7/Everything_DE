"""
models/data_loader.py
Utilities to read staging/participants tables from postgres into pandas.
"""
from sqlalchemy import text
from models.db import get_engine
import pandas as pd

def fetch_staging(limit: int | None = None) -> pd.DataFrame:
    engine = get_engine()
    sql = "SELECT * FROM staging_participants"
    if limit:
        sql += f" LIMIT {int(limit)}"
    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn)
    return df

def fetch_participants(limit: int | None = None) -> pd.DataFrame:
    engine = get_engine()
    sql = """
    SELECT p.*, t.track_name, c.country_name
    FROM participants p
    LEFT JOIN tracks t ON p.track_id = t.id
    LEFT JOIN countries c ON p.country_id = c.id
    ORDER BY p.id
    """
    if limit:
        sql += f" LIMIT {int(limit)}"
    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn)
    return df
