# src/data_utils.py
from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine, text
from src.config import settings

_engine = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(settings.db_uri, future=True)
    return _engine


def load_clean_data(limit: int | None = None) -> pd.DataFrame:
    
    engine = get_engine()
    sql = """
    SELECT
      p.id as participant_id,
      p.id_no,
      p.timestamp,
      p.age_range,
      p.gender,
      c.country_name,
      t.track_name,
      -- prefer explicit values in graduation_outcomes if available, else participants' fields
      COALESCE(g.total_score, p.total_score) as total_score,
      COALESCE(g.graduation_status, p.graduation_status) as graduation_status,
      p.years_experience,
      p.hours_per_week,
      p.skill_level,
      p.heard_about,
      p.cohort,
      p.sheet
    FROM participants p
    LEFT JOIN countries c ON p.country_id = c.id
    LEFT JOIN tracks t ON p.track_id = t.id
    LEFT JOIN graduation_outcomes g ON g.participant_id = p.id
    ORDER BY p.id
    """
    if limit:
        sql += f" LIMIT {int(limit)}"

    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn)

    # Normalize types
    if "total_score" in df.columns:
        df["total_score"] = pd.to_numeric(df["total_score"], errors="coerce")
    for col in ["years_experience", "hours_per_week", "skill_level"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "graduation_status" in df.columns:
        # ensure boolean (None/NaN stays NaN)
        df["graduation_status"] = df["graduation_status"].apply(
            lambda v: bool(v) if pd.notna(v) else None
        )

    return df
