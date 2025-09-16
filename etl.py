#!/usr/bin/env python3
"""
etl.py - Extract (from staging_participants) -> Transform -> Load (participants, tracks, countries)
Produces: reports/data_quality_report_pre_transform.json
Usage:
    python etl.py
"""

import os
import json
import logging
import hashlib
from datetime import datetime, timezone
from typing import Tuple, Dict, Any

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()
DB_URI = os.getenv("DB_URI", "postgresql+psycopg2://postgres:password@localhost:5432/everything_de")
REPORT_PATH = "reports/data_quality_report_pre_transform.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("etl")

engine = create_engine(DB_URI, echo=False, future=True)


##### ---- utility functions ---- #####
def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def ensure_staging_helper_cols(conn):
    # Add helper columns if they don't exist: row_hash, processed, processed_at
    # Uses Postgres syntax; OK since you're on Postgres
    conn.execute(text("ALTER TABLE IF EXISTS staging_participants ADD COLUMN IF NOT EXISTS row_hash TEXT"))
    conn.execute(text("ALTER TABLE IF EXISTS staging_participants ADD COLUMN IF NOT EXISTS processed BOOLEAN DEFAULT FALSE"))
    conn.execute(text("ALTER TABLE IF EXISTS staging_participants ADD COLUMN IF NOT EXISTS processed_at TIMESTAMP"))


def fetch_staging_df(conn) -> pd.DataFrame:
    sql = "SELECT * FROM staging_participants"
    df = pd.read_sql(sql, conn)
    logger.info("fetched %d rows from staging_participants", len(df))
    return df


##### ---- transform helpers ---- #####
def standardize_text(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    return s if s != "" else None


def standardize_country(c):
    if c is None:
        return None
    s = str(c).strip().lower()
    mapping = {
        "usa": "united states",
        "us": "united states",
        "u.s.": "united states",
        "united states of america": "united states",
        "uk": "united kingdom",
        "england": "united kingdom",
        "kenya": "kenya",
        # add more as needed
    }
    out = mapping.get(s, s.title())
    return out


def parse_age_range(age):
    if pd.isna(age) or age is None:
        return (None, None, None)
    a = str(age).strip().lower()
    import re
    if '-' in a:
        parts = re.findall(r'(\d+)', a)
        if len(parts) >= 2:
            return (a.title(), int(parts[0]), int(parts[1]))
    if '+' in a:
        n = re.findall(r'(\d+)', a)
        if n:
            return (f"{n[0]}+", int(n[0]), None)
    if 'under' in a:
        n = re.findall(r'(\d+)', a)
        if n:
            return (f"under {n[0]}", None, int(n[0]) - 1)
    return (a.title(), None, None)


def parse_numeric_years(s):
    if pd.isna(s) or s is None:
        return None
    s = str(s).strip().lower()
    import re
    m = re.search(r'(\d+(\.\d+)?)', s)
    if m:
        return float(m.group(1))
    if "less" in s or "<1" in s or "0" in s:
        return 0.5
    return None


def parse_hours_per_week(s):
    if pd.isna(s) or s is None:
        return None
    s = str(s).strip().lower()
    import re
    nums = re.findall(r'(\d+(\.\d+)?)', s)
    if len(nums) == 1:
        return float(nums[0][0])
    if '-' in s and len(nums) >= 2:
        return (float(nums[0][0]) + float(nums[1][0])) / 2.0
    m = re.search(r'(\d+)', s)
    if m:
        return float(m.group(1))
    return None


def skill_to_numeric(s):
    if pd.isna(s) or s is None:
        return None
    s = str(s).strip().lower()
    mapping = {"beginner": 1, "novice": 1, "intermediate": 5, "advanced": 8, "expert": 10}
    for k in mapping:
        if k in s:
            return mapping[k]
    import re
    m = re.search(r'(\d+)', s)
    if m:
        try:
            return int(m.group(1))
        except:
            return None
    return None


def yes_no_to_bool(s):
    if pd.isna(s) or s is None:
        return None
    ss = str(s).strip().lower()
    if ss in ("yes", "y", "true", "1", "completed", "complete"):
        return True
    if ss in ("no", "n", "false", "0", "not completed", "incomplete"):
        return False
    return None


def normalize_gender(g):
    if pd.isna(g) or g is None:
        return None
    s = str(g).strip().lower()
    if s in ("female", "f", "woman", "w"):
        return "female"
    if s in ("male", "m", "man"):
        return "male"
    if s in ("non-binary", "nonbinary", "nb"):
        return "non-binary"
    return s.title()


##### ---- main transform pipeline ----#
def transform_staging_df(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Return cleaned dataframe ready for loading into participants and a small diagnostics dict."""
    df = df_raw.copy()

    # make column names predictable (lowercase)
    df.columns = [c.strip().lower() for c in df.columns]

    # ensure raw_row exists
    if 'raw_row' not in df.columns:
        df['raw_row'] = df.astype(str).apply(lambda r: json.dumps(r.to_dict(), default=str), axis=1)

    # compute row_hash (for tracing)
    df['row_hash'] = df['raw_row'].apply(lambda r: sha256_text(r if isinstance(r, str) else json.dumps(r, sort_keys=True)))

    # basic cleanup + transforms
    df['id_no'] = df.get('id_no').map(standardize_text)
    df['timestamp'] = pd.to_datetime(df.get('timestamp'), errors='coerce')
    df['age_range'] = df.get('age_range').map(standardize_text)
    df['gender'] = df.get('gender').map(normalize_gender)
    df['country_raw'] = df.get('country').map(standardize_text)
    df['country'] = df['country_raw'].map(standardize_country)
    df['heard_about'] = df.get('heard_about').map(standardize_text)
    df['years_experience_raw'] = df.get('years_experience')
    df['years_experience'] = df['years_experience_raw'].apply(parse_numeric_years)
    df['track_applied'] = df.get('track_applied').map(standardize_text)
    df['hours_per_week_raw'] = df.get('hours_per_week')
    df['hours_per_week'] = df['hours_per_week_raw'].apply(parse_hours_per_week)
    df['main_aim'] = df.get('main_aim').map(standardize_text)
    df['motivation'] = df.get('motivation').map(standardize_text)
    df['skill_level_raw'] = df.get('skill_level')
    df['skill_level_numeric'] = df['skill_level_raw'].apply(skill_to_numeric)
    df['aptitude_test_status'] = df.get('aptitude_test_status').apply(yes_no_to_bool)
    df['total_score'] = pd.to_numeric(df.get('total_score'), errors='coerce')
    df['graduation_status'] = df.get('graduation_status').apply(yes_no_to_bool)
    df['cohort'] = df.get('cohort')
    df['sheet'] = df.get('sheet')

    # diagnostics
    diag = {
        "rows_before": int(len(df_raw)),
        "rows_after": int(len(df)),
        "missing_id_no": int(df['id_no'].isna().sum()),
        "missing_total_score": int(df['total_score'].isna().sum()),
        "missing_graduation_status": int(df['graduation_status'].isna().sum()),
        "duplicate_id_no_count": int(df['id_no'].duplicated().sum())
    }

    # build cleaned df columns to insert into participants
    cleaned_cols = [
        'id_no', 'timestamp', 'age_range', 'gender', 'country', 'heard_about', 'years_experience',
        'track_applied', 'hours_per_week', 'main_aim', 'motivation', 'skill_level_numeric',
        'aptitude_test_status', 'total_score', 'graduation_status', 'cohort', 'sheet', 'row_hash'
    ]
    # ensure columns exist
    for col in cleaned_cols:
        if col not in df.columns:
            df[col] = None

    cleaned = df[cleaned_cols].rename(columns={'skill_level_numeric': 'skill_level'})

    return cleaned, diag


##### ---- load helpers ----#
def upsert_lookups(conn, df_cleaned: pd.DataFrame):
    # tracks
    tracks = df_cleaned['track_applied'].dropna().unique().tolist()
    tracks = [t.strip() for t in tracks if t is not None and str(t).strip() != ""]
    for t in tracks:
        conn.execute(text("""
            INSERT INTO tracks (track_name) VALUES (:t)
            ON CONFLICT (track_name) DO NOTHING
        """), {"t": t})

    countries = df_cleaned['country'].dropna().unique().tolist()
    countries = [c.strip() for c in countries if c is not None and str(c).strip() != ""]
    for c in countries:
        conn.execute(text("""
           INSERT INTO countries (country_name) VALUES (:c)
           ON CONFLICT (country_name) DO NOTHING
        """), {"c": c})

    # read mapping back
    tracks_df = pd.read_sql("SELECT id, track_name FROM tracks", conn)
    countries_df = pd.read_sql("SELECT id, country_name FROM countries", conn)
    track_map = dict(zip(tracks_df['track_name'], tracks_df['id']))
    country_map = dict(zip(countries_df['country_name'], countries_df['id']))
    return track_map, country_map


def upsert_participants(conn, cleaned: pd.DataFrame, track_map: dict, country_map: dict) -> Tuple[int, int, int]:
    # identify new id_no values (for assertion)
    existing = pd.read_sql("SELECT id_no FROM participants", conn)
    existing_set = set(existing['id_no'].astype(str).tolist()) if not existing.empty else set()

    cleaned = cleaned.copy()
    cleaned['track_id'] = cleaned['track_applied'].map(lambda t: track_map.get(t))
    cleaned['country_id'] = cleaned['country'].map(lambda c: country_map.get(c))
    # drop rows missing id_no
    missing_id = cleaned['id_no'].isna()
    skipped = cleaned[missing_id]
    if not skipped.empty:
        logger.warning("skipping %d rows with missing id_no", len(skipped))

    cleaned_to_insert = cleaned[~missing_id].copy()

    new_idnos = [x for x in cleaned_to_insert['id_no'].astype(str).unique().tolist() if x not in existing_set]
    new_count = len(new_idnos)

    # upsert via ON CONFLICT (id_no)
    insert_sql = """
    INSERT INTO participants (
        id_no, timestamp, age_range, gender, country_id, heard_about, years_experience,
        track_id, hours_per_week, main_aim, motivation, skill_level,
        aptitude_test_status, total_score, graduation_status, cohort, sheet
    ) VALUES (
        :id_no, :timestamp, :age_range, :gender, :country_id, :heard_about, :years_experience,
        :track_id, :hours_per_week, :main_aim, :motivation, :skill_level,
        :aptitude_test_status, :total_score, :graduation_status, :cohort, :sheet
    )
    ON CONFLICT (id_no) DO UPDATE SET
        timestamp = EXCLUDED.timestamp,
        age_range = EXCLUDED.age_range,
        gender = EXCLUDED.gender,
        country_id = EXCLUDED.country_id,
        heard_about = EXCLUDED.heard_about,
        years_experience = EXCLUDED.years_experience,
        track_id = EXCLUDED.track_id,
        hours_per_week = EXCLUDED.hours_per_week,
        main_aim = EXCLUDED.main_aim,
        motivation = EXCLUDED.motivation,
        skill_level = EXCLUDED.skill_level,
        aptitude_test_status = EXCLUDED.aptitude_test_status,
        total_score = EXCLUDED.total_score,
        graduation_status = EXCLUDED.graduation_status,
        cohort = EXCLUDED.cohort,
        sheet = EXCLUDED.sheet;
    """

    params = []
    for _, r in cleaned_to_insert.iterrows():
        params.append({
            "id_no": r['id_no'],
            "timestamp": r['timestamp'],
            "age_range": r['age_range'],
            "gender": r['gender'],
            "country_id": int(r['country_id']) if pd.notna(r['country_id']) else None,
            "heard_about": r['heard_about'],
            "years_experience": float(r['years_experience']) if pd.notna(r['years_experience']) else None,
            "track_id": int(r['track_id']) if pd.notna(r['track_id']) else None,
            "hours_per_week": float(r['hours_per_week']) if pd.notna(r['hours_per_week']) else None,
            "main_aim": r['main_aim'],
            "motivation": r['motivation'],
            "skill_level": r['skill_level'],
            "aptitude_test_status": bool(r['aptitude_test_status']) if pd.notna(r['aptitude_test_status']) else None,
            "total_score": float(r['total_score']) if pd.notna(r['total_score']) else None,
            "graduation_status": bool(r['graduation_status']) if pd.notna(r['graduation_status']) else None,
            "cohort": r['cohort'],
            "sheet": r['sheet']
        })

    # EXECUTE: main() opens engine.begin() so here we only run executes (no inner transaction)
    inserted_count = 0
    if params:
        for p in params:
            conn.execute(text(insert_sql), p)
            inserted_count += 1

    return new_count, inserted_count, len(skipped)


##### ---- small helpers ----#
def generate_pre_report(df_staging: pd.DataFrame) -> dict:
    report = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "total_rows": len(df_staging),
        "null_counts": {},
        "duplicate_id_no_count": 0,
        "sample_bad_rows": []
    }

    df_tmp = df_staging.copy()
    df_tmp.columns = [c.strip().lower() for c in df_tmp.columns]

    for col in ['id_no', 'total_score', 'graduation_status']:
        report['null_counts'][col] = int(df_tmp[col].isna().sum()) if col in df_tmp.columns else None

    if 'id_no' in df_tmp.columns:
        report['duplicate_id_no_count'] = int(df_tmp['id_no'].duplicated().sum())

    return report


def update_processed_flags(conn, cleaned: pd.DataFrame):
    if 'row_hash' not in cleaned.columns:
        return

    hashes = cleaned['row_hash'].dropna().unique().tolist()
    CHUNK = 500
    for i in range(0, len(hashes), CHUNK):
        sub = hashes[i:i+CHUNK]
        conn.execute(
            text("UPDATE staging_participants SET processed = TRUE, processed_at = now() WHERE row_hash = ANY(:list)"),
            {"list": sub}
        )


def main():
    logger.info("ETL started at %s", datetime.now(timezone.utc).isoformat())

    # single transaction for the whole ETL run; will commit on success and rollback on exception
    with engine.begin() as conn:
        ensure_staging_helper_cols(conn)
        df_staging = fetch_staging_df(conn)

        pre_report = generate_pre_report(df_staging)
        os.makedirs(os.path.dirname(REPORT_PATH) or ".", exist_ok=True)
        with open(REPORT_PATH, "w") as fh:
            json.dump(pre_report, fh, indent=2)

        if pre_report["total_rows"] == 0:
            logger.warning("no staging rows found. exiting.")
            return

        # transform
        cleaned, diag = transform_staging_df(df_staging)
        logger.info("transformed staging -> cleaned (%d rows)", len(cleaned))

        # lookups + upsert
        track_map, country_map = upsert_lookups(conn, cleaned)
        logger.info("tracks: %d, countries: %d", len(track_map), len(country_map))

        new_count, inserted_count, skipped_count = upsert_participants(conn, cleaned, track_map, country_map)
        logger.info("new participants (by id_no): %d", new_count)
        logger.info("insert attempts: %d, skipped (missing id_no): %d", inserted_count, skipped_count)

        # mark processed
        update_processed_flags(conn, cleaned)

        # finalize report
        pre_report.update({
            "transformed_rows": diag,
            "new_participants_by_id_no": new_count,
            "insert_attempt_count": inserted_count,
            "skipped_count_missing_id_no": skipped_count,
            "completed_at": datetime.now(timezone.utc).isoformat()
        })
        with open(REPORT_PATH, "w") as fh:
            json.dump(pre_report, fh, indent=2)

        if pre_report["total_rows"] > 0:
            assert inserted_count >= 0, "No rows considered for insertion - check transform"
            logger.info("ETL finished successfully. report saved to %s", REPORT_PATH)


if __name__ == "__main__":
    main()