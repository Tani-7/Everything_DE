import pandas as pd
from sqlalchemy import create_engine, text
from src.config import settings


def normalize_experience(val):
    """Convert years_experience strings into numeric values."""
    if not val or pd.isna(val):
        return None
    val = str(val).lower().strip()
    if "less than six" in val:
        return 0.5
    if "less than 1" in val or "under 1" in val:
        return 0.5
    if "1" in val and "year" in val:
        return 1
    if "2" in val:
        return 2
    if "3" in val:
        return 3
    if "4" in val:
        return 4
    if "5" in val:
        return 5
    try:
        return float(val)
    except ValueError:
        return None


def normalize_hours(val):
    """Convert hours_per_week strings into numeric values."""
    if not val or pd.isna(val):
        return None
    val = str(val).lower().strip()
    if "less than 6" in val:
        return 5
    if "6-10" in val:
        return 8
    if "11-20" in val:
        return 15
    if "21-30" in val:
        return 25
    if "30+" in val or "more than 30" in val:
        return 35
    try:
        return float(val)
    except ValueError:
        return None


def run_etl():
    """ETL pipeline to populate normalized tables from staging_participants."""

    engine = create_engine(settings.db_uri)

    with engine.begin() as conn:
        # 1) read raw staging data
        staging_df = pd.read_sql("SELECT * FROM staging_participants", conn)

        if staging_df.empty:
            print("⚠️ staging_participants is empty, nothing to process.")
            return

        # 2) load lookup tables
        countries = staging_df["country"].dropna().unique().tolist()
        for country in countries:
            conn.execute(
                text("INSERT INTO countries (country_name) VALUES (:c) ON CONFLICT DO NOTHING"),
                {"c": country},
            )

        tracks = staging_df["track_applied"].dropna().unique().tolist()
        for track in tracks:
            conn.execute(
                text("INSERT INTO tracks (track_name) VALUES (:t) ON CONFLICT DO NOTHING"),
                {"t": track},
            )

        # refresh lookup maps
        country_rows = conn.execute(
            text("SELECT country_name, id FROM countries")
        ).mappings().all()
        country_map = {row["country_name"]: row["id"] for row in country_rows}

        track_rows = conn.execute(
            text("SELECT track_name, id FROM tracks")
        ).mappings().all()
        track_map = {row["track_name"]: row["id"] for row in track_rows}

        # 3) transform + load participants
        for _, row in staging_df.iterrows():
            country_id = country_map.get(row["country"])
            track_id = track_map.get(row["track_applied"])

            conn.execute(
                text("""
                    INSERT INTO participants (
                        id_no, timestamp, age_range, gender, country_id,
                        heard_about, years_experience, track_id, hours_per_week,
                        main_aim, motivation, skill_level, aptitude_test_status,
                        total_score, graduation_status, cohort, sheet
                    )
                    VALUES (
                        :id_no, :timestamp, :age_range, :gender, :country_id,
                        :heard_about, :years_experience, :track_id, :hours_per_week,
                        :main_aim, :motivation, :skill_level, :aptitude_test_status,
                        :total_score, :graduation_status, :cohort, :sheet
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
                        sheet = EXCLUDED.sheet
                """),
                {
                    "id_no": row["id_no"],
                    "timestamp": row["timestamp"],
                    "age_range": row["age_range"],
                    "gender": row["gender"],
                    "country_id": country_id,
                    "heard_about": row["heard_about"],
                    "years_experience": normalize_experience(row["years_experience"]),
                    "track_id": track_id,
                    "hours_per_week": normalize_hours(row["hours_per_week"]),
                    "main_aim": row["main_aim"],
                    "motivation": row["motivation"],
                    "skill_level": row["skill_level"],
                    "aptitude_test_status": row["aptitude_test_status"],
                    "total_score": row["total_score"],
                    "graduation_status": row["graduation_status"],
                    "cohort": row["cohort"],
                    "sheet": row["sheet"],
                },
            )

        print(f"✅ ETL complete: {len(staging_df)} rows processed.")


if __name__ == "__main__":
    run_etl()
