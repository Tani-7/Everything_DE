import os
import pandas as pd
import json
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

DB_URI = os.getenv("DB_URI")  # <-- define it
if not DB_URI:
    raise RuntimeError("DB_URI not set. Did you create your .env?")

engine = create_engine(DB_URI)

# Normalizing the header names
COLUMN_MAP = {
    "Timestamp": "timestamp",
    "ID No.": "id_no",
    "Id. No": "id_no",
    "Age range": "age_range",
    "Gender": "gender",
    "Country": "country",
    "Where did you hear about Everything Data?": "heard_about",
    "How many years of learning experience do you have in the field of data?": "years_experience",
    "Which track are you applying for?": "track_applied",
    "How many hours per week can you commit to learning?": "hours_per_week",
    "What is your main aim for joining the mentorship program?": "main_aim",
    "What is your motivation to join the Everything Data mentorship program?": "motivation",
    "How best would you describe your skill level in the track you are applying for?": "skill_level",
    "Have you completed the everything data aptitude test for your track?": "aptitude_test_status",
    "Total score": "total_score",
    "Graduated": "graduation_status",
}

def normalize_dataframe(df, cohort_label, sheet_name):
    # rename headers
    df = df.rename(columns=COLUMN_MAP)

    # keep only known columns
    df = df[list(COLUMN_MAP.values())]

    # --- conversions ---
    # timestamp → datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # total_score → numeric
    df["total_score"] = pd.to_numeric(df["total_score"], errors="coerce")

    # aptitude_test_status + graduation_status → boolean
    def yes_no_to_bool(val):
        if isinstance(val, str):
            val = val.strip().lower()
            if val in ["yes", "y", "true", "1"]:
                return True
            if val in ["no", "n", "false", "0"]:
                return False
        return None

    df["aptitude_test_status"] = df["aptitude_test_status"].map(yes_no_to_bool)
    df["graduation_status"] = df["graduation_status"].map(yes_no_to_bool)

    # add cohort + sheet label
    df["cohort"] = cohort_label
    df["sheet"] = sheet_name

    # add raw row as JSON (use original values, not converted, if you prefer)
    df["raw_row"] = df.apply(lambda r: json.dumps(r.to_dict(), default=str), axis=1)

    return df

def load_gsheet(sheet_url, cohort_label):
    print(f"Loading Google Sheet ({cohort_label})...")

    # convert share link -> export link
    file_id = sheet_url.split("/d/")[1].split("/")[0]
    xlsx_url = f"https://docs.google.com/spreadsheets/d/{file_id}/export?format=xlsx"

    # read Excel directly from URL
    all_sheets = pd.read_excel(xlsx_url, sheet_name=None)

    for sheet_name, df in all_sheets.items():
        print(f"  Processing sheet: {sheet_name}")
        df = normalize_dataframe(df, cohort_label, sheet_name)

        # remove duplicate column names
        df = df.loc[:, ~df.columns.duplicated()].copy()
        df.columns = df.columns.str.strip().str.lower()

        ### ⬇️ NOW insert each sheet separately
        df.to_sql(
            "staging_participants",
            engine,
            if_exists="append",   # append rows for each sheet
            index=False,
            method="multi",
            dtype=None
        )

if __name__ == "__main__":
    gsheet_url = "https://docs.google.com/spreadsheets/d/1NGGpjVinFZZIWV7NY95g-A8H2RNoGy5YXIxmet4aKQ4/edit?usp=sharing"
    load_gsheet(gsheet_url, "C3_ALL")

    with engine.connect() as conn:
        res = conn.execute(text("SELECT COUNT(*) FROM staging_participants"))
        print("Row count in staging_participants:", res.scalar())
