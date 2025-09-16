#!/usr/bin/env python3
"""
eda.py - Exploratory Data Analysis for mentorship dataset.
Saves figures to reports/figures and summary CSV/JSON to reports/.
Run: python eda.py
"""

import os
import json
import logging
from datetime import datetime, timezone

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# optional stats
from scipy import stats

load_dotenv()
DB_URI = os.getenv("DB_URI")
if not DB_URI:
    raise RuntimeError("DB_URI not set in .env")

# output paths
OUT_DIR = "reports"
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("eda")

engine = create_engine(DB_URI, future=True)

# ----- helpers -----
def parse_age_range(age):
    """Return (age_bucket, age_min, age_max). Similar logic to ETL parse_age_range."""
    import re
    if pd.isna(age) or age is None:
        return (None, None, None)
    a = str(age).strip().lower()
    if '-' in a:
        parts = re.findall(r'(\d+)', a)
        if len(parts) >= 2:
            return (a.title(), int(parts[0]), int(parts[1]))
    if '+' in a:
        m = re.findall(r'(\d+)', a)
        if m:
            return (f"{m[0]}+", int(m[0]), None)
    if 'under' in a:
        m = re.findall(r'(\d+)', a)
        if m:
            return (f"under {m[0]}", None, int(m[0]) - 1)
    return (a.title(), None, None)

def ensure_numeric(df, col):
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    else:
        df[col] = np.nan

def save_fig(fig, fname):
    p = os.path.join(FIG_DIR, fname)
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    logger.info("saved figure: %s", p)

# ----- load cleaned data with human-readable track/country names -----
QUERY = """
SELECT p.*,
       t.track_name,
       c.country_name
FROM participants p
LEFT JOIN tracks t ON p.track_id = t.id
LEFT JOIN countries c ON p.country_id = c.id
"""
df = pd.read_sql(QUERY, engine)
logger.info("loaded %d rows from participants (joined with tracks/countries)", len(df))

# ----- data quality & initial diagnostics -----
dq = {
    "fetched_at": datetime.now(timezone.utc).isoformat(),
    "rows": len(df),
    "null_counts": {},
    "duplicates_id_no": 0,
    "numeric_ranges": {}
}

for col in ['id_no', 'total_score', 'graduation_status', 'hours_per_week', 'years_experience']:
    dq['null_counts'][col] = int(df[col].isna().sum()) if col in df.columns else None

if 'id_no' in df.columns:
    dq['duplicates_id_no'] = int(df['id_no'].duplicated().sum())

# quick numeric summaries
for col in ['total_score', 'hours_per_week', 'years_experience']:
    if col in df.columns:
        series = pd.to_numeric(df[col], errors='coerce').dropna()
        if len(series) > 0:
            dq['numeric_ranges'][col] = {
                "count": int(series.count()),
                "min": float(series.min()),
                "max": float(series.max()),
                "mean": float(series.mean()),
                "median": float(series.median()),
                "std": float(series.std())
            }
        else:
            dq['numeric_ranges'][col] = None

# save DQ JSON
with open(os.path.join(OUT_DIR, "data_quality_eda.json"), "w") as fh:
    json.dump(dq, fh, indent=2)
logger.info("saved data quality summary to reports/data_quality_eda.json")

# ----- feature engineering for EDA -----
# numeric casting
ensure_numeric(df, 'total_score')
ensure_numeric(df, 'hours_per_week')
ensure_numeric(df, 'years_experience')

# graduation boolean ensure
if 'graduation_status' in df.columns:
    df['graduated'] = df['graduation_status'].apply(lambda x: True if x in (1, True, 't', 'true', 'True', 'Yes', 'yes', 'YES') else (False if pd.notna(x) else None))
else:
    df['graduated'] = None

# age parsing
age_vals = df.get('age_range', pd.Series([None]*len(df)))
parsed = age_vals.apply(lambda x: pd.Series(parse_age_range(x), index=['age_bucket','age_min','age_max']))
df = pd.concat([df, parsed], axis=1)

# standardized country/track names already in track_name/country_name
if 'track_name' not in df.columns:
    df['track_name'] = df.get('track_applied')

if 'country_name' not in df.columns:
    df['country_name'] = df.get('country')

# ----- descriptive tables -----
# counts by country/gender/age_bucket/track
counts_country = df['country_name'].value_counts(dropna=True).rename_axis('country').reset_index(name='participants')
counts_gender = df['gender'].value_counts(dropna=True).rename_axis('gender').reset_index(name='participants')
counts_age = df['age_bucket'].value_counts(dropna=True).rename_axis('age_bucket').reset_index(name='participants')
counts_track = df['track_name'].value_counts(dropna=True).rename_axis('track').reset_index(name='participants')

counts_country.to_csv(os.path.join(OUT_DIR, "participants_by_country.csv"), index=False)
counts_gender.to_csv(os.path.join(OUT_DIR, "participants_by_gender.csv"), index=False)
counts_age.to_csv(os.path.join(OUT_DIR, "participants_by_agebucket.csv"), index=False)
counts_track.to_csv(os.path.join(OUT_DIR, "participants_by_track.csv"), index=False)
logger.info("saved participant count CSVs to reports/")

# summary stats
num_cols = ['total_score', 'hours_per_week', 'years_experience']
summary = df[num_cols].describe().transpose()
summary['median'] = df[num_cols].median()
summary.to_csv(os.path.join(OUT_DIR, "summary_stats.csv"))
logger.info("saved summary stats to reports/summary_stats.csv")

# graduation rates by group (track, country, age_bucket)
def graduation_rate_groupby(df, group_col):
    g = df.groupby(group_col).agg(total=('id_no','count'), graduates=('graduated', lambda s: int(s.sum() if s.notna().any() else 0)))
    g['graduation_rate'] = (g['graduates'] / g['total']).fillna(0)
    return g.reset_index()

grad_by_track = graduation_rate_groupby(df, 'track_name')
grad_by_country = graduation_rate_groupby(df, 'country_name')
grad_by_age = graduation_rate_groupby(df, 'age_bucket')

grad_by_track.to_csv(os.path.join(OUT_DIR, "graduation_rates_by_track.csv"), index=False)
grad_by_country.to_csv(os.path.join(OUT_DIR, "graduation_rates_by_country.csv"), index=False)
grad_by_age.to_csv(os.path.join(OUT_DIR, "graduation_rates_by_agebucket.csv"), index=False)
logger.info("saved graduation rate CSVs to reports/")

# ----- PLOTS -----
# 1) bar: graduation rate by track (sorted)
fig, ax = plt.subplots(figsize=(8,4))
g = grad_by_track.sort_values('graduation_rate', ascending=False)
ax.bar(g['track_name'].astype(str), g['graduation_rate'])
ax.set_xticklabels(g['track_name'], rotation=45, ha='right')
ax.set_ylabel("Graduation rate")
ax.set_title("Graduation rate by track")
save_fig(fig, "viz_1_graduation_by_track.png")

# 2) histogram: total_score distribution with mean & median lines
fig, ax = plt.subplots(figsize=(6,4))
scores = df['total_score'].dropna()
ax.hist(scores, bins=20)
ax.axvline(scores.mean(), linestyle='--', linewidth=1)
ax.axvline(scores.median(), linestyle=':', linewidth=1)
ax.set_xlabel("Total score")
ax.set_ylabel("Count")
ax.set_title("Distribution of total_score (mean = dashed, median = dotted)")
save_fig(fig, "viz_2_total_score_hist.png")

# 3) scatter: total_score vs hours_per_week, split by graduated
fig, ax = plt.subplots(figsize=(6,5))
mask_grad = df['graduated'] == True
mask_not = df['graduated'] == False
ax.scatter(df.loc[mask_not, 'hours_per_week'], df.loc[mask_not, 'total_score'], label='Not graduated', alpha=0.7, s=20)
ax.scatter(df.loc[mask_grad, 'hours_per_week'], df.loc[mask_grad, 'total_score'], label='Graduated', alpha=0.7, s=20)
ax.set_xlabel("Hours per week")
ax.set_ylabel("Total score")
ax.set_title("Total score vs Hours per week (graduated vs not)")
ax.legend()
save_fig(fig, "viz_3_score_vs_hours_scatter.png")

# 4) violin: score by graduation_status
fig, ax = plt.subplots(figsize=(6,4))
groups = [
    df.loc[df['graduated']==True, 'total_score'].dropna().values,
    df.loc[df['graduated']==False, 'total_score'].dropna().values
]
ax.violinplot(groups, showmeans=True)
ax.set_xticks([1,2])
ax.set_xticklabels(['Graduated', 'Not graduated'])
ax.set_ylabel("Total score")
ax.set_title("Total score distribution by graduation status (violin)")
save_fig(fig, "viz_4_score_violin_by_graduation.png")

# 5) bar: participants by country (top 10)
top_countries = counts_country.head(10)
fig, ax = plt.subplots(figsize=(8,4))
ax.bar(top_countries['country'], top_countries['participants'])
ax.set_xticklabels(top_countries['country'], rotation=45, ha='right')
ax.set_ylabel("Participants")
ax.set_title("Top 10 participant countries")
save_fig(fig, "viz_5_participants_by_country_top10.png")

# ----- quick stats: t-test & chi-square where possible -----
tt_result = None
chi2_result = None
try:
    grad_scores = df.loc[df['graduated']==True, 'total_score'].dropna()
    nongrad_scores = df.loc[df['graduated']==False, 'total_score'].dropna()
    if len(grad_scores) > 2 and len(nongrad_scores) > 2:
        tt_result = stats.ttest_ind(grad_scores, nongrad_scores, equal_var=False, nan_policy='omit')
        logger.info("t-test on total_score (grad vs not): stat=%.3f p=%.3f", tt_result.statistic, tt_result.pvalue)
except Exception as e:
    logger.warning("t-test failed: %s", e)

# chi-square: track_name vs graduation (if both present)
try:
    ct = pd.crosstab(df['track_name'], df['graduated'])
    if ct.shape[0] > 1 and ct.shape[1] > 1:
        chi2_result = stats.chi2_contingency(ct)
        logger.info("chi2 test track vs graduation: chi2=%.3f p=%.3g", chi2_result[0], chi2_result[1])
except Exception as e:
    logger.warning("chi-square test failed: %s", e)

# Save summary JSON for EDA
eda_summary = {
    "dq": dq,
    "ttest_total_score_grad_vs_not": {
        "statistic": float(tt_result.statistic) if tt_result else None,
        "pvalue": float(tt_result.pvalue) if tt_result else None
    } if tt_result else None,
    "chi2_track_vs_grad": {
        "chi2": float(chi2_result[0]) if chi2_result else None,
        "pvalue": float(chi2_result[1]) if chi2_result else None
    } if chi2_result else None
}
with open(os.path.join(OUT_DIR, "eda_summary.json"), "w") as fh:
    json.dump(eda_summary, fh, indent=2)
logger.info("saved EDA summary to reports/eda_summary.json")

logger.info("EDA complete. Figures and CSVs are in the reports/ directory.")
