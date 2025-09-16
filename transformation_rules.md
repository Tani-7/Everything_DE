# transform_rules.md

## Purpose
Documented mapping of how fields from staging_participants are cleaned and moved into participants.

## Column mapping
- `id_no` -> participants.id_no (text, required). If missing, row is skipped and logged.
- `timestamp` -> participants.timestamp (TIMESTAMP). parsed with pandas.to_datetime(errors='coerce').
- `age_range` -> participants.age_range (text). additionally parse into age_min/age_max during EDA if needed.
- `gender` -> normalized to 'female'/'male'/'non-binary'/TitleCase fallback.
- `country` -> normalized to lowercase-title mapping using `standardize_country()`; stored in countries lookup.
- `track_applied` -> canonical track_name in `tracks` lookup.
- `years_experience` -> numeric (float). parse first numeric or map 'less than 1' -> 0.5.
- `hours_per_week` -> numeric (float). if given as range '10-15' -> mean.
- `skill_level` -> map textual levels to numeric scores (beginner->1, intermediate->5, advanced->8, expert->10) or None when unknown.
- `aptitude_test_status` -> boolean: yes/true/1 -> True; no/false/0 -> False; otherwise NULL.
- `total_score` -> numeric (float).
- `graduation_status` -> boolean similarly to aptitude.

## Lookups
- tracks: store unique `track_applied` values. Upsert using `ON CONFLICT DO NOTHING`.
- countries: store normalized country_name. Upsert using `ON CONFLICT DO NOTHING`.

## Idempotency
- Participants are upserted by `id_no` with `ON CONFLICT (id_no) DO UPDATE`.
- staging rows are left intact; optional `row_hash` and `processed` columns added to mark progress.

## Quality checks
- Produce `data_quality_report_pre_transform.json` with stats:
  - total_rows
  - missing counts for critical fields (id_no, total_score, graduation_status)
  - duplicates in id_no
  - sample 10 problematic rows
