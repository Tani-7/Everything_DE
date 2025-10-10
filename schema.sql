-- schema.sql
-- clean creation order, safe for resets

-- drop in dependency order
DROP TABLE IF EXISTS graduation_outcomes CASCADE;
DROP TABLE IF EXISTS participants CASCADE;
DROP TABLE IF EXISTS staging_participants CASCADE;
DROP TABLE IF EXISTS tracks CASCADE;
DROP TABLE IF EXISTS countries CASCADE;

-- d) countries (lookup, must come before participants)
CREATE TABLE countries (
    id SERIAL PRIMARY KEY,
    country_name TEXT UNIQUE NOT NULL
);

-- c) tracks (lookup, must come before participants)
CREATE TABLE tracks (
    id SERIAL PRIMARY KEY,
    track_name TEXT UNIQUE NOT NULL
);

-- a) staging participants (raw dump, no FKs)
CREATE TABLE staging_participants (
    timestamp TIMESTAMP,
    id_no TEXT,
    age_range TEXT,
    gender TEXT,
    country TEXT,
    heard_about TEXT,
    years_experience TEXT,
    track_applied TEXT,
    hours_per_week TEXT,
    main_aim TEXT,
    motivation TEXT,
    skill_level TEXT,
    aptitude_test_status BOOLEAN,
    total_score NUMERIC(5,2),
    graduation_status BOOLEAN,
    cohort TEXT,
    sheet TEXT,
    raw_row JSONB
);

-- b) participants (core table, normalized FKs to lookups)
CREATE TABLE participants (
    id SERIAL PRIMARY KEY,
    id_no TEXT UNIQUE NOT NULL,
    timestamp TIMESTAMP,
    age_range TEXT,
    gender TEXT,
    country_id INT REFERENCES countries(id),
    heard_about TEXT,
    years_experience NUMERIC,
    track_id INT REFERENCES tracks(id),
    hours_per_week NUMERIC,
    main_aim TEXT,
    motivation TEXT,
    skill_level TEXT,
    aptitude_test_status BOOLEAN,
    total_score NUMERIC,
    graduation_status BOOLEAN,
    cohort TEXT,
    sheet TEXT
);

-- e) graduation outcomes (derived)
CREATE TABLE graduation_outcomes (
    id SERIAL PRIMARY KEY,
    participant_id INT REFERENCES participants(id),
    total_score NUMERIC(5,2),
    graduation_status BOOLEAN,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
