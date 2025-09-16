-- a) Staging participants (lookup)
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

-- b) participants (core table)
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
    sheet TEXT\c everything_de
);

-- c) tracks (lookup)
CREATE TABLE tracks (
    id SERIAL PRIMARY KEY,
    track_name TEXT UNIQUE NOT NULL
);

-- d) countries (lookup)
CREATE TABLE countries (
    id SERIAL PRIMARY KEY,
    country_name TEXT UNIQUE NOT NULL
);
