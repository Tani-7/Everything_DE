DB_USER=postgres
DB_NAME=everything_de

# run schema.sql to (re)create tables
schema:
	psql -U $(DB_USER) -d $(DB_NAME) -f schema.sql

# drop & rebuild everything using reset_schema.sql
reset:
	psql -U $(DB_USER) -d $(DB_NAME) -f reset_schema.sql

# open interactive psql shell
psql:
	psql -U $(DB_USER) -d $(DB_NAME)

# run ingestion script (python)
ingest:
	python ingest_raw.py
