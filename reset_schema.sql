-- Error proofing the database
-- dropping tables in reverse dependency order
DROP TABLE IF EXISTS participants CASCADE;
DROP TABLE IF EXISTS staging_participants CASCADE;
DROP TABLE IF EXISTS tracks CASCADE;
DROP TABLE IF EXISTS countries CASCADE;

-- including  schema definitions
\i schema.sql
