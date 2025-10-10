"""
models/db.py
SQLAlchemy engine helper for PostgreSQL. Use get_engine() to get a shared engine.
"""
from sqlalchemy import create_engine
from sqlalchemy.engine.url import URL
from urllib.parse import quote_plus
import os
from dotenv import load_dotenv

load_dotenv()

# prefer DB_URI environment variable (full sqlalchemy URL)
DB_URI = os.getenv("DB_URI")
if not DB_URI:
    # fallback to manual components (for convenience)
    user = os.getenv("POSTGRES_USER", "postgres")
    password = os.getenv("POSTGRES_PASSWORD", "")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "postgres")
    # ensure password safe quoting
    DB_URI = f"postgresql+psycopg2://{user}:{quote_plus(password)}@{host}:{port}/{db}"

_engine = None

def get_engine():
    """Lazily create a SQLAlchemy engine and return it."""
    global _engine
    if _engine is None:
        _engine = create_engine(DB_URI, future=True, echo=False)
    return _engine
