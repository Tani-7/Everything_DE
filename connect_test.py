from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os

load_dotenv()
DB_URI = os.getenv("DB_URI")
engine = create_engine(DB_URI)

with engine.connect() as conn:
    res = conn.execute(text("SELECT version();"))
    print("Postgres version:", res.scalar())
