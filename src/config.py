# src/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

class Settings(BaseSettings):
    # main connection details
    postgres_user: str = "postgres"
    postgres_password: str = "Tani-7"
    postgres_db: str = "everything_de"
    postgres_host: str = "localhost"
    postgres_port: int = 5432

    # custom vars
    db_uri: str | None = None
    model_dir: str = "src/models"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"  # ignore unrelated vars like other app envs
    )

    @property
    def db_url(self):
        """return the effective db connection url"""
        if self.db_uri:
            return self.db_uri
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

@lru_cache()
def get_settings():
    """cache settings instance so it’s only loaded once"""
    return Settings()

settings = get_settings()
