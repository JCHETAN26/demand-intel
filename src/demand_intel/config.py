"""Runtime settings, loaded from environment variables / `.env`."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="DI_", extra="ignore")

    data_dir: Path = REPO_ROOT / "data"
    evaluation_dir: Path = REPO_ROOT / "evaluation"
    database_url: str = "postgresql+psycopg://demand_intel:demand_intel@localhost:5432/demand_intel"
    mlflow_tracking_uri: str = "http://localhost:5000"
    s3_bucket: str | None = None

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def validated_dir(self) -> Path:
        return self.data_dir / "validated"

    @property
    def quarantine_dir(self) -> Path:
        return self.data_dir / "quarantine"


settings = Settings()
