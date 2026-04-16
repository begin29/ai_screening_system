from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm_base_url: str = "http://localhost:1234/v1"
    llm_api_key: str = "lm-studio"
    llm_model: str = "mistral"
    llm_timeout: float = 60.0
    llm_max_resume_chars: int = 5000
    llm_max_jd_chars: int = 1500
    llm_max_chunk_chars: int = 4000
    llm_max_concurrency: int = 1

    database_url: str = "sqlite:///./screening.db"

    max_file_mb: int = 10

    @property
    def max_file_bytes(self) -> int:
        return self.max_file_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
