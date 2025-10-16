from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    # Security / limits
    api_keys_csv: str | None = None  # comma-separated list of allowed API keys
    rate_limit_per_minute: int = 60
    
    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()