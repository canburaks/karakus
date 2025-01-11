from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    HUGGINGFACE_API_KEY: str = ""
    HUGGINGFACE_LOCAL_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    HUGGINGFACE_USE_LOCAL: bool = False

    class Config:
        env_file = ".env.local"
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "ignore"  # Allow extra fields for testing


@lru_cache()
def get_huggingface_settings() -> Settings:
    return Settings()
