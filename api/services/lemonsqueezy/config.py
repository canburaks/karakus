from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings


class LemonSqueezySettings(BaseSettings):
    """Settings for LemonSqueezy API integration."""

    LEMONSQUEEZY_API_KEY: str = ""
    LEMONSQUEEZY_STORE_ID: str = ""
    LEMONSQUEEZY_WEBHOOK_SECRET: Optional[str] = None
    LEMONSQUEEZY_BASE_URL: str = "https://api.lemonsqueezy.com/v1"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


@lru_cache()
def get_lemonsqueezy_settings() -> LemonSqueezySettings:
    """Get cached LemonSqueezy settings."""
    return LemonSqueezySettings()
