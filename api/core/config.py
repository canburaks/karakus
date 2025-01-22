from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings


class AppSettings(BaseSettings):
    ENV: Literal["dev", "prod"] = "dev"
    BUCKET_NAME: str = ""
    BUCKET_REGION: str = ""
    BUCKET_ACCESS_ID: str = ""
    BUCKET_SECRET_KEY: str = ""

    class Config:
        env_file: str = ".env"
        extra: str = "allow"


@lru_cache()
def get_app_settings() -> AppSettings:
    return AppSettings()
