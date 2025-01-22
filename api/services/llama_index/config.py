from functools import lru_cache

from pydantic_settings import BaseSettings


class LlamaIndexSettings(BaseSettings):
    """Settings for LlamaIndex configuration"""

    BUCKET_NAME: str = ""
    BUCKET_REGION: str = ""
    BUCKET_ACCESS_ID: str = ""
    SUPABASE_URL: str = ""
    SUPABASE_KEY: str = ""

    # PDF Loader settings
    PDF_LOADER_TIMEOUT: int = 30
    PDF_LOADER_MAX_RETRIES: int = 3
    PDF_LOADER_RETRY_DELAY: int = 1
    PDF_LOADER_VERIFY_SSL: bool = True

    class Config:
        env_file: str = ".env"
        extra: str = "allow"


@lru_cache()
def get_llama_index_settings() -> LlamaIndexSettings:
    """Get cached LlamaIndex settings"""
    return LlamaIndexSettings()
