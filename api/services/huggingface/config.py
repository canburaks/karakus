from enum import Enum
from functools import lru_cache
from typing import Any, Dict, Optional

from pydantic import BaseModel
from pydantic_settings import BaseSettings


class SentenceTransformerModel(str, Enum):
    BERT_BASE_EMRECAN = "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr"
    MINI_LM_L12_V2 = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class TransformerModel(str, Enum):
    BERT_BASE_DBMDZ = "dbmdz/bert-base-turkish-cased"


class HuggingFaceConfig(BaseModel):
    api_key: str
    base_url: str
    default_model: str
    use_local: bool = True


class Settings(BaseSettings):
    HUGGINGFACE_API_KEY: str = ""
    HUGGINGFACE_BASE_URL: str = "https://api-inference.huggingface.co/models"
    HUGGINGFACE_LOCAL_MODEL: str = TransformerModel.BERT_BASE_DBMDZ
    HUGGINGFACE_USE_LOCAL: bool = False

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"  # Allow extra fields for testing


@lru_cache()
def get_huggingface_env_settings() -> Settings:
    return Settings()


hf_env_vars = get_huggingface_env_settings()


@lru_cache()
def get_huggingface_settings(
    api_key: str = hf_env_vars.HUGGINGFACE_API_KEY,
    base_url: str = hf_env_vars.HUGGINGFACE_BASE_URL,
    default_model: str = hf_env_vars.HUGGINGFACE_LOCAL_MODEL,
    use_local: bool = hf_env_vars.HUGGINGFACE_USE_LOCAL,
) -> HuggingFaceConfig:
    return HuggingFaceConfig(
        api_key=api_key,
        base_url=base_url,
        default_model=default_model,
        use_local=use_local,
    )
