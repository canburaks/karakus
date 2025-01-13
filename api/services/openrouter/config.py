from enum import Enum
from functools import lru_cache
from typing import Any, Dict

from pydantic import BaseModel
from pydantic_settings import BaseSettings


class ModelProvider(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GOOGLE = "google"
    MISTRAL = "mistral"
    QWEN = "qwen"


class OpenRouterModel(str, Enum):
    CLAUDE_3_SONNET = "anthropic/claude-3-sonnet"
    CLAUDE_3_OPUS = "anthropic/claude-3-opus"
    GPT_4 = "openai/gpt-4"
    GPT_4_TURBO = "openai/gpt-4-turbo-preview"
    GEMINI_PRO = "google/gemini-pro"
    MISTRAL_LARGE = "mistral/mistral-large"
    QWEN_70B = "qwen/qwen-2.5-72b-instruct"


MODEL_CONFIGS: Dict[OpenRouterModel, Dict[str, Any]] = {
    OpenRouterModel.CLAUDE_3_SONNET: {
        "context_length": 16000,
        "provider": ModelProvider.ANTHROPIC,
        "supports_tools": True,
        "supports_vision": True,
    },
    OpenRouterModel.CLAUDE_3_OPUS: {
        "context_length": 32000,
        "provider": ModelProvider.ANTHROPIC,
        "supports_tools": True,
        "supports_vision": True,
    },
    OpenRouterModel.GPT_4_TURBO: {
        "context_length": 128000,
        "provider": ModelProvider.OPENAI,
        "supports_tools": True,
        "supports_vision": True,
    },
    OpenRouterModel.QWEN_70B: {
        "context_length": 128000,
        "provider": ModelProvider.QWEN,
        "supports_tools": True,
        "supports_vision": True,
    },
}


class Settings(BaseSettings):
    OPENROUTER_API_KEY: str = ""
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    OPENROUTER_DEFAULT_MODEL: OpenRouterModel = OpenRouterModel.QWEN_70B

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "ignore"  # Allow extra fields for testing


openrouter_env_vars = Settings()


class OpenRouterConfig(BaseModel):
    api_key: str
    base_url: str = "https://openrouter.ai/api/v1"
    default_model: OpenRouterModel = OpenRouterModel.QWEN_70B


@lru_cache()
def get_openrouter_settings(
    api_key: str = openrouter_env_vars.OPENROUTER_API_KEY,
    base_url: str = openrouter_env_vars.OPENROUTER_BASE_URL,
    default_model: OpenRouterModel = openrouter_env_vars.OPENROUTER_DEFAULT_MODEL,
) -> OpenRouterConfig:
    return OpenRouterConfig(
        api_key=api_key,
        base_url=base_url,
        default_model=default_model,
    )
