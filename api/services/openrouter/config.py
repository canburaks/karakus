from enum import Enum
from functools import lru_cache
from typing import Any, Dict

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

    class Config:
        env_file = ".env.local"
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "ignore"  # Allow extra fields for testing


@lru_cache()
def get_openrouter_settings() -> Settings:
    return Settings()
