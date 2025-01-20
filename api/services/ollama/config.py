from enum import Enum
from functools import lru_cache
from typing import Any, Dict

from pydantic import BaseModel
from pydantic_settings import BaseSettings


class OllamaModel(str, Enum):
    """Available Ollama models."""

    MISTRAL = "mistral"
    LLAMA2 = "llama2"
    CODELLAMA = "codellama"
    MISTRAL_TURKISH = "brooqs/mistral-turkish-v2"
    QWEN_7B = "qwen2.5:7b"
    QWEN_70B = "qwen2.5:70b"


MODEL_CONFIGS: Dict[OllamaModel, Dict[str, Any]] = {
    OllamaModel.MISTRAL: {
        "context_length": 8192,
        "supports_tools": True,
        "supports_vision": False,
    },
    OllamaModel.LLAMA2: {
        "context_length": 4096,
        "supports_tools": True,
        "supports_vision": False,
    },
    OllamaModel.CODELLAMA: {
        "context_length": 16384,
        "supports_tools": True,
        "supports_vision": False,
    },
    OllamaModel.MISTRAL_TURKISH: {
        "context_length": 8192,
        "supports_tools": True,
        "supports_vision": False,
    },
    OllamaModel.QWEN_7B: {
        "context_length": 32768,
        "supports_tools": True,
        "supports_vision": False,
    },
    OllamaModel.QWEN_70B: {
        "context_length": 32768,
        "supports_tools": True,
        "supports_vision": True,
    },
}


class Settings(BaseSettings):
    """Ollama settings from environment variables."""

    OLLAMA_URL: str = "http://localhost:11434"
    OLLAMA_MODEL_NAME: OllamaModel = OllamaModel.QWEN_70B

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "ignore"  # Allow extra fields for testing


ollama_env_vars = Settings()


class OllamaConfig(BaseModel):
    """Configuration for Ollama service."""

    base_url: str = ollama_env_vars.OLLAMA_URL
    default_model: OllamaModel = OllamaModel.QWEN_7B


@lru_cache()
def get_ollama_settings(
    base_url: str = ollama_env_vars.OLLAMA_URL,
    default_model: OllamaModel = ollama_env_vars.OLLAMA_MODEL_NAME,
) -> OllamaConfig:
    """Get cached Ollama settings."""
    return OllamaConfig(
        base_url=base_url,
        default_model=default_model,
    )
