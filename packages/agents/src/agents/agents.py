from typing import Any, Literal, Type, TypeVar

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.models.openai import OpenAIModel

from .configs import AgentsSettings

settings = AgentsSettings()

PROVIDERS = Literal["openai", "anthropic", "gemini", "ollama", "openrouter", "deekseek"]
T = TypeVar("T", bound=BaseModel)
D = TypeVar("D", bound=BaseModel)


def create_agent(
    provider: str,
    model_name: str,
    result_type: Type[T] | None = None,
    deps_type: Type[D] | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    **kwargs: Any,
) -> Agent[D, T]:
    """Factory function to create an Agent with the appropriate model configuration."""
    match provider.lower():
        case "openai":
            model = OpenAIModel(
                model_name=model_name,
                api_key=api_key or settings.OPENAI_API_KEY,
            )
        case "anthropic":
            model = OpenAIModel(
                model_name=model_name,
                api_key=api_key or settings.ANTHROPIC_API_KEY,
            )
        case "ollama":
            model = OllamaModel(
                model_name=model_name,
                base_url=base_url or settings.OLLAMA_URL,
            )
        case "openrouter":
            model = OpenAIModel(
                model_name=model_name,
                api_key=api_key or settings.OPENROUTER_API_KEY,
                base_url=base_url or "https://openrouter.ai/api/v1",
            )
        case "deepseek":
            model = OpenAIModel(
                model_name=model_name,
                api_key=api_key or settings.DEEPSEEK_API_KEY,
                base_url=base_url or "https://api.deepseek.com",
            )
        case _:
            raise ValueError(f"Unsupported provider: {provider}")

    return Agent[D, T](model=model, **kwargs)
