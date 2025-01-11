from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings


class AgentsSettings(BaseSettings):
    OPENAI_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""
    GEMINI_API_KEY: str = ""
    OLLAMA_URL: str = ""
    OLLAMA_MODEL_NAME: str = ""
    OPENROUTER_API_KEY: str = ""
    DEEPSEEK_API_KEY: str = ""
    ENV: Literal["dev", "prod"] = "dev"

    class Config:
        env_file: str = ".env"
        extra: str = "allow"


@lru_cache()
def get_agents_settings() -> AgentsSettings:
    print("\n--get_agents_settings")
    return AgentsSettings()
