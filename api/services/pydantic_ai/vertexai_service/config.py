from functools import lru_cache
from types import NoneType
from typing import Literal, Optional

from pydantic_settings import BaseSettings

GeminiModelName = Literal[
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b",
    "gemini-1.5-pro",
    "gemini-1.0-pro",
    "gemini-2.0-flash-exp",
]

VertexAiModelNames = Literal[
    "google-vertex:gemini-1.5-flash",
    "google-vertex:gemini-1.5-pro",
    "google-vertex:gemini-2.0-flash-exp",
]

VertexAiRegion = Literal[
    "us-central1",
    "us-east1",
    "us-east4",
    "us-south1",
    "us-west1",
    "us-west2",
    "us-west3",
    "us-west4",
    "us-east5",
    "europe-central2",
    "europe-north1",
    "europe-southwest1",
    "europe-west1",
    "europe-west2",
    "europe-west3",
    "europe-west4",
    "europe-west6",
    "europe-west8",
    "europe-west9",
    "europe-west12",
    "africa-south1",
    "asia-east1",
    "asia-east2",
    "asia-northeast1",
    "asia-northeast2",
    "asia-northeast3",
    "asia-south1",
    "asia-southeast1",
    "asia-southeast2",
    "australia-southeast1",
    "australia-southeast2",
    "me-central1",
    "me-central2",
    "me-west1",
    "northamerica-northeast1",
    "northamerica-northeast2",
    "southamerica-east1",
    "southamerica-west1",
]


class GoogleVertexAISettings(BaseSettings):
    GEMINI_API_KEY: Optional[str] = ""
    VERTEXAI_REGION: VertexAiRegion = "europe-west2"
    VERTEXAI_MODEL: GeminiModelName = "gemini-2.0-flash-exp"
    VERTEXAI_SERVICE_ACCOUNT_FILE: str = ""

    class Config:
        env_file: str = ".env"
        extra: str = "allow"


@lru_cache()
def get_vertexai_settings() -> GoogleVertexAISettings:
    return GoogleVertexAISettings()
