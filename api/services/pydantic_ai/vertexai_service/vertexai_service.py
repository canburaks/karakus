from typing import Any, Generic, Optional, Sequence, TypeVar

from pandas import Series
from pydantic_ai.agent import Agent
from pydantic_ai.models import AgentModel, KnownModelName
from pydantic_ai.models.vertexai import VertexAIModel
from pydantic_ai.result import ResultData
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import Tool, ToolFuncEither

from .config import GeminiModelName, get_vertexai_settings

# Define generic type variables
D = TypeVar("D")  # Dependency type
R = TypeVar("R")  # Result type
_vertexai_settings = get_vertexai_settings()


class VertexAIService:
    def __init__(
        self,
        model_name: GeminiModelName = _vertexai_settings.VERTEXAI_MODEL,
        api_key: Optional[str] = None,
    ):
        self.model = VertexAIModel(
            model_name=model_name,
            service_account_file=_vertexai_settings.VERTEXAI_SERVICE_ACCOUNT_FILE,
        )
        self.api_key: Optional[str] = api_key

    def get_agent[D, R](self, **kwargs) -> Agent[D, R]:  # type: ignore
        return Agent[D, R](model=self.model, **kwargs)
