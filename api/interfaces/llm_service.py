from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Sequence, AsyncGenerator, TypeVar, Protocol

from pydantic import BaseModel
from api.models.ai import ChatMessage, ChatResponse, EmbeddingResponse

T = TypeVar('T')

class LLMService(ABC):
    """Abstract base class for LLM providers"""

    @abstractmethod
    def generate_text(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any
    ) -> ChatResponse:
        """Synchronous chat completion"""
        pass

    @abstractmethod
    async def stream_text(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any
    ) -> AsyncGenerator[str, None]:
        """Asynchronous streaming chat completion"""
        yield ""
    
    @abstractmethod
    def generate_object(self, prompt: str, **kwargs: Any) -> List[str]:
        """Generate structured output from a prompt"""
        pass

    @abstractmethod
    async def stream_object(self, prompt: str, **kwargs: Any) -> List[str]:
        """Asynchronous structured output generation from a prompt"""
        pass

    @abstractmethod
    def get_embeddings(
        self,
        texts: Union[str, List[str]],
        **kwargs: Any
    ) -> List[List[float]]:
        """Synchronous text embedding generation"""
        pass

    @abstractmethod
    async def aget_embeddings(
        self,
        texts: Union[str, List[str]],
        **kwargs: Any
    ) -> List[List[float]]:
        """Asynchronous text embedding generation"""
        pass

    @abstractmethod
    def get_available_models(self) -> List[str]:
        """Get list of available models for the provider"""
        pass
