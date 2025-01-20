from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Dict, List, Optional, Type, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class BaseLangChainService(ABC):
    """Base class for LangChain integrations."""

    @abstractmethod
    async def generate_text(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        stop_sequences: Optional[List[str]] = None,
        **kwargs: Dict[str, Any],
    ) -> str:
        """Generate text using the LangChain integration.

        Args:
            prompt: The input prompt
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            stop_sequences: Optional list of stop sequences
            **kwargs: Additional model-specific parameters

        Returns:
            Generated text response
        """
        pass

    @abstractmethod
    async def generate_chat_response(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        stop_sequences: Optional[List[str]] = None,
        **kwargs: Dict[str, Any],
    ) -> str:
        """Generate a complete chat response."""
        pass

    @abstractmethod
    async def stream_chat_response(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        stop_sequences: Optional[List[str]] = None,
        **kwargs: Dict[str, Any],
    ) -> AsyncGenerator[str, None]:
        """Stream chat response chunks."""
        pass

    @abstractmethod
    async def generate_structured_output(
        self,
        messages: List[Dict[str, str]],
        output_schema: Type[T],
        temperature: float = 0.0,
        **kwargs: Dict[str, Any],
    ) -> T:
        """Generate structured output matching the given Pydantic schema."""
        pass
