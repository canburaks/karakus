import json
from enum import Enum
from typing import Any, AsyncGenerator, Dict, List, Optional, Type, TypeVar

from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_community.llms import Ollama
from langchain_ollama import ChatOllama, OllamaLLM
from pydantic import BaseModel

from .base import BaseLangChainService, T


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


class OllamaService(BaseLangChainService):
    """LangChain integration for Ollama."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model_name: str = OllamaModel.QWEN_7B,
        temperature: float = 0.0,
    ):
        """Initialize the Ollama service.

        Args:
            base_url: Ollama API base URL
            model: Model name to use
        """
        self.base_url = base_url
        self.model_name = model_name
        self.model = OllamaLLM(model=model_name, base_url=base_url)
        self.llm = ChatOllama(model=model_name, temperature=temperature)
        self.json_llm = ChatOllama(
            model=model_name,
            temperature=0.0,
            format="json",
            num_predict=8000,
            num_ctx=4096,
            num_thread=6,
        )

    async def generate_text(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        stop_sequences: Optional[List[str]] = None,
        **kwargs: Dict[str, Any],
    ) -> str:
        """Generate text using Ollama."""
        # Configure parameters
        generation_kwargs = {}
        if max_tokens is not None:
            generation_kwargs["max_tokens"] = max_tokens
        if stop_sequences is not None:
            generation_kwargs["stop"] = stop_sequences
        generation_kwargs["temperature"] = temperature

        # Generate response
        response = await self.model.agenerate([prompt], **generation_kwargs)
        return response.generations[0][0].text.strip()

    async def generate_chat_response(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs: Dict[str, Any],
    ) -> str:
        """Generate a complete chat response."""
        langchain_messages = []
        for msg in messages:
            if msg["role"] == "system":
                langchain_messages.append(SystemMessage(content=msg["content"]))
            elif msg["role"] == "user":
                langchain_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                langchain_messages.append(AIMessage(content=msg["content"]))

        generation_kwargs = {}
        if max_tokens is not None:
            generation_kwargs["max_tokens"] = max_tokens
        if stop_sequences is not None:
            generation_kwargs["stop"] = stop_sequences
        if temperature is not None:
            generation_kwargs["temperature"] = temperature

        response = await self.llm.agenerate([langchain_messages], **generation_kwargs)
        return response.generations[0][0].text.strip()

    async def stream_chat_response(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs: Dict[str, Any],
    ) -> AsyncGenerator[str, None]:
        """Stream chat response chunks."""
        langchain_messages = []
        for msg in messages:
            if msg["role"] == "system":
                langchain_messages.append(SystemMessage(content=msg["content"]))
            elif msg["role"] == "user":
                langchain_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                langchain_messages.append(AIMessage(content=msg["content"]))

        generation_kwargs = {}
        if max_tokens is not None:
            generation_kwargs["max_tokens"] = max_tokens
        if stop_sequences is not None:
            generation_kwargs["stop"] = stop_sequences

        async for chunk in self.llm.astream(langchain_messages, **generation_kwargs):
            if isinstance(chunk.content, str):
                yield chunk.content
            elif isinstance(chunk.content, list):
                for item in chunk.content:
                    if isinstance(item, str):
                        yield item

    async def generate_structured_output(
        self,
        message: str,
        output_schema: Type[T],
        **kwargs: Dict[str, Any],
    ) -> T:
        """Generate structured output matching the given Pydantic schema."""

        # Generate response with minimal parameters
        structured_llm = self.json_llm.with_structured_output(
            schema=output_schema, method="json_schema"
        )

        # Get response and ensure it's properly typed
        response = structured_llm.invoke(message)
        if isinstance(response, output_schema):
            return response
        elif isinstance(response, dict):
            return output_schema.model_validate(response)
        else:
            raise ValueError(f"Unexpected response type: {type(response)}")
