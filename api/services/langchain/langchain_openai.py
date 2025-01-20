from typing import Any, Dict, List, Optional, cast

from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAI

from .base import BaseLangChainService


class OpenAIService(BaseLangChainService):
    """LangChain integration for OpenAI."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-3.5-turbo",
        organization: Optional[str] = None,
    ):
        """Initialize the OpenAI service.

        Args:
            api_key: OpenAI API key
            model: Model name to use
            organization: Optional organization ID
        """
        self.api_key = api_key
        self.model = model
        self.organization = organization

        # Initialize models
        common_args = {
            "openai_api_key": api_key,
            "openai_organization": organization,
        }

        self.llm = OpenAI(model=model, **common_args)
        self.chat_model = ChatOpenAI(model=model, **common_args)

    async def generate_text(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        stop_sequences: Optional[List[str]] = None,
        **kwargs: Dict[str, Any],
    ) -> str:
        """Generate text using OpenAI."""
        # Configure parameters
        generation_kwargs = {}
        if max_tokens is not None:
            generation_kwargs["max_tokens"] = max_tokens
        if stop_sequences is not None:
            generation_kwargs["stop"] = stop_sequences
        generation_kwargs["temperature"] = temperature

        # Generate response
        response = await self.llm.agenerate([prompt], **generation_kwargs)
        return response.generations[0][0].text.strip()

    async def generate_chat_response(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        stop_sequences: Optional[List[str]] = None,
        **kwargs: Dict[str, Any],
    ) -> str:
        """Generate a chat response using OpenAI."""
        # Convert messages to LangChain format
        langchain_messages = []
        for msg in messages:
            if msg["role"] == "system":
                langchain_messages.append(SystemMessage(content=msg["content"]))
            elif msg["role"] == "user":
                langchain_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                langchain_messages.append(AIMessage(content=msg["content"]))

        # Configure parameters
        generation_kwargs = {}
        if max_tokens is not None:
            generation_kwargs["max_tokens"] = max_tokens
        if stop_sequences is not None:
            generation_kwargs["stop"] = stop_sequences
        generation_kwargs["temperature"] = temperature

        # Generate response
        response = await self.chat_model.agenerate(
            [langchain_messages], **generation_kwargs
        )
        return response.generations[0][0].text.strip()
