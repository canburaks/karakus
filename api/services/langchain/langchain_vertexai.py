from typing import Any, Dict, List, Optional

from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_google_vertexai import ChatVertexAI, VertexAI

from .base import BaseLangChainService


class VertexAIService(BaseLangChainService):
    """LangChain integration for Google Vertex AI."""

    def __init__(
        self,
        project_id: str,
        location: str = "us-central1",
        model_name: str = "text-bison",
        credentials: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the Vertex AI service.

        Args:
            project_id: Google Cloud project ID
            location: Google Cloud region
            model_name: Model name to use
            credentials: Optional credentials dictionary
        """
        self.project_id = project_id
        self.location = location
        self.model_name = model_name

        # Initialize models
        common_args = {
            "project": project_id,
            "location": location,
            "credentials": credentials,
        }

        self.llm = VertexAI(model_name=model_name, **common_args)
        self.chat_model = ChatVertexAI(model_name=model_name, **common_args)

    async def generate_text(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        stop_sequences: Optional[List[str]] = None,
        **kwargs: Dict[str, Any],
    ) -> str:
        """Generate text using Vertex AI."""
        # Configure parameters
        generation_kwargs = {}
        if max_tokens is not None:
            generation_kwargs["max_output_tokens"] = max_tokens
        if stop_sequences is not None:
            generation_kwargs["stop_sequences"] = stop_sequences
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
        """Generate a chat response using Vertex AI."""
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
            generation_kwargs["max_output_tokens"] = max_tokens
        if stop_sequences is not None:
            generation_kwargs["stop_sequences"] = stop_sequences
        generation_kwargs["temperature"] = temperature

        # Generate response
        response = await self.chat_model.agenerate(
            [langchain_messages], **generation_kwargs
        )
        return response.generations[0][0].text.strip()
