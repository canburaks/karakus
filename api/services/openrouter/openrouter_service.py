import json
from typing import AsyncGenerator, List, Optional, TypedDict

import httpx
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from pydantic import BaseModel

from api.services.openrouter.config import (
    MODEL_CONFIGS,
    OpenRouterModel,
    get_openrouter_settings,
)

from .config import OpenRouterConfig


class ToolCall(TypedDict):
    """
    Type definition for tool calls in chat completions.

    Attributes:
        id (str): Unique identifier for the tool call
        name (str): Name of the tool being called
        arguments (str): JSON string of arguments for the tool
    """

    id: str
    name: str
    arguments: str


class OpenRouterService:
    """
    Service for interacting with OpenRouter's API, providing access to various AI models.

    Handles chat completions and embeddings generation with support for streaming
    and tool calls.

    Attributes:
        client (httpx.AsyncClient): Async client for API requests
        default_model (OpenRouterModel): Default model to use for requests
    """

    def __init__(self, config: OpenRouterConfig):
        self.client = httpx.AsyncClient(
            base_url=config.base_url,
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "HTTP-Referer": "http://localhost:3000",
                "X-Title": "AI SDK Python Streaming",
            },
            timeout=60.0,
        )
        self.default_model = config.default_model

    async def create_embeddings(
        self, texts: List[str], model: str = "openai/text-embedding-3-small"
    ) -> List[List[float]]:
        """
        Generate embeddings for given texts using specified model.

        Args:
            texts (List[str]): List of texts to generate embeddings for
            model (str): Model to use for embedding generation

        Returns:
            List[List[float]]: List of embedding vectors

        Raises:
            httpx.HTTPError: If API request fails
        """
        response = await self.client.post(
            "/embeddings", json={"model": model, "input": texts}
        )
        response.raise_for_status()
        return response.json()["data"]

    async def stream_chat(
        self,
        messages: List[ChatCompletionMessageParam],
        model: Optional[OpenRouterModel] = None,
        tools: Optional[List[dict]] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Stream chat completions with optional tool support.

        Args:
            messages (List[ChatCompletionMessageParam]): Chat messages
            model (Optional[OpenRouterModel]): Model to use
            tools (Optional[List[dict]]): List of tools available to the model

        Yields:
            str: Chunks of the response in a streaming format

        Raises:
            ValueError: If model doesn't support tools but tools are provided
            httpx.HTTPError: If API request fails
        """
        selected_model = model or self.default_model
        model_config = MODEL_CONFIGS[selected_model]

        if tools and not model_config["supports_tools"]:
            raise ValueError(f"Model {selected_model} does not support tools")

        payload = {"model": selected_model, "messages": messages, "stream": True}

        if tools:
            payload["tools"] = tools

        async with self.client.stream(
            "POST", "/chat/completions", json=payload
        ) as response:
            response.raise_for_status()
            draft_tool_calls: List[ToolCall] = []
            draft_tool_calls_index = -1

            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        continue

                    try:
                        chunk = json.loads(data)
                        choice = chunk["choices"][0]

                        if choice.get("finish_reason") == "stop":
                            continue
                        elif choice.get("finish_reason") == "tool_calls":
                            for tool_call in draft_tool_calls:
                                yield f'9:{{"toolCallId":"{tool_call["id"]}","toolName":"{tool_call["name"]}","args":{tool_call["arguments"]}}}\n'

                        delta = choice.get("delta", {})
                        if delta.get("tool_calls"):
                            for tool_call in delta["tool_calls"]:
                                id = tool_call.get("id")
                                function = tool_call.get("function", {})
                                name = function.get("name")
                                arguments = function.get("arguments", "")

                                if id:
                                    draft_tool_calls_index += 1
                                    draft_tool_calls.append(
                                        {"id": id, "name": name, "arguments": ""}
                                    )
                                else:
                                    draft_tool_calls[draft_tool_calls_index][
                                        "arguments"
                                    ] += arguments
                        elif delta.get("content"):
                            yield f'0:{json.dumps(delta["content"])}\n'
                    except json.JSONDecodeError:
                        continue

    async def close(self):
        await self.client.aclose()


def get_openrouter_service(config: OpenRouterConfig) -> OpenRouterService:
    return OpenRouterService(config)
