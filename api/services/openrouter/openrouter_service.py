from typing import Any, AsyncGenerator, List, Optional, TypedDict, Union, Dict, Sequence, cast

import httpx
import json
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from pydantic import BaseModel
from fastapi import HTTPException
import uuid
from datetime import datetime

from api.services.openrouter.config import (
    MODEL_CONFIGS,
    OpenRouterModel,
    get_openrouter_settings,
)
from api.models.ai import ChatMessage, ChatResponse
from api.interfaces.llm_service import LLMService

from .config import OpenRouterConfig
from api.core.logger import log


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


class OpenRouterService(LLMService):
    """
    Service for interacting with OpenRouter's API, providing access to various AI models.

    Handles chat completions and embeddings generation with support for streaming
    and tool calls.

    Attributes:
        client (httpx.AsyncClient): Async client for API requests
        default_model (OpenRouterModel): Default model to use for requests
    """

    def __init__(self, config: OpenRouterConfig):
        log.info(f"Initializing OpenRouter service with base URL: {config.base_url}")
        log.info(f"Using default_model: {config.default_model}")
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

    def get_available_models(self) -> List[str]:
        """Get list of available models for the provider"""
        return list(MODEL_CONFIGS.keys())

    def generate_text(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any
    ) -> ChatResponse:
        """Synchronous chat completion"""
        model = kwargs.get("model", self.default_model)
        try:
            with httpx.Client(base_url=self.client.base_url, headers=self.client.headers) as client:
                response = client.post(
                    "/chat/completions",
                    json={
                        "model": model,
                        "messages": [m.model_dump(exclude_unset=True) for m in messages],
                        "temperature": kwargs.get("temperature", 0.7),
                        "max_tokens": kwargs.get("max_tokens"),
                        "stream": False
                    },
                )
                response.raise_for_status()
                data = response.json()
                
                return ChatResponse(
                    id=data.get("id", f"chatcmpl-{uuid.uuid4().hex}"),
                    object="chat.completion",
                    created=data.get("created", int(datetime.now().timestamp())),
                    model=model,
                    choices=data.get("choices", []),
                    usage=data.get("usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
                )
        except httpx.HTTPError as e:
            log.error(f"HTTP error during text generation: {str(e)}", exc_info=True)
            if isinstance(e, httpx.HTTPStatusError):
                raise HTTPException(status_code=e.response.status_code, detail=f"API request failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"API request failed: {str(e)}")
        except Exception as e:
            log.error(f"Text generation failed: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Text generation failed: {str(e)}")

    async def stream_text(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any
    ) -> AsyncGenerator[str, None]:
        """Asynchronous streaming chat completion"""
        model = kwargs.get("model", self.default_model)
        is_first_chunk = True
        try:
            async with self.client.stream(
                "POST",
                "/chat/completions",
                json={
                    "model": model,
                    "messages": [m.model_dump(exclude_unset=True) for m in messages],
                    "temperature": kwargs.get("temperature", 0.7),
                    "max_tokens": kwargs.get("max_tokens"),
                    "stream": True
                },
            ) as response:
                response.raise_for_status()
                
                # Send initial chunk with role
                if is_first_chunk:
                    initial_chunk = {
                        "id": f"chatcmpl-{uuid.uuid4().hex}",
                        "object": "chat.completion.chunk",
                        "created": int(datetime.now().timestamp()),
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "delta": {"role": "assistant", "content": ""},
                            "logprobs": None,
                            "finish_reason": None
                        }]
                    }
                    yield f"data: {json.dumps(initial_chunk)}\n\n"
                    is_first_chunk = False

                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        try:
                            raw_data = json.loads(line[6:])
                            if raw_data == "[DONE]":
                                yield "data: [DONE]\n\n"
                                continue

                            # Convert to OpenAI format
                            openai_format = {
                                "id": raw_data.get("id", f"chatcmpl-{uuid.uuid4().hex}"),
                                "object": "chat.completion.chunk",
                                "created": raw_data.get("created", int(datetime.now().timestamp())),
                                "model": model,
                                "choices": []
                            }

                            for choice in raw_data.get("choices", []):
                                delta = {}
                                
                                # Handle content chunks
                                if choice.get("delta", {}).get("content"):
                                    delta["content"] = choice["delta"]["content"]
                                # Handle finish chunk
                                elif choice.get("finish_reason") == "stop":
                                    delta = {}  # Empty delta for finish chunk

                                openai_format["choices"].append({
                                    "index": choice.get("index", 0),
                                    "delta": delta,
                                    "logprobs": None,
                                    "finish_reason": choice.get("finish_reason")
                                })

                            # Only yield if there's content or it's a finish chunk
                            if openai_format["choices"] and (
                                any(c["delta"].get("content") for c in openai_format["choices"]) or
                                any(c["finish_reason"] == "stop" for c in openai_format["choices"])
                            ):
                                if "usage" in raw_data:
                                    openai_format["usage"] = raw_data["usage"]
                                yield f"data: {json.dumps(openai_format)}\n\n"

                        except json.JSONDecodeError:
                            continue

        except httpx.HTTPStatusError as e:
            error_data = json.dumps({
                "error": {
                    "message": str(e),
                    "type": "invalid_request_error",
                    "code": e.response.status_code,
                    "param": None,
                }
            })
            yield f"data: {error_data}\n\n"
        except httpx.HTTPError as e:
            log.error(f"HTTP error during streaming: {str(e)}", exc_info=True)
            if isinstance(e, httpx.HTTPStatusError):
                raise HTTPException(status_code=e.response.status_code, detail=f"API request failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"API request failed: {str(e)}")
        except Exception as e:
            log.error(f"Text streaming failed: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Text streaming failed: {str(e)}")

    def generate_object(self, prompt: str, **kwargs: Any) -> List[str]:
        """Generate structured output from a prompt"""
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            response = loop.run_until_complete(self._async_generate_object(prompt, **kwargs))
            return response
        finally:
            loop.close()

    async def _async_generate_object(self, prompt: str, **kwargs: Any) -> List[str]:
        """Internal async implementation of generate_object"""
        model = kwargs.get("model", self.default_model)
        log.info(f"Generating object with model {model}")
        try:
            response = await self.client.post(
                "/chat/completions",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": kwargs.get("temperature", 0.7),
                    "max_tokens": kwargs.get("max_tokens", 100),
                    "response_format": {"type": "json_object"},
                },
            )
            response.raise_for_status()
            data = response.json()
            return [data["choices"][0]["message"]["content"]]
        except Exception as e:
            log.error(f"Object generation failed: {str(e)}", exc_info=True)
            raise

    async def stream_object(self, prompt: str, **kwargs: Any) -> List[str]:
        """Asynchronous structured output generation"""
        response = await self._async_generate_object(prompt, **kwargs)
        return response

    def get_embeddings(
        self,
        texts: Union[str, List[str]],
        **kwargs: Any
    ) -> List[List[float]]:
        """Synchronous text embedding generation"""
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            response = loop.run_until_complete(self._async_get_embeddings(texts, **kwargs))
            return response
        finally:
            loop.close()

    async def _async_get_embeddings(
        self,
        texts: Union[str, List[str]],
        **kwargs: Any
    ) -> List[List[float]]:
        """Internal async implementation of get_embeddings"""
        if isinstance(texts, str):
            texts = [texts]
        model = kwargs.get("model", "openai/text-embedding-3-small")
        log.info(f"Generating embeddings for {len(texts)} texts using model {model}")
        try:
            response = await self.client.post(
                "/embeddings",
                json={"model": model, "input": texts},
            )
            response.raise_for_status()
            data = response.json()
            return [item["embedding"] for item in data["data"]]
        except Exception as e:
            log.error(f"Embeddings generation failed: {str(e)}", exc_info=True)
            raise

    async def aget_embeddings(
        self,
        texts: Union[str, List[str]],
        **kwargs: Any
    ) -> List[List[float]]:
        """Asynchronous text embedding generation"""
        return await self._async_get_embeddings(texts, **kwargs)


def get_openrouter_service(config: OpenRouterConfig = get_openrouter_settings()) -> OpenRouterService:
    return OpenRouterService(config=config)
