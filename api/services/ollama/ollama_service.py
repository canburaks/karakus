import json
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

import httpx
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from pydantic import BaseModel, Field

from api.core.logger import log
from api.services.ollama.config import MODEL_CONFIGS, OllamaConfig, OllamaModel


class OllamaOptions(BaseModel):
    """Model options for Ollama requests."""

    num_keep: Optional[int] = None
    seed: Optional[int] = None
    num_predict: Optional[int] = None
    top_k: Optional[int] = 20
    top_p: Optional[float] = 0.9
    tfs_z: Optional[float] = 0.5
    typical_p: Optional[float] = 0.7
    repeat_last_n: Optional[int] = 64
    temperature: Optional[float] = 0.8
    repeat_penalty: Optional[float] = 1.1
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    mirostat: Optional[int] = 0
    mirostat_tau: Optional[float] = 5.0
    mirostat_eta: Optional[float] = 0.1
    penalize_newline: Optional[bool] = False
    stop: Optional[List[str]] = None
    numa: Optional[bool] = False
    num_ctx: Optional[int] = None
    num_batch: Optional[int] = 512
    num_gqa: Optional[int] = None
    num_gpu: Optional[int] = 1
    main_gpu: Optional[int] = 0
    low_vram: Optional[bool] = False
    f16_kv: Optional[bool] = True
    vocab_only: Optional[bool] = False
    use_mmap: Optional[bool] = True
    use_mlock: Optional[bool] = False
    rope_frequency_base: Optional[float] = 10000.0
    rope_frequency_scale: Optional[float] = 1.0
    num_thread: Optional[int] = None


class ModelDetails(BaseModel):
    """Model details from process status."""

    model: str
    digest: str
    size: int
    size_vram: Optional[int] = None
    details: Dict[str, Any]
    expires_at: Optional[str] = None


class ProcessResponse(BaseModel):
    """Response from process status."""

    models: List[ModelDetails]


class ToolCall(BaseModel):
    """Tool call information."""

    id: str = Field(..., description="Unique identifier for the tool call")
    name: str = Field(..., description="Name of the tool being called")
    arguments: Dict[str, Any] = Field(..., description="Arguments for the tool call")


class ChatMessage(BaseModel):
    """Chat message with optional tool calls."""

    role: str
    content: str
    tool_calls: Optional[List[ToolCall]] = None
    name: Optional[str] = None


class ChatResponse(BaseModel):
    """Response from chat endpoint."""

    model: str
    created_at: str
    message: ChatMessage
    done: bool
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    prompt_eval_duration: Optional[int] = None
    eval_count: Optional[int] = None
    eval_duration: Optional[int] = None


class OllamaService:
    """
    Service for interacting with Ollama's API, providing access to local LLM models.

    Handles chat completions, embeddings generation, and model management with support for streaming.

    Attributes:
        client (httpx.AsyncClient): Async client for API requests
        default_model (OllamaModel): Default model to use for requests
    """

    def __init__(self, config: OllamaConfig):
        self.client = httpx.AsyncClient(
            base_url=config.base_url,
            headers={"Content-Type": "application/json"},
            timeout=httpx.Timeout(
                12000.0, connect=60.0
            ),  # 5 minutes total timeout, 60s connect timeout
        )
        self.default_model = config.default_model

    async def create_embeddings(
        self, texts: List[str], model: Optional[OllamaModel] = None
    ) -> List[List[float]]:
        """
        Generate embeddings for given texts using specified model.

        Args:
            texts (List[str]): List of texts to generate embeddings for
            model (Optional[OllamaModel]): Model to use for embedding generation

        Returns:
            List[List[float]]: List of embedding vectors

        Raises:
            httpx.HTTPError: If API request fails
        """
        selected_model = model or self.default_model
        embeddings = []

        for text in texts:
            response = await self.client.post(
                "/api/embeddings",
                json={"model": selected_model, "prompt": text},
            )
            response.raise_for_status()
            embeddings.append(response.json()["embedding"])

        return embeddings

    async def generate(
        self,
        prompt: str,
        model: Optional[OllamaModel] = None,
        stream: bool = True,
        raw: bool = False,
        format: Optional[str] = None,
        options: Optional[OllamaOptions] = None,
        images: Optional[List[str]] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Generate text using the specified model.

        Args:
            prompt (str): The prompt to generate from
            model (Optional[OllamaModel]): Model to use
            stream (bool): Whether to stream the response
            raw (bool): Whether to use raw mode (bypass templating)
            format (Optional[str]): Response format (e.g., 'json')
            options (Optional[OllamaOptions]): Model options
            images (Optional[List[str]]): List of base64-encoded images for multimodal models

        Yields:
            str: Generated text chunks

        Raises:
            httpx.HTTPError: If API request fails
        """
        selected_model = model or self.default_model

        payload = {
            "model": selected_model,
            "prompt": prompt,
            "stream": stream,
            "raw": raw,
        }

        if format:
            payload["format"] = format

        if options:
            payload["options"] = options.model_dump(exclude_none=True)

        if images:
            payload["images"] = images

        async with self.client.stream("POST", "/generate", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                    if chunk.get("done"):
                        continue
                    if "response" in chunk:
                        yield f"0:{json.dumps(chunk['response'])}\n"
                except json.JSONDecodeError:
                    continue

    async def chat(
        self,
        messages: List[ChatCompletionMessageParam],
        model: Optional[OllamaModel] = None,
        stream: bool = True,
        format: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        images: Optional[List[str]] = None,
    ) -> Union[AsyncGenerator[str, None], ChatResponse]:
        """
        Chat with the model using a list of messages.

        Args:
            messages (List[ChatCompletionMessageParam]): Chat messages
            model (Optional[OllamaModel]): Model to use
            stream (bool): Whether to stream the response
            format (Optional[str]): Response format (e.g., 'json')
            options (Optional[Dict[str, Any]]): Model options
            tools (Optional[List[Dict[str, Any]]]): List of available tools
            images (Optional[List[str]]): List of base64-encoded images for multimodal models

        Returns:
            Union[AsyncGenerator[str, None], ChatResponse]: Streamed responses or single response

        Raises:
            httpx.HTTPError: If API request fails
        """
        try:
            selected_model = model or self.default_model

            # Convert messages to Ollama format
            formatted_messages = []
            for msg in messages:
                role = (
                    "system"
                    if msg.get("role") == "system"
                    else "user"
                    if msg.get("role") == "user"
                    else "assistant"
                )
                content = msg.get("content", "")
                formatted_messages.append({"role": role, "content": content})

            payload = {
                "model": selected_model,
                "messages": formatted_messages,
                "stream": stream,
            }

            if format:
                payload["format"] = format

            if options:
                payload["options"] = options

            if tools:
                payload["tools"] = tools

            if images:
                payload["images"] = images

            if not stream:
                response = await self.client.post("/api/chat", json=payload)
                response.raise_for_status()
                return ChatResponse(**response.json())

            async def response_generator() -> AsyncGenerator[str, None]:
                async with self.client.stream(
                    "POST",
                    "/api/chat",
                    json=payload,
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        try:
                            chunk = json.loads(line)
                            if chunk.get("done"):
                                continue
                            if "message" in chunk and "content" in chunk["message"]:
                                yield f"0:{json.dumps(chunk['message']['content'])}\n"
                        except json.JSONDecodeError:
                            continue

            return response_generator()
        except httpx.RequestError as e:
            log.error(f"Request error occurred: {str(e)}")
            raise
        except Exception as e:
            log.error(f"Unexpected error in chat: {str(e)}")
            raise

    async def pull_model(self, model: str) -> AsyncGenerator[str, None]:
        """
        Pull a model from the Ollama library.

        Args:
            model (str): Name of the model to pull

        Yields:
            str: Status updates during pull

        Raises:
            httpx.HTTPError: If API request fails
        """
        async with self.client.stream(
            "POST", "/pull", json={"name": model}
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line:
                    yield line

    async def list_models(self) -> List[Dict[str, Any]]:
        """
        List all available models.

        Returns:
            List[Dict[str, Any]]: List of model information

        Raises:
            httpx.HTTPError: If API request fails
        """
        response = await self.client.get("/api/tags")
        response.raise_for_status()
        return response.json()["models"]

    async def delete_model(self, model: str) -> Dict[str, Any]:
        """
        Delete a model.

        Args:
            model (str): Name of the model to delete

        Returns:
            Dict[str, Any]: Response from the server

        Raises:
            httpx.HTTPError: If API request fails
        """
        response = await self.client.delete("/delete", params={"name": model})
        response.raise_for_status()
        return response.json()

    async def show_model(self, model: str) -> Dict[str, Any]:
        """
        Show information about a model.

        Args:
            model (str): Name of the model

        Returns:
            Dict[str, Any]: Model information

        Raises:
            httpx.HTTPError: If API request fails
        """
        response = await self.client.post("/show", json={"name": model})
        response.raise_for_status()
        return response.json()

    async def copy_model(self, source: str, destination: str) -> Dict[str, Any]:
        """
        Copy a model.

        Args:
            source (str): Source model name
            destination (str): Destination model name

        Returns:
            Dict[str, Any]: Response from the server

        Raises:
            httpx.HTTPError: If API request fails
        """
        response = await self.client.post(
            "/copy",
            json={"source": source, "destination": destination},
        )
        response.raise_for_status()
        return response.json()

    async def create_model(
        self,
        name: str,
        modelfile: str,
        path: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Create a model from a Modelfile.

        Args:
            name (str): Name for the new model
            modelfile (str): Contents of the Modelfile
            path (Optional[str]): Path to the model files

        Yields:
            str: Status updates during creation

        Raises:
            httpx.HTTPError: If API request fails
        """
        payload = {"name": name, "modelfile": modelfile}
        if path:
            payload["path"] = path

        async with self.client.stream("POST", "/create", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line:
                    yield line

    async def get_model_status(self) -> ProcessResponse:
        """
        Get status of running models including CPU/GPU usage.

        Returns:
            ProcessResponse: Status of running models

        Raises:
            httpx.HTTPError: If API request fails
        """
        response = await self.client.get("/api/ps")
        response.raise_for_status()
        return ProcessResponse(**response.json())

    async def close(self):
        """Close the async client connection."""
        await self.client.aclose()


# Global instance
_ollama_instance = None


def get_ollama_service(config: OllamaConfig) -> OllamaService:
    """
    Get or create a singleton instance of OllamaService.

    Args:
        config (OllamaConfig): Service configuration

    Returns:
        OllamaService: Singleton service instance
    """
    global _ollama_instance
    if _ollama_instance is None:
        _ollama_instance = OllamaService(config)
    return _ollama_instance
