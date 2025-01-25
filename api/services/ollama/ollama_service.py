from typing import Any, AsyncGenerator, Dict, List, Optional, Union, Sequence

import httpx
import json
import asyncio
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from pydantic import BaseModel, Field
import uuid
from datetime import datetime

from api.core.logger import log
from api.models.ai import ChatMessage, ChatResponse, Message, Role
from api.interfaces.llm_service import LLMService
from api.services.ollama.config import MODEL_CONFIGS, OllamaConfig, OllamaModel
from api.services.ollama.config import get_ollama_settings


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


class OllamaService(LLMService):
    def __init__(self, config: OllamaConfig = get_ollama_settings()):
        log.info(f"Initializing Ollama service with base URL: {config.base_url}")
        self.config = config
        self.client = httpx.AsyncClient(
            base_url=self.config.base_url,
            headers={"Content-Type": "application/json"},
            timeout=httpx.Timeout(12000.0, connect=60.0),
        )
        self.default_model = self.config.default_model

    def generate_text(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any
    ) -> ChatResponse:
        """Synchronous chat completion"""
        model = kwargs.get("model", self.default_model)
        log.debug(f"Generating text with model {model}")
        try:
            with httpx.Client(
                base_url=self.client.base_url,
                headers=self.client.headers,
                timeout=self.client.timeout
            ) as client:
                response = client.post(
                    "/api/chat",
                    json={
                        "model": model,
                        "messages": [{"role": msg.role, "content": msg.content} for msg in messages],
                        "stream": False
                    },
                )
                response.raise_for_status()
                data = response.json()
                content = data.get("message", {}).get("content", "")
                return ChatResponse(
                    id=f"chatcmpl-{uuid.uuid4().hex}",
                    object="chat.completion",
                    created=int(datetime.now().timestamp()),
                    model=str(model),
                    choices=[{
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop"
                    }],
                    usage={"prompt_tokens": 0, "completion_tokens": data.get("total_tokens", 0), "total_tokens": data.get("total_tokens", 0)}
                )
        except Exception as e:
            log.error(f"Text generation failed: {str(e)}", exc_info=True)
            raise

    async def _async_generate_text(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any
    ) -> ChatResponse:
        """Internal async implementation of generate_text"""
        response = await self.chat(
            messages=[{"role": msg.role, "content": msg.content} for msg in messages],
            model=kwargs.get("model", self.default_model),
            stream=False
        )
        if isinstance(response, dict):
            content = response.get("message", {}).get("content", "")
            return ChatResponse(
                id=f"chatcmpl-{uuid.uuid4().hex}",
                object="chat.completion",
                created=int(datetime.now().timestamp()),
                model=str(kwargs.get("model", self.default_model)),
                choices=[{
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop"
                }],
                usage={"prompt_tokens": 0, "completion_tokens": response.get("total_tokens", 0), "total_tokens": response.get("total_tokens", 0)}
            )
        return ChatResponse(
            id=f"chatcmpl-{uuid.uuid4().hex}",
            object="chat.completion",
            created=int(datetime.now().timestamp()),
            model=str(kwargs.get("model", self.default_model)),
            choices=[{
                "index": 0,
                "message": {"role": "assistant", "content": ""},
                "finish_reason": "stop"
            }],
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        )

    async def stream_text(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any
    ) -> AsyncGenerator[str, None]:
        """Asynchronous streaming chat completion"""
        generator = await self.chat(
            messages=[{"role": msg.role, "content": msg.content} for msg in messages],
            model=kwargs.get("model", self.default_model),
            stream=True
        )
        if isinstance(generator, AsyncGenerator):
            async for chunk in generator:
                yield chunk

    def generate_object(self, prompt: str, **kwargs: Any) -> List[str]:
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            response = loop.run_until_complete(self._async_generate_object(prompt, **kwargs))
            return response
        finally:
            loop.close()

    async def _async_generate_object(self, prompt: str, **kwargs: Any) -> List[str]:
        response = await self.chat(
            messages=[{"role": "user", "content": prompt}],
            model=kwargs.get("model", self.default_model),
            format="json",
            stream=False
        )
        if isinstance(response, dict):
            return [response.get("message", {}).get("content", "")]
        return []

    async def stream_object(self, prompt: str, **kwargs: Any) -> List[str]:
        response = await self._async_generate_object(prompt, **kwargs)
        return response

    def get_embeddings(
        self,
        texts: Union[str, List[str]],
        **kwargs: Any
    ) -> List[List[float]]:
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
        if isinstance(texts, str):
            texts = [texts]
        return await self.create_embeddings(texts=texts, model=kwargs.get("model"))

    async def aget_embeddings(
        self,
        texts: Union[str, List[str]],
        **kwargs: Any
    ) -> List[List[float]]:
        return await self._async_get_embeddings(texts, **kwargs)

    def get_available_models(self) -> List[str]:
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            models = loop.run_until_complete(self.list_models())
            return [model["name"] for model in models]
        finally:
            loop.close()

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
        log.debug(f"Generating embeddings for {len(texts)} texts using model {selected_model}")
        try:
            embeddings = []
            for i, text in enumerate(texts):
                response = await self.client.post(
                    "/api/embeddings",
                    json={"model": selected_model, "prompt": text},
                )
                response.raise_for_status()
                embeddings.append(response.json()["embedding"])
                log.debug(f"Generated embedding {i+1}/{len(texts)}")
            log.debug(f"Successfully generated all embeddings with dimensions: {len(embeddings)}x{len(embeddings[0])}")
            return embeddings
        except Exception as e:
            log.error(f"Failed to generate embeddings: {str(e)}", exc_info=True)
            raise

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
        messages: List[Dict[str, Any]],
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
            messages (List[Dict[str, Any]]): Chat messages
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
        selected_model = model or self.default_model
        log.debug(f"Starting chat with model {selected_model}, {len(messages)} messages")
        try:
            payload = {
                "model": selected_model,
                "messages": messages,
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
                log.debug(f"Chat completed successfully with model {selected_model}")
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

            log.debug(f"Chat completed successfully with model {selected_model}")
            return response_generator()
        except httpx.RequestError as e:
            log.error(f"Request error occurred: {str(e)}")
            raise
        except Exception as e:
            log.error(f"Chat failed with model {selected_model}: {str(e)}", exc_info=True)
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


def get_ollama_service(config: OllamaConfig = get_ollama_settings()) -> OllamaService:
    """
    Get or create a singleton instance of OllamaService.

    Args:
        config (OllamaConfig): Service configuration

    Returns:
        OllamaService: Singleton service instance
    """
    global _ollama_instance
    if _ollama_instance is None:
        _ollama_instance = OllamaService()
    return _ollama_instance
