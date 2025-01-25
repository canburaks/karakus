# A class that will be used in ai routes to generate text, embeddings, etc.
from typing import Any, Dict, List, Optional, Union, Sequence, AsyncGenerator

from api.interfaces.llm_service import LLMService
from api.models.ai import ChatMessage, ChatResponse
from api.services.huggingface import get_huggingface_service
from api.services.ollama import get_ollama_service
from api.services.openrouter import get_openrouter_service
from api.core.logger import log


class LLM(LLMService):
    """A wrapper class that implements LLMService interface for different LLM providers"""

    def __init__(self, service: Any):
        self.service = service
        log.info(f"Initialized LLM service with provider: {service.__class__.__name__}")

    def generate_text(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any
    ) -> ChatResponse:
        """Synchronous chat completion"""
        log.info(f"Generating text with {len(messages)} messages. kwargs: {kwargs}")
        try:
            response = self.service.generate_text(messages, **kwargs)
            log.info("Text generation successful")
            return response
        except Exception as e:
            log.error(f"Text generation failed: {str(e)}", exc_info=True)
            raise

    async def stream_text(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any
    ) -> AsyncGenerator[str, None]:
        """Asynchronous streaming chat completion"""
        log.info(f"Streaming text with {len(messages)} messages. kwargs: {kwargs}")
        try:
            async for chunk in self.service.stream_text(messages, **kwargs):
                log.info(f"Yielding chunk: {chunk}")
                yield chunk
            log.info("Text streaming successful")
        except Exception as e:
            log.error(f"Text streaming failed: {str(e)}", exc_info=True)
            raise
    
    def generate_object(self, prompt: str, **kwargs: Any) -> List[str]:
        """Generate structured output from a prompt"""
        log.info(f"Generating object from prompt. kwargs: {kwargs}")
        try:
            response = self.service.generate_object(prompt, **kwargs)
            log.info("Object generation successful")
            return response
        except Exception as e:
            log.error(f"Object generation failed: {str(e)}", exc_info=True)
            raise

    async def stream_object(self, prompt: str, **kwargs: Any) -> List[str]:
        """Asynchronous structured output generation from a prompt"""
        log.info(f"Streaming object from prompt. kwargs: {kwargs}")
        try:
            response = await self.service.stream_object(prompt, **kwargs)
            log.info("Object streaming successful")
            return response
        except Exception as e:
            log.error(f"Object streaming failed: {str(e)}", exc_info=True)
            raise

    def get_embeddings(
        self,
        texts: Union[str, List[str]],
        **kwargs: Any
    ) -> List[List[float]]:
        """Synchronous text embedding generation"""
        text_count = len(texts) if isinstance(texts, list) else 1
        log.info(f"Generating embeddings for {text_count} texts. kwargs: {kwargs}")
        try:
            embeddings = self.service.get_embeddings(texts, **kwargs)
            log.info("Embeddings generation successful")
            return embeddings
        except Exception as e:
            log.error(f"Embeddings generation failed: {str(e)}", exc_info=True)
            raise

    async def aget_embeddings(
        self,
        texts: Union[str, List[str]],
        **kwargs: Any
    ) -> List[List[float]]:
        """Asynchronous text embedding generation"""
        text_count = len(texts) if isinstance(texts, list) else 1
        log.info(f"Generating async embeddings for {text_count} texts. kwargs: {kwargs}")
        try:
            embeddings = await self.service.aget_embeddings(texts, **kwargs)
            log.info("Async embeddings generation successful")
            return embeddings
        except Exception as e:
            log.error(f"Async embeddings generation failed: {str(e)}", exc_info=True)
            raise

    def get_available_models(self) -> List[str]:
        """Get list of available models for the provider"""
        log.info("Fetching available models")
        try:
            models = self.service.get_available_models()
            log.info(f"Found {len(models)} available models")
            return models
        except Exception as e:
            log.error(f"Failed to fetch available models: {str(e)}", exc_info=True)
            raise


# Factory functions with logging
def ollama_llm():
    log.info("Creating Ollama LLM instance")
    ollama_service = get_ollama_service()
    return LLM(service=ollama_service)

def openrouter_llm():
    log.info("Creating OpenRouter LLM instance")
    openrouter_service = get_openrouter_service()
    return LLM(service=openrouter_service)

def huggingface_llm():
    log.info("Creating HuggingFace LLM instance")
    huggingface_service = get_huggingface_service()
    return LLM(service=huggingface_service)
