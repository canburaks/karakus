from typing import Any, List, Optional, Union, Sequence, Dict, cast

import httpx
import torch
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, pipeline, Pipeline
import numpy as np
import asyncio
import json
from typing import AsyncGenerator
import uuid
from datetime import datetime

from api.core.logger import log
from api.models.ai import ChatMessage, ChatResponse
from api.interfaces.llm_service import LLMService
from .config import HuggingFaceConfig, get_huggingface_settings


class HuggingFaceService(LLMService):
    """
    Service class for handling HuggingFace model operations including embeddings generation
    using both API and local models. Supports sentence transformers and regular transformers.

    Attributes:
        config (HuggingFaceConfig): Configuration for the service
        client (httpx.AsyncClient): Async client for API calls
        sentence_transformer (SentenceTransformer): Local sentence transformer model
        tokenizer (AutoTokenizer): Tokenizer for local transformer model
        model (AutoModel): Local transformer model
    """

    def __init__(self, config: HuggingFaceConfig):
        """
        Initialize the HuggingFace service with given configuration.

        Args:
            config (HuggingFaceConfig): Service configuration including API keys and model settings
        """
        log.info(f"Initializing HuggingFace service with model: {config.default_model}")
        self.config = config
        self.client = httpx.AsyncClient(
            base_url=config.base_url,
            headers={"Authorization": f"Bearer {config.api_key}"},
            timeout=30.0,
        )
        self.sentence_transformer: Optional[SentenceTransformer] = None
        self.tokenizer = None
        self.model = None
        self.text_generator: Optional[Pipeline] = None

    def generate_text(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any
    ) -> ChatResponse:
        """Synchronous chat completion"""
        model_name = kwargs.get("model", self.config.default_model)
        log.info(f"Generating text with model {model_name}")
        try:
            if self.text_generator is None:
                log.info(f"Initializing text generator pipeline with model: {model_name}")
                self.text_generator = pipeline(
                    "text-generation",
                    model=model_name,
                    tokenizer=model_name,
                    device=0 if torch.cuda.is_available() else -1
                )

            prompt = "\n".join([f"{msg.role}: {msg.content}" for msg in messages])
            log.info(f"Generated prompt with {len(prompt.split())} words")

            outputs = self.text_generator(
                prompt,
                max_length=kwargs.get("max_tokens", 100),
                temperature=kwargs.get("temperature", 0.7),
                num_return_sequences=1
            )
            
            generated_text = str(outputs[0].get("generated_text", "")) if isinstance(outputs, list) and len(outputs) > 0 else ""
            log.info(f"Successfully generated text with {len(generated_text.split())} words")
            
            return ChatResponse(
                id=f"chatcmpl-{uuid.uuid4().hex}",
                object="chat.completion",
                created=int(datetime.now().timestamp()),
                model=str(model_name),
                choices=[{
                    "index": 0,
                    "message": {"role": "assistant", "content": generated_text},
                    "finish_reason": "stop"
                }],
                usage={"prompt_tokens": 0, "completion_tokens": len(generated_text.split()), "total_tokens": len(generated_text.split())}
            )
        except Exception as e:
            log.error(f"Text generation failed with model {model_name}: {str(e)}", exc_info=True)
            raise

    async def _async_generate_text(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any
    ) -> ChatResponse:
        """Internal async implementation of generate_text"""
        model_name = kwargs.get("model", self.config.default_model)
        log.info(f"Generating text with model {model_name}")
        try:
            if self.text_generator is None:
                log.info(f"Initializing text generator pipeline with model: {model_name}")
                self.text_generator = pipeline(
                    "text-generation",
                    model=model_name,
                    tokenizer=model_name,
                    device=0 if torch.cuda.is_available() else -1
                )

            prompt = "\n".join([f"{msg.role}: {msg.content}" for msg in messages])
            log.info(f"Generated prompt with {len(prompt.split())} words")

            outputs = self.text_generator(
                prompt,
                max_length=kwargs.get("max_tokens", 100),
                temperature=kwargs.get("temperature", 0.7),
                num_return_sequences=1
            )
            
            generated_text = str(outputs[0].get("generated_text", "")) if isinstance(outputs, list) and len(outputs) > 0 else ""
            log.info(f"Successfully generated text with {len(generated_text.split())} words")
            
            return ChatResponse(
                id=f"chatcmpl-{uuid.uuid4().hex}",
                object="chat.completion",
                created=int(datetime.now().timestamp()),
                model=str(model_name),
                choices=[{
                    "index": 0,
                    "message": {"role": "assistant", "content": generated_text},
                    "finish_reason": "stop"
                }],
                usage={"prompt_tokens": 0, "completion_tokens": len(generated_text.split()), "total_tokens": len(generated_text.split())}
            )
        except Exception as e:
            log.error(f"Text generation failed with model {model_name}: {str(e)}", exc_info=True)
            raise

    async def stream_text(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any
    ) -> AsyncGenerator[str, None]:
        """Asynchronous streaming chat completion"""
        model_name = kwargs.get("model", self.config.default_model)
        log.info(f"Streaming text with model {model_name}")
        try:
            if self.text_generator is None:
                log.info(f"Initializing text generator pipeline with model: {model_name}")
                self.text_generator = pipeline(
                    "text-generation",
                    model=model_name,
                    tokenizer=model_name,
                    device=0 if torch.cuda.is_available() else -1
                )

            prompt = "\n".join([f"{msg.role}: {msg.content}" for msg in messages])
            log.info(f"Generated prompt with {len(prompt.split())} words")

            outputs = self.text_generator(
                prompt,
                max_length=kwargs.get("max_tokens", 100),
                temperature=kwargs.get("temperature", 0.7),
                num_return_sequences=1,
                return_full_text=False
            )
            
            if isinstance(outputs, list) and len(outputs) > 0:
                text = str(outputs[0].get("generated_text", ""))
                # Simulate streaming by yielding chunks
                chunk_size = 4  # Adjust chunk size as needed
                for i in range(0, len(text), chunk_size):
                    chunk = text[i:i + chunk_size]
                    yield f"0:{json.dumps(chunk)}\n"
                    await asyncio.sleep(0.1)  # Add small delay between chunks
        except Exception as e:
            log.error(f"Text streaming failed with model {model_name}: {str(e)}", exc_info=True)
            raise

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
        model_name = kwargs.get("model", self.config.default_model)
        log.info(f"Generating object from prompt with model {model_name}")
        try:
            if self.text_generator is None:
                log.info(f"Initializing text generator pipeline with model: {model_name}")
                self.text_generator = pipeline(
                    "text-generation",
                    model=model_name,
                    tokenizer=model_name,
                    device=0 if torch.cuda.is_available() else -1
                )

            json_prompt = f"{prompt}\nGenerate a valid JSON response:"
            log.info(f"Generated JSON prompt with {len(json_prompt.split())} words")
            
            outputs = self.text_generator(
                json_prompt,
                max_length=kwargs.get("max_tokens", 100),
                temperature=kwargs.get("temperature", 0.7),
                num_return_sequences=1
            )
            
            if isinstance(outputs, list) and len(outputs) > 0:
                generated_text = str(outputs[0].get("generated_text", ""))
                log.info(f"Successfully generated JSON text with {len(generated_text.split())} words")
                return [generated_text]
            log.warning("No output generated, returning empty string")
            return [""]
        except Exception as e:
            log.error(f"Object generation failed with model {model_name}: {str(e)}", exc_info=True)
            raise

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
        return await self.get_sentence_embeddings(texts)

    async def aget_embeddings(
        self,
        texts: Union[str, List[str]],
        **kwargs: Any
    ) -> List[List[float]]:
        """Asynchronous text embedding generation"""
        try:
            if isinstance(texts, str):
                texts = [texts]
            embeddings = await self.get_sentence_embeddings(texts)
            # Ensure all values are float type
            return [[float(val) for val in emb] for emb in embeddings]
        except Exception as e:
            log.error(f"Embedding generation failed: {str(e)}", exc_info=True)
            raise

    def get_available_models(self) -> List[str]:
        # Return a list of commonly used models
        return [
            "gpt2",
            "gpt2-medium",
            "gpt2-large",
            "gpt2-xl",
            "facebook/opt-125m",
            "facebook/opt-350m",
            "facebook/opt-1.3b",
            "EleutherAI/pythia-70m",
            "EleutherAI/pythia-160m",
            "EleutherAI/pythia-410m",
        ]

    async def get_sentence_embeddings(self, texts: List[str]) -> List[List[float]]:
        log.info(f"Generating sentence embeddings for {len(texts)} texts")
        default_embedding: List[List[float]] = [[0.0] * 768]
        
        if not self.sentence_transformer:
            try:
                log.info("Initializing sentence transformer model")
                self.initialize_sentence_transformer()
            except Exception as e:
                log.error(f"Failed to initialize sentence transformer: {str(e)}", exc_info=True)
                return default_embedding
                
        if self.sentence_transformer is None:
            return default_embedding
            
        try:
            embeddings = self.sentence_transformer.encode(texts)
            if isinstance(embeddings, (np.ndarray, torch.Tensor)):
                result = embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings.detach().cpu().numpy().tolist()
                return cast(List[List[float]], result)
            return default_embedding
        except Exception as e:
            log.error(f"Failed to generate embeddings: {str(e)}", exc_info=True)
            return default_embedding

    def initialize_sentence_transformer(self, model_name: Optional[str] = None):
        """
        Initialize a local sentence transformer model.

        Args:
            model_name (Optional[str]): Name of the model to initialize. Uses default from config if None.
        """
        model_name = model_name or self.config.default_model
        self.sentence_transformer = SentenceTransformer(model_name)

    async def close(self):
        """Close the async client connection."""
        await self.client.aclose()


# Global instance
_huggingface_instance = None


def get_huggingface_service() -> HuggingFaceService:
    """
    Get or create a singleton instance of HuggingFaceService.

    Returns:
        HuggingFaceService: Singleton service instance
    """
    global _huggingface_instance
    if _huggingface_instance is None:
        _huggingface_instance = HuggingFaceService(config=get_huggingface_settings())
    return _huggingface_instance
