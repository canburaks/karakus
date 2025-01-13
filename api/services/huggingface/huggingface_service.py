from typing import Any, List, Optional

import httpx
import torch
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer

from api.core.logger import log

from .config import HuggingFaceConfig, get_huggingface_settings


class HuggingFaceService:
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
        self.config = config
        self.client = httpx.AsyncClient(
            base_url=config.base_url,
            headers={"Authorization": f"Bearer {config.api_key}"},
            timeout=30.0,
        )
        self.sentence_transformer = None
        self.tokenizer = None
        self.model = None

    async def initialize_local_model(self, model_name: Optional[str] = None):
        """
        Initialize a local transformer model and tokenizer.

        Args:
            model_name (Optional[str]): Name of the model to initialize. Uses default from config if None.
        """
        model_name = model_name or self.config.default_model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def initialize_sentence_transformer(self, model_name: Optional[str] = None):
        """
        Initialize a local sentence transformer model.

        Args:
            model_name (Optional[str]): Name of the model to initialize. Uses default from config if None.
        """
        model_name = model_name or self.config.default_model
        self.sentence_transformer = SentenceTransformer(model_name)

    async def get_embeddings_api(
        self, texts: List[str], model: Optional[str] = None
    ) -> List[List[float]]:
        """
        Get embeddings using HuggingFace's API endpoint.

        Args:
            texts (List[str]): List of texts to get embeddings for
            model (Optional[str]): Model to use for embeddings. Uses default from config if None.

        Returns:
            List[List[float]]: List of embeddings vectors

        Raises:
            ValueError: If API response structure is unexpected
            httpx.HTTPError: If API request fails
        """
        model_name = model or self.config.default_model
        response = await self.client.post(
            f"/{model_name}", json={"inputs": texts, "wait_for_model": True}
        )
        response.raise_for_status()
        data = response.json()

        def validate_and_convert(embeddings: List[List[Any]]) -> List[List[float]]:
            return [[float(val) for val in emb] for emb in embeddings]

        if isinstance(data, list):
            embeddings = validate_and_convert(data)
        elif isinstance(data, dict) and "embeddings" in data:
            embeddings = validate_and_convert(data["embeddings"])
        else:
            raise ValueError(
                f"Unexpected API response structure from model {model_name}"
            )

        return embeddings

    def get_embeddings_local(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings using a local transformer model.

        Args:
            texts (List[str]): List of texts to get embeddings for

        Returns:
            List[List[float]]: List of embeddings vectors

        Raises:
            ValueError: If model is not initialized
        """
        if not self.model or not self.tokenizer:
            raise ValueError(
                "Model not initialized. Call initialize_local_model first."
            )

        encoded_input = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt"
        )
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            embeddings = model_output.last_hidden_state[:, 0, :].numpy()
        return embeddings.tolist()

    def get_sentence_embeddings(self, texts: List[str] | str) -> List[List[float]]:
        """
        Get embeddings using a local sentence transformer model.

        Args:
            texts (Union[List[str], str]): Single text or list of texts to get embeddings for

        Returns:
            List[List[float]]: List of embeddings vectors

        Raises:
            ValueError: If sentence transformer is not initialized
        """
        if not self.sentence_transformer:
            raise ValueError(
                "Sentence transformer not initialized. Call initialize_sentence_transformer first."
            )
        if isinstance(texts, str):
            texts = [texts]
        embeddings = self.sentence_transformer.encode(texts)
        log.info(f"\n\nembeddings shape: {embeddings.shape}")
        if len(embeddings.shape) == 1:
            return [[float(x) for x in embeddings]]
        return [[float(x) for x in emb] for emb in embeddings]

    async def close(self):
        """Close the async client connection."""
        await self.client.aclose()


# Global instance
_huggingface_instance = None


def get_huggingface_service(config: HuggingFaceConfig) -> HuggingFaceService:
    """
    Get or create a singleton instance of HuggingFaceService.

    Args:
        config (HuggingFaceConfig): Service configuration

    Returns:
        HuggingFaceService: Singleton service instance
    """
    global _huggingface_instance
    if _huggingface_instance is None:
        _huggingface_instance = HuggingFaceService(config=config)
        if config.use_local:
            _huggingface_instance.initialize_sentence_transformer(
                model_name=config.default_model
            )
    return _huggingface_instance
