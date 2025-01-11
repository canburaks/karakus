from typing import Any, List, Optional

import httpx
import torch
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer


class HuggingFaceConfig(BaseModel):
    api_key: str
    base_url: str = "https://api-inference.huggingface.co/models"
    default_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    use_local: bool = False


class HuggingFaceService:
    def __init__(self, config: HuggingFaceConfig):
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
        """Initialize local transformer model"""
        model_name = model_name or self.config.default_model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def initialize_sentence_transformer(self, model_name: Optional[str] = None):
        """Initialize sentence transformer model"""
        model_name = model_name or self.config.default_model
        self.sentence_transformer = SentenceTransformer(model_name)

    async def get_embeddings_api(
        self, texts: List[str], model: Optional[str] = None
    ) -> List[List[float]]:
        """Get embeddings using HuggingFace API"""
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
        """Get embeddings using local transformer model"""
        if not self.model or not self.tokenizer:
            raise ValueError(
                "Model not initialized. Call initialize_local_model first."
            )

        # Tokenize and get embeddings
        encoded_input = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt"
        )
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            embeddings = model_output.last_hidden_state[:, 0, :].numpy()
        return embeddings.tolist()

    def get_sentence_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings using sentence transformer"""
        if not self.sentence_transformer:
            raise ValueError(
                "Sentence transformer not initialized. Call initialize_sentence_transformer first."
            )

        embeddings = self.sentence_transformer.encode(texts)
        print(f"embeddings: {embeddings}")
        # Ensure 2D array and convert to list of lists with float values
        if len(embeddings.shape) == 1:
            return [[float(x) for x in embeddings]]
        return [[float(x) for x in emb] for emb in embeddings]

    async def close(self):
        await self.client.aclose()


# Global instance
_huggingface_instance = None


def get_huggingface_service(config: HuggingFaceConfig) -> HuggingFaceService:
    global _huggingface_instance
    if _huggingface_instance is None:
        _huggingface_instance = HuggingFaceService(config)
        if config.use_local:
            _huggingface_instance.initialize_sentence_transformer(config.default_model)
    return _huggingface_instance
