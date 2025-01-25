import os
from unittest.mock import AsyncMock, Mock, patch

import httpx
import numpy as np
import pytest
import torch

from api.core.logger import log
from api.services.huggingface import (
    HuggingFaceConfig,
    HuggingFaceService,
    SentenceTransformerModel,
    TransformerModel,
    get_huggingface_service,
    get_huggingface_settings,
)


@pytest.fixture
def service() -> HuggingFaceService:
    settings = get_huggingface_settings(
        use_local=True,
        default_model=SentenceTransformerModel.MINI_LM_L12_V2,
    )
    return get_huggingface_service(settings) # type: ignore


@pytest.mark.asyncio
async def test_sentence_transformer(service: HuggingFaceService):
    result = await service.get_sentence_embeddings(["This is a test sentence."])
    log.info(f"\n\nSentence transformer result: {result}")
    assert result is not None
    assert len(result) > 0
