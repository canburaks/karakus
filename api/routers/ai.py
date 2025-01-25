import json
from datetime import datetime
from typing import List, Optional, Dict, Any, Union

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.responses import JSONResponse

from api.core.parser import parse_pdf, read_text_file
from api.models.ai import (
    ChatRequest,
    ChatResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    Message,
    Role,
)
from api.core.llm import openrouter_llm, ollama_llm
from api.core.logger import log
from api.interfaces.llm_service import LLMService
from typing import cast

router = APIRouter(
    prefix="/ai",
    tags=["ai"],
    responses={404: {"description": "Not found"}},
)


@router.post("/chat/completions", response_model=ChatResponse)
async def create_chat_completion(
    request: ChatRequest,
    llm_service: LLMService = Depends(openrouter_llm),
) -> Union[ChatResponse, StreamingResponse]:
    """OpenAI-compatible chat completion endpoint"""
    try:
        log.info(f"\n\nRequest: {request.__dict__}")
        if request.stream:
            return StreamingResponse(
                llm_service.stream_text(
                    messages=request.messages,
                    model=request.model,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                ),
                media_type="text/event-stream",
            )

        response = llm_service.generate_text(
            messages=request.messages,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        return response

    except HTTPException as e:
        raise HTTPException(
            status_code=e.status_code,
            detail={
                "error": {
                    "message": e.detail,
                    "type": "invalid_request_error",
                    "code": e.status_code,
                }
            },
        )


@router.post("/chat/stream")
async def stream_chat_completion(
    request: ChatRequest,
    llm_service=Depends(openrouter_llm),
):
    """Stream a chat completion."""
    try:
        return StreamingResponse(
            llm_service.stream_text(
                messages=request.messages,
                model=request.model,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            ),
            media_type="text/event-stream",
        )
    except Exception as e:
        log.error(f"Chat streaming failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(
    request: EmbeddingRequest,
    llm_service: LLMService = Depends(openrouter_llm),
) -> EmbeddingResponse:
    """OpenAI-compatible embeddings endpoint"""
    try:
        embeddings = await llm_service.aget_embeddings(
            texts=request.input if isinstance(request.input, list) else [request.input],
            model=request.model or "openai/text-embedding-3-small",
        )
        return EmbeddingResponse(
            object="list",
            data=[
                {
                    "object": "embedding",
                    "embedding": [float(x) for x in emb],
                    "index": i,
                }
                for i, emb in enumerate(embeddings)
            ],
            model=request.model or "openai/text-embedding-3-small",
            usage={"prompt_tokens": 0, "total_tokens": 0},
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "message": str(e),
                    "type": "api_error",
                    "code": "internal_error",
                }
            },
        )
