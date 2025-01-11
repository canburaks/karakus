import json
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from api.models.ai import (
    ChatRequest,
    ChatResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    Message,
    Role,
)
from api.services.openrouter import get_openrouter_service
from api.core.parser import parse_pdf, read_text_file


router = APIRouter(
    prefix="/ai",
    tags=["ai"],
    responses={404: {"description": "Not found"}},
)


async def process_file_content(file: UploadFile) -> str:
    """Process uploaded file content and return as string."""
    try:
        if file.content_type == "application/pdf":
            return await parse_pdf(file)
        else:
            return await read_text_file(file)
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must be text-based")


@router.post("/chat/completions", response_model=ChatResponse)
async def create_chat_completion(
    request: ChatRequest,
    file: Optional[UploadFile] = File(None),
    openrouter_service=Depends(get_openrouter_service),
):
    """Create a chat completion with optional file input."""
    messages = request.messages

    if file:
        file_content = await process_file_content(file)
        messages.append(
            Message(role=Role.USER, content=f"Content from file: {file_content}")
        )

    try:
        response = await openrouter_service.stream_chat(
            messages=[msg.model_dump() for msg in messages],
            model=request.model,
        )
        
        # Collect all chunks
        content = ""
        async for chunk in response:
            if chunk.startswith("0:"):
                content += json.loads(chunk[2:])

        return ChatResponse(
            id=f"chatcmpl-{datetime.now().timestamp()}",
            model=request.model or "default",
            created=int(datetime.now().timestamp()),
            choices=[{"message": {"role": "assistant", "content": content}}],
            usage={"total_tokens": len(content.split())},  # Approximate
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/stream")
async def stream_chat_completion(
    request: ChatRequest,
    file: Optional[UploadFile] = File(None),
    openrouter_service=Depends(get_openrouter_service),
):
    """Stream a chat completion with optional file input."""
    messages = request.messages

    if file:
        file_content = await process_file_content(file)
        messages.append(
            Message(role=Role.USER, content=f"Content from file: {file_content}")
        )

    try:
        return StreamingResponse(
            openrouter_service.stream_chat(
                messages=[msg.model_dump() for msg in messages],
                model=request.model,
            ),
            media_type="text/event-stream",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
    model: str = Form("openai/text-embedding-3-small"),
    openrouter_service=Depends(get_openrouter_service),
):
    """Create embeddings from text or file input."""
    if not file and not text:
        raise HTTPException(
            status_code=400, detail="Either file or text must be provided"
        )

    try:
        if file:
            input_text = await process_file_content(file)
        elif text:
            input_text = text
        else:
            raise HTTPException(status_code=400, detail="No input provided")

        embeddings = await openrouter_service.create_embeddings(
            texts=[input_text], model=model
        )

        return EmbeddingResponse(
            model=model,
            data=embeddings,
            usage={"total_tokens": len(input_text.split())},  # Approximate
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 