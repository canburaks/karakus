from enum import Enum
from typing import List, Optional, Union

from fastapi import UploadFile
from pydantic import BaseModel, Field


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"


class Message(BaseModel):
    role: Role
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]
    model: Optional[str] = Field(default="openai/text-embedding-3-small")
    stream: bool = False
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, ge=1)


class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]]
    model: Optional[str] = Field(default="openai/text-embedding-3-small")


class EmbeddingResponse(BaseModel):
    model: str
    data: List[List[float]]
    usage: dict


class ChatResponse(BaseModel):
    id: str
    model: str
    created: int
    choices: List[dict]
    usage: dict 