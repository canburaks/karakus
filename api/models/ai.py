from enum import Enum
from typing import List, Optional, Union, Dict, Any
from datetime import datetime
import uuid

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


class ChatMessage(BaseModel):
    role: str
    content: str
    name: Optional[str] = None
    function_call: Optional[dict] = None


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    model: str
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    n: Optional[int] = 1
    stop: Optional[Union[str, List[str]]] = None


class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]]
    model: Optional[str] = Field(default="openai/text-embedding-3-small")


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[dict]
    model: str
    usage: dict


class ChatResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(datetime.now().timestamp()))
    model: str
    choices: List[Dict[str, Any]] = Field(
        default_factory=lambda: [{
            "index": 0,
            "message": {"role": "assistant", "content": ""},
            "finish_reason": "stop"
        }]
    )
    usage: Dict[str, int] = Field(
        default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    )
