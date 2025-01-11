import json
from typing import AsyncGenerator, List, Optional

import httpx
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from pydantic import BaseModel

from api.services.openrouter.config import MODEL_CONFIGS, OpenRouterModel, get_openrouter_settings


class OpenRouterConfig(BaseModel):
    api_key: str
    base_url: str = "https://openrouter.ai/api/v1"
    default_model: OpenRouterModel = OpenRouterModel.QWEN_70B


class OpenRouterService:
    def __init__(self):
        settings = get_openrouter_settings()
        self.client = httpx.AsyncClient(
            base_url=settings.OPENROUTER_BASE_URL,
            headers={
                "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
                "HTTP-Referer": "http://localhost:3000",
                "X-Title": "AI SDK Python Streaming",
            },
            timeout=60.0,
        )
        self.default_model = OpenRouterModel.QWEN_70B

    async def create_embeddings(
        self, texts: List[str], model: str = "openai/text-embedding-3-small"
    ) -> List[List[float]]:
        
        response = await self.client.post(
            "/embeddings", json={"model": model, "input": texts}
        )
        response.raise_for_status()
        return response.json()["data"]

    async def stream_chat(
        self,
        messages: List[ChatCompletionMessageParam],
        model: Optional[OpenRouterModel] = None,
        tools: Optional[List[dict]] = None,
    ) -> AsyncGenerator[str, None]:
        selected_model = model or self.default_model
        model_config = MODEL_CONFIGS[selected_model]

        if tools and not model_config["supports_tools"]:
            raise ValueError(f"Model {selected_model} does not support tools")

        payload = {"model": selected_model, "messages": messages, "stream": True}

        if tools:
            payload["tools"] = tools

        async with self.client.stream(
            "POST", "/chat/completions", json=payload
        ) as response:
            response.raise_for_status()
            draft_tool_calls = []
            draft_tool_calls_index = -1

            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        continue

                    try:
                        chunk = json.loads(data)
                        choice = chunk["choices"][0]

                        if choice.get("finish_reason") == "stop":
                            continue
                        elif choice.get("finish_reason") == "tool_calls":
                            for tool_call in draft_tool_calls:
                                yield f'9:{{"toolCallId":"{tool_call["id"]}","toolName":"{tool_call["name"]}","args":{tool_call["arguments"]}}}\n'

                        delta = choice.get("delta", {})
                        if delta.get("tool_calls"):
                            for tool_call in delta["tool_calls"]:
                                id = tool_call.get("id")
                                function = tool_call.get("function", {})
                                name = function.get("name")
                                arguments = function.get("arguments", "")

                                if id:
                                    draft_tool_calls_index += 1
                                    draft_tool_calls.append(
                                        {"id": id, "name": name, "arguments": ""}
                                    )
                                else:
                                    draft_tool_calls[draft_tool_calls_index][
                                        "arguments"
                                    ] += arguments
                        elif delta.get("content"):
                            yield f'0:{json.dumps(delta["content"])}\n'
                    except json.JSONDecodeError:
                        continue

    async def close(self):
        await self.client.aclose()


def get_openrouter_service() -> OpenRouterService:
    return OpenRouterService()
