from dataclasses import dataclass
from typing import Any

import httpx
from loguru import logger
from pydantic import BaseModel
from pydantic_ai import Agent, ModelRetry, RunContext, capture_run_messages
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.usage import UsageLimits

from ..agents import create_agent

logger.add(sink="static/logs.log", format="{time} {level} {message}", level="DEBUG")


@dataclass
class DepsWithHttpClient:
    test_label: str
    http_client: httpx.AsyncClient


class RouletteDeps(BaseModel):
    winning_number: int


class RouletteResult(BaseModel):
    is_winner: bool
    message: str


agent: Agent[RouletteDeps, RouletteResult] = create_agent(
    result_type=RouletteResult,
    deps_type=RouletteDeps,
    provider="openrouter",
    model_name="qwen/qwen-2.5-72b-instruct",
    system_prompt=(
        "Use the `roulette_wheel` function to see if the "
        "customer has won based on the number they provide."
    ),
)


@agent.tool
def roulette_wheel(ctx: RunContext[RouletteDeps], square: int) -> str:
    logger.info("Test agent roulette_wheel: {} {}", ctx, square, feature="f-strings")
    """check if the square is a winner"""
    return "winner" if square == ctx.deps else "loser"


def test_agent_with_tool_call() -> Any:
    with capture_run_messages() as messages:
        try:
            deps = RouletteDeps(winning_number=18)
            result = agent.run_sync(
                user_prompt="Put my money on square eighteen",
                deps=deps,
                usage_limits=UsageLimits(response_tokens_limit=100),
            )
            logger.info("result.data: {}", result, feature="f-strings")
            logger.info("try messages:{}", messages[0], feature="f-strings")
            assert result
        except UsageLimitExceeded as e:
            logger.info("UsageLimitExceeded:{}", messages, feature="f-strings")
            raise e
        except Exception as e:
            logger.error("Test agent error: 1 -> {e}")
            logger.info("except messages:{}", messages, feature="f-strings")
            raise e


@agent.system_prompt
async def get_system_prompt() -> str:
    async with httpx.AsyncClient() as client:
        response = await client.get(
            url="https://example.com",
            headers={"Authorization": "Bearer test_token"},
        )
        response.raise_for_status()
        return f"Prompt: {response.text}"


@agent.result_validator
async def validate_result(
    ctx: RunContext[RouletteDeps], final_response: RouletteResult
) -> RouletteResult:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://example.com#validate",
            headers={"Authorization": "Bearer test_token"},
            json=final_response.model_dump(),
        )
        if response.status_code == 400:
            raise ModelRetry(f"invalid response: {response.text}")
        response.raise_for_status()
        return final_response


agent = create_agent(provider="openrouter", model_name="qwen/qwen-2.5-72b-instruct")


class Foobar(BaseModel):
    """This is a Foobar"""

    x: int
    y: str
    z: float = 3.14


@agent.tool_plain
def foobar(f: Foobar) -> str:
    return str(object=f)
