"""Thin LLM client wrapper.

All model calls route through here so usage/cost logging and provider swaps
stay in one place. Feature code must not import a provider SDK directly.
"""

import time
from dataclasses import dataclass
from typing import Protocol

import anthropic


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float


class LLMClient(Protocol):
    def complete(
        self,
        *,
        messages: list[dict],
        model: str,
        system: str | None = None,
        max_tokens: int = 1024,
    ) -> LLMResponse: ...


class AnthropicClient:
    def __init__(self, api_key: str) -> None:
        self._client = anthropic.Anthropic(api_key=api_key)

    def complete(
        self,
        *,
        messages: list[dict],
        model: str,
        system: str | None = None,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        started = time.perf_counter()
        request: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if system is not None:
            request["system"] = system
        response = self._client.messages.create(**request)
        latency_ms = (time.perf_counter() - started) * 1000
        text = "".join(block.text for block in response.content if block.type == "text")
        return LLMResponse(
            text=text,
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_ms=latency_ms,
        )
