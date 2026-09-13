"""Thin LLM client wrapper.

All model calls route through here so usage/cost logging and provider swaps
stay in one place. Feature code must not import a provider SDK directly.

Model ids prefixed with ``openrouter/`` are sent to OpenRouter (OpenAI-compatible
chat completions, stdlib only); anything else goes to Anthropic.
"""

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol

import anthropic

from research_agent.llm.usage import log_usage

OPENROUTER_PREFIX = "openrouter/"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_TIMEOUT_S = 120


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    stop_reason: str | None = None

    @property
    def truncated(self) -> bool:
        return self.stop_reason == "max_tokens"


class LLMClient(Protocol):
    def complete(
        self,
        *,
        messages: list[dict],
        model: str,
        system: str | None = None,
        max_tokens: int = 1024,
        task: str = "llm",
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
        task: str = "llm",
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
        result = LLMResponse(
            text=text,
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_ms=latency_ms,
            stop_reason=response.stop_reason,
        )
        log_usage(task, result)
        return result


class OpenRouterClient:
    """OpenAI-compatible chat completions via OpenRouter, stdlib only."""

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    def complete(
        self,
        *,
        messages: list[dict],
        model: str,
        system: str | None = None,
        max_tokens: int = 1024,
        task: str = "llm",
    ) -> LLMResponse:
        # ``model`` arrives routed (e.g. "openrouter/deepseek/deepseek-v4.1-flash");
        # the API gets the bare id, usage logging keeps the routed id.
        bare_id = model.removeprefix(OPENROUTER_PREFIX)
        payload_messages = ([{"role": "system", "content": system}] if system else []) + list(
            messages
        )
        payload = json.dumps(
            {"model": bare_id, "messages": payload_messages, "max_tokens": max_tokens}
        ).encode("utf-8")

        started = time.perf_counter()
        data = None
        for attempt in range(3):
            request = urllib.request.Request(
                OPENROUTER_URL,
                data=payload,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
            )
            try:
                with urllib.request.urlopen(request, timeout=OPENROUTER_TIMEOUT_S) as response:
                    data = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as error:
                # ponytail: 3 attempts on transient errors; back off linearly.
                if attempt < 2 and error.code in (408, 429, 500, 502, 503, 524):
                    time.sleep(2 * (attempt + 1))
                    continue
                raise
        latency_ms = (time.perf_counter() - started) * 1000
        if data is None:
            raise RuntimeError("OpenRouter request failed without a response")

        choice = data["choices"][0]
        usage = data.get("usage", {})
        result = LLMResponse(
            text=choice["message"]["content"] or "",
            model=model,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            latency_ms=latency_ms,
            stop_reason="max_tokens" if choice.get("finish_reason") == "length" else None,
        )
        log_usage(task, result)
        return result


class RoutedClient:
    """Dispatches per model prefix: ``openrouter/...`` to OpenRouter, else Anthropic."""

    def __init__(
        self,
        anthropic_client: AnthropicClient | None,
        openrouter_client: OpenRouterClient | None,
    ) -> None:
        self._anthropic = anthropic_client
        self._openrouter = openrouter_client

    def complete(
        self,
        *,
        messages: list[dict],
        model: str,
        system: str | None = None,
        max_tokens: int = 1024,
        task: str = "llm",
    ) -> LLMResponse:
        if model.startswith(OPENROUTER_PREFIX):
            if self._openrouter is None:
                raise ValueError("OPENROUTER_API_KEY is not set")
            return self._openrouter.complete(
                messages=messages,
                model=model,
                system=system,
                max_tokens=max_tokens,
                task=task,
            )
        if self._anthropic is None:
            raise ValueError("ANTHROPIC_API_KEY is not set")
        return self._anthropic.complete(
            messages=messages,
            model=model,
            system=system,
            max_tokens=max_tokens,
            task=task,
        )


def create_client(settings) -> LLMClient:
    """Build the dispatching client from whatever keys the environment provides."""
    anthropic_client = (
        AnthropicClient(settings.anthropic_api_key) if settings.anthropic_api_key else None
    )
    openrouter_client = (
        OpenRouterClient(settings.openrouter_api_key) if settings.openrouter_api_key else None
    )
    return RoutedClient(anthropic_client, openrouter_client)
