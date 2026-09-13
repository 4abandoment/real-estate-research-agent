"""Thin LLM client wrapper.

All model calls route through here so usage/cost logging and provider swaps
stay in one place. Feature code must not import a provider SDK directly.

Model ids pick the gateway by prefix: ``openrouter/...`` goes to OpenRouter,
``opencode/...`` to OpenCode Zen; anything else goes to Anthropic. Both
gateways speak OpenAI-compatible chat completions (stdlib only).
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
OPENCODE_PREFIX = "opencode/"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENCODE_ZEN_URL = "https://opencode.ai/zen/v1/chat/completions"
GATEWAY_TIMEOUT_S = 180
# 402 included: observed flaky on OpenRouter even with account credits.
RETRYABLE_STATUS = (402, 408, 429, 500, 502, 503, 524)
# opencode.ai sits behind Cloudflare, which rejects urllib's default agent.
GATEWAY_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"


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


class OpenAICompatibleClient:
    """OpenAI-compatible chat completions (OpenRouter / OpenCode Zen), stdlib only."""

    def __init__(self, api_key: str, *, base_url: str, model_prefix: str, label: str) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._model_prefix = model_prefix
        self._label = label

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
        bare_id = model.removeprefix(self._model_prefix)
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
                self._base_url,
                data=payload,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                    "User-Agent": GATEWAY_UA,
                },
            )
            try:
                with urllib.request.urlopen(request, timeout=GATEWAY_TIMEOUT_S) as response:
                    data = json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as error:
                # ponytail: 3 attempts on transient errors; back off linearly.
                if attempt < 2 and error.code in RETRYABLE_STATUS:
                    time.sleep(2 * (attempt + 1))
                    continue
                body = error.read().decode("utf-8", "replace")
                raise RuntimeError(f"{self._label} HTTP {error.code}: {body[:300]}") from error
            except (urllib.error.URLError, TimeoutError) as error:
                # Network-level failures (read timeout, connection reset).
                if attempt < 2:
                    time.sleep(2 * (attempt + 1))
                    continue
                raise RuntimeError(f"{self._label} request failed: {error}") from error
            # Gateways also return HTTP 200 with an upstream error body.
            error = data.get("error")
            if error is None:
                break
            code = error.get("code") if isinstance(error, dict) else None
            if attempt < 2 and code in RETRYABLE_STATUS:
                time.sleep(2 * (attempt + 1))
                data = None
                continue
            break
        latency_ms = (time.perf_counter() - started) * 1000
        if data is None:
            raise RuntimeError(f"{self._label} request failed without a response")
        if data.get("error") is not None:
            raise RuntimeError(f"{self._label} error: {data['error']}")
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"{self._label} response had no choices: {str(data)[:300]}")

        choice = choices[0]
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


class OpenRouterClient(OpenAICompatibleClient):
    """OpenRouter gateway (``openrouter/...`` model ids)."""

    def __init__(self, api_key: str) -> None:
        super().__init__(
            api_key,
            base_url=OPENROUTER_URL,
            model_prefix=OPENROUTER_PREFIX,
            label="OpenRouter",
        )


class OpenCodeZenClient(OpenAICompatibleClient):
    """OpenCode Zen gateway (``opencode/...`` model ids)."""

    def __init__(self, api_key: str) -> None:
        super().__init__(
            api_key,
            base_url=OPENCODE_ZEN_URL,
            model_prefix=OPENCODE_PREFIX,
            label="OpenCode Zen",
        )


class RoutedClient:
    """Dispatches per model prefix: openrouter/ and opencode/, else Anthropic."""

    def __init__(
        self,
        anthropic_client: AnthropicClient | None,
        openrouter_client: OpenRouterClient | None,
        opencode_client: OpenCodeZenClient | None = None,
    ) -> None:
        self._anthropic = anthropic_client
        self._openrouter = openrouter_client
        self._opencode = opencode_client

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
            client = self._openrouter
        elif model.startswith(OPENCODE_PREFIX):
            if self._opencode is None:
                raise ValueError("OPENCODE_API_KEY is not set")
            client = self._opencode
        else:
            if self._anthropic is None:
                raise ValueError("ANTHROPIC_API_KEY is not set")
            client = self._anthropic
        return client.complete(
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
    opencode_client = (
        OpenCodeZenClient(settings.opencode_api_key) if settings.opencode_api_key else None
    )
    return RoutedClient(anthropic_client, openrouter_client, opencode_client)
