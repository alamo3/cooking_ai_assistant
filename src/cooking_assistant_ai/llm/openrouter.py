"""OpenRouter (and any OpenAI-compatible) backend for the same LLM interface.

The orchestrator builds messages in Ollama's shape, so this module translates them to the
OpenAI chat-completions shape on the way out:

* assistant tool calls need an ``id`` and a JSON *string* for ``arguments``
* tool results are matched to that id via ``tool_call_id`` rather than a name

Tool schemas already match: ``ToolSpec.ollama_schema()`` emits ``{"type": "function",
"function": {...}}``, which is exactly what OpenAI expects.

Usage and cost are accumulated per instance so a meal's spend can be reported.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Union

from cooking_assistant_ai.llm.client import LLM, Chunk, ToolCallRequest

log = logging.getLogger(__name__)

DEFAULT_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
# Chosen by the messy-recovery benchmark: fast, strong at replanning, and cheap enough that
# a week of cooking costs pence ($0.75/M in, $3.75/M out).
DEFAULT_OR_MODEL = os.environ.get("COOK_OPENROUTER_MODEL", "google/gemini-3.8-flash")


def default_reasoning_for(model: str) -> Optional[str]:
    """Reasoning effort when COOK_OPENROUTER_REASONING is not set.

    Off wherever it is allowed: a hidden chain of thought measured 11,621 reasoning tokens
    and 262 s for 90 spoken words, which is unusable when someone is standing at a hob.
    Gemini's endpoint rejects disabling it outright ("Reasoning is mandatory"), so it gets
    the smallest setting it will accept instead.
    """
    return "low" if "gemini" in model.lower() else "off"


@dataclass
class Usage:
    """Token and cost totals across every request this client has made."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    requests: int = 0

    def add(self, raw: Dict[str, Any]) -> None:
        self.requests += 1
        self.prompt_tokens += int(raw.get("prompt_tokens") or 0)
        self.completion_tokens += int(raw.get("completion_tokens") or 0)
        cost = raw.get("cost")
        if cost is None:
            cost = (raw.get("cost_details") or {}).get("upstream_inference_cost")
        if cost is not None:
            self.cost_usd += float(cost)

    def summary(self) -> str:
        return (f"{self.requests} requests, {self.prompt_tokens} in / {self.completion_tokens} out, "
                f"${self.cost_usd:.4f}")


def to_openai_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Ollama-shaped message list -> OpenAI-shaped, wiring up tool_call ids."""
    out: List[Dict[str, Any]] = []
    pending: List[str] = []  # ids of the most recent assistant's tool calls, in order
    counter = 0
    for m in messages:
        role = m.get("role")
        if role == "assistant" and m.get("tool_calls"):
            calls = []
            pending = []
            for c in m["tool_calls"]:
                fn = c.get("function", {})
                call_id = c.get("id") or f"call_{counter}"
                counter += 1
                pending.append(call_id)
                args = fn.get("arguments", {})
                calls.append({
                    "id": call_id,
                    "type": "function",
                    "function": {"name": fn.get("name"),
                                 "arguments": args if isinstance(args, str) else json.dumps(args)},
                })
            out.append({"role": "assistant", "content": m.get("content") or None, "tool_calls": calls})
        elif role == "tool":
            call_id = m.get("tool_call_id") or (pending.pop(0) if pending else f"call_{counter}")
            out.append({"role": "tool", "tool_call_id": call_id, "content": m.get("content", "")})
        else:
            out.append({"role": role, "content": m.get("content", "")})
    return out


class OpenRouterLLM(LLM):
    def __init__(self, model: str = DEFAULT_OR_MODEL, api_key: Optional[str] = None,
                 base_url: str = DEFAULT_BASE_URL, temperature: float = 0.3,
                 timeout: float = 120.0, reasoning_effort: Optional[str] = None,
                 app_title: str = "Cooking Assistant", transport: Any = None,
                 retries: int = 2, retry_base: float = 0.5, max_tokens: int = 4096):
        # Short backoff on purpose: a cook waiting at the stove is better served by failing
        # over to the local model in ~1.5 s than by three polite retries against a busy
        # provider. FallbackLLM catches what these retries do not.
        import httpx

        self.model = model
        self.api_key = api_key if api_key is not None else os.environ.get("OPENROUTER_API_KEY", "")
        if not self.api_key and transport is None:
            raise RuntimeError("no OpenRouter API key: set OPENROUTER_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.reasoning_effort = (reasoning_effort
                                 or os.environ.get("COOK_OPENROUTER_REASONING")
                                 or default_reasoning_for(self.model))
        self.retries = retries
        self.retry_base = retry_base
        self.max_tokens = int(os.environ.get("COOK_OPENROUTER_MAX_TOKENS", max_tokens))
        self.usage = Usage()
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-Title": app_title,  # OpenRouter attribution, not required
        }
        self._client = httpx.AsyncClient(base_url=self.base_url, headers=headers,
                                         timeout=timeout, transport=transport)

    # -- request building ---------------------------------------------------

    def _body(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]],
              stream: bool) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "model": self.model,
            "messages": to_openai_messages(messages),
            "temperature": self.temperature,
            # Without this, providers default max_tokens to the whole remaining context and
            # some then reject their own default ("max_tokens: 100352 exceeds maximum
            # 98304"). A kitchen reply plus its tool calls needs a fraction of this.
            "max_tokens": self.max_tokens,
            "stream": stream,
        }
        if tools:
            body["tools"] = tools
        if stream:
            body["stream_options"] = {"include_usage": True}
        if self.reasoning_effort:
            # "off" is the cloud equivalent of Ollama's think=False. Qwen3.5/3.6/3.7 and GLM
            # all reason by default, which costs minutes per turn across 5-9 tool rounds.
            if self.reasoning_effort in ("off", "false", "none", "disabled"):
                body["reasoning"] = {"enabled": False}
            else:
                body["reasoning"] = {"effort": self.reasoning_effort}
        return body

    def _backoff(self, attempt: int) -> float:
        return min(self.retry_base * (2 ** attempt), 8.0)

    @staticmethod
    def _retryable(status: int) -> bool:
        # 429 is a rate limit (common on free tiers); 5xx is the provider having a moment.
        return status == 429 or 500 <= status < 600

    async def _post(self, body: Dict[str, Any]):
        import asyncio

        last = ""
        for attempt in range(self.retries + 1):
            r = await self._client.post("/chat/completions", json=body)
            if r.status_code < 400:
                return r
            last = f"OpenRouter {r.status_code}: {r.text[:300]}"
            if not self._retryable(r.status_code) or attempt == self.retries:
                break
            delay = self._backoff(attempt)
            log.warning("%s; retrying in %.1fs", last[:120], delay)
            await asyncio.sleep(delay)
        raise RuntimeError(last)

    # -- streaming ----------------------------------------------------------

    async def stream(self, messages: List[Dict[str, Any]],
                     tools: Optional[List[Dict[str, Any]]] = None) -> AsyncIterator[Chunk]:
        import asyncio

        body = self._body(messages, tools, stream=True)
        # Tool call arguments arrive as fragments keyed by index; assemble then emit at the end.
        partial: Dict[int, Dict[str, str]] = {}
        # One request per attempt: the response is consumed inside the same context it was
        # opened in, so a retry never leaves a half-read stream (or pays for two).
        for attempt in range(self.retries + 1):
            delay = 0.0
            async with self._client.stream("POST", "/chat/completions", json=body) as response:
                if response.status_code >= 400:
                    detail = (await response.aread()).decode("utf-8", "replace")[:300]
                    err = f"OpenRouter {response.status_code}: {detail}"
                    if not self._retryable(response.status_code) or attempt == self.retries:
                        raise RuntimeError(err)
                    delay = self._backoff(attempt)
                    log.warning("%s; retrying in %.1fs", err[:120], delay)
                else:
                    async for line in response.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            event = json.loads(data)
                        except ValueError:
                            continue
                        if event.get("usage"):
                            self.usage.add(event["usage"])
                        choices = event.get("choices") or []
                        if not choices:
                            continue
                        delta = choices[0].get("delta") or {}
                        for tc in delta.get("tool_calls") or []:
                            slot = partial.setdefault(int(tc.get("index", 0)), {"name": "", "arguments": ""})
                            fn = tc.get("function") or {}
                            if fn.get("name"):
                                slot["name"] = fn["name"]
                            if fn.get("arguments"):
                                slot["arguments"] += fn["arguments"]
                        text = delta.get("content") or ""
                        if text:
                            yield Chunk(text=text)
            if delay == 0.0:
                break
            await asyncio.sleep(delay)

        calls: List[ToolCallRequest] = []
        for _index, slot in sorted(partial.items()):
            if not slot["name"]:
                continue
            try:
                args = json.loads(slot["arguments"] or "{}")
            except ValueError:
                log.warning("unparsable tool arguments from %s: %r", self.model, slot["arguments"][:200])
                args = {}
            calls.append(ToolCallRequest(name=slot["name"], args=dict(args or {})))
        yield Chunk(tool_calls=calls, done=True)

    # -- one-shot -----------------------------------------------------------

    async def search(self, messages: List[Dict[str, Any]],
                     json_schema: Optional[Dict[str, Any]] = None,
                     max_results: int = 4) -> str:
        """One completion with OpenRouter's web plugin enabled.

        Kept off the main conversation deliberately: search is billed per result (roughly
        $0.004 each), so it runs only on the turn that actually needs the internet rather
        than on every "what's next?".
        """
        body = self._body(messages, None, stream=False)
        body["plugins"] = [{"id": "web", "max_results": max_results}]
        if json_schema:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "result", "strict": False, "schema": json_schema},
            }
        r = await self._post(body)
        payload = r.json()
        if payload.get("usage"):
            self.usage.add(payload["usage"])  # search results are billed too
        choices = payload.get("choices") or []
        if not choices:
            return ""
        return ((choices[0].get("message") or {}).get("content") or "").strip()

    async def complete(self, messages: List[Dict[str, Any]],
                       json_schema: Optional[Dict[str, Any]] = None) -> str:
        body = self._body(messages, None, stream=False)
        if json_schema:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "result", "strict": False, "schema": json_schema},
            }
        r = await self._post(body)
        payload = r.json()
        if payload.get("usage"):
            self.usage.add(payload["usage"])
        choices = payload.get("choices") or []
        if not choices:
            return ""
        return (choices[0].get("message") or {}).get("content") or ""

    async def warm(self) -> None:
        return None  # nothing to load; the provider is always ready

    async def aclose(self) -> None:
        await self._client.aclose()


class FallbackLLM(LLM):
    """Try `primary`, fall back to `secondary` on failure.

    A stream only falls back if it fails before emitting anything, so the cook never hears
    half an answer from one model followed by a whole answer from another.

    `secondary` may be a callable returning an LLM instead of an LLM. It is then built on the
    first fallback and not before, which is what keeps a local model out of VRAM while the
    cloud is healthy: constructing OllamaLLM is cheap, but warming it loads ~19 GB of weights
    that would then sit there for the whole keep_alive window doing nothing.
    """

    def __init__(self, primary: LLM, secondary: Union[LLM, Callable[[], LLM]]):
        self.primary = primary
        self._make_secondary = secondary
        # An instance is used as-is; anything else callable is a factory. Checking for a
        # .stream attribute would not do: a class has one too, and would be mistaken for a
        # built instance, which is exactly the eager construction this avoids.
        already_built = isinstance(secondary, LLM) or not callable(secondary)
        self._secondary: Optional[LLM] = secondary if already_built else None  # type: ignore[assignment]
        self.fallbacks = 0

    @property
    def secondary(self) -> LLM:
        if self._secondary is None:
            log.info("first fallback: building the local backend now")
            self._secondary = self._make_secondary()  # type: ignore[operator]
        return self._secondary

    @property
    def secondary_loaded(self) -> bool:
        """False while the cloud has never failed, i.e. while no VRAM is being held."""
        return self._secondary is not None

    async def stream(self, messages: List[Dict[str, Any]],
                     tools: Optional[List[Dict[str, Any]]] = None) -> AsyncIterator[Chunk]:
        started = False
        try:
            async for chunk in self.primary.stream(messages, tools):
                started = True
                yield chunk
            return
        except Exception as e:
            if started:
                raise
            self.fallbacks += 1
            log.warning("primary LLM failed (%s); falling back to %s", e, type(self.secondary).__name__)
        async for chunk in self.secondary.stream(messages, tools):
            yield chunk

    async def complete(self, messages: List[Dict[str, Any]],
                       json_schema: Optional[Dict[str, Any]] = None) -> str:
        try:
            return await self.primary.complete(messages, json_schema)
        except Exception as e:
            self.fallbacks += 1
            log.warning("primary LLM failed (%s); falling back", e)
            return await self.secondary.complete(messages, json_schema)

    async def search(self, messages, json_schema=None, max_results: int = 4) -> str:
        search = getattr(self.primary, "search", None)
        if search is None:
            raise RuntimeError("this backend cannot search the web")
        return await search(messages, json_schema, max_results)

    async def warm(self) -> None:
        """Warm the primary only. Warming the local fallback would load the model into VRAM
        at startup for a backend that may never be used."""
        warm = getattr(self.primary, "warm", None)
        if warm:
            try:
                await warm()
            except Exception as e:  # pragma: no cover - warming is best effort
                log.warning("warm-up failed for %s: %s", type(self.primary).__name__, e)

    async def aclose(self) -> None:
        built = [self.primary] + ([self._secondary] if self._secondary is not None else [])
        for llm in built:
            close = getattr(llm, "aclose", None)
            if close:
                await close()
