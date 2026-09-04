"""LLM client abstraction. The orchestrator only sees `stream(messages, tools)`.

OllamaLLM talks to a local Ollama server. ScriptedLLM replays canned responses so the
whole loop (tools, timers, proactive turns) can be tested without a model.
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Sequence, Union

log = logging.getLogger(__name__)


def _is_think_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "think" in text or "reasoning" in text

DEFAULT_MODEL = os.environ.get("COOK_MODEL", "kitchen:gemma-q8")
DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_NUM_CTX = int(os.environ.get("COOK_NUM_CTX", "16384"))  # spec budgets ~4.3K per turn at 16K


def _keep_alive_value(raw: str) -> Union[int, str]:
    """Ollama takes a duration string ("30m") or a number of seconds; -1 (int) means forever."""
    raw = raw.strip()
    try:
        return int(raw)
    except ValueError:
        return raw


# How long Ollama keeps the model in VRAM after a request. Pinning it forever ("-1") starves
# the desktop compositor and anything else using the GPU between meals. 30m stays resident
# through a whole cook and releases afterwards; the next question then pays one reload.
DEFAULT_KEEP_ALIVE: Union[int, str] = _keep_alive_value(os.environ.get("COOK_KEEP_ALIVE", "30m"))


def _think_value(raw: str) -> Union[bool, str, None]:
    """COOK_THINK: 'false' (default) hides chain-of-thought on hybrid models like Qwen3;
    'low' / 'medium' / 'high' set the reasoning effort of models that need one, such as
    gpt-oss; 'none' omits the field entirely."""
    raw = raw.strip().lower()
    if raw in ("false", "off", "0", "no"):
        return False
    if raw in ("true", "on", "1", "yes"):
        return True
    if raw in ("none", ""):
        return None
    return raw


_THINK_ENV = os.environ.get("COOK_THINK")
_UNSET = object()  # "caller said nothing", distinct from think=None ("omit the field")


def default_think_for(model: str) -> Union[bool, str, None]:
    """Reasoning setting for a model when COOK_THINK is not set.

    gpt-oss is built around reasoning: forcing it off made the planning turn take 150 s and
    thrash (one task, no briefing), while "low" finished in 11 s with a complete plan.
    Hybrid models such as Qwen3 are the opposite: their chain-of-thought is slow, adds
    nothing here, and must not reach the cook, so it stays off.
    """
    if _THINK_ENV is not None:
        return _think_value(_THINK_ENV)
    return "low" if "gpt-oss" in model.lower() else False


DEFAULT_THINK: Union[bool, str, None] = _think_value(_THINK_ENV) if _THINK_ENV is not None else False


@dataclass
class ToolCallRequest:
    name: str
    args: Dict[str, Any]


@dataclass
class Chunk:
    text: str = ""
    tool_calls: List[ToolCallRequest] = field(default_factory=list)
    done: bool = False


class LLM:
    """Protocol-ish base class."""

    async def stream(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None) -> AsyncIterator[Chunk]:  # pragma: no cover
        raise NotImplementedError
        yield Chunk()  # noqa: unreachable, keeps this an async generator

    async def complete(self, messages: List[Dict[str, Any]], json_schema: Optional[Dict[str, Any]] = None) -> str:  # pragma: no cover
        raise NotImplementedError


# --------------------------------------------------------------------------- Ollama

class OllamaLLM(LLM):
    def __init__(self, model: str = DEFAULT_MODEL, host: str = DEFAULT_HOST,
                 think: Any = _UNSET,
                 temperature: float = 0.3, num_ctx: Optional[int] = DEFAULT_NUM_CTX,
                 keep_alive: Union[int, str] = DEFAULT_KEEP_ALIVE):
        import ollama  # local import so tests never need the package's server

        self.model = model
        self.think: Union[bool, str, None] = default_think_for(model) if think is _UNSET else think
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.keep_alive = keep_alive
        self._client = ollama.AsyncClient(host=host)

    def _options(self) -> Dict[str, Any]:
        opts: Dict[str, Any] = {"temperature": self.temperature}
        if self.num_ctx:
            opts["num_ctx"] = self.num_ctx
        return opts

    async def stream(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None) -> AsyncIterator[Chunk]:
        kwargs: Dict[str, Any] = dict(
            model=self.model, messages=messages, stream=True,
            options=self._options(), keep_alive=self.keep_alive,
        )
        if tools:
            kwargs["tools"] = tools
        if self.think is not None:
            kwargs["think"] = self.think
        try:
            response = await self._client.chat(**kwargs)
        except Exception as e:
            # Some models refuse the think field outright (gpt-oss always reasons, and
            # non-reasoning models reject it). Drop it once and remember.
            if "think" not in kwargs or not _is_think_error(e):
                raise
            log.info("model %s rejected think=%r; disabling it", self.model, self.think)
            self.think = None
            kwargs.pop("think", None)
            response = await self._client.chat(**kwargs)
        async for part in response:
            msg = getattr(part, "message", None)
            calls: List[ToolCallRequest] = []
            for tc in (getattr(msg, "tool_calls", None) or []):
                fn = tc.function
                args = fn.arguments
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except ValueError:
                        args = {"_raw": args}
                calls.append(ToolCallRequest(name=fn.name, args=dict(args or {})))
            yield Chunk(text=(getattr(msg, "content", "") or ""), tool_calls=calls, done=bool(getattr(part, "done", False)))

    async def complete(self, messages: List[Dict[str, Any]], json_schema: Optional[Dict[str, Any]] = None) -> str:
        kwargs: Dict[str, Any] = dict(model=self.model, messages=messages, stream=False,
                                      options=self._options(), keep_alive=self.keep_alive)
        if json_schema:
            kwargs["format"] = json_schema
        if self.think is not None:
            kwargs["think"] = self.think
        resp = await self._client.chat(**kwargs)
        return resp.message.content or ""

    async def aclose(self) -> None:
        inner = getattr(self._client, "_client", None)
        if inner is not None and hasattr(inner, "aclose"):
            try:
                await inner.aclose()
            except Exception:  # pragma: no cover - best effort at shutdown
                pass

    async def warm(self) -> None:
        """Load the model into VRAM before the first real turn, with the same context size
        real turns use (a different num_ctx would make Ollama reload the model)."""
        await self._client.chat(model=self.model, messages=[{"role": "user", "content": "hi"}],
                                options={**self._options(), "num_predict": 1}, keep_alive=self.keep_alive)


# --------------------------------------------------------------------------- text fallback

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)

# Chat-template control tokens that occasionally leak into content. Left alone they are
# spoken aloud: a real reply began "<|tool_call> I've started the air fryer timer".
_CONTROL_MARKUP = re.compile(
    r"<\|[^|>]*\|?>?"                                  # <|tool_call|>, <|im_end|>, <|channel|>
    r"|</?(?:tool_call|tool_response|think|thinking|analysis|final|assistant|system)\b[^>]*>",
    re.I,
)


def strip_control_markup(text: str) -> str:
    """Remove leaked template tokens so they are never spoken or shown."""
    return _CONTROL_MARKUP.sub("", text)


def extract_text_tool_calls(text: str) -> "tuple[str, List[ToolCallRequest]]":
    """Some models emit tool calls as text. Pull them out; return (clean_text, calls)."""
    calls: List[ToolCallRequest] = []
    for raw in _TOOL_CALL_RE.findall(text):
        try:
            obj = json.loads(raw)
        except ValueError:
            continue
        name = obj.get("name")
        args = obj.get("arguments", obj.get("parameters", {}))
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError:
                args = {}
        if name:
            calls.append(ToolCallRequest(name=str(name), args=dict(args or {})))
    clean = _TOOL_CALL_RE.sub("", text).strip()
    return clean, calls


# --------------------------------------------------------------------------- scripted (tests)

ScriptItem = Union[str, Sequence[Chunk], Callable[[List[Dict[str, Any]]], Sequence[Chunk]]]


def say(text: str) -> List[Chunk]:
    """Split text into a few streamed chunks."""
    words = text.split(" ")
    out: List[Chunk] = []
    for i in range(0, len(words), 3):
        out.append(Chunk(text=" ".join(words[i:i + 3]) + (" " if i + 3 < len(words) else "")))
    if out:
        out[-1].done = True
    return out


def call(name: str, **args: Any) -> List[Chunk]:
    return [Chunk(tool_calls=[ToolCallRequest(name, args)], done=True)]


class ScriptedLLM(LLM):
    """Replays scripted responses in order. Each item is text, a chunk list, or a callable."""

    def __init__(self, script: Optional[List[ScriptItem]] = None):
        self.script: List[ScriptItem] = list(script or [])
        self.calls: List[List[Dict[str, Any]]] = []

    def push(self, *items: ScriptItem) -> None:
        self.script.extend(items)

    async def stream(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None) -> AsyncIterator[Chunk]:
        self.calls.append(list(messages))
        if not self.script:
            yield Chunk(text="(no scripted response)", done=True)
            return
        item = self.script.pop(0)
        if callable(item) and not isinstance(item, (str, list, tuple)):
            chunks = item(messages)
        elif isinstance(item, str):
            chunks = say(item)
        else:
            chunks = item
        if hasattr(chunks, "__aiter__"):
            async for c in chunks:  # type: ignore[union-attr]
                yield c
        else:
            for c in chunks:
                yield c

    async def complete(self, messages: List[Dict[str, Any]], json_schema: Optional[Dict[str, Any]] = None) -> str:
        self.calls.append(list(messages))
        item = self.script.pop(0) if self.script else "{}"
        return item if isinstance(item, str) else "".join(c.text for c in item)  # type: ignore[union-attr]
