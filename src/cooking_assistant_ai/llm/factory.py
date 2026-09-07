"""Pick the LLM backend from configuration.

COOK_LLM:
  cloud       (default) OpenRouter first, local Ollama only if it errors. The local model is
              not built or warmed until that happens, so it holds no VRAM while the internet
              is up, which is nearly always.
  openrouter  cloud only, no local safety net
  ollama      local only, COOK_MODEL

Cloud is the default because the local model costs ~19 GB of VRAM to keep resident and is
slower and weaker at planning, while a week of cooking on Gemini Flash costs a few pence.
Without an API key this degrades to local rather than failing: a kitchen assistant that
refuses to start because of a missing key is worse than a slower one.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from cooking_assistant_ai.llm.client import DEFAULT_MODEL, LLM, OllamaLLM
from cooking_assistant_ai.llm.openrouter import DEFAULT_OR_MODEL, FallbackLLM, OpenRouterLLM

log = logging.getLogger(__name__)

DEFAULT_BACKEND = "cloud"


def build_llm(kind: Optional[str] = None, model: Optional[str] = None) -> LLM:
    kind = (kind or os.environ.get("COOK_LLM", DEFAULT_BACKEND)).lower()
    if kind in ("ollama", "local"):
        return OllamaLLM(model=model or DEFAULT_MODEL)
    if kind in ("openrouter", "or"):
        return OpenRouterLLM(model=os.environ.get("COOK_OPENROUTER_MODEL", DEFAULT_OR_MODEL))
    if kind in ("cloud", "fallback", "hybrid", ""):
        try:
            remote = OpenRouterLLM(model=os.environ.get("COOK_OPENROUTER_MODEL", DEFAULT_OR_MODEL))
        except RuntimeError as e:
            log.warning("cloud backend unavailable (%s); using the local model instead", e)
            return OllamaLLM(model=model or DEFAULT_MODEL)
        # A factory, not an instance: nothing local is constructed, connected to or loaded
        # into VRAM unless a cloud request actually fails.
        return FallbackLLM(remote, lambda: OllamaLLM(model=model or DEFAULT_MODEL))
    raise ValueError(f"unknown COOK_LLM '{kind}' (cloud | openrouter | ollama)")
