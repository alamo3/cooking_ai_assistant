"""Pick the LLM backend from configuration.

COOK_LLM:
  ollama      (default) local Ollama, COOK_MODEL
  openrouter  OpenRouter, COOK_OPENROUTER_MODEL, key from OPENROUTER_API_KEY
  cloud       cloud first, local Ollama whenever it errors (recommended for a kitchen:
              the meal continues if the internet drops mid-cook)
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from cooking_assistant_ai.llm.client import DEFAULT_MODEL, LLM, OllamaLLM
from cooking_assistant_ai.llm.openrouter import DEFAULT_OR_MODEL, FallbackLLM, OpenRouterLLM

log = logging.getLogger(__name__)


def build_llm(kind: Optional[str] = None, model: Optional[str] = None) -> LLM:
    kind = (kind or os.environ.get("COOK_LLM", "ollama")).lower()
    if kind in ("ollama", "local", ""):
        return OllamaLLM(model=model or DEFAULT_MODEL)
    if kind in ("openrouter", "or"):
        return OpenRouterLLM(model=os.environ.get("COOK_OPENROUTER_MODEL", DEFAULT_OR_MODEL))
    if kind in ("cloud", "fallback", "hybrid"):
        local = OllamaLLM(model=model or DEFAULT_MODEL)
        try:
            remote = OpenRouterLLM(model=os.environ.get("COOK_OPENROUTER_MODEL", DEFAULT_OR_MODEL))
        except RuntimeError as e:
            log.warning("cloud backend unavailable (%s); using local only", e)
            return local
        return FallbackLLM(remote, local)
    raise ValueError(f"unknown COOK_LLM '{kind}' (ollama | openrouter | cloud)")
