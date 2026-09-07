"""Cloud first, and no local model in VRAM until the cloud actually fails.

The local model costs ~19 GB of resident VRAM. Warming it at startup for a backend that may
never be used is the whole problem this covers, so the test asserts the local side is not
merely unwarmed but never even constructed.
"""
from __future__ import annotations

import pytest

from cooking_assistant_ai.llm.client import LLM
from cooking_assistant_ai.llm.openrouter import FallbackLLM, default_reasoning_for


class FakeCloud(LLM):
    def __init__(self, fail: bool = False):
        self.fail = fail
        self.warmed = False

    async def stream(self, messages, tools=None):
        if self.fail:
            raise RuntimeError("no internet")
        yield type("C", (), {"text": "from the cloud"})()

    async def warm(self):
        self.warmed = True


class FakeLocal(LLM):
    built = 0

    def __init__(self):
        FakeLocal.built += 1
        self.warmed = False

    async def stream(self, messages, tools=None):
        yield type("C", (), {"text": "from the local model"})()

    async def warm(self):
        self.warmed = True  # would load ~19 GB into VRAM


@pytest.fixture(autouse=True)
def _reset():
    FakeLocal.built = 0


def test_cloud_is_the_default_backend(monkeypatch):
    monkeypatch.delenv("COOK_LLM", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    from cooking_assistant_ai.llm.factory import build_llm

    llm = build_llm()
    assert isinstance(llm, FallbackLLM), "cloud should be the default, with a local safety net"
    assert not llm.secondary_loaded, "the local model must not be built at startup"


def test_no_key_degrades_to_local_rather_than_failing(monkeypatch):
    monkeypatch.delenv("COOK_LLM", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    from cooking_assistant_ai.llm.factory import build_llm

    llm = build_llm()
    assert type(llm).__name__ == "OllamaLLM", "a missing key must not stop the kitchen working"


@pytest.mark.asyncio
async def test_warming_never_touches_the_local_model():
    llm = FallbackLLM(FakeCloud(), FakeLocal)
    await llm.warm()
    assert llm.primary.warmed
    assert FakeLocal.built == 0, "warm-up constructed the local model and would hold its VRAM"


@pytest.mark.asyncio
async def test_the_local_model_is_built_on_the_first_failure_only():
    llm = FallbackLLM(FakeCloud(fail=True), FakeLocal)
    assert FakeLocal.built == 0

    said = [c.text async for c in llm.stream([])]
    assert said == ["from the local model"] and llm.fallbacks == 1
    assert FakeLocal.built == 1 and llm.secondary_loaded

    said = [c.text async for c in llm.stream([])]      # a second failure reuses it
    assert said == ["from the local model"]
    assert FakeLocal.built == 1, "the local model was rebuilt instead of reused"


@pytest.mark.asyncio
async def test_a_healthy_cloud_never_builds_the_local_model():
    llm = FallbackLLM(FakeCloud(), FakeLocal)
    for _ in range(3):
        assert [c.text async for c in llm.stream([])] == ["from the cloud"]
    assert FakeLocal.built == 0 and llm.fallbacks == 0
    assert not llm.secondary_loaded


@pytest.mark.asyncio
async def test_closing_does_not_build_a_local_model_just_to_close_it():
    llm = FallbackLLM(FakeCloud(), FakeLocal)
    await llm.aclose()
    assert FakeLocal.built == 0


def test_reasoning_default_matches_the_model():
    # Gemini's endpoint rejects disabling reasoning; everything else pays 262 s for it.
    assert default_reasoning_for("google/gemini-3.8-flash") == "low"
    assert default_reasoning_for("z-ai/glm-5.3-flash") == "off"
