from __future__ import annotations

import json

import httpx
import pytest

from cooking_assistant_ai.llm.client import Chunk, ScriptedLLM
from cooking_assistant_ai.llm.openrouter import FallbackLLM, OpenRouterLLM, to_openai_messages


def sse(*events: dict) -> bytes:
    body = "".join(f"data: {json.dumps(e)}\n\n" for e in events)
    return (body + "data: [DONE]\n\n").encode()


def make(handler) -> OpenRouterLLM:
    return OpenRouterLLM(model="test/model", api_key="k", transport=httpx.MockTransport(handler))


# --------------------------------------------------------------- message translation

def test_ollama_messages_become_openai_messages_with_tool_call_ids():
    converted = to_openai_messages([
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "plan it"},
        {"role": "assistant", "content": "",
         "tool_calls": [{"function": {"name": "add_task", "arguments": {"label": "rice"}}},
                        {"function": {"name": "get_plan", "arguments": {}}}]},
        {"role": "tool", "content": '{"ok": true}', "tool_name": "add_task"},
        {"role": "tool", "content": '{"ok": true}', "tool_name": "get_plan"},
    ])
    assistant = converted[2]
    assert [c["type"] for c in assistant["tool_calls"]] == ["function", "function"]
    # arguments must be a JSON string, not an object
    assert assistant["tool_calls"][0]["function"]["arguments"] == '{"label": "rice"}'
    ids = [c["id"] for c in assistant["tool_calls"]]
    # each tool result is matched to its call by id, in order
    assert converted[3]["tool_call_id"] == ids[0] and "tool_name" not in converted[3]
    assert converted[4]["tool_call_id"] == ids[1]


# --------------------------------------------------------------- streaming

async def test_stream_yields_text_then_assembled_tool_calls():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, content=sse(
            {"choices": [{"delta": {"content": "Planning "}}]},
            {"choices": [{"delta": {"content": "now."}}]},
            # arguments arrive in fragments and must be concatenated per index
            {"choices": [{"delta": {"tool_calls": [
                {"index": 0, "id": "c1", "function": {"name": "add_task", "arguments": '{"lab'}}]}}]},
            {"choices": [{"delta": {"tool_calls": [
                {"index": 0, "function": {"arguments": 'el": "rice"}'}}]}}]},
            {"usage": {"prompt_tokens": 1200, "completion_tokens": 40, "cost": 0.00031}, "choices": []},
        ), headers={"content-type": "text/event-stream"})

    llm = make(handler)
    tools = [{"type": "function", "function": {"name": "add_task", "parameters": {}}}]
    chunks = [c async for c in llm.stream([{"role": "user", "content": "plan"}], tools)]
    assert "".join(c.text for c in chunks) == "Planning now."
    calls = [c for chunk in chunks for c in chunk.tool_calls]
    assert len(calls) == 1 and calls[0].name == "add_task" and calls[0].args == {"label": "rice"}
    assert chunks[-1].done
    assert seen["body"]["tools"] == tools and seen["body"]["stream"] is True
    assert llm.usage.prompt_tokens == 1200 and llm.usage.cost_usd == pytest.approx(0.00031)
    await llm.aclose()


async def test_malformed_tool_arguments_do_not_crash_the_turn():
    def handler(_r):
        return httpx.Response(200, content=sse(
            {"choices": [{"delta": {"tool_calls": [
                {"index": 0, "id": "c1", "function": {"name": "get_plan", "arguments": "{not json"}}]}}]},
        ), headers={"content-type": "text/event-stream"})

    llm = make(handler)
    calls = [c for chunk in [x async for x in llm.stream([{"role": "user", "content": "x"}])] for c in chunk.tool_calls]
    assert len(calls) == 1 and calls[0].name == "get_plan" and calls[0].args == {}
    await llm.aclose()


async def test_http_error_is_raised_with_detail():
    llm = make(lambda _r: httpx.Response(402, text="insufficient credits"))
    llm.retries = 0
    with pytest.raises(RuntimeError, match="402"):
        [c async for c in llm.stream([{"role": "user", "content": "x"}])]
    await llm.aclose()


async def test_rate_limits_are_retried_but_client_errors_are_not():
    """Free tiers return 429 constantly; a 400 is our own bug and must surface at once."""
    attempts = {"n": 0}

    def flaky(_r):
        attempts["n"] += 1
        if attempts["n"] < 3:
            return httpx.Response(429, text="rate-limited upstream")
        return httpx.Response(200, content=sse({"choices": [{"delta": {"content": "ok"}}]}),
                              headers={"content-type": "text/event-stream"})

    llm = make(flaky)
    llm.retry_base = 0.01
    chunks = [c async for c in llm.stream([{"role": "user", "content": "x"}])]
    assert "".join(c.text for c in chunks) == "ok" and attempts["n"] == 3
    await llm.aclose()

    tries = {"n": 0}

    def bad_request(_r):
        tries["n"] += 1
        return httpx.Response(400, text="malformed tool schema")

    llm2 = make(bad_request)
    llm2.retry_base = 0.01
    with pytest.raises(RuntimeError, match="400"):
        [c async for c in llm2.stream([{"role": "user", "content": "x"}])]
    assert tries["n"] == 1  # not retried
    await llm2.aclose()


async def test_complete_sends_a_json_schema_and_records_usage():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={
            "choices": [{"message": {"content": '{"title": "Toast"}'}}],
            "usage": {"prompt_tokens": 900, "completion_tokens": 30, "cost": 0.0002},
        })

    llm = make(handler)
    out = await llm.complete([{"role": "user", "content": "extract"}], json_schema={"type": "object"})
    assert out == '{"title": "Toast"}'
    assert seen["body"]["response_format"]["json_schema"]["schema"] == {"type": "object"}
    assert llm.usage.completion_tokens == 30 and "$0.0002" in llm.usage.summary()
    await llm.aclose()


def test_missing_api_key_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        OpenRouterLLM(model="test/model")


# --------------------------------------------------------------- fallback

async def test_fallback_uses_local_when_the_cloud_fails_before_speaking():
    cloud = make(lambda _r: httpx.Response(500, text="upstream down"))
    local = ScriptedLLM(["Rice takes fifteen minutes."])
    llm = FallbackLLM(cloud, local)
    chunks = [c async for c in llm.stream([{"role": "user", "content": "how long"}])]
    assert "fifteen minutes" in "".join(c.text for c in chunks) and llm.fallbacks == 1
    await cloud.aclose()


async def test_fallback_does_not_restart_a_stream_that_already_spoke():
    """Half an answer from one model plus a whole answer from another would be worse."""
    def handler(_r):
        return httpx.Response(200, content=b'data: {"choices":[{"delta":{"content":"Half "}}]}\n\n',
                              headers={"content-type": "text/event-stream"})

    class Broken(OpenRouterLLM):
        async def stream(self, messages, tools=None):
            yield Chunk(text="Half ")
            raise RuntimeError("connection dropped")

    local = ScriptedLLM(["A whole different answer."])
    llm = FallbackLLM(Broken(model="m", api_key="k", transport=httpx.MockTransport(handler)), local)
    with pytest.raises(RuntimeError, match="connection dropped"):
        [c async for c in llm.stream([{"role": "user", "content": "x"}])]
    assert llm.fallbacks == 0
