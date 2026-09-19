"""Running out of max_tokens must not look like a broken model.

`max_tokens` is one budget shared by thinking and reply. Measured against a real recipe at a
small budget, overrunning it produced JSON stopping mid-string with reasoning off, and an
entirely empty `content` with reasoning on - the latter reaching the caller as
"Expecting value: line 1 column 1 (char 0)", which explains nothing.
"""
from __future__ import annotations

import json

import httpx
import pytest

from cooking_assistant_ai.llm.openrouter import MAX_TOKENS_CEILING, OpenRouterLLM

SCHEMA = {"type": "object", "properties": {"x": {"type": "string"}}}


def sse(*events: dict) -> bytes:
    body = "".join(f"data: {json.dumps(e)}\n\n" for e in events)
    return (body + "data: [DONE]\n\n").encode()


def make(handler) -> OpenRouterLLM:
    return OpenRouterLLM(model="test/model", api_key="k", transport=httpx.MockTransport(handler))


def reply(content: str, finish: str = "stop", reasoning_tokens: int = 0) -> httpx.Response:
    return httpx.Response(200, json={
        "choices": [{"message": {"content": content}, "finish_reason": finish}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30,
                  "completion_tokens_details": {"reasoning_tokens": reasoning_tokens}},
    })


# --------------------------------------------------------------------- one-shot JSON

@pytest.mark.asyncio
async def test_truncated_json_is_retried_on_a_bigger_budget():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body["max_tokens"])
        if len(seen) == 1:
            return reply('{"x": "half a str', finish="length")
        return reply('{"x": "whole"}')

    llm = make(handler)
    assert json.loads(await llm.complete([{"role": "user", "content": "go"}], SCHEMA)) == {"x": "whole"}
    assert seen == [4096, 16384], seen


@pytest.mark.asyncio
async def test_empty_content_from_reasoning_is_the_same_fault():
    """With reasoning on, the budget goes to thinking and content comes back empty."""
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content)["max_tokens"])
        if len(seen) == 1:
            return reply("", finish="length", reasoning_tokens=4096)
        return reply('{"x": "thought about it"}')

    llm = make(handler)
    out = await llm.complete([{"role": "user", "content": "go"}], SCHEMA)
    assert json.loads(out) == {"x": "thought about it"}
    assert len(seen) == 2


@pytest.mark.asyncio
async def test_giving_up_says_what_to_change():
    def handler(request: httpx.Request) -> httpx.Response:
        return reply("", finish="length", reasoning_tokens=9000)

    llm = make(handler)
    with pytest.raises(RuntimeError) as caught:
        await llm.complete([{"role": "user", "content": "go"}], SCHEMA)
    message = str(caught.value)
    assert "COOK_OPENROUTER_MAX_TOKENS" in message
    assert "COOK_OPENROUTER_REASONING" in message
    assert "9000" in message  # how much went to thinking rather than answering


@pytest.mark.asyncio
async def test_the_retry_never_exceeds_the_ceiling():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content)["max_tokens"])
        return reply("{", finish="length")

    llm = make(handler)
    llm.max_tokens = MAX_TOKENS_CEILING // 2
    with pytest.raises(RuntimeError):
        await llm.complete([{"role": "user", "content": "go"}], SCHEMA)
    assert max(seen) <= MAX_TOKENS_CEILING


@pytest.mark.asyncio
async def test_a_whole_answer_is_not_retried():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return reply('{"x": "fine"}')

    llm = make(handler)
    await llm.complete([{"role": "user", "content": "go"}], SCHEMA)
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_truncated_prose_is_returned_rather_than_retried():
    """Without a schema a cut-off answer is still readable, and a redo costs real money."""
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return reply("the sauce should be reduced until it", finish="length")

    llm = make(handler)
    out = await llm.complete([{"role": "user", "content": "go"}])
    assert out.startswith("the sauce")
    assert len(calls) == 1


# ----------------------------------------------------------------------- streaming

@pytest.mark.asyncio
async def test_a_tool_call_cut_off_mid_arguments_is_dropped():
    """A set_timer whose arguments were truncated would otherwise start a timer with no
    duration, and the cook would be told it was running."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=sse(
            {"choices": [{"delta": {"tool_calls": [
                {"index": 0, "function": {"name": "set_timer",
                                          "arguments": '{"label": "rice", "seco'}}]}}]},
            {"choices": [{"delta": {}, "finish_reason": "length"}]},
        ))

    llm = make(handler)
    chunks = [c async for c in llm.stream([{"role": "user", "content": "go"}])]
    assert chunks[-1].tool_calls == []


@pytest.mark.asyncio
async def test_a_complete_tool_call_still_arrives():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=sse(
            {"choices": [{"delta": {"tool_calls": [
                {"index": 0, "function": {"name": "set_timer",
                                          "arguments": '{"label": "rice", "seconds": 600}'}}]}}]},
            {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
        ))

    llm = make(handler)
    chunks = [c async for c in llm.stream([{"role": "user", "content": "go"}])]
    assert chunks[-1].tool_calls[0].name == "set_timer"
    assert chunks[-1].tool_calls[0].args["seconds"] == 600
