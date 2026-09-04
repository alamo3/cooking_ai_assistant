from __future__ import annotations

from cooking_assistant_ai.llm.client import _is_think_error, _keep_alive_value, _think_value, default_think_for


def test_keep_alive_accepts_durations_and_seconds():
    assert _keep_alive_value("30m") == "30m"
    assert _keep_alive_value("-1") == -1 and isinstance(_keep_alive_value("-1"), int)
    assert _keep_alive_value(" 600 ") == 600


def test_think_values():
    assert _think_value("false") is False and _think_value("off") is False
    assert _think_value("true") is True
    assert _think_value("none") is None and _think_value("") is None
    assert _think_value("low") == "low" and _think_value("HIGH ") == "high"


def test_reasoning_default_depends_on_the_model():
    # gpt-oss is built around reasoning: off, it thrashes and takes ten times as long.
    assert default_think_for("kitchen:gpt-oss") == "low"
    assert default_think_for("gpt-oss:20b") == "low"
    # hybrid models keep their chain-of-thought off, so it never reaches the cook
    assert default_think_for("qwen-kitchen") is False
    assert default_think_for("qwen3:14b") is False


def test_think_error_detection():
    assert _is_think_error(Exception('registry.ollama.ai: "think" not supported'))
    assert _is_think_error(Exception("model does not support reasoning"))
    assert not _is_think_error(Exception("connection refused"))
