"""The cook log.

Two cooks went badly and both post-mortems were guesswork, because the only record was a
session snapshot that keeps forty turns and then overwrites itself. These pin the properties
that make the log worth trusting: it captures rejections (the interesting part), it survives
a crash because it is written as it happens, and it can never take the kitchen down.
"""
from __future__ import annotations

import json

from cooking_assistant_ai.storage.journal import Journal, list_logs, render, summarize


def test_a_cook_is_recorded_as_it_happens(tmp_path):
    j = Journal("s1", directory=tmp_path)
    j.heard("what's next?")
    j.tool("add_task", {"label": "rice", "appliance": "rice_cooker"}, True, "added rice")
    j.tool("set_timer", {"task_id": "rice"}, False, "finishes when it finishes")
    j.said("Rice is on. Tell me when it clicks off.")

    # written immediately, not at the end: a power cut must not lose the cook
    lines = j.path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 4 and all(json.loads(x)["at"] for x in lines)

    text = render(j.path)
    assert "COOK: what's next?" in text
    assert "ok  add_task" in text and "REJECTED set_timer" in text
    assert "finishes when it finishes" in text
    assert "SAID: Rice is on." in text


def test_the_summary_surfaces_rejections(tmp_path):
    j = Journal("s2", directory=tmp_path)
    j.heard("go")
    for _ in range(3):
        j.tool("add_task", {}, True, "ok")
    j.tool("add_task", {"appliance": "air_fryer"}, False, "this kitchen has no air fryer")

    info = summarize(j.path)
    assert info["turns"] == 1 and info["tool_calls"] == 4 and info["rejected"] == 1
    assert info["by_tool"]["add_task"] == 4
    assert "no air fryer" in info["rejections"][0]


def test_a_broken_log_never_stops_the_cook(tmp_path):
    j = Journal("s3", directory=tmp_path / "nope")
    j.directory = tmp_path / "file-not-a-dir"
    j.directory.write_text("in the way", encoding="utf-8")
    j.heard("this must not raise")       # directory cannot be created
    assert j._broken
    j.said("and neither must this")      # still silent once disabled


def test_unserialisable_arguments_do_not_break_a_line(tmp_path):
    j = Journal("s4", directory=tmp_path)
    j.tool("odd", {"when": object(), "big": "x" * 5000}, True, "")
    rec = json.loads(j.path.read_text(encoding="utf-8").strip())
    assert rec["name"] == "odd" and "truncated" in json.dumps(rec["args"])


def test_a_torn_final_line_is_skipped(tmp_path):
    """A hard kill mid-write leaves half a line; reading must not explode."""
    j = Journal("s5", directory=tmp_path)
    j.heard("one")
    with j.path.open("a", encoding="utf-8") as fh:
        fh.write('{"kind": "cook", "text": "hal')
    assert summarize(j.path)["turns"] == 1
    assert "one" in render(j.path)


def test_logs_are_listed_newest_first(tmp_path):
    for sid in ("a", "b"):
        Journal(sid, directory=tmp_path).heard("hi")
    names = [p.name for p in list_logs(tmp_path)]
    assert len(names) == 2 and names == sorted(names, reverse=True)
