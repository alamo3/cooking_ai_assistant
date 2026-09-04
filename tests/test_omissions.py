from __future__ import annotations

import pytest

from cooking_assistant_ai.core.claims import implied_state_change, missed_state_change, omission_prompt


def kind(text):
    o = implied_state_change(text)
    return o.kind if o else None


def test_statements_that_move_the_kitchen_on_are_detected():
    assert kind("the chicken is seared and it's going in the oven now") == "started"
    assert kind("rice is on and simmering with the lid on") == "started"
    assert kind("the sprouts are in the air fryer") == "started"
    assert kind("I've rinsed the rice") == "done"
    assert kind("that's done") == "done"
    assert kind("I'm skipping the resting step") == "skipped"


def test_questions_and_chatter_are_never_flagged():
    """The cook asks far more than they report; a false nudge wastes a turn."""
    for text in [
        "how long left on the chicken",
        "is the rice supposed to be covered",
        "what should I do while that's roasting",
        "should I start the brussels sprouts yet",
        "can I use dried thyme instead of fresh",
        "what temperature should the oven be",
        "how much salt in total across everything",
        "I'm out of butter, what should I use",
        "what's next",
        "remind me the oven temperature again",
        "[SYSTEM] Timer \"rice\" completed. Tell the cook what to do now.",
    ]:
        assert kind(text) is None, text


def test_an_omission_is_only_reported_when_no_matching_tool_ran():
    text = "the chicken is seared and it's going in the oven now"
    assert missed_state_change(text, []) is not None
    assert missed_state_change(text, ["get_plan"]) is not None      # unrelated tool: still missed
    assert missed_state_change(text, ["start_task"]) is None        # recorded
    assert missed_state_change(text, ["mark_complete"]) is None
    assert missed_state_change("how long left", []) is None


def test_the_nudge_asks_rather_than_dictates():
    o = implied_state_change("I've rinsed the rice")
    prompt = omission_prompt(o)
    assert prompt.startswith("[SYSTEM]") and "I've rinsed the rice" in prompt
    # it must leave room for the model to disagree or ask, not force a tool call
    assert "ask them" in prompt and "If it did not" in prompt
