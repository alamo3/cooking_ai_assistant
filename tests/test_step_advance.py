"""What the cook says about their own progress, and what is deliberately not inferred.

This file once also tested a reply-side detector that read the model's own wording and guessed
which step it had moved to. It is gone. Over a real cook with four recipes loaded it fired 33
times on shared cooking vocabulary - once on a reply about missing pantry items, once on the
literal reply "NOTHING" - and because each nudge costs a round whose first answer has already
been spoken, it made the assistant repeat itself. advance_step is the mechanism; guessing from
text was never going to be one.
"""
from __future__ import annotations

import pytest

from cooking_assistant_ai.core.claims import implied_state_change, missed_state_change


@pytest.mark.parametrize("said", [
    "Alright, I've mashed the tofu into the oil pan using my potato masher.",
    "Okay, I'm ready for the spices.",
    "Alright, sounds like we're done here.",
    "I've chopped the onions.",
    "That's done.",
    "Okay, I've added the spices.",
])
def test_the_cook_saying_they_did_it_registers(said):
    """All taken from real cooks, including three the verb list used to miss."""
    found = implied_state_change(said)
    assert found is not None, said
    assert {"advance_step", "mark_complete"} & found.needs


@pytest.mark.parametrize("said", [
    "So how much of the spices do I need?",
    "When are we supposed to add the milk?",
    "300 grams of tofu.",
    "I think the tofu needs a few more minutes.",
    "What am I supposed to season the noodles with?",
])
def test_questions_and_statements_of_fact_do_not(said):
    assert implied_state_change(said) is None, said


def test_recording_it_settles_the_nudge():
    assert missed_state_change("I've chopped the onions.", ["advance_step"]) is None
    assert missed_state_change("I've chopped the onions.", ["mark_complete"]) is None
    assert missed_state_change("I've chopped the onions.", ["get_plan"]) is not None


def test_a_system_prompt_is_never_treated_as_the_cook_speaking():
    assert implied_state_change("[SYSTEM] 6 minutes since last exchange.") is None
