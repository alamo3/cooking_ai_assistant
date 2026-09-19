"""Recording that the kitchen moved on, from a real cook where it never happened.

Journal 2026-09-19-4bd8b62a: seventeen turns of The Best Tofu Scramble, ten tool calls, not
one mark_complete. The model narrated all three steps correctly and in order while the tablet
sat on step 1 from the first word to the last. These are the utterances from that session.
"""
from __future__ import annotations

from datetime import datetime

import pytest

from cooking_assistant_ai.core.claims import implied_state_change, step_moved_on
from cooking_assistant_ai.model.types import Overlay, Recipe, Session, Step

STEPS = [
    "Heat the olive oil in a pan over medium heat (I prefer a cast iron pan, or non-stick). "
    "Mash the block of tofu right in the pan, with a potato masher or a fork. You can also "
    "crumble it into the pan with your hands. Cook, stirring frequently, for 3-4 minutes "
    "until the water from the tofu is mostly gone.",
    "Now add the nutritional yeast, salt, turmeric and garlic powder. Cook and stir "
    "constantly for about 5 minutes. Continue to cook for longer, not stirring for a few "
    "minutes at a time, until golden spots form on some of the tofu.",
    "Pour the soy milk into the pan, and stir to mix. Serve immediately with sliced avocado, "
    "hot sauce, parsley, steamed kale, toast or any other breakfast item.",
]


def make_session(done=()) -> Session:
    recipe = Recipe(id="r016", title="The Best Tofu Scramble", servings=2, ingredients=(),
                    steps=tuple(Step(id=f"r016-s{n}", text=t)
                                for n, t in enumerate(STEPS, start=1)))
    s = Session(id="s1", started_at=datetime(2026, 9, 19, 0, 14))
    s.add_recipe(recipe)
    s.completed_steps.update(done)
    return s


# ------------------------------------------------------- the model advancing on its own

def test_narrating_step_two_while_step_one_is_open_is_noticed():
    said = ("Go ahead and add two tablespoons of nutritional yeast, half a teaspoon of kala "
            "namak, a quarter teaspoon of turmeric and a quarter teaspoon of garlic powder.")
    found = step_moved_on(said, make_session())
    assert found is not None
    assert "mark_complete" in found.needs
    assert "step 1" in found.describe()
    assert "The Best Tofu Scramble" in found.describe()


def test_narrating_the_last_step_while_earlier_ones_are_open_is_noticed():
    said = "Pour two tablespoons of soy milk into the tofu pan and stir to mix, then serve it up."
    assert step_moved_on(said, make_session()) is not None


def test_the_current_step_is_not_an_advance():
    """"Mash it into the oiled pan" is step 1, which is where the cook already is."""
    said = "Go ahead and mash it into the oiled pan now."
    assert step_moved_on(said, make_session()) is None


def test_nothing_to_be_behind_on_once_earlier_steps_are_marked():
    said = "Pour two tablespoons of soy milk into the tofu pan and stir to mix."
    done = {"r016-s1", "r016-s2"}
    assert step_moved_on(said, make_session(done)) is None


def test_chatter_matches_nothing():
    for said in ("Nice work. Enjoy your scramble.",
                 "Timer set for four minutes.",
                 "Sure, give it a few more minutes. Tell me when you're ready.",
                 ""):
        assert step_moved_on(said, make_session()) is None, said


def test_an_empty_kitchen_is_never_flagged():
    assert step_moved_on("add the spices", Session(id="s", started_at=datetime.now())) is None


def test_a_single_open_step_cannot_be_an_advance():
    assert step_moved_on("Pour the soy milk in and stir to mix, then serve.",
                         make_session({"r016-s1", "r016-s2"})) is None


# ------------------------------------------------------------ what the cook actually said

@pytest.mark.parametrize("said", [
    "Alright, I've mashed the tofu into the oil pan using my potato masher.",
    "Okay, I'm ready for the spices.",
    "Alright, sounds like we're done here.",
])
def test_phrases_from_the_recorded_cook_now_register(said):
    found = implied_state_change(said)
    assert found is not None, said
    assert "mark_complete" in found.needs


@pytest.mark.parametrize("said", [
    "So how much of the spices do I need?",
    "When are we supposed to add the milk?",
    "300 grams of tofu.",
    "I think the tofu needs a few more minutes.",
])
def test_questions_and_statements_of_fact_still_do_not(said):
    assert implied_state_change(said) is None, said
