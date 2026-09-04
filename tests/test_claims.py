from __future__ import annotations

from datetime import timedelta

from cooking_assistant_ai.core.claims import find_claims, unjustified_claims
from cooking_assistant_ai.model.types import Timer


def kinds(sentence):
    return sorted(c.kind for c in find_claims(sentence))


def test_timer_claims_are_detected():
    assert kinds("I've set a timer for 25 minutes.") == ["timer"]
    assert kinds("I'll set a 15-minute timer for the rice simmering.") == ["timer"]
    assert kinds("Rice timer is set for fifteen minutes.") == ["timer"]
    assert kinds("Timer's on. Flip the chicken.") == ["timer"]
    assert kinds("Let me start a timer for the sprouts.") == ["timer"]


def test_ordinary_speech_is_not_flagged():
    assert kinds("There are 15 minutes left on the rice simmering timer.") == []
    assert kinds("It will go off at 2:58 PM.") == []
    assert kinds("You can use olive oil instead of butter for searing.") == []
    assert kinds("Flip the thighs and tuck the garlic around them.") == []
    assert kinds("Set the oven to 425 and pat the chicken dry.") == []


def test_other_claim_kinds():
    assert kinds("I've marked the sear as complete.") == ["complete"]
    assert kinds("I've started the roast.") == ["started"]
    assert kinds("I'll remember that you prefer less salt.") == ["remember"]
    assert kinds("I've added the rice to the plan.") == ["plan"]
    assert kinds("I've swapped butter for olive oil in the recipe.") == ["recipe"]


def test_plan_ready_claim_needs_tasks_to_exist(session):
    from cooking_assistant_ai.model.types import Task

    assert kinds("The plan is ready.") == ["plan"]
    assert kinds("Everything is scheduled.") == ["plan"]
    assert unjustified_claims("The plan is ready.", [], session)  # no tasks: a lie
    session.tasks["t_001"] = Task(id="t_001", label="rice", recipe_id="r002", step_ids=[],
                                  appliance=None, temp_f=None, duration_s=600)
    assert not unjustified_claims("The plan is ready.", [], session)  # tasks exist: fair remark


def test_justification_by_tools_and_running_timers(session):
    s = "I've set a timer for 25 minutes."
    assert unjustified_claims(s, [], session)
    assert not unjustified_claims(s, ["set_timer"], session)
    # a status remark about an existing timer is fine without a new tool call
    session.timers["tm_001"] = Timer(id="tm_001", label="rice simmering", task_id=None, step_id=None,
                                     end_at=session.started_at + timedelta(minutes=10))
    assert not unjustified_claims("Your rice timer is still running with 8 minutes left.", [], session)
    assert unjustified_claims("I've set a chicken timer for 25 minutes.", [], session)
