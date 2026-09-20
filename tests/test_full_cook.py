"""A whole cook, start to plate, with the invariants checked after every turn.

Until this file the longest conversation any test ran was two turns, which is why a string of
things reached a real kitchen instead of a test run: a recipe marched through without being
cooked, the assistant repeating itself, mass prep grouping nothing because the data it groups
by was never filled in. All of those are multi-turn failures and none of them can be seen from
a single exchange.

The model here is a fake, but a reactive one: it reads the session each turn and decides what
to do, rather than replaying a fixed script. That means the orchestrator, the tools, the
scheduler and the state rendering all run for real, and the fake is free to misbehave in the
ways the real one did.
"""
from __future__ import annotations

import pytest

from cooking_assistant_ai.llm.client import Chunk, ScriptedLLM, ToolCallRequest
from cooking_assistant_ai.llm.llm_orchestrator import Notice, Orchestrator, Speech, ToolCalled


class Kitchen:
    """Drives a cook and records everything, so invariants can be checked as it goes."""

    def __init__(self, session, store, clock):
        self.session = session
        self.store = store
        self.clock = clock
        self.spoken: list[str] = []
        self.tools: list[ToolCalled] = []
        self.notices: list[str] = []
        self.really_done: set[str] = set()   # what the cook actually did, our shadow truth
        self.events: list = []

    async def __call__(self, ev):
        self.events.append(ev)
        if isinstance(ev, Speech):
            self.spoken.append(ev.text)
        elif isinstance(ev, ToolCalled):
            self.tools.append(ev)
        elif isinstance(ev, Notice):
            self.notices.append(ev.text)

    # -- what the fake model does with a turn -------------------------------

    def open_steps(self, recipe):
        ov = self.session.overlays[recipe.id]
        return [s for s in recipe.steps
                if s.id not in self.session.completed_steps and s.id not in ov.skipped_steps]

    def current(self, recipe):
        left = self.open_steps(recipe)
        return left[0] if left else None

    def next_dish(self):
        """The dish furthest along that still has work, else one not begun."""
        going = [r for r in self.session.recipes.values() if self.current(r)
                 and any(s.id in self.session.completed_steps for s in r.steps)]
        if going:
            return going[0]
        return next((r for r in self.session.recipes.values() if self.current(r)), None)


def text_chunks(text):
    return [Chunk(text=text), Chunk(done=True)]


def tool_chunk(name, **args):
    return [Chunk(tool_calls=[ToolCallRequest(name=name, args=args)], done=True)]


@pytest.fixture
async def cook(session, store, clock):
    # all three seeded recipes stay loaded; the two-at-once cap is part of what is tested
    k = Kitchen(session, store, clock)
    llm = ScriptedLLM()
    o = Orchestrator(session, llm, store, k, clock=clock, idle_interval_s=0)
    o.start()
    yield o, llm, k
    await o.stop()


# ------------------------------------------------------------------ a cook that behaves

async def test_a_whole_cook_records_only_what_was_actually_done(cook):
    """Walk two dishes to the end, one step per turn, and check nothing was invented."""
    o, llm, k = cook
    turns = 0
    while turns < 40:
        dish = k.next_dish()
        if dish is None:
            break
        step = k.current(dish)

        # the model moves the cook on, then the cook does it
        llm.push(tool_chunk("advance_step", step_id=step.id),
                 text_chunks(f"Right: {step.text[:40]}"))
        await o.submit("what now?")
        await o.wait_idle()

        # nothing may be recorded as done that the cook has not done
        assert k.session.completed_steps <= k.really_done, (
            f"turn {turns}: {sorted(k.session.completed_steps - k.really_done)} "
            f"recorded without being cooked")

        llm.push(tool_chunk("mark_complete", step_ids=[step.id]), text_chunks("Good."))
        await o.submit(f"done with {step.text[:20]}")
        await o.wait_idle()
        k.really_done.add(step.id)
        turns += 1

    assert turns > 8, f"only got through {turns} steps"
    assert k.session.completed_steps == k.really_done


async def test_nothing_is_said_twice_over_a_long_cook(cook):
    """The assistant repeating itself was the loudest complaint, and it came from an extra
    round whose first answer had already been spoken."""
    o, llm, k = cook
    said = []
    for n in range(12):
        dish = k.next_dish()
        step = k.current(dish)
        line = f"Step {n}: {step.text[:50]}"
        llm.push(tool_chunk("advance_step", step_id=step.id), text_chunks(line))
        await o.submit("what now?")
        await o.wait_idle()
        llm.push(tool_chunk("mark_complete", step_ids=[step.id]), text_chunks(f"Noted {n}."))
        await o.submit("done")
        await o.wait_idle()
        k.really_done.add(step.id)
        said.append(line)

    whole = "".join(k.spoken)
    for line in said:
        assert whole.count(line) == 1, f"spoken twice: {line!r}"


async def test_the_model_still_sees_the_start_of_the_cook_at_the_end(cook):
    """Five exchanges of history meant a four-minute memory over a hundred-minute cook."""
    o, llm, k = cook
    # it has to actually call remember: "Noted" with no tool behind it is an unbacked
    # claim, and the gate drops it before the cook hears it, which is its job.
    llm.push(tool_chunk("remember", text="the cook dislikes mushrooms"),
             text_chunks("Mushrooms are off the list."))
    await o.submit("remember I hate mushrooms")
    await o.wait_idle()

    for n in range(14):
        llm.push(text_chunks(f"Fine {n}."))
        await o.submit(f"turn {n}")
        await o.wait_idle()

    last = llm.calls[-1]
    whole = chr(10).join(m["content"] for m in last)
    assert "I hate mushrooms" in whole              # what the cook said, fifteen turns back
    assert "Mushrooms are off the list." in whole   # and what was answered


# ------------------------------------------------------- a cook that does not behave

async def test_a_model_that_jumps_ahead_cannot_fabricate_a_cooked_dish(cook):
    """The jambalaya: advance_step used to sweep every earlier step into completed_steps."""
    o, llm, k = cook
    dish = k.session.recipes["r001"]
    last = dish.steps[-1]

    llm.push(tool_chunk("advance_step", step_id=last.id),
             text_chunks("And that's the chicken done."))
    await o.submit("what now?")
    await o.wait_idle()

    assert k.session.completed_steps == set()
    rejected = [t for t in k.tools if not t.result["ok"]]
    assert rejected and "still open" in rejected[0].result["reason"]


async def test_a_model_that_opens_every_dish_is_held_to_two(cook):
    """Four dishes in the air at once is what the cook shouted down."""
    o, llm, k = cook
    for rid in ("r001", "r002", "r003"):
        recipe = k.session.recipes[rid]
        first = recipe.steps[0]
        llm.push(tool_chunk("advance_step", step_id=first.id), text_chunks("On it."))
        await o.submit("start it")
        await o.wait_idle()
        llm.push(tool_chunk("mark_complete", step_ids=[first.id]), text_chunks("Done."))
        await o.submit("done")
        await o.wait_idle()
        k.really_done.add(first.id)

    # The cap stops the model *sending* the cook to a third dish.
    refusals = [t for t in k.tools if t.name == "advance_step" and not t.result["ok"]]
    assert any("three at once" in t.result["reason"] for t in refusals), \
        "the third advance_step should have been refused"
    third = k.session.recipes["r003"]
    advanced = [t.args.get("step_id") for t in k.tools
                if t.name == "advance_step" and t.result["ok"]]
    assert third.steps[0].id not in advanced

    # mark_complete is deliberately not capped: if the cook says they did something then
    # recording it is never the wrong answer, and a cap that makes the state lie about the
    # kitchen is worse than no cap at all.
    assert third.steps[0].id in k.session.completed_steps


async def test_state_stays_coherent_all_the_way_through(cook):
    """Render the whole state every turn: a crash here is a crash on the tablet."""
    from cooking_assistant_ai.core.render import state_dict

    o, llm, k = cook
    for n in range(10):
        dish = k.next_dish()
        step = k.current(dish)
        llm.push(tool_chunk("advance_step", step_id=step.id), text_chunks(f"Go {n}."))
        await o.submit("next")
        await o.wait_idle()
        llm.push(tool_chunk("mark_complete", step_ids=[step.id]), text_chunks("Yes."))
        await o.submit("done")
        await o.wait_idle()
        k.really_done.add(step.id)

        state = state_dict(k.session, k.clock.now(), k.store)
        for recipe in state["progress"]["recipes"]:
            cur = recipe["current_step"]
            done = set(recipe["completed_steps"])
            assert done <= k.really_done
            if cur is not None:
                step_row = next(s for s in recipe["steps"] if s["n"] == cur)
                assert step_row["status"] == "pending"
                assert step_row["id"] not in done
        # a finished task never keeps a timer running
        for timer in k.session.timers.values():
            if timer.status == "running" and timer.task_id:
                assert k.session.tasks[timer.task_id].status != "complete"


async def test_a_cook_side_nudge_does_not_make_it_say_everything_twice(cook):
    """The reply-side matcher was removed for causing this; the cook-side rule remains and
    takes the same path, so it needs the same guarantee."""
    o, llm, k = cook
    llm.push(text_chunks("Lovely, the onions are ready."),
             tool_chunk("mark_complete", step_ids=["r001-s2"]),
             text_chunks("Marked the seasoning done."))
    await o.submit("I've chopped the onions")
    await o.wait_idle()

    assert any("drifted" in n for n in k.notices), "the nudge should have fired"
    whole = "".join(k.spoken)
    assert whole.count("Lovely, the onions are ready.") == 1, whole
