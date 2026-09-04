from __future__ import annotations

import asyncio
from typing import List

import pytest

from cooking_assistant_ai.core.tools import dispatch
from cooking_assistant_ai.llm.client import Chunk, ScriptedLLM, ToolCallRequest, call, say
from cooking_assistant_ai.llm.llm_orchestrator import (
    Notice,
    Orchestrator,
    Speech,
    SpeechEnd,
    SpeechStart,
    StateChanged,
    ToolCalled,
)
from cooking_assistant_ai.model.events import IdleTick
from cooking_assistant_ai.speech.sentences import SentenceSplitter


class Collector:
    def __init__(self):
        self.events: List = []

    async def __call__(self, ev):
        self.events.append(ev)

    def spoken(self) -> str:
        return "".join(e.text for e in self.events if isinstance(e, Speech))

    def tools(self) -> List[ToolCalled]:
        return [e for e in self.events if isinstance(e, ToolCalled)]


@pytest.fixture
async def orch(session, store, clock):
    llm = ScriptedLLM()
    out = Collector()
    o = Orchestrator(session, llm, store, out, clock=clock, idle_interval_s=0)
    o.start()
    yield o, llm, out
    await o.stop()


async def test_tool_round_trip_and_context(orch):
    o, llm, out = orch
    llm.push(
        call("set_timer", label="rice simmering", duration_s=900, on_complete_hint="take it off the heat"),
        "Rice timer is on for fifteen minutes.",
    )
    await o.submit("the rice is simmering, set a timer")
    await o.wait_idle()
    assert out.tools()[0].name == "set_timer" and out.tools()[0].result["ok"]
    assert "fifteen minutes" in out.spoken()
    # the tool result was fed back as a tool message and the state was pushed
    second_call = llm.calls[1]
    assert second_call[-1]["role"] == "tool" and "rice simmering" in second_call[-1]["content"]
    assert any(isinstance(e, StateChanged) for e in out.events)
    # context: system prompt carries state, transcript recorded
    first_system = llm.calls[0][0]["content"]
    assert "RECIPE: Roast Chicken Thighs" in first_system and "TIMELINE" in first_system
    assert o.session.transcript[-1].role == "assistant"
    # next turn sees the transcript verbatim
    llm.push("Sure.")
    await o.submit("thanks")
    await o.wait_idle()
    roles = [m["role"] for m in llm.calls[2]]
    assert roles == ["system", "user", "assistant", "user"]


async def test_rejected_tool_is_fed_back(orch):
    o, llm, out = orch
    llm.push(call("set_timer", label="timer", duration_s=60), "That label was rejected; let me fix it.")
    await o.submit("start a timer")
    await o.wait_idle()
    assert out.tools()[0].result["ok"] is False
    assert "too generic" in llm.calls[1][-1]["content"]


async def test_timer_fires_proactively_with_hint(orch, clock):
    o, llm, out = orch
    llm.push(call("set_timer", label="chicken resting", duration_s=600, on_complete_hint="slice and plate"), "Timer set.")
    await o.submit("chicken is resting")
    await o.wait_idle()
    llm.push("Chicken's rested. Slice it and plate up.")
    clock.advance(601)
    await asyncio.sleep(0.1)
    await o.wait_idle()
    prompt = llm.calls[2][-1]["content"]
    assert prompt.startswith('[SYSTEM] Timer "chicken resting" completed at 6:40 PM')
    assert "slice and plate" in prompt
    starts = [e for e in out.events if isinstance(e, SpeechStart)]
    assert starts[-1].proactive is True
    assert o.session.timers["tm_001"].status == "fired"


async def test_multiple_timers_batch_into_one_turn(orch, clock):
    o, llm, out = orch
    llm.push(call("set_timer", label="rice resting", duration_s=60), "ok")
    await o.submit("a"); await o.wait_idle()
    llm.push(call("set_timer", label="sprouts done", duration_s=60), "ok")
    await o.submit("b"); await o.wait_idle()
    llm.push("Rice and sprouts are both ready.")
    clock.advance(61)
    await asyncio.sleep(0.1)
    await o.wait_idle()
    prompt = llm.calls[-1][-1]["content"]
    assert prompt.count("[SYSTEM] Timer") == 2 and "one breath" in prompt
    assert len(llm.calls) == 5  # two user turns x2 rounds + one batched timer turn


async def test_idle_tick_gating_and_nothing_suppression(orch, clock):
    o, llm, out = orch
    # nothing planned -> gated without an LLM call
    await o.queue.put(IdleTick(600)); await o.wait_idle()
    assert llm.calls == []
    llm.push(call("add_task", label="rice", recipe_id="r002", step_ids=["r002-s2", "r002-s3"]), "Rice is on the plan.")
    await o.submit("plan the rice"); await o.wait_idle()
    n = len(llm.calls)
    # too soon after the last exchange -> gated
    await o.queue.put(IdleTick(10)); await o.wait_idle()
    assert len(llm.calls) == n
    clock.advance(600)
    llm.push("NOTHING")
    await o.queue.put(IdleTick(600)); await o.wait_idle()
    assert len(llm.calls) == n + 1
    assert "[SYSTEM]" in llm.calls[-1][-1]["content"] and "Next action" in llm.calls[-1][-1]["content"]
    assert not any(isinstance(e, SpeechStart) and e.proactive for e in out.events)
    assert o.session.transcript[-1].role == "assistant"  # suppressed prompt not kept


async def test_text_embedded_tool_calls_are_parsed(orch):
    o, llm, out = orch
    llm.push([Chunk(text='<tool_call>{"name": "get_plan", "arguments": {}}</tool_call>', done=True)], "Here is the plan.")
    await o.submit("what's the plan"); await o.wait_idle()
    assert out.tools()[0].name == "get_plan"
    assert "<tool_call>" not in out.spoken()


async def test_barge_in_cancels_generation(orch):
    o, llm, out = orch

    async def slow():
        yield Chunk(text="Let me think ")
        await asyncio.sleep(5)
        yield Chunk(text="never reached", done=True)

    llm.push(lambda messages: slow())
    await o.submit("long question")
    await asyncio.sleep(0.05)
    assert o.barge_in()
    await o.wait_idle()
    assert "never reached" not in out.spoken()
    assert isinstance(out.events[-1], SpeechEnd)


async def test_unbacked_claim_triggers_correction_and_is_never_spoken(orch):
    o, llm, out = orch
    llm.push(
        "Flip the chicken now. I've set a timer for 25 minutes. Then rest it.",
        call("set_timer", label="chicken roast", duration_s=1500, on_complete_hint="pull it out"),
        "Timer is set for twenty-five minutes, then rest it.",
    )
    await o.submit("chicken's in the oven"); await o.wait_idle()
    spoken = out.spoken()
    assert spoken.startswith("Flip the chicken now.")           # safe sentence streamed immediately
    assert "25 minutes" not in spoken                             # the unbacked claim never reached the cook
    assert "Timer is set for twenty-five minutes" in spoken       # the corrected, tool-backed version did
    correction = llm.calls[1][-1]["content"]
    assert correction.startswith("[SYSTEM] Your reply claimed") and "set_timer" in correction
    assert o.session.timers["tm_001"].label == "chicken roast"
    assert o.session.transcript[-1].text == spoken.strip()


async def test_claim_backed_by_tool_in_same_turn_streams_without_correction(orch):
    o, llm, out = orch
    # text and the backing tool call arrive in the same response, as Qwen emits them
    llm.push(
        [Chunk(text="I'll set a timer for the rice."), Chunk(tool_calls=[ToolCallRequest("set_timer", {"label": "rice simmering", "duration_s": 900})], done=True)],
        "Done, fifteen minutes.",
    )
    await o.submit("rice is on"); await o.wait_idle()
    assert out.spoken().startswith("I'll set a timer for the rice.")
    assert len(llm.calls) == 2 and "[SYSTEM]" not in llm.calls[1][-1]["content"]


async def test_claim_still_unbacked_after_correction_is_dropped(orch):
    o, llm, out = orch
    llm.push("I've marked the sear as complete.", "I've marked it complete, promise.")
    await o.submit("sear is done"); await o.wait_idle()
    assert out.spoken().strip() == ""
    notices = [e for e in out.events if isinstance(e, Notice)]
    assert any("dropped unbacked claim" in n.text for n in notices)


def test_tool_calls_are_dispatched_in_dependency_order():
    from cooking_assistant_ai.llm.llm_orchestrator import order_tool_calls

    calls = [
        ToolCallRequest("add_task", {"label": "chicken roast", "after": "Chicken Sear", "must_finish_by": "plating"}),
        ToolCallRequest("set_timer", {"label": "x", "duration_s": 5}),
        ToolCallRequest("add_task", {"label": "rice simmer", "after": "rice boil"}),
        ToolCallRequest("add_task", {"label": "chicken sear", "before": "chicken roast"}),
        ToolCallRequest("add_task", {"label": "rice boil"}),
    ]
    names = [c.args["label"] for c in order_tool_calls(calls)]
    assert names.index("chicken sear") < names.index("chicken roast")
    assert names.index("rice boil") < names.index("rice simmer")
    assert names == ["chicken sear", "chicken roast", "x", "rice boil", "rice simmer"]  # otherwise stable
    # a reference cycle must not hang or drop calls
    cyc = [ToolCallRequest("add_task", {"label": "a", "after": "b"}), ToolCallRequest("add_task", {"label": "b", "after": "a"})]
    assert len(order_tool_calls(cyc)) == 2


async def test_plan_emitted_out_of_order_needs_no_retry(orch):
    o, llm, out = orch
    llm.push(
        [Chunk(tool_calls=[
            ToolCallRequest("add_task", {"label": "roast", "recipe_id": "r001", "step_ids": ["r001-s5"], "after": "sear"}),
            ToolCallRequest("add_task", {"label": "sear", "recipe_id": "r001", "step_ids": ["r001-s3"], "appliance": "stovetop"}),
        ], done=True)],
        "Planned.",
    )
    await o.submit("plan the chicken"); await o.wait_idle()
    assert all(t.result["ok"] for t in out.tools())
    assert o.session.find_task("roast").depends_on == [o.session.find_task("sear").id]


async def test_empty_generation_is_retried_once(orch):
    """Every model tested sometimes returns nothing at all; one retry recovers the turn."""
    o, llm, out = orch
    llm.push([Chunk(text="", done=True)], "Rice takes fifteen minutes.")
    await o.submit("how long for the rice"); await o.wait_idle()
    assert "fifteen minutes" in out.spoken()
    assert any("returned nothing" in e.text for e in out.events if isinstance(e, Notice))
    assert len(llm.calls) == 2


async def test_a_turn_that_only_called_tools_is_not_retried(orch):
    o, llm, out = orch
    llm.push(call("get_plan"), [Chunk(text="", done=True)])
    await o.submit("what's the plan"); await o.wait_idle()
    assert len(llm.calls) == 2  # tool round then a silent finish, no third attempt
    assert not any("returned nothing" in e.text for e in out.events if isinstance(e, Notice))


async def test_current_prompt_is_not_duplicated_in_the_next_context(orch):
    o, llm, out = orch
    llm.push("Sure.", "Of course.")
    await o.submit("first question"); await o.wait_idle()
    await o.submit("second question"); await o.wait_idle()
    roles = [m["role"] for m in llm.calls[1]]
    assert roles == ["system", "user", "assistant", "user"]
    assert llm.calls[1][1]["content"] == "first question"
    assert llm.calls[1][-1]["content"] == "second question"


async def test_leaked_template_tokens_are_never_spoken(orch):
    """A real cloud reply began '<|tool_call> I've started the air fryer timer'."""
    o, llm, out = orch
    llm.push([Chunk(text="<|tool_call> Twelve minutes for the sprouts. ", done=True),
              Chunk(text="<|im_end|>Shake the basket halfway.", done=True)])
    await o.submit("sprouts are in"); await o.wait_idle()
    spoken = out.spoken()
    assert "tool_call" not in spoken and "im_end" not in spoken and "<|" not in spoken
    assert "Twelve minutes for the sprouts." in spoken and "Shake the basket halfway." in spoken


def test_control_markup_stripping():
    from cooking_assistant_ai.llm.client import strip_control_markup

    assert strip_control_markup("<|tool_call> hello") == " hello"
    assert strip_control_markup("a<|im_end|>b") == "ab"
    assert strip_control_markup("<think>x</think>done") == "xdone"
    assert strip_control_markup("normal text, 5 < 6 > 4") == "normal text, 5 < 6 > 4"


async def test_unrecorded_progress_is_nudged_not_forced(orch):
    """The exact drift seen in a real session: the cook says the chicken went in the oven
    and the turn ends with no state change."""
    o, llm, out = orch
    llm.push(
        "Great, it'll take about 25 minutes.",                       # says nothing to the state
        call("start_task", task_id="chicken roast"),                  # after the nudge
        "Chicken's roasting, 25 minutes.",
    )
    dispatch(o.ctx, "add_task", {"label": "chicken roast", "recipe_id": "r001",
                                 "step_ids": ["r001-s5"], "appliance": "oven", "temp_f": 425})
    await o.submit("the chicken is seared and it's going in the oven now"); await o.wait_idle()
    nudge = llm.calls[1][-1]["content"]
    assert nudge.startswith("[SYSTEM] That turn changed nothing in the state")
    assert o.session.find_task("chicken roast").status == "active"
    assert any("state may have drifted" in e.text for e in out.events if isinstance(e, Notice))


async def test_no_nudge_when_the_cook_only_asked_a_question(orch):
    o, llm, out = orch
    llm.push("About twenty minutes.")
    await o.submit("how long left on the chicken"); await o.wait_idle()
    assert len(llm.calls) == 1
    assert not any("drifted" in e.text for e in out.events if isinstance(e, Notice))


async def test_no_nudge_when_the_model_already_recorded_it(orch):
    o, llm, out = orch
    llm.push(call("mark_complete", step_ids=["r002-s1"]), "Rice rinsed, noted.")
    await o.submit("I've rinsed the rice"); await o.wait_idle()
    assert len(llm.calls) == 2  # tool round then reply, no nudge round
    assert not any("drifted" in e.text for e in out.events if isinstance(e, Notice))


def test_drift_warnings_surface_contradictions(session, clock):
    from datetime import timedelta

    from cooking_assistant_ai.core.scheduler import drift_warnings
    from cooking_assistant_ai.core.tools import ToolContext, dispatch
    from cooking_assistant_ai.storage.db import Store

    ctx = ToolContext(session, clock, Store(":memory:"))
    dispatch(ctx, "add_task", {"label": "chicken roast", "recipe_id": "r001",
                               "step_ids": ["r001-s5"], "appliance": "oven", "temp_f": 425})
    dispatch(ctx, "set_timer", {"label": "chicken roasting", "duration_s": 1500,
                                "task_id": "chicken roast"})
    warnings = drift_warnings(session, clock.now())
    assert any("still pending" in w and "start_task" in w for w in warnings)
    dispatch(ctx, "start_task", {"task_id": "chicken roast"})
    assert drift_warnings(session, clock.now()) == []
    # an active task long past its window is also worth asking about
    clock.advance(1500 + 6 * 60)
    assert any("due to finish" in w for w in drift_warnings(session, clock.now()))


def test_sentence_splitter_streams_early():
    s = SentenceSplitter()
    out = s.feed("Put the rice on. Then ")
    assert out == ["Put the rice on."]
    out += s.feed("sear the chicken for 1.")
    out += s.feed("5 min. Done!")
    assert out == ["Put the rice on.", "Then sear the chicken for 1.5 min."]
    assert s.flush() == ["Done!"]
    assert s.feed("Add 2 tbsp. of oil now. ") == ["Add 2 tbsp. of oil now."]
