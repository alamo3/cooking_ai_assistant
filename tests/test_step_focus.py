"""What the tablet shows is what the model decided, not what the browser inferred.

The step panel used to pick "the dish furthest along that still has work" - a rule invented in
the browser. With two dishes interleaved it showed whichever happened to have more steps ticked
off, so the cook watched one recipe run to the end while the model was walking them between
both. advance_step already makes this decision; the session simply never recorded it.
"""
from __future__ import annotations

import json
from datetime import datetime

import pytest

from cooking_assistant_ai.core.render import recipe_view, state_dict
from cooking_assistant_ai.core.tools import dispatch


def test_advancing_puts_that_dish_at_the_front(ctx):
    dispatch(ctx, "advance_step", {"step_id": 1, "recipe_id": "r001"})
    assert ctx.session.focus == ["r001"]
    dispatch(ctx, "advance_step", {"step_id": 1, "recipe_id": "r002"})
    assert ctx.session.focus == ["r002", "r001"]


def test_going_back_to_a_dish_moves_it_up_without_duplicating(ctx):
    for rid in ("r001", "r002", "r001"):
        dispatch(ctx, "advance_step", {"step_id": 1, "recipe_id": rid})
    assert ctx.session.focus == ["r001", "r002"]


def test_a_refused_advance_does_not_change_the_order(ctx):
    dispatch(ctx, "advance_step", {"step_id": 1, "recipe_id": "r001"})
    before = list(ctx.session.focus)
    assert not dispatch(ctx, "advance_step", {"step_id": 6, "recipe_id": "r002"}).ok
    assert ctx.session.focus == before


def test_the_view_reports_where_each_dish_sits(ctx):
    dispatch(ctx, "advance_step", {"step_id": 1, "recipe_id": "r001"})
    dispatch(ctx, "advance_step", {"step_id": 1, "recipe_id": "r002"})
    views = {r.id: recipe_view(ctx.session, r) for r in ctx.session.recipes.values()}
    assert views["r002"]["focus"] == 0
    assert views["r001"]["focus"] == 1
    assert views["r003"]["focus"] == -1          # never sent there


def test_it_survives_a_restart(ctx):
    from cooking_assistant_ai.storage.sessions import from_dict, to_dict

    dispatch(ctx, "advance_step", {"step_id": 1, "recipe_id": "r001"})
    dispatch(ctx, "advance_step", {"step_id": 1, "recipe_id": "r002"})
    back = from_dict(json.loads(json.dumps(to_dict(ctx.session))))
    assert back.focus == ["r002", "r001"]


def test_a_dish_that_is_gone_is_dropped_on_restore(ctx):
    from cooking_assistant_ai.storage.sessions import from_dict, to_dict

    dispatch(ctx, "advance_step", {"step_id": 1, "recipe_id": "r001"})
    raw = json.loads(json.dumps(to_dict(ctx.session)))
    raw["focus"] = ["r001", "r999"]
    assert from_dict(raw).focus == ["r001"]


# ------------------------------------------------------------- the tablet's own code

def run_picker(recipes):
    """Execute the page's dishesOnTheGo against this state."""
    dukpy = pytest.importorskip("dukpy")
    src = open("src/cooking_assistant_ai/web/app.js", encoding="utf-8").read()
    start = src.index("  function dishesOnTheGo(recipes) {")
    end = src.index("  function renderStepPanel(recipes) {")
    js = """
    %s
    JSON.stringify(dishesOnTheGo(%s).map(function (d) { return d.r.id; }));
    """ % (src[start:end], json.dumps(recipes))
    return json.loads(dukpy.evaljs(js))


def test_the_panel_orders_dishes_the_way_the_model_moved_between_them(ctx):
    dispatch(ctx, "advance_step", {"step_id": 1, "recipe_id": "r001"})
    dispatch(ctx, "advance_step", {"step_id": 1, "recipe_id": "r002"})
    state = state_dict(ctx.session, datetime(2026, 9, 20, 19, 0), ctx.store)
    assert run_picker(state["progress"]["recipes"]) == ["r002", "r001"]


def test_the_panel_never_shows_more_than_two(ctx):
    for rid in ("r001", "r002"):
        dispatch(ctx, "advance_step", {"step_id": 1, "recipe_id": rid})
    dispatch(ctx, "advance_step", {"step_id": 1, "recipe_id": "r003", "anyway": True})
    state = state_dict(ctx.session, datetime(2026, 9, 20, 19, 0), ctx.store)
    assert len(run_picker(state["progress"]["recipes"])) == 2


def test_before_anything_is_advanced_the_first_step_is_still_shown(ctx):
    state = state_dict(ctx.session, datetime(2026, 9, 20, 19, 0), ctx.store)
    assert len(run_picker(state["progress"]["recipes"])) == 1
