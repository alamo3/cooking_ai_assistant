"""Getting ahead during dead time, on request.

A cook reached the noodle dish with unboiled noodles and unsliced green onions, and neither
had a prep step to find: the recipe writes "6 green onions, sliced" in the ingredient list and
never mentions slicing again. So this hands over ingredients as written rather than guessing
which verbs count as prep.
"""
from __future__ import annotations

from cooking_assistant_ai.core.plan import in_flight, not_started, prep_ahead
from cooking_assistant_ai.core.tools import dispatch


def start(ctx, rid, upto):
    r = ctx.session.recipes[rid]
    dispatch(ctx, "mark_complete", {"step_ids": [s.id for s in r.steps[:upto]]})


def test_it_looks_at_the_dishes_not_yet_begun(ctx):
    start(ctx, "r001", 1)
    assert [r.id for r in not_started(ctx.session)] == ["r002", "r003"]
    assert [r.id for r in in_flight(ctx.session)] == ["r001"]


def test_a_finished_dish_is_not_in_flight(ctx):
    r = ctx.session.recipes["r001"]
    dispatch(ctx, "mark_complete", {"step_ids": [s.id for s in r.steps]})
    assert in_flight(ctx.session) == []


def test_it_covers_the_next_two_by_default(ctx):
    rows = prep_ahead(ctx.session, 2)
    covered = {name for row in rows for name in row["recipes"]}
    assert covered <= {"Roast Chicken Thighs", "Jasmine Rice"}


def test_shared_ingredients_collapse_into_one_line(ctx):
    """Chopping the onions twice because two dishes want them is the thing to avoid."""
    rows = prep_ahead(ctx.session, 3)
    for row in rows:
        assert len(row["recipes"]) == len(set(row["recipes"]))
    keys = [row["key"] for row in rows]
    assert len(keys) == len(set(keys))


def test_shared_lines_come_first(ctx):
    rows = prep_ahead(ctx.session, 3)
    counts = [len(row["recipes"]) for row in rows]
    assert counts == sorted(counts, reverse=True)


def test_names_are_handed_over_as_the_recipe_writes_them(ctx):
    rows = prep_ahead(ctx.session, 3)
    assert all(row["as_written"] for row in rows)


def test_mixed_units_are_not_added_up(ctx):
    rows = prep_ahead(ctx.session, 3)
    for row in rows:
        if row["unit"] is None:
            assert row["total"] == ""   # refuses to invent a total it cannot compute


def test_the_tool_reports_when_everything_has_started(ctx):
    for rid in ("r001", "r002", "r003"):
        start(ctx, rid, 1)
    out = dispatch(ctx, "prep_ahead", {"recipes": 2})
    assert out.ok and "has been started" in out.message


def test_the_tool_names_the_dishes_it_covers(ctx):
    out = dispatch(ctx, "prep_ahead", {"recipes": 1})
    assert out.ok and "PREP AHEAD for Roast Chicken Thighs" in out.message
