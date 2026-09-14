"""Finding out the coconut milk is gone, forty minutes in, with three pans going.

The cook's complaint was that nothing checked the pantry before a session started. This is
built as a derived report rather than a stored warning, so it clears itself the moment the
gap is closed — a warning that cries wolf is one the model learns to skip past.
"""
from __future__ import annotations

import pytest

from cooking_assistant_ai.core import preflight
from cooking_assistant_ai.core.context import SYSTEM_PROMPT, _state_blocks
from cooking_assistant_ai.core.tools import dispatch


def _empty_pantry(ctx, *names):
    for name in names:
        ctx.store.conn.execute("DELETE FROM inventory WHERE name LIKE ?", (f"%{name}%",))
    ctx.store.conn.commit()


def test_it_finds_what_is_missing_before_anything_starts(ctx):
    dispatch(ctx, "load_recipe", {"recipe": "r001"})
    _empty_pantry(ctx, "garlic", "lemon")

    report = preflight.check(ctx.store, ctx.session)
    missing = {g.need.name for g in report.blocking}
    assert "garlic cloves" in missing and "lemon" in missing
    assert not report.ok and report.checked > 0


def test_it_suggests_a_substitution_for_each(ctx):
    dispatch(ctx, "load_recipe", {"recipe": "r001"})
    _empty_pantry(ctx, "garlic", "lemon", "thyme")

    text = preflight.render(preflight.check(ctx.store, ctx.session))
    assert "garlic powder" in text and "lime" in text and "oregano" in text
    assert "before they start anything" in text


def test_substituting_clears_the_warning_on_its_own(ctx):
    """Derived, not stored: closing the gap has to silence it with no bookkeeping."""
    dispatch(ctx, "load_recipe", {"recipe": "r001"})
    _empty_pantry(ctx, "lemon")
    assert any(g.need.name == "lemon" for g in preflight.check(ctx.store, ctx.session).blocking)

    assert dispatch(ctx, "substitute", {"recipe_id": "r001", "ingredient_id": "lemon",
                                        "replacement": "lime"}).ok
    after = preflight.check(ctx.store, ctx.session)
    assert not any("lemon" in g.need.name for g in after.blocking)


def test_adding_the_stock_clears_it_too(ctx):
    dispatch(ctx, "load_recipe", {"recipe": "r001"})
    _empty_pantry(ctx, "thyme")
    assert any("thyme" in g.need.name for g in preflight.check(ctx.store, ctx.session).blocking)

    assert dispatch(ctx, "add_stock", {"name": "thyme", "amount": 6}).ok
    assert not any("thyme" in g.need.name for g in preflight.check(ctx.store, ctx.session).blocking)


def test_it_is_in_every_turn_of_context_until_resolved(ctx):
    dispatch(ctx, "load_recipe", {"recipe": "r001"})
    _empty_pantry(ctx, "garlic")

    blocks = _state_blocks(ctx.session, ctx.clock.now(), ctx.store)
    assert "MISSING INGREDIENTS" in blocks
    # above the recipes, where it cannot be scrolled past
    assert blocks.index("MISSING INGREDIENTS") < blocks.index("RECIPE:")
    assert "MISSING INGREDIENTS block is present" in SYSTEM_PROMPT

    dispatch(ctx, "add_stock", {"name": "garlic", "amount": 8})
    assert "MISSING INGREDIENTS" not in _state_blocks(ctx.session, ctx.clock.now(), ctx.store)


def test_staples_alone_do_not_earn_a_block(ctx):
    """"water is not recorded" every turn is noise, and noise gets tuned out."""
    dispatch(ctx, "load_recipe", {"recipe": "r001"})
    _empty_pantry(ctx, "butter", "salt")

    report = preflight.check(ctx.store, ctx.session)
    assert report.gaps and report.ok, "staples should not be blocking"
    assert preflight.render(report) == ""
    assert "butter" in preflight.render(report, staples_too=True)


def test_scaling_is_taken_into_account(ctx):
    dispatch(ctx, "load_recipe", {"recipe": "r002"})
    dispatch(ctx, "add_stock", {"name": "jasmine rice", "amount": 2, "unit": "cup"})
    assert preflight.check(ctx.store, ctx.session).ok      # 2 cups covers one batch

    dispatch(ctx, "scale", {"recipe_id": "r002", "factor": 8.0})
    short = [g for g in preflight.check(ctx.store, ctx.session).blocking if "rice" in g.need.name]
    assert short, "eight batches of rice cannot come out of two cups"
    assert "more jasmine rice" in short[0].need.shortfall_text()


def test_a_unit_mismatch_does_not_cry_wolf(ctx):
    """A kilo in the pantry against a recipe asking for cups is unknowable, not missing."""
    dispatch(ctx, "load_recipe", {"recipe": "r002"})
    dispatch(ctx, "add_stock", {"name": "jasmine rice", "amount": 1, "unit": "kg"})
    assert not any("rice" in g.need.name
                   for g in preflight.check(ctx.store, ctx.session).blocking)


def test_the_tool_reports_a_clean_pantry_plainly(ctx):
    dispatch(ctx, "load_recipe", {"recipe": "r003"})
    r = dispatch(ctx, "check_ingredients", {})
    assert r.ok and ("in stock" in r.message or "MISSING" in r.message)


def test_checking_with_nothing_loaded_says_so(ctx):
    from cooking_assistant_ai.model.types import Session
    from cooking_assistant_ai.core.tools import ToolContext

    bare = ToolContext(Session(id="b", started_at=ctx.clock.now()), ctx.clock, ctx.store)
    r = dispatch(bare, "check_ingredients", {})
    assert not r.ok and "no recipes are loaded" in r.reason


def test_add_stock_rejects_a_negative_amount(ctx):
    r = dispatch(ctx, "add_stock", {"name": "rice", "amount": -5})
    assert not r.ok and "cannot be negative" in r.reason


def test_accepting_a_swap_does_not_produce_a_fresh_warning(ctx):
    """Swapping lemon for lime must not immediately complain that you have no lime."""
    dispatch(ctx, "load_recipe", {"recipe": "r001"})
    _empty_pantry(ctx, "lemon", "lime")
    dispatch(ctx, "substitute", {"recipe_id": "r001", "ingredient_id": "lemon",
                                 "replacement": "lime"})
    names = {g.need.name for g in preflight.check(ctx.store, ctx.session).blocking}
    assert "lime" not in names and "lemon" not in names


def test_an_ingredient_whose_steps_are_all_skipped_is_not_missing(ctx):
    dispatch(ctx, "load_recipe", {"recipe": "r001"})
    _empty_pantry(ctx, "garlic")
    assert any("garlic" in g.need.name for g in preflight.check(ctx.store, ctx.session).blocking)

    garlic = next(i for i in ctx.store.get_recipe("r001").ingredients if "garlic" in i.name)
    for step in ctx.store.get_recipe("r001").steps:
        if garlic.id in step.ingredient_ids:
            dispatch(ctx, "skip_step", {"step_id": step.id, "reason": "no garlic"})
    assert not any("garlic" in g.need.name
                   for g in preflight.check(ctx.store, ctx.session).blocking)
