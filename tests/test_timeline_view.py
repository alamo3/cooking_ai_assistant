"""The timeline is a picture now, so the picture is what gets checked.

It was a four-column text table that went unread for weeks. The useful fact in a plan is that
two pans overlap and where the gap is, and that is a shape rather than a column of times.
These run the page's own renderTimeline in a JS engine and assert on the geometry it produced,
because there is no browser here and a wrong bar is invisible in a diff.
"""
from __future__ import annotations

import json
from datetime import datetime

import pytest

from cooking_assistant_ai.core.render import state_dict
from cooking_assistant_ai.core.tools import dispatch

HARNESS = """
var stub = {};
function $(sel) {
  if (!stub[sel]) stub[sel] = { textContent: "", innerHTML: "", hidden: false, children: [],
    style: { setProperty: function (k, v) { this[k] = v; } },
    appendChild: function (c) { this.children.push(c); } };
  return stub[sel];
}
function el(tag, cls, text) {
  return { tag: tag, cls: cls || "", text: text == null ? "" : text, children: [], title: "",
           style: { setProperty: function (k, v) { this[k] = v; } },
           appendChild: function (c) { this.children.push(c); } };
}
var document = { createTextNode: function (t) {
  return { tag: "#text", cls: "", text: t, children: [] }; } };
function fmtTime(iso) {
  if (!iso) return "--:--";
  var d = new Date(iso), h = d.getHours(), m = d.getMinutes();
  var ampm = h >= 12 ? "PM" : "AM"; h = h % 12 || 12;
  return "" + h + ":" + (m < 10 ? "0" : "") + m + " " + ampm;
}
function fmtWindow(a, b) { return fmtTime(a) + "-" + fmtTime(b); }
var app = { state: __STATE__ };
__CODE__
renderTimeline(app.state.timeline);
var rows = [], nowline = null, axis = [], key = [];
(stub["#timeline"].children || []).forEach(function (c) {
  (c.children || []).forEach(function (g) {
    if (g.cls === "gantt-now") nowline = g.style.left;
    if ((g.cls || "").indexOf("gantt-row") === 0) {
      rows.push({ lane: g.children[0].text,
                  bars: (g.children[1].children || []).map(function (b) {
                    return { label: b.children[0] ? b.children[0].text : "",
                             left: parseFloat(b.style.left), width: parseFloat(b.style.width),
                             cls: b.cls, hue: b.style["--hue"] }; }) });
    }
  });
  if (c.cls === "gantt-axis") axis = c.children.map(function (x) { return x.text; });
  if (c.cls === "gantt-key") key = c.children.map(function (x) {
    return { hue: x.children[0].style["--hue"],
             dish: x.children[1] ? x.children[1].text : "" }; });
});
JSON.stringify({ rows: rows, now: nowline, axis: axis, key: key,
                 empty: (stub["#timeline"].children || []).some(function (c) {
                   return c.cls === "empty"; }) });
"""


def draw(session, store, now):
    dukpy = pytest.importorskip("dukpy")
    src = open("src/cooking_assistant_ai/web/app.js", encoding="utf-8").read()
    code = src[src.index("  const DISH_HUES = ["):src.index("  async function renderChoices() {")]
    state = state_dict(session, now, store)
    js = HARNESS.replace("__STATE__", json.dumps(state)).replace("__CODE__", code)
    return json.loads(dukpy.evaljs(js))


@pytest.fixture
def planned(ctx, prepped):
    """Two dishes interleaved over two burners, as in a real cook."""
    for args in (
        {"recipe_id": "r001", "label": "sear thighs", "step_ids": ["r001-s3"],
         "duration_s": 300, "appliance": "stovetop"},
        {"recipe_id": "r002", "label": "rice boil", "step_ids": ["r002-s2"],
         "duration_s": 600, "appliance": "stovetop"},
        {"recipe_id": "r002", "label": "rice simmer", "step_ids": ["r002-s3"],
         "duration_s": 900, "appliance": "stovetop", "after": "rice boil"},
    ):
        assert dispatch(ctx, "add_task", args).ok, args["label"]
    return ctx


def test_each_piece_of_equipment_gets_its_own_lane(planned):
    out = draw(planned.session, planned.store, planned.clock.now())
    lanes = [r["lane"] for r in out["rows"]]
    assert lanes == sorted(lanes)                 # stable order, no jumping about
    assert all(l.startswith("Burner") for l in lanes)
    assert len(lanes) == 2                        # two pans, two burners


def test_a_bar_sits_where_its_task_does(planned):
    out = draw(planned.session, planned.store, planned.clock.now())
    bars = [b for r in out["rows"] for b in r["bars"]]
    assert {b["label"] for b in bars} == {"sear thighs", "rice boil", "rice simmer"}
    for b in bars:
        assert 0 <= b["left"] <= 100 and b["width"] > 0
        assert b["left"] + b["width"] <= 100.5


def test_the_longer_task_draws_the_longer_bar(planned):
    out = draw(planned.session, planned.store, planned.clock.now())
    by = {b["label"]: b for r in out["rows"] for b in r["bars"]}
    assert by["rice simmer"]["width"] > by["rice boil"]["width"] > by["sear thighs"]["width"]


def test_tasks_chained_after_one_another_do_not_overlap(planned):
    out = draw(planned.session, planned.store, planned.clock.now())
    by = {b["label"]: b for r in out["rows"] for b in r["bars"]}
    boil, simmer = by["rice boil"], by["rice simmer"]
    assert simmer["left"] >= boil["left"] + boil["width"] - 0.1


def test_a_dish_keeps_one_colour_across_its_tasks(planned):
    out = draw(planned.session, planned.store, planned.clock.now())
    by = {b["label"]: b for r in out["rows"] for b in r["bars"]}
    assert by["rice boil"]["hue"] == by["rice simmer"]["hue"]
    assert by["sear thighs"]["hue"] != by["rice boil"]["hue"]
    assert len(out["key"]) == 2 and all(k["dish"] for k in out["key"])


def test_a_started_task_is_drawn_as_running(planned):
    # start_task refuses a task whose prep is outstanding, which is the point of that gate;
    # assert the dispatch rather than assuming it, or this tests nothing.
    assert dispatch(planned, "mark_complete",
                    {"step_ids": ["r001-s1", "r001-s2"]}).ok
    assert dispatch(planned, "start_task", {"task_id": "sear thighs"}).ok
    out = draw(planned.session, planned.store, planned.clock.now())
    by = {b["label"]: b for r in out["rows"] for b in r["bars"]}
    assert "active" in by["sear thighs"]["cls"]
    assert "active" not in by["rice boil"]["cls"]


def test_the_axis_reads_in_the_cook_s_own_clock(planned):
    """It briefly showed 11:08 PM for a 7:08 PM plan, from a needless round-trip via UTC."""
    out = draw(planned.session, planned.store, planned.clock.now())
    assert out["axis"] and all(x.endswith("AM") or x.endswith("PM") for x in out["axis"])
    assert out["axis"][0].startswith("6:")        # the fixture clock is 6:30 PM


def test_now_is_marked_across_the_lanes(planned):
    out = draw(planned.session, planned.store, planned.clock.now())
    assert out["now"] and "calc(" in out["now"]


def test_an_empty_plan_says_so_rather_than_drawing_nothing(ctx):
    out = draw(ctx.session, ctx.store, ctx.clock.now())
    assert out["empty"] and not out["rows"]
