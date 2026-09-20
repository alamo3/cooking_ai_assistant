"""The timeline is a picture, so the picture is what gets checked.

It began as a four-column text table nobody read, became a Gantt chart - lanes of equipment
with proportional bars - and then the same chart again in pixels per minute. Both were scales,
and a scale cannot draw this: a thirty-second step and a forty-minute simmer have to be equally
readable. It is a sequence now. Events sit evenly along the line in the order they happen and
the time between them is written on the segment that joins them.

These run the page's own renderTimeline in a JS engine and assert on the geometry it produced,
because there is no browser here and a wrong bar is invisible in a diff.
"""
from __future__ import annotations

import json

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
function requestAnimationFrame(f) {}
function applianceIcon(family) {
  return { tag: "svg", cls: "appl-icon", text: "", children: [], family: family,
           style: { setProperty: function () {} }, appendChild: function () {} };
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
var out = { events: [], dots: [], gaps: [], now: null, key: [], empty: false };
function scan(node) {
  var c = node.cls || "";
  if (c.indexOf("tl-event") === 0) out.events.push({
    left: parseFloat(node.style.left), cls: c,
    when: node.children[0].children[0].text,
    takes: node.children[0].children[1].text,
    what: node.children[1].text,
    where: node.children[2] ? node.children[2].children.map(function (x) {
      return x.family || x.text || ""; }).join(" ") : "",
    hue: node.style["--hue"] });
  if (c.indexOf("tl-dot") === 0) out.dots.push({ left: parseFloat(node.style.left), cls: c });
  if (c === "tl-gap") out.gaps.push({ left: parseFloat(node.style.left),
      text: node.children[0].text });
  if (c === "tl-now") out.now = parseFloat(node.style.left);
  if (c === "tl-key-item") out.key.push(node.children[1] ? node.children[1].text : "");
  if (c === "empty") out.empty = true;
  (node.children || []).forEach(scan);
}
(stub["#timeline"].children || []).forEach(scan);
JSON.stringify(out);
"""


def draw(ctx):
    dukpy = pytest.importorskip("dukpy")
    src = open("src/cooking_assistant_ai/web/app.js", encoding="utf-8").read()
    code = src[src.index("  const DISH_HUES = ["):src.index("  async function renderChoices() {")]
    state = state_dict(ctx.session, ctx.clock.now(), ctx.store)
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


def by_label(out):
    return {e["what"]: e for e in out["events"]}


@pytest.fixture
def lopsided(ctx, prepped):
    """Thirty seconds beside forty minutes: what no proportional layout can draw."""
    for args in (
        {"recipe_id": "r001", "label": "quick flip", "step_ids": ["r001-s4"],
         "duration_s": 30, "appliance": "stovetop"},
        {"recipe_id": "r001", "label": "long roast", "step_ids": ["r001-s5"],
         "duration_s": 2400, "appliance": "oven", "after": "quick flip"},
    ):
        assert dispatch(ctx, "add_task", args).ok, args["label"]
    return ctx


def test_every_task_becomes_an_event_on_the_line(planned):
    out = draw(planned)
    assert set(by_label(out)) == {"sear thighs", "rice boil", "rice simmer"}
    assert len(out["dots"]) == 3


def test_events_are_spaced_evenly_whatever_they_take(lopsided):
    """The whole reason for the rewrite: spacing by duration made one of these a smear."""
    out = draw(lopsided)
    xs = sorted(d["left"] for d in out["dots"])
    assert len(xs) == 2
    assert xs[1] - xs[0] == pytest.approx(168, abs=1)   # one slot, regardless


def test_a_thirty_second_step_and_a_forty_minute_one_read_the_same(lopsided):
    out = draw(lopsided)
    ev = by_label(out)
    assert ev["quick flip"]["takes"] == "30s"
    assert ev["long roast"]["takes"] == "40 min"


def test_the_wait_between_tasks_is_written_on_the_line(lopsided):
    out = draw(lopsided)
    assert len(out["gaps"]) == 1
    assert out["gaps"][0]["text"] == "30s"          # the flip finishes, the roast begins


def test_there_is_one_gap_label_between_each_pair(planned):
    out = draw(planned)
    assert len(out["gaps"]) == len(out["events"]) - 1
    assert all(g["text"] for g in out["gaps"])


def test_tasks_starting_together_say_so_rather_than_showing_nothing(ctx, prepped):
    for args in (
        {"recipe_id": "r001", "label": "sear thighs", "step_ids": ["r001-s3"],
         "duration_s": 300, "appliance": "stovetop"},
        {"recipe_id": "r002", "label": "rice boil", "step_ids": ["r002-s2"],
         "duration_s": 600, "appliance": "stovetop"},
    ):
        assert dispatch(ctx, "add_task", args).ok
    out = draw(ctx)
    assert out["gaps"][0]["text"] == "same time"


def test_events_run_left_to_right_in_the_order_they_happen(planned):
    out = draw(planned)
    assert out["dots"] == sorted(out["dots"], key=lambda d: d["left"])


def test_cards_alternate_above_and_below_the_line(planned):
    """That is how a timeline keeps its labels apart."""
    out = draw(planned)
    sides = ["up" if "up" in e["cls"] else "down" for e in out["events"]]
    assert sides == ["up", "down", "up"]


def test_no_card_hangs_off_the_left_edge(planned):
    """The first card sat at -26px and lost its border and half its time."""
    out = draw(planned)
    assert all(e["left"] >= 0 for e in out["events"])
    assert out["now"] >= 0


def test_a_dish_keeps_one_colour(planned):
    out = draw(planned)
    ev = by_label(out)
    assert ev["rice boil"]["hue"] == ev["rice simmer"]["hue"]
    assert ev["sear thighs"]["hue"] != ev["rice boil"]["hue"]
    assert len(out["key"]) == 2 and all(out["key"])


def test_each_event_says_when_and_where(planned):
    out = draw(planned)
    for e in out["events"]:
        assert e["when"].endswith("AM") or e["when"].endswith("PM")
        assert "Burner" in e["where"] and "stovetop" in e["where"]


def test_a_running_task_is_drawn_as_running(planned):
    assert dispatch(planned, "mark_complete", {"step_ids": ["r001-s1", "r001-s2"]}).ok
    assert dispatch(planned, "start_task", {"task_id": "sear thighs"}).ok
    out = draw(planned)
    assert "active" in by_label(out)["sear thighs"]["cls"]
    assert "active" not in by_label(out)["rice boil"]["cls"]


def test_now_is_marked(planned):
    out = draw(planned)
    assert out["now"] is not None and out["now"] >= 0


def test_an_empty_plan_says_so_rather_than_drawing_nothing(ctx):
    out = draw(ctx)
    assert out["empty"] and not out["events"]
