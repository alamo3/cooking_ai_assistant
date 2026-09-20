"""The timeline is a picture, so the picture is what gets checked.

It began as a four-column text table nobody read, became a Gantt chart - lanes of equipment
with proportional bars, which is a project plan and truncated every label to "chi..." in a
narrow column - and is now a timeline proper: one line, events pinned along it in the order
they happen, each leading to the next.

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
var out = { events: [], dots: [], runs: [], stalks: [], now: null, key: [], empty: false };
function scan(node) {
  var c = node.cls || "";
  if (c.indexOf("tl-event") === 0) out.events.push({
    left: parseFloat(node.style.left), cls: c,
    when: node.children[0].text, what: node.children[1].text,
    where: node.children[2] ? node.children[2].children.map(function (x) {
      return x.family || x.text || ""; }).join(" ") : "",
    hue: node.style["--hue"] });
  if (c.indexOf("tl-dot") === 0) out.dots.push({ left: parseFloat(node.style.left), cls: c });
  if (c.indexOf("tl-run") === 0) out.runs.push({ left: parseFloat(node.style.left),
      width: parseFloat(node.style.width), cls: c });
  if (c.indexOf("tl-stalk") === 0) out.stalks.push({ left: parseFloat(node.style.left),
      width: parseFloat(node.style.width), cls: c });
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


def test_every_task_becomes_an_event_on_the_line(planned):
    out = draw(planned)
    assert set(by_label(out)) == {"sear thighs", "rice boil", "rice simmer"}
    assert len(out["dots"]) == 3 and len(out["runs"]) == 3


def test_events_run_left_to_right_in_the_order_they_happen(planned):
    out = draw(planned)
    assert out["dots"] == sorted(out["dots"], key=lambda d: d["left"])
    assert all(d["left"] >= 0 for d in out["dots"])


def test_how_long_a_task_runs_is_drawn_on_the_line(planned):
    out = draw(planned)
    runs = sorted(out["runs"], key=lambda r: r["width"])
    # 5 min sear, 10 min boil, 15 min simmer, at a fixed pixels-per-minute
    assert runs[0]["width"] < runs[1]["width"] < runs[2]["width"]
    assert runs[1]["width"] == pytest.approx(runs[0]["width"] * 2, rel=0.01)


def test_cards_alternate_above_and_below_the_line(planned):
    """That is how a historical timeline keeps its labels apart."""
    out = draw(planned)
    sides = ["up" if "up" in e["cls"] else "down" for e in out["events"]]
    assert sides == ["up", "down", "up"]


def test_a_card_only_slides_when_its_own_side_is_crowded(planned):
    """Counting both sides together cascaded: every card ended up evenly spaced and none of
    them anywhere near its own time."""
    out = draw(planned)
    dots = sorted(d["left"] for d in out["dots"])
    cards = sorted(e["left"] for e in out["events"])
    # the first two are opposite each other, so neither needs to move
    assert cards[0] == dots[0]
    assert cards[1] == dots[1]
    # and nothing is ever dragged left of its own moment
    assert all(c >= d - 0.01 for c, d in zip(cards, dots))


def test_the_dot_stays_on_the_true_time_when_a_card_slides(planned):
    out = draw(planned)
    leaning = [s for s in out["stalks"] if s["width"] > 0]
    for stalk in leaning:
        assert any(abs(d["left"] - stalk["left"]) < 0.01 for d in out["dots"])


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
        assert "Burner" in e["where"]
        assert "stovetop" in e["where"]          # the drawn appliance icon rides along


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
