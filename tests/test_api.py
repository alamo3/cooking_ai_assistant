from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from cooking_assistant_ai.api.app import Settings, create_app
from cooking_assistant_ai.core.tools import dispatch
from cooking_assistant_ai.llm.client import ScriptedLLM, call
from cooking_assistant_ai.storage.db import Store


@pytest.fixture
def client():
    llm = ScriptedLLM()
    app = create_app(Settings(no_llm=True, idle_interval_s=0, warm_model=False), llm=llm, store=Store(":memory:"))
    with TestClient(app) as c:
        c.llm = llm  # type: ignore[attr-defined]
        yield c


def test_http_surface(client):
    assert client.get("/health").json()["ok"]
    recipes = client.get("/recipes").json()
    assert [r["id"] for r in recipes] == ["r001", "r002", "r003"]
    assert client.get("/recipes/r002").json()["title"] == "Jasmine Rice"
    new = client.post("/recipes", json={"title": "Toast", "servings": 1,
                                        "ingredients": [{"name": "bread", "amount": 2}],
                                        "steps": [{"text": "Toast it.", "duration_s": 120}]})
    assert new.status_code == 201 and new.json()["id"] == "r004"
    assert client.delete("/recipes/r004").status_code == 204
    assert client.get("/recipes/r004").status_code == 404
    inv = client.patch("/inventory", json={"items": [{"name": "saffron", "amount": 1, "unit": "g"}], "remove": ["thyme"]}).json()
    names = [i["name"] for i in inv]
    assert "saffron" in names and "thyme" not in names


def test_session_lifecycle_and_state(client):
    r = client.post("/session", json={"recipe_ids": ["r001", "rice"], "target_plating": "7:30 PM"})
    assert r.status_code == 201
    sid = r.json()["session_id"]
    state = client.get(f"/session/{sid}/state").json()
    assert {x["id"] for x in state["progress"]["recipes"]} == {"r001", "r002"}
    assert state["target_plating"].endswith("19:30:00")
    assert client.delete(f"/session/{sid}").status_code == 204
    assert client.get(f"/session/{sid}/state").status_code == 404


def test_import_recipe_from_pasted_text(client):
    import json
    client.llm.push(json.dumps({
        "title": "Garlic Toast", "servings": 2,
        "ingredients": [{"name": "bread", "amount": 2, "unit": None}, {"name": "butter", "amount": 1, "unit": "tbsp"}],
        "steps": [{"text": "Butter the bread.", "duration_s": None, "appliance": None, "temp_f": None},
                  {"text": "Toast at 400F for 5 minutes.", "duration_s": 300, "appliance": "oven", "temp_f": 400}],
    }))
    r = client.post("/recipes", json={"text": "Garlic toast\n2 slices bread\n1 tbsp butter\nButter the bread. Toast at 400F for 5 minutes."})
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["id"] == "r004" and body["steps"][1]["appliance"] == "oven"
    prompt = client.llm.calls[-1][-1]["content"]
    assert "PASTED RECIPE TEXT" in prompt and "1 tbsp butter" in prompt
    assert client.post("/recipes", json={"text": "hi"}).status_code == 422


def test_ui_tool_endpoints_and_state_shape(client):
    sid = client.post("/session", json={"recipe_ids": ["r001"]}).json()["session_id"]
    state = client.post(f"/session/{sid}/recipes", json={"recipe": "rice"}).json()
    recipes = {r["id"]: r for r in state["progress"]["recipes"]}
    assert set(recipes) == {"r001", "r002"}
    rice = recipes["r002"]
    assert rice["current_step"] == 1 and rice["steps"][0]["status"] == "pending"
    assert rice["ingredients"][0]["text"] == "1 1/2 cup jasmine rice"
    state = client.post(f"/session/{sid}/steps/r002-s1/complete").json()
    rice = next(r for r in state["progress"]["recipes"] if r["id"] == "r002")
    assert rice["current_step"] == 2 and rice["steps"][0]["status"] == "done"
    assert client.post(f"/session/{sid}/recipes", json={"recipe": "nope"}).status_code == 400
    assert client.delete(f"/session/{sid}/timers/tm_999").status_code == 400
    state = client.delete(f"/session/{sid}/recipes/r002").json()
    assert [r["id"] for r in state["progress"]["recipes"]] == ["r001"]
    assert state["plan"]["now_id"] == "r001-s1" and state["plan"]["items"][0]["kind"] == "step"
    assert client.post(f"/session/{sid}/prep/prep_nothing/complete").status_code == 400


def test_start_cooking_loads_selection_and_briefs(client):
    client.llm.push(
        call("add_task", label="rice", recipe_id="r002", step_ids=["r002-s2", "r002-s3"], appliance="stovetop", must_finish_by="plating"),
        "Here's the plan: about forty minutes, one burner and the oven. First, rinse the rice.",
    )
    sid = client.post("/session", json={"recipe_ids": ["r003"]}).json()["session_id"]
    with client.websocket_connect(f"/ws/session?session_id={sid}") as ws:
        ws.receive_json(); ws.receive_json()
        r = client.post(f"/session/{sid}/start", json={"recipe_ids": ["r001", "rice"], "minutes_from_now": 60})
        assert r.status_code == 200, r.text
        state = r.json()
        assert {x["id"] for x in state["progress"]["recipes"]} == {"r001", "r002"}  # sprouts unloaded, chicken+rice in
        assert state["target_plating"] is not None
        assert state["plan"]["summary"]["recipes"][0]["title"] == "Roast Chicken Thighs"
        seen = []
        while True:
            m = ws.receive_json()
            seen.append(m["type"])
            if m["type"] == "speech_end":
                break
        assert "tool_call" in seen and "speech_start" in seen
    prompt = client.llm.calls[0][-1]["content"]
    assert prompt.startswith("[SYSTEM] The cook chose these recipes on the tablet: Roast Chicken Thighs, Jasmine Rice.")
    assert "PLAN SUMMARY" in prompt and "brief the cook" in prompt
    tool_result = client.llm.calls[1][-1]["content"]
    assert "PLAN SUMMARY" in tool_result and "Appliances:" in tool_result and "Ingredients to get out" in tool_result
    assert client.post(f"/session/{sid}/start", json={"recipe_ids": []}).status_code == 400
    assert client.post(f"/session/{sid}/start", json={"recipe_ids": ["nope"]}).status_code == 404


def test_static_app_is_served(client):
    r = client.get("/app/")
    assert r.status_code == 200 and "Kitchen Assistant" in r.text
    assert client.get("/app/app.js").status_code == 200
    assert client.get("/", follow_redirects=False).status_code in (302, 307)


def test_open_mic_listen_flow():
    """Stream audio over the websocket: VAD segments it, STT transcribes, the model answers."""
    import base64

    from cooking_assistant_ai.speech.stt import STT
    from tests.test_listener import chunks, noise, silence

    class FakeSTT(STT):
        async def transcribe(self, pcm16, sample_rate=16000):
            return "how long left on the rice"

    llm = ScriptedLLM(["About eight minutes."])
    app = create_app(Settings(no_llm=True, idle_interval_s=0, warm_model=False, vad="energy"),
                     llm=llm, store=Store(":memory:"), stt=FakeSTT())
    with TestClient(app) as c:
        sid = c.post("/session", json={"recipe_ids": ["r002"]}).json()["session_id"]
        with c.websocket_connect(f"/ws/session?session_id={sid}") as ws:
            ws.receive_json(); ws.receive_json()
            ws.send_json({"type": "listen", "on": True})
            assert ws.receive_json() == {"type": "listen", "on": True}
            for chunk in chunks(silence(0.5) + noise(1.0) + silence(1.0)):
                ws.send_json({"type": "audio_chunk", "data": base64.b64encode(chunk).decode()})
            seen = []
            while True:
                m = ws.receive_json()
                seen.append(m["type"])
                if m["type"] == "speech_end":
                    break
            assert seen[:2] == ["vad", "vad"]
            assert "transcript" in seen and "speech_start" in seen
            ws.send_json({"type": "listen", "on": False})
            assert ws.receive_json() == {"type": "listen", "on": False}


def test_websocket_turn(client):
    client.llm.push(call("set_timer", label="rice simmering", duration_s=900), "Timer's on.")
    sid = client.post("/session", json={"recipe_ids": ["r002"]}).json()["session_id"]
    with client.websocket_connect(f"/ws/session?session_id={sid}") as ws:
        assert ws.receive_json()["type"] == "session"
        assert ws.receive_json()["type"] == "state"
        ws.send_json({"type": "text", "text": "rice is simmering, timer please"})
        types = []
        while True:
            msg = ws.receive_json()
            types.append(msg["type"])
            if msg["type"] == "speech_end":
                break
        assert types[0] == "transcript"
        assert "tool_call" in types and "state" in types and "text" in types
        assert types.index("tool_call") < types.index("speech_start")
        ws.send_json({"type": "set_proactivity", "value": 0.2})
        assert ws.receive_json()["proactivity"] == 0.2
        ws.send_json({"type": "audio", "data": ""})
        assert "no STT" in ws.receive_json()["text"]


def test_substitution_is_saved_to_the_library(client):
    """A swap edits the stored recipe, so the Recipes screen shows it with or without a session."""
    sid = client.post("/session", json={"recipe_ids": ["r001"]}).json()["session_id"]
    ctx = client.app.state.manager.get(sid).orchestrator.ctx
    assert dispatch(ctx, "substitute", {"recipe_id": "r001", "ingredient_id": "butter",
                                        "replacement": "olive oil"}).ok

    step = lambda d: next(s["text"] for s in d["steps"] if "skillet" in s["text"])
    for view in (client.get("/recipes/r001").json(),
                 client.get(f"/recipes/r001?session={sid}").json()):
        assert "olive oil" in step(view) and "butter" not in step(view)
    assert not any(i["name"] == "butter" for i in client.get("/recipes/r001").json()["ingredients"])
    assert client.get("/recipes/r001?session=nosuch").status_code == 200  # unknown session, no crash
