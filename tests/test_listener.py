from __future__ import annotations

import math
import random
from typing import List

import pytest

from cooking_assistant_ai.speech.listener import Listener, clean_transcript, is_hallucination
from cooking_assistant_ai.speech.stt import STT
from cooking_assistant_ai.speech.vad import EnergyVAD

RATE = 16000


def silence(seconds: float) -> bytes:
    return b"\x00\x00" * int(RATE * seconds)


def noise(seconds: float, amp: float = 0.3, seed: int = 1) -> bytes:
    rng = random.Random(seed)
    out = bytearray()
    for _ in range(int(RATE * seconds)):
        v = int(max(-1, min(1, rng.uniform(-amp, amp))) * 32767)
        out += v.to_bytes(2, "little", signed=True)
    return bytes(out)


def chunks(pcm: bytes, ms: int = 100):
    n = RATE * ms // 1000 * 2
    for i in range(0, len(pcm), n):
        yield pcm[i:i + n]


class FakeSTT(STT):
    def __init__(self, text: str):
        self.text = text
        self.clips: List[bytes] = []

    async def transcribe(self, pcm16: bytes, sample_rate: int = 16000) -> str:
        self.clips.append(pcm16)
        return self.text


def make(text: str, playing: bool = False):
    stt = FakeTT = FakeSTT(text)
    heard: List[str] = []
    events: List[tuple] = []
    barge: List[int] = []

    async def on_utt(t):
        heard.append(t)

    async def on_ev(kind, data):
        events.append((kind, data))

    async def on_barge():
        barge.append(1)

    lst = Listener(stt, EnergyVAD(), on_utt, on_ev, is_playing=lambda: playing, on_barge_in=on_barge)
    return lst, stt, heard, events, barge


async def test_utterance_is_cut_with_pre_roll_and_transcribed():
    lst, stt, heard, events, _ = make("how long left on the rice")
    audio = silence(1.0) + noise(1.2) + silence(1.0)
    for c in chunks(audio):
        await lst.feed(c)
    await lst.wait_pending()
    assert heard == ["how long left on the rice"]
    assert [e for e in events if e[0] == "vad"] == [("vad", {"speaking": True}), ("vad", {"speaking": False})]
    clip_s = len(stt.clips[0]) / 2 / RATE
    assert 1.4 <= clip_s <= 1.9  # 1.2 s of speech + pre-roll + pad, not the whole 3.2 s stream


async def test_short_blips_and_hallucinations_are_dropped():
    lst, stt, heard, events, _ = make("Thank you.")
    for c in chunks(silence(0.5) + noise(0.15) + silence(1.0)):
        await lst.feed(c)
    await lst.wait_pending()
    assert heard == [] and stt.clips == []  # too short: never sent to STT
    for c in chunks(noise(1.0) + silence(1.0)):
        await lst.feed(c)
    await lst.wait_pending()
    assert heard == [] and len(stt.clips) == 1  # transcribed, then dropped as a hallucination
    assert ("dropped", {"reason": "empty", "text": "Thank you."}) in events


async def test_echo_of_assistant_speech_is_dropped_but_real_speech_passes():
    lst, stt, heard, events, barge = make("the rice timer is set for fifteen minutes", playing=True)
    lst.note_spoken("Rice timer is set for fifteen minutes. I'll tell you when it's done.")
    for c in chunks(silence(0.5) + noise(1.0) + silence(1.0)):
        await lst.feed(c)
    await lst.wait_pending()
    assert heard == []
    assert any(e[0] == "dropped" and e[1]["reason"] == "echo" for e in events)
    assert barge == [1]  # sustained speech while playing triggers exactly one barge-in
    # different words while playing: a real interruption
    stt.text = "wait, make that twenty minutes"
    for c in chunks(silence(0.2) + noise(1.0, seed=2) + silence(1.0)):
        await lst.feed(c)
    await lst.wait_pending()
    assert heard == ["wait, make that twenty minutes"]


async def test_transcript_mode_barges_in_only_on_real_speech():
    stt = FakeSTT("the rice timer is set for fifteen minutes")
    heard, barge = [], []

    async def on_utt(t):
        heard.append(t)

    async def on_barge():
        barge.append(1)

    lst = Listener(stt, EnergyVAD(), on_utt, None, is_playing=lambda: True, on_barge_in=on_barge, barge_in_mode="transcript")
    lst.note_spoken("Rice timer is set for fifteen minutes.")
    for c in chunks(noise(1.0) + silence(1.0)):
        await lst.feed(c)
    await lst.wait_pending()
    assert barge == [] and heard == []  # echo: no interruption at all
    stt.text = "no wait, make it twenty"
    for c in chunks(silence(0.2) + noise(1.0, seed=3) + silence(1.0)):
        await lst.feed(c)
    await lst.wait_pending()
    assert barge == [1] and heard == ["no wait, make it twenty"]


async def test_no_barge_in_when_not_playing():
    lst, stt, heard, events, barge = make("hello", playing=False)
    for c in chunks(noise(1.0) + silence(1.0)):
        await lst.feed(c)
    await lst.wait_pending()
    assert barge == [] and heard == ["hello"]


def test_transcript_cleaning():
    assert clean_transcript(" [BLANK_AUDIO]  set a timer (laughs) ") == "set a timer"
    assert is_hallucination("Thanks for watching!") and is_hallucination(".") and not is_hallucination("rice is on")
