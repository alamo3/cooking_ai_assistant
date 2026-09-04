"""Opt-in integration test: Kokoro speaks, faster-whisper listens.

Downloads model weights on first run, so it only runs with COOK_SPEECH_TESTS=1.
"""
from __future__ import annotations

import os
import re

import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("COOK_SPEECH_TESTS"), reason="set COOK_SPEECH_TESTS=1")

SENTENCE = "Turn off the heat and let the rice sit covered for ten minutes."


async def test_tts_to_stt_round_trip():
    pytest.importorskip("kokoro")
    pytest.importorskip("faster_whisper")
    from cooking_assistant_ai.speech.stt import FasterWhisperSTT
    from cooking_assistant_ai.speech.tts import KokoroTTS

    tts = KokoroTTS()
    pcm = await tts.synthesize(SENTENCE)
    assert pcm and len(pcm) > 24000  # more than half a second of 24 kHz int16

    stt = FasterWhisperSTT(model_size="small")
    heard = await stt.transcribe(pcm, sample_rate=tts.sample_rate)
    norm = lambda s: re.sub(r"[^a-z ]", "", s.lower())  # noqa: E731
    want, got = norm(SENTENCE).split(), norm(heard).split()
    overlap = len(set(want) & set(got)) / len(set(want))
    assert overlap >= 0.8, f"heard: {heard!r}"
