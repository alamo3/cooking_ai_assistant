"""Noise the microphone picks up must not reach the cook as a reply.

From two real cooks: whisper transcribed "I'll see you next time!" and "I'll see you in the
next video." out of an empty kitchen, and a podcast playing nearby arrived as two long
sentences about AI in education. The model correctly decided it had nothing to say to any of
it and answered with its no-op token, NOTHING - which was only suppressed on idle turns, so on
a cook's turn the tablet said the word "Nothing" out loud. The cook replied "Sorry, what was
that?".
"""
from __future__ import annotations

import pytest

from cooking_assistant_ai.llm.llm_orchestrator import is_nothing
from cooking_assistant_ai.speech.listener import is_hallucination, too_unsure


# ----------------------------------------------------------------- the no-op token

@pytest.mark.parametrize("said", ["NOTHING", "NOTHING.", "Nothing", "nothing.", "  NOTHING  "])
def test_the_no_op_token_is_recognised(said):
    assert is_nothing(said)


@pytest.mark.parametrize("said", [
    "Nothing else needs doing.",
    "There's nothing left on the hob.",
    "Nothing to add to the curry yet.",
    "",
])
def test_a_sentence_that_merely_contains_it_is_not(said):
    assert not is_nothing(said)


# --------------------------------------------------------------- whisper sign-offs

@pytest.mark.parametrize("said", [
    "I'll see you next time!",
    "I'll see you in the next video.",
    "See you next time.",
    "Thanks for watching!",
    "Thank you for watching",
    "Please subscribe",
    "Subtitles by the Amara.org community",
    "Stay tuned",
    "",
    "   ",
])
def test_the_silence_hallucinations_are_dropped(said):
    assert is_hallucination(said)


@pytest.mark.parametrize("said", [
    "I've chopped the onions.",
    "See if the rice is done, will you?",
    "Thanks, that worked.",
    "How much garlic for the curry?",
    "Set a timer for ten minutes.",
    "Nothing else needs doing.",
])
def test_real_speech_survives(said):
    assert not is_hallucination(said), said


# ----------------------------------------------------- what the decoder thought of it

def test_a_confident_transcript_is_kept():
    assert not too_unsure(no_speech=0.05, logprob=-0.25)


def test_the_decoder_saying_there_was_no_speech_drops_it():
    assert too_unsure(no_speech=0.95, logprob=-0.3)


def test_a_transcript_it_was_guessing_at_is_dropped():
    assert too_unsure(no_speech=0.1, logprob=-2.4)


def test_a_backend_that_reports_nothing_is_never_penalised():
    """None means "not reported", which must never be read as "bad"."""
    assert not too_unsure(no_speech=None, logprob=None)
    assert not too_unsure(no_speech=None, logprob=-0.3)
    assert not too_unsure(no_speech=0.2, logprob=None)


def test_noise_transcribed_as_a_short_word_is_caught_here_not_by_a_wordlist():
    """A real cook collected "kal kal" off the microphone. It is three characters, like
    "yes", "off" and "hot", so no length rule or list of nonsense syllables can tell them
    apart - but the decoder knows it was guessing."""
    assert not is_hallucination("kal kal")      # deliberately not on any list
    assert too_unsure(no_speech=0.9, logprob=-1.8)


def test_a_cook_over_a_noisy_extractor_is_still_heard():
    """The thresholds are loose on purpose: losing a real instruction is the worse error."""
    assert not too_unsure(no_speech=0.55, logprob=-0.9)


# ------------------------------------------------------- through the listener for real

class FakeSTT:
    """Stands in for whisper, returning a fixed decode with whatever confidence we choose."""

    def __init__(self, heard):
        self.heard = heard

    async def transcribe(self, pcm16, sample_rate=16000):
        return self.heard.text

    async def listen(self, pcm16, sample_rate=16000):
        return self.heard


async def run_listener(heard):
    from cooking_assistant_ai.speech.listener import Listener
    from cooking_assistant_ai.speech.vad import EnergyVAD

    through, dropped = [], []

    async def on_utterance(text):
        through.append(text)

    async def on_event(kind, data):
        if kind == "dropped":
            dropped.append(data.get("reason"))

    listener = Listener(FakeSTT(heard), EnergyVAD(), on_utterance,
                        on_event=on_event, min_utterance_s=0.1)
    await listener.feed(b"\x00\x40" * 8000)      # half a second over the RMS gate
    await listener.feed(b"\x00\x00" * 16000)     # silence closes the utterance
    for task in list(listener._tasks):
        await task
    return through, dropped


async def test_a_confident_cook_gets_through():
    from cooking_assistant_ai.speech.stt import Heard

    through, dropped = await run_listener(Heard("add the garlic now", 0.05, -0.2))
    assert through == ["add the garlic now"] and dropped == []


async def test_a_sign_off_into_an_empty_kitchen_is_dropped():
    from cooking_assistant_ai.speech.stt import Heard

    through, dropped = await run_listener(Heard("I'll see you next time!", 0.1, -0.3))
    assert through == [] and dropped == ["empty"]


async def test_noise_the_decoder_was_unsure_about_is_dropped():
    from cooking_assistant_ai.speech.stt import Heard

    through, dropped = await run_listener(Heard("kal kal", 0.95, -1.9))
    assert through == [] and dropped == ["unsure"]


async def test_a_backend_without_confidence_still_passes_speech_through():
    from cooking_assistant_ai.speech.stt import Heard

    through, dropped = await run_listener(Heard("add the garlic now", None, None))
    assert through == ["add the garlic now"] and dropped == []
