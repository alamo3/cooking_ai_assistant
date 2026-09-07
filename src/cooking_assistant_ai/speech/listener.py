"""Open-microphone listener: continuous 16 kHz PCM16 in, finished utterances out.

Transport-agnostic. The websocket layer feeds audio chunks and supplies callbacks:

* on_utterance(text)         a transcribed, cleaned, non-echo utterance
* on_event(kind, data)       UI hints: ("vad", {"speaking": bool}), ("dropped", {...})
* is_playing()               whether the client is currently playing assistant audio
* on_barge_in()              speech persisted while the assistant was playing

Echo handling: the browser's AEC removes most of the assistant's own voice. What leaks
through tends to be transcribed as fragments of what was just said, so utterances that
began while audio was playing (or right after) are dropped when most of their words appear
in recently spoken sentences. Whisper's silence hallucinations are dropped as well.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from collections import deque
from typing import Any, Awaitable, Callable, Deque, Dict, Optional, Set, Tuple

from cooking_assistant_ai.speech.stt import STT
from cooking_assistant_ai.speech.vad import VAD

log = logging.getLogger(__name__)

_HALLUCINATIONS = {
    "thank you", "thanks", "thank you for watching", "thanks for watching", "you", "bye", "goodbye",
    "subtitles by the amara org community", "the end", "so", "um", "uh", "hmm", "oh", "okay", "ok",
    "please subscribe", "like and subscribe", "music", "applause", "silence",
}
_STOPWORDS = {"the", "a", "an", "and", "to", "of", "it", "is", "for", "on", "in", "at", "that", "this", "i", "you", "your"}


def clean_transcript(text: str) -> str:
    text = re.sub(r"\[[^\]]*\]|\([^)]*\)", " ", text)  # [BLANK_AUDIO], (laughs)
    return re.sub(r"\s+", " ", text).strip()


def is_hallucination(text: str) -> bool:
    norm = re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()
    return not norm or norm in _HALLUCINATIONS or len(norm) < 2


def _words(text: str) -> Set[str]:
    return {w for w in re.findall(r"[a-z0-9']+", text.lower()) if w not in _STOPWORDS}


class Listener:
    def __init__(self, stt: STT, vad: VAD,
                 on_utterance: Callable[[str], Awaitable[None]],
                 on_event: Optional[Callable[[str, Dict[str, Any]], Awaitable[None]]] = None,
                 is_playing: Optional[Callable[[], bool]] = None,
                 on_barge_in: Optional[Callable[[], Awaitable[None]]] = None,
                 min_utterance_s: float = 0.4, max_utterance_s: float = 25.0,
                 pre_roll_ms: int = 300,
                 # 400 ms of "speech" is met by a pan lid, a tap or an extractor fan:
                 # Silero scores plenty of kitchen noise as voice. Nearly a second of
                 # sustained speech is still responsive but ignores almost all of it.
                 barge_in_after_ms: int = int(os.environ.get("COOK_BARGE_IN_MS", "900")),
                 echo_window_s: float = 30.0, echo_grace_s: float = 1.5,
                 barge_in_mode: str = "voice"):
        # barge_in_mode: "voice" interrupts after barge_in_after_ms of sustained speech while
        # playing (instant, relies on the client's echo cancellation); "transcript" waits until
        # the utterance is transcribed and passes the echo guard (about a second slower, can
        # never be triggered by the assistant's own voice); "off" disables voice interruption.
        self.stt = stt
        self.vad = vad
        self.on_utterance = on_utterance
        self.on_event = on_event
        self.is_playing = is_playing or (lambda: False)
        self.on_barge_in = on_barge_in
        self.rate = 16000
        self.min_samples = int(min_utterance_s * self.rate)
        self.max_samples = int(max_utterance_s * self.rate)
        self.pre_roll = int(pre_roll_ms * self.rate / 1000)
        self.barge_in_after = int(barge_in_after_ms * self.rate / 1000)
        self.echo_window_s = echo_window_s
        self.echo_grace_s = echo_grace_s
        if barge_in_mode not in ("voice", "transcript", "off"):
            raise ValueError(f"unknown barge_in_mode '{barge_in_mode}'")
        self.barge_in_mode = barge_in_mode

        self._buf = bytearray()
        self._buf_start = 0  # absolute sample index of _buf[0]
        self._pos = 0  # absolute samples fed so far
        self._speech_start: Optional[int] = None  # clip start (includes pre-roll)
        self._speech_raw_start = 0  # where the VAD said speech began
        self._started_while_playing = False
        self._barged = False
        self._last_playing_at = 0.0
        self._spoken: Deque[Tuple[float, str]] = deque()
        self._tasks: Set[asyncio.Task] = set()
        self.utterances = 0
        self.dropped = 0

    # -- assistant speech bookkeeping (for the echo guard) -------------------

    def note_spoken(self, text: str) -> None:
        now = time.monotonic()
        self._spoken.append((now, text))
        while self._spoken and now - self._spoken[0][0] > self.echo_window_s:
            self._spoken.popleft()

    def _recently_playing(self) -> bool:
        if self.is_playing():
            self._last_playing_at = time.monotonic()
            return True
        return time.monotonic() - self._last_playing_at < self.echo_grace_s

    def is_echo(self, text: str) -> bool:
        words = _words(text)
        if not words:
            return False
        now = time.monotonic()
        recent: Set[str] = set()
        for at, s in self._spoken:
            if now - at <= self.echo_window_s:
                recent |= _words(s)
        if not recent:
            return False
        overlap = len(words & recent) / len(words)
        return overlap >= 0.7 if len(words) >= 3 else words <= recent

    # -- audio in ------------------------------------------------------------

    async def _emit(self, kind: str, data: Dict[str, Any]) -> None:
        if self.on_event:
            await self.on_event(kind, data)

    async def feed(self, pcm16: bytes) -> None:
        self._buf += pcm16
        self._pos += len(pcm16) // 2
        playing_now = self._recently_playing()

        for kind, idx in self.vad.process(pcm16):
            if kind == "start" and self._speech_start is None:
                self._speech_start = max(self._buf_start, idx - self.pre_roll)
                self._speech_raw_start = idx
                self._started_while_playing = playing_now
                self._barged = False
                await self._emit("vad", {"speaking": True})
            elif kind == "end" and self._speech_start is not None:
                await self._finish(min(idx, self._pos))

        if self._speech_start is not None:
            if (self.barge_in_mode == "voice" and self.is_playing() and not self._barged and self.on_barge_in
                    and self._pos - self._speech_start - self.pre_roll >= self.barge_in_after):
                self._barged = True
                await self.on_barge_in()
            if self._pos - self._speech_start >= self.max_samples:
                await self._finish(self._pos)  # runaway utterance (background talk): cut it
                self.vad.reset()
        else:
            keep = self.pre_roll + self.rate  # idle: keep pre-roll plus a little slack
            if self._pos - self._buf_start > keep:
                self._trim(self._pos - keep)

    async def _finish(self, end: int) -> None:
        start = self._speech_start or self._buf_start
        speech_samples = max(0, end - self._speech_raw_start)
        self._speech_start = None
        await self._emit("vad", {"speaking": False})
        utter = self._slice(start, end)
        self._trim(end)
        task = asyncio.create_task(self._handle(utter, speech_samples, self._started_while_playing))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def _slice(self, start: int, end: int) -> bytes:
        a = max(0, (start - self._buf_start) * 2)
        b = max(a, (end - self._buf_start) * 2)
        return bytes(self._buf[a:b])

    def _trim(self, upto: int) -> None:
        n = max(0, (upto - self._buf_start) * 2)
        del self._buf[:n]
        self._buf_start = upto

    async def _handle(self, pcm: bytes, speech_samples: int, started_while_playing: bool) -> None:
        if speech_samples < self.min_samples:
            self.dropped += 1
            await self._emit("dropped", {"reason": "too short", "seconds": round(speech_samples / self.rate, 2)})
            return
        try:
            text = clean_transcript(await self.stt.transcribe(pcm, self.rate))
        except Exception as e:
            log.exception("transcription failed")
            await self._emit("error", {"text": f"transcription failed: {e}"})
            return
        if is_hallucination(text):
            self.dropped += 1
            await self._emit("dropped", {"reason": "empty", "text": text})
            return
        if started_while_playing and self.is_echo(text):
            self.dropped += 1
            await self._emit("dropped", {"reason": "echo", "text": text})
            return
        if (self.barge_in_mode == "transcript" and started_while_playing and self.on_barge_in
                and self._recently_playing()):
            await self.on_barge_in()
        self.utterances += 1
        await self.on_utterance(text)

    async def wait_pending(self, timeout: float = 30.0) -> None:
        """Test helper: wait for in-flight transcriptions."""
        if self._tasks:
            await asyncio.wait(list(self._tasks), timeout=timeout)

    def reset(self) -> None:
        self.vad.reset()
        self._buf = bytearray()
        self._buf_start = self._pos
        self._speech_start = None
