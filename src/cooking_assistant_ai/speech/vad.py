"""Voice activity detection over a 16 kHz PCM16 stream.

Both detectors consume audio in any chunk size and emit ("start", sample_index) and
("end", sample_index) events with *absolute* sample indices, so the listener can cut the
utterance out of its own buffer.
"""
from __future__ import annotations

import os
from typing import List, Optional, Tuple

Event = Tuple[str, int]


class VAD:
    sample_rate = 16000
    name = "vad"

    def process(self, pcm16: bytes) -> List[Event]:  # pragma: no cover
        raise NotImplementedError

    def reset(self) -> None:  # pragma: no cover
        raise NotImplementedError


class SileroVAD(VAD):
    """Silero VAD v5+ via the `silero-vad` package (torch). ~2.5 ms per 32 ms window on CPU."""

    name = "silero"
    WINDOW = 512  # samples per model call at 16 kHz

    def __init__(self, threshold: float = 0.5, min_silence_ms: int = 500, speech_pad_ms: int = 100):
        import numpy as np  # type: ignore
        import torch  # type: ignore
        from silero_vad import VADIterator, load_silero_vad  # type: ignore

        self._np = np
        self._torch = torch
        self._model = load_silero_vad()
        self._make = lambda: VADIterator(self._model, threshold=threshold, sampling_rate=16000,
                                         min_silence_duration_ms=min_silence_ms, speech_pad_ms=speech_pad_ms)
        self._it = self._make()
        self._pending = np.zeros(0, dtype=np.float32)

    def process(self, pcm16: bytes) -> List[Event]:
        np = self._np
        audio = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
        self._pending = np.concatenate([self._pending, audio])
        events: List[Event] = []
        while len(self._pending) >= self.WINDOW:
            win = self._pending[:self.WINDOW].copy()
            self._pending = self._pending[self.WINDOW:]
            result = self._it(self._torch.from_numpy(win), return_seconds=False)
            if result:
                if "start" in result:
                    events.append(("start", int(result["start"])))
                if "end" in result:
                    events.append(("end", int(result["end"])))
        return events

    def reset(self) -> None:
        self._it = self._make()
        self._pending = self._np.zeros(0, dtype=self._np.float32)


class EnergyVAD(VAD):
    """RMS gate with hangover. No dependencies; used in tests and as a fallback."""

    name = "energy"
    WINDOW = 512

    def __init__(self, threshold: float = 0.02, min_speech_ms: int = 64, min_silence_ms: int = 500, speech_pad_ms: int = 100):
        self.threshold = threshold
        self.min_speech_windows = max(1, min_speech_ms * 16 // self.WINDOW)
        self.min_silence_windows = max(1, min_silence_ms * 16 // self.WINDOW)
        self.pad = speech_pad_ms * 16
        self.reset()

    def reset(self) -> None:
        self._pending = bytearray()
        self._pos = 0
        self._speech_run = 0
        self._silence_run = 0
        self._in_speech = False
        self._speech_start: Optional[int] = None
        self._last_speech_end = 0

    def process(self, pcm16: bytes) -> List[Event]:
        import array

        self._pending += pcm16
        events: List[Event] = []
        nbytes = self.WINDOW * 2
        while len(self._pending) >= nbytes:
            win = array.array("h", bytes(self._pending[:nbytes]))
            del self._pending[:nbytes]
            rms = (sum(s * s for s in win) / len(win)) ** 0.5 / 32768.0
            loud = rms > self.threshold
            if loud:
                self._speech_run += 1
                self._silence_run = 0
                if not self._in_speech and self._speech_run >= self.min_speech_windows:
                    self._in_speech = True
                    start = self._pos - (self._speech_run - 1) * self.WINDOW
                    events.append(("start", max(0, start - self.pad)))
                self._last_speech_end = self._pos + self.WINDOW
            else:
                self._silence_run += 1
                self._speech_run = 0
                if self._in_speech and self._silence_run >= self.min_silence_windows:
                    self._in_speech = False
                    events.append(("end", self._last_speech_end + self.pad))
            self._pos += self.WINDOW
        return events


def build_vad(kind: Optional[str] = None) -> Optional[VAD]:
    kind = (kind or os.environ.get("COOK_VAD", "silero")).lower()
    if kind in ("none", "", "off"):
        return None
    if kind == "energy":
        return EnergyVAD()
    if kind == "silero":
        try:
            # 0.5 is Silero's general-purpose default and lets clatter and running water
            # through as speech. A kitchen wants fewer false positives than missed words.
            return SileroVAD(threshold=float(os.environ.get("COOK_VAD_THRESHOLD", "0.65")))
        except ImportError:
            return EnergyVAD()
    raise ValueError(f"unknown VAD '{kind}'")
