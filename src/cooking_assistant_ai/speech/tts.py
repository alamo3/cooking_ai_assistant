"""Text-to-speech (Phase 5). Output is PCM16 mono bytes at `sample_rate`.

NullTTS returns None so the websocket sends text only. KokoroTTS is an optional
adapter selected via COOK_TTS=kokoro; it has not been exercised in this repo.
"""
from __future__ import annotations

import asyncio
import os
from typing import Optional


class TTS:
    sample_rate: int = 24000

    async def synthesize(self, text: str) -> Optional[bytes]:  # pragma: no cover
        raise NotImplementedError

    async def warm(self) -> None:
        """First synthesis pays JIT/model-load cost; do it before the cook is waiting."""
        if self.available:
            await self.synthesize("Ready when you are.")

    @property
    def available(self) -> bool:
        return True


class NullTTS(TTS):
    @property
    def available(self) -> bool:
        return False

    async def synthesize(self, text: str) -> Optional[bytes]:
        return None


class KokoroTTS(TTS):
    """Kokoro-82M via the `kokoro` package. Runs on CPU in well under real time."""

    def __init__(self, voice: str = "af_heart", lang_code: str = "a", speed: float = 1.0):
        from kokoro import KPipeline  # type: ignore

        self._pipe = KPipeline(lang_code=lang_code, repo_id="hexgrad/Kokoro-82M")
        self.voice = voice
        self.speed = speed
        self.sample_rate = 24000

    async def synthesize(self, text: str) -> Optional[bytes]:
        import numpy as np  # type: ignore

        def _run() -> bytes:
            parts = []
            for result in self._pipe(text, voice=self.voice, speed=self.speed):
                audio = result.audio if hasattr(result, "audio") else result[2]
                if audio is None:
                    continue
                if hasattr(audio, "detach"):
                    audio = audio.detach().cpu().numpy()
                parts.append(np.asarray(audio, dtype=np.float32))
            if not parts:
                return b""
            pcm = np.concatenate(parts)
            return (np.clip(pcm, -1, 1) * 32767).astype(np.int16).tobytes()

        return await asyncio.get_event_loop().run_in_executor(None, _run)


def build_tts(kind: Optional[str] = None) -> TTS:
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    kind = (kind or os.environ.get("COOK_TTS", "none")).lower()
    if kind in ("none", "", "null"):
        return NullTTS()
    if kind == "kokoro":
        return KokoroTTS(voice=os.environ.get("COOK_TTS_VOICE", "af_heart"))
    raise ValueError(f"unknown TTS backend '{kind}'")
