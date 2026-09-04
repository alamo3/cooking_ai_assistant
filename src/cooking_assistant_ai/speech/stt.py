"""Speech-to-text (Phase 4). Input is PCM16 mono bytes at any sample rate; adapters
resample to 16 kHz.

Backends (COOK_STT):
* ``none``            NullSTT, the default.
* ``whisper.cpp``     WhisperCppSTT: drives the user's whisper.cpp build through its
                      ``whisper-server`` (model stays loaded, one HTTP call per utterance).
* ``faster-whisper``  FasterWhisperSTT: CTranslate2 on CPU.
"""
from __future__ import annotations

import asyncio
import io
import logging
import os
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path
from typing import Any, List, Optional, Sequence

log = logging.getLogger(__name__)


def _resample_pcm16(pcm16: bytes, src_rate: int, dst_rate: int = 16000) -> bytes:
    if src_rate == dst_rate:
        return pcm16
    try:
        import numpy as np  # type: ignore

        audio = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32)
        n = int(len(audio) * dst_rate / src_rate)
        out = np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio)
        return out.astype(np.int16).tobytes()
    except ImportError:  # pragma: no cover - numpy is in the speech extra
        import audioop  # type: ignore

        return audioop.ratecv(pcm16, 2, 1, src_rate, dst_rate, None)[0]


def pcm16_to_wav(pcm16: bytes, sample_rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm16)
    return buf.getvalue()


class STT:
    async def transcribe(self, pcm16: bytes, sample_rate: int = 16000) -> str:  # pragma: no cover
        raise NotImplementedError

    async def warm(self) -> None:
        if self.available:
            await self.transcribe(b"\x00\x00" * 16000)  # one second of silence

    async def aclose(self) -> None:
        return None

    @property
    def available(self) -> bool:
        return True


class NullSTT(STT):
    @property
    def available(self) -> bool:
        return False

    async def transcribe(self, pcm16: bytes, sample_rate: int = 16000) -> str:
        raise RuntimeError("no STT backend configured (set COOK_STT=faster-whisper)")


class WhisperCppSTT(STT):
    """Talks to whisper.cpp's ``whisper-server``. Spawns it if no ``url`` is given.

    Latency-oriented defaults: greedy decoding (beam 1, best-of 1), no temperature fallback,
    no timestamps. The server keeps the model resident, so each utterance costs one encode
    plus a few decode steps.
    """

    def __init__(self, url: Optional[str] = None, binary: Optional[str] = None, model: Optional[str] = None,
                 host: str = "127.0.0.1", port: int = 8178, threads: int = 8, prompt: Optional[str] = None,
                 extra_args: Sequence[str] = (), start_timeout: float = 90.0, transport: Any = None):
        import httpx

        self.external = url is not None
        self.url = (url or f"http://{host}:{port}").rstrip("/")
        self.binary = binary
        self.model = model
        self.host, self.port, self.threads = host, port, threads
        self.prompt = prompt
        self.extra_args = list(extra_args)
        self.start_timeout = start_timeout
        self._proc: Optional[subprocess.Popen] = None
        self._client = httpx.AsyncClient(timeout=60, transport=transport)
        self._ready = self.external or transport is not None

    # -- discovery ----------------------------------------------------------

    @staticmethod
    def find_binary(root: Optional[Path] = None) -> Optional[Path]:
        roots = [root] if root else [Path.cwd(), Path(__file__).resolve().parents[3]]
        exe = "whisper-server.exe" if sys.platform == "win32" else "whisper-server"
        for r in roots:
            for rel in ("whisper.cpp/build-vulkan/bin/Release", "whisper.cpp/build-vulkan/bin",
                        "whisper.cpp/build/bin/Release", "whisper.cpp/build/bin"):
                p = r / rel / exe
                if p.exists():
                    return p
        return None

    @staticmethod
    def find_model(root: Optional[Path] = None, prefer: Sequence[str] = ("ggml-small.en.bin", "ggml-base.en.bin")) -> Optional[Path]:
        roots = [root] if root else [Path.cwd(), Path(__file__).resolve().parents[3]]
        for r in roots:
            for name in prefer:
                for rel in ("whisper.cpp", "whisper.cpp/models"):
                    p = r / rel / name
                    if p.exists():
                        return p
        return None

    # -- process ------------------------------------------------------------

    async def start(self) -> None:
        if self._ready:
            return
        binary = Path(self.binary).resolve() if self.binary else self.find_binary()
        if binary is None or not binary.exists():
            raise RuntimeError("whisper-server binary not found; build whisper.cpp or set COOK_WHISPER_CPP_BIN")
        if self.model:
            model: Optional[Path] = Path(self.model).resolve()
        else:
            # base.en: 39 ms on Vulkan / 0.33 s on CPU for a 4 s utterance; small.en is
            # 0.59 s / 1.25 s. Set COOK_WHISPER_CPP_MODEL to trade latency for accuracy.
            model = self.find_model(prefer=("ggml-base.en.bin", "ggml-small.en.bin"))
        if model is None or not model.exists():
            raise RuntimeError("no ggml whisper model found; set COOK_WHISPER_CPP_MODEL")
        cmd: List[str] = [
            str(binary), "-m", str(model), "--host", self.host, "--port", str(self.port),
            "-t", str(self.threads), "-nt", "-bs", "1", "-bo", "1", "-nf", "-l", "en",
            *self.extra_args,
        ]
        log.info("starting whisper-server: %s", " ".join(cmd))
        self._log = tempfile.NamedTemporaryFile(prefix="whisper-server-", suffix=".log", delete=False)
        self._proc = subprocess.Popen(cmd, cwd=str(binary.parent), stdout=self._log, stderr=subprocess.STDOUT)
        deadline = time.time() + self.start_timeout
        while time.time() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError(f"whisper-server exited with code {self._proc.returncode}: {self._log_tail()}")
            try:
                r = await self._client.get(self.url + "/", timeout=2)
                if r.status_code < 500:
                    self._ready = True
                    return
            except Exception:
                pass
            await asyncio.sleep(0.2)
        raise RuntimeError("whisper-server did not become ready in time")

    def _log_tail(self, n: int = 6) -> str:
        try:
            self._log.flush()
            with open(self._log.name, "r", errors="replace") as f:
                lines = [ln.strip() for ln in f.readlines() if ln.strip()]
            return " | ".join(lines[-n:])
        except Exception:
            return "(no log)"

    async def aclose(self) -> None:
        await self._client.aclose()
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None

    # -- inference ----------------------------------------------------------

    async def transcribe(self, pcm16: bytes, sample_rate: int = 16000) -> str:
        if not self._ready:
            await self.start()
        wav = pcm16_to_wav(_resample_pcm16(pcm16, sample_rate), 16000)
        data = {"response_format": "json", "temperature": "0.0", "temperature_inc": "0.0",
                "no_timestamps": "true", "language": "en"}
        if self.prompt:
            data["prompt"] = self.prompt
        r = await self._client.post(self.url + "/inference", files={"file": ("audio.wav", wav, "audio/wav")}, data=data)
        r.raise_for_status()
        body = r.json()
        if "error" in body:
            raise RuntimeError(f"whisper-server: {body['error']}")
        return str(body.get("text", "")).strip()

    async def warm(self) -> None:
        await self.start()
        await super().warm()


class FasterWhisperSTT(STT):
    """faster-whisper (CTranslate2). CPU int8 by default: CTranslate2 has no AMD GPU path
    on Windows, and whisper-small int8 transcribes a short utterance in well under a second."""

    def __init__(self, model_size: str = "small", device: str = "cpu", compute_type: str = "int8"):
        from faster_whisper import WhisperModel  # type: ignore

        self._model = WhisperModel(model_size, device=device, compute_type=compute_type)

    @staticmethod
    def _to_16k(audio, sample_rate: int):
        import numpy as np  # type: ignore

        if sample_rate == 16000:
            return audio
        n = int(len(audio) * 16000 / sample_rate)
        return np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio).astype(np.float32)

    async def transcribe(self, pcm16: bytes, sample_rate: int = 16000) -> str:
        import numpy as np  # type: ignore

        audio = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
        audio = self._to_16k(audio, sample_rate)

        def _run() -> str:
            segments, _info = self._model.transcribe(audio, language="en", vad_filter=True, beam_size=1)
            return " ".join(seg.text.strip() for seg in segments).strip()

        return await asyncio.get_event_loop().run_in_executor(None, _run)


def build_stt(kind: Optional[str] = None) -> STT:
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    kind = (kind or os.environ.get("COOK_STT", "none")).lower()
    if kind in ("none", "", "null"):
        return NullSTT()
    if kind in ("whisper.cpp", "whispercpp", "whisper-cpp"):
        return WhisperCppSTT(
            url=os.environ.get("COOK_WHISPER_CPP_URL") or None,
            binary=os.environ.get("COOK_WHISPER_CPP_BIN") or None,
            model=os.environ.get("COOK_WHISPER_CPP_MODEL") or None,
            port=int(os.environ.get("COOK_WHISPER_CPP_PORT", "8178")),
            threads=int(os.environ.get("COOK_WHISPER_CPP_THREADS", "8")),
            prompt=os.environ.get("COOK_WHISPER_PROMPT") or None,
        )
    if kind in ("faster-whisper", "whisper"):
        return FasterWhisperSTT(
            model_size=os.environ.get("COOK_WHISPER_MODEL", "small"),
            device=os.environ.get("COOK_WHISPER_DEVICE", "cpu"),
            compute_type=os.environ.get("COOK_WHISPER_COMPUTE", "int8"),
        )
    raise ValueError(f"unknown STT backend '{kind}'")
