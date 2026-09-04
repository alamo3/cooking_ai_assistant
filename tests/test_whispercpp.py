from __future__ import annotations

import io
import wave

import httpx

from cooking_assistant_ai.speech.stt import WhisperCppSTT, pcm16_to_wav


async def test_whispercpp_adapter_posts_16k_wav_and_parses_text():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["content_type"] = request.headers["content-type"]
        seen["body"] = request.content
        return httpx.Response(200, json={"text": "  the rice is done \n"})

    stt = WhisperCppSTT(url="http://stub:1", transport=httpx.MockTransport(handler))
    pcm24k = b"\x00\x01" * 24000  # one second at 24 kHz
    text = await stt.transcribe(pcm24k, sample_rate=24000)
    assert text == "the rice is done"
    assert seen["url"] == "http://stub:1/inference"
    assert seen["content_type"].startswith("multipart/form-data")
    body = seen["body"]
    assert b'name="response_format"' in body and b"json" in body
    assert b'name="no_timestamps"' in body
    # the uploaded WAV is 16 kHz mono PCM16, one second long
    start = body.index(b"RIFF")
    with wave.open(io.BytesIO(body[start:start + 44 + 32000 + 64]), "rb") as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1 and w.getsampwidth() == 2
        assert abs(w.getnframes() - 16000) < 5
    await stt.aclose()


def test_pcm16_to_wav_header():
    wav = pcm16_to_wav(b"\x00\x00" * 160, 16000)
    with wave.open(io.BytesIO(wav), "rb") as w:
        assert w.getnframes() == 160 and w.getframerate() == 16000
