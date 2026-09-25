"""Microphone capture for the CLI. Needs the optional `mic` extra."""

from __future__ import annotations

import io
import wave

from whisper_flow.providers.base import Audio

SAMPLE_RATE = 16_000


def encode_wav(pcm: bytes, sample_rate: int = SAMPLE_RATE, channels: int = 1) -> bytes:
    """Wrap 16-bit little-endian PCM in a WAV container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


def record_until_enter(prompt: str = "錄音中，按 Enter 結束…") -> Audio:
    try:
        import numpy as np
        import sounddevice as sd
    except ImportError as e:
        raise SystemExit("microphone support is not installed: run `uv sync --extra mic`") from e

    chunks: list = []

    def on_audio(indata, frames, time_info, status) -> None:
        chunks.append(indata.copy())

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16", callback=on_audio):
        input(prompt)
    if not chunks:
        raise SystemExit("no audio captured")
    pcm = np.concatenate(chunks).tobytes()
    return Audio(encode_wav(pcm), "wav")
