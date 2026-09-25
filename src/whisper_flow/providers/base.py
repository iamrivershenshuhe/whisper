"""Provider-neutral interfaces. Vendors implement these; the pipeline only sees these."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

AUDIO_MIME = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "mpga": "audio/mpeg",
    "mpeg": "audio/mpeg",
    "m4a": "audio/mp4",
    "mp4": "audio/mp4",
    "webm": "audio/webm",
    "ogg": "audio/ogg",
    "flac": "audio/flac",
}


@dataclass(frozen=True)
class Audio:
    data: bytes
    # File extension without the dot, e.g. "wav" or "webm".
    format: str = "wav"

    def __post_init__(self) -> None:
        # The format ends up in a file name on disk, so keep it to a bare extension.
        if self.format not in AUDIO_MIME:
            raise ValueError(f"unsupported audio format: {self.format!r}")

    @property
    def filename(self) -> str:
        return f"audio.{self.format}"

    @property
    def mime_type(self) -> str:
        return AUDIO_MIME.get(self.format, "application/octet-stream")


@dataclass(frozen=True)
class TranscriptionHints:
    # Free-text context that biases recognition (script, domain, spelling).
    prompt: str = ""
    # Terms that must be recognized exactly (the user's dictionary).
    keywords: tuple[str, ...] = ()
    languages: tuple[str, ...] = ()


@dataclass(frozen=True)
class Transcript:
    text: str
    languages: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class Message:
    role: Literal["system", "user", "assistant"]
    content: str


class ProviderError(RuntimeError):
    """A vendor call failed. Message is safe to show to the user."""


class Transcriber(Protocol):
    model: str

    async def transcribe(self, audio: Audio, hints: TranscriptionHints) -> Transcript: ...


class ChatModel(Protocol):
    model: str

    async def complete(self, messages: list[Message]) -> str: ...
