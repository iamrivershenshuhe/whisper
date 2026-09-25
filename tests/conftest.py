from __future__ import annotations

import pytest

from whisper_flow.audio import encode_wav
from whisper_flow.config import Config, Snippet
from whisper_flow.history import History
from whisper_flow.pipeline import Dictation
from whisper_flow.providers import Audio, Message, ProviderError, Transcript, TranscriptionHints


class FakeTranscriber:
    model = "fake-asr"

    def __init__(self, text: str = "嗯 明天下午五點 不對 六點開會", fail: bool = False) -> None:
        self.text = text
        self.fail = fail
        self.calls: list[TranscriptionHints] = []

    async def transcribe(self, audio: Audio, hints: TranscriptionHints) -> Transcript:
        self.calls.append(hints)
        if self.fail:
            raise ProviderError("asr down")
        return Transcript(self.text, ("zh",))


class FakeChat:
    model = "fake-llm"

    def __init__(self, reply: str = "明天下午六點開會。", fail: bool = False) -> None:
        self.reply = reply
        self.fail = fail
        self.calls: list[list[Message]] = []

    async def complete(self, messages: list[Message]) -> str:
        self.calls.append(messages)
        if self.fail:
            raise ProviderError("llm down")
        return self.reply


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def audio() -> Audio:
    return Audio(encode_wav(b"\x00\x00" * 1600), "wav")


@pytest.fixture
def config(tmp_path) -> Config:
    return Config.model_validate(
        {
            "dictionary": ["Claude", "uv"],
            "snippets": [Snippet(triggers=["電子郵件", "my email"], text="me@example.com")],
            "history": {"data_dir": tmp_path},
        }
    )


@pytest.fixture
def history(config) -> History:
    return History(config.history.resolved_dir())


@pytest.fixture
def transcriber() -> FakeTranscriber:
    return FakeTranscriber()


@pytest.fixture
def chat() -> FakeChat:
    return FakeChat()


@pytest.fixture
def dictation(config, transcriber, chat, history) -> Dictation:
    return Dictation(config, transcriber, chat, history)
