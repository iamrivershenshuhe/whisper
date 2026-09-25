"""The dictation pipeline: audio → transcript → snippet or LLM cleanup → text."""

from __future__ import annotations

import logging
import time

from pydantic import BaseModel

from whisper_flow.config import Config
from whisper_flow.history import History
from whisper_flow.providers import (
    Audio,
    ChatModel,
    ProviderError,
    Transcriber,
    build_chat,
    build_transcriber,
)
from whisper_flow.text import (
    cleanup_messages,
    match_snippet,
    resolve_style,
    transcription_hints,
)

log = logging.getLogger(__name__)


class Timings(BaseModel):
    asr_ms: int
    llm_ms: int
    total_ms: int


class DictationResult(BaseModel):
    id: int | None
    text: str
    raw_text: str
    style: str
    snippet: str | None = None
    languages: list[str] = []
    asr_model: str
    # Set only when the LLM actually rewrote the text.
    llm_model: str | None = None
    timings: Timings
    # Non-fatal problems, e.g. cleanup failed and the raw transcript was used.
    warnings: list[str] = []


class EntryNotFound(LookupError):
    pass


class Dictation:
    def __init__(
        self,
        config: Config,
        transcriber: Transcriber,
        chat: ChatModel | None,
        history: History | None,
    ) -> None:
        self.config = config
        self.transcriber = transcriber
        self.chat = chat
        self.history = history

    @classmethod
    def from_config(cls, config: Config) -> Dictation:
        history = None
        if config.history.enabled:
            history = History(config.history.resolved_dir(), save_audio=config.history.save_audio)
        chat = build_chat(config) if config.cleanup.enabled else None
        return cls(config, build_transcriber(config), chat, history)

    async def dictate(self, audio: Audio, app: str | None = None) -> DictationResult:
        entry_id = self.history.create(audio, app) if self.history else None
        return await self._run(entry_id, audio, app)

    async def retry(self, entry_id: int) -> DictationResult:
        if self.history is None:
            raise EntryNotFound("history is disabled")
        entry = self.history.get(entry_id)
        audio = self.history.load_audio(entry_id)
        if entry is None or audio is None:
            raise EntryNotFound(f"no stored audio for entry {entry_id}")
        self.history.update(entry_id, status="pending", error=None)
        return await self._run(entry_id, audio, entry.app)

    async def _run(self, entry_id: int | None, audio: Audio, app: str | None) -> DictationResult:
        try:
            result = await self._process(audio, app)
        except Exception as e:
            if self.history and entry_id is not None:
                self.history.update(entry_id, status="error", error=str(e))
            raise
        result.id = entry_id
        if self.history and entry_id is not None:
            self.history.update(
                entry_id,
                status="done",
                style=result.style,
                raw_text=result.raw_text,
                final_text=result.text,
                snippet=result.snippet,
                languages=",".join(result.languages) or None,
                asr_model=result.asr_model,
                llm_model=result.llm_model,
                asr_ms=result.timings.asr_ms,
                llm_ms=result.timings.llm_ms,
                total_ms=result.timings.total_ms,
                error="; ".join(result.warnings) or None,
            )
        return result

    async def _process(self, audio: Audio, app: str | None) -> DictationResult:
        start = time.perf_counter()
        transcript = await self.transcriber.transcribe(audio, transcription_hints(self.config))
        asr_done = time.perf_counter()

        style = resolve_style(self.config, app)
        raw = transcript.text
        text = raw
        snippet = match_snippet(raw, self.config.snippets)
        warnings: list[str] = []
        llm_model = None

        if snippet is not None:
            text = snippet.text
        elif raw and self.chat is not None:
            messages = cleanup_messages(self.config, raw, style, app)
            try:
                text = await self.chat.complete(messages)
                llm_model = self.chat.model
            except ProviderError as e:
                log.warning("cleanup failed, falling back to raw transcript: %s", e)
                warnings.append(str(e))
        llm_done = time.perf_counter()

        return DictationResult(
            id=None,
            text=text,
            raw_text=raw,
            style=style.name,
            snippet=snippet.triggers[0] if snippet else None,
            languages=list(transcript.languages),
            asr_model=self.transcriber.model,
            llm_model=llm_model,
            timings=Timings(
                asr_ms=_ms(asr_done - start),
                llm_ms=_ms(llm_done - asr_done),
                total_ms=_ms(llm_done - start),
            ),
            warnings=warnings,
        )


def _ms(seconds: float) -> int:
    return round(seconds * 1000)
