"""Local HTTP API for desktop clients. Binds to localhost only by default; no auth."""

from __future__ import annotations

from pathlib import PurePath
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from whisper_flow import __version__
from whisper_flow.config import Config
from whisper_flow.history import History, HistoryEntry
from whisper_flow.pipeline import Dictation, DictationResult, EntryNotFound
from whisper_flow.providers import Audio, ProviderError
from whisper_flow.providers.base import AUDIO_MIME

_MIME_TO_FORMAT = {mime: fmt for fmt, mime in reversed(AUDIO_MIME.items())}


def create_app(config: Config | None = None, dictation: Dictation | None = None) -> FastAPI:
    if dictation is None:
        dictation = Dictation.from_config(config or Config())
    app = FastAPI(title="whisper-flow", version=__version__)

    def history() -> History:
        if dictation.history is None:
            raise HTTPException(404, "history is disabled")
        return dictation.history

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "asr_model": dictation.transcriber.model,
            "llm_model": dictation.chat.model if dictation.chat else None,
        }

    @app.post("/v1/dictate")
    async def dictate(
        audio: Annotated[UploadFile, File(description="Recorded speech")],
        app_name: Annotated[
            str | None,
            Form(alias="app", description="Foreground app: process name, window title, or URL"),
        ] = None,
    ) -> DictationResult:
        data = await audio.read()
        if not data:
            raise HTTPException(400, "empty audio")
        fmt = _audio_format(audio)
        try:
            return await dictation.dictate(Audio(data, fmt), app_name)
        except ProviderError as e:
            raise HTTPException(502, str(e)) from e

    @app.get("/v1/history")
    def list_history(limit: int = 50, offset: int = 0, q: str | None = None) -> list[HistoryEntry]:
        return history().list(limit=min(limit, 500), offset=offset, query=q)

    @app.get("/v1/history/{entry_id}")
    def get_history(entry_id: int) -> HistoryEntry:
        entry = history().get(entry_id)
        if entry is None:
            raise HTTPException(404, "not found")
        return entry

    @app.get("/v1/history/{entry_id}/audio")
    def get_audio(entry_id: int) -> FileResponse:
        path = history().audio_path(entry_id)
        if path is None:
            raise HTTPException(404, "no audio stored for this entry")
        return FileResponse(path, media_type=AUDIO_MIME[path.suffix.lstrip(".")])

    @app.post("/v1/history/{entry_id}/retry")
    async def retry(entry_id: int) -> DictationResult:
        try:
            return await dictation.retry(entry_id)
        except EntryNotFound as e:
            raise HTTPException(404, str(e)) from e
        except ProviderError as e:
            raise HTTPException(502, str(e)) from e

    return app


def _audio_format(upload: UploadFile) -> str:
    if upload.filename:
        ext = PurePath(upload.filename).suffix.lstrip(".").lower()
        if ext in AUDIO_MIME:
            return ext
    mime = (upload.content_type or "").split(";")[0].strip()
    if mime in _MIME_TO_FORMAT:
        return _MIME_TO_FORMAT[mime]
    supported = ", ".join(AUDIO_MIME)
    raise HTTPException(415, f"unsupported audio type; use one of: {supported}")
