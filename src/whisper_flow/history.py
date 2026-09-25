"""Local history: every dictation's audio and text, kept in SQLite + audio files.

An entry is written *before* any network call, so audio is never lost even if
transcription fails; failed entries can be retried later.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from whisper_flow.providers.base import Audio

Status = Literal["pending", "done", "error"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS dictations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at  TEXT NOT NULL,
    status      TEXT NOT NULL,
    app         TEXT,
    style       TEXT,
    raw_text    TEXT,
    final_text  TEXT,
    snippet     TEXT,
    languages   TEXT,
    asr_model   TEXT,
    llm_model   TEXT,
    audio_file  TEXT,
    asr_ms      INTEGER,
    llm_ms      INTEGER,
    total_ms    INTEGER,
    error       TEXT
);
CREATE INDEX IF NOT EXISTS dictations_created_at ON dictations(created_at);
"""


class HistoryEntry(BaseModel):
    id: int
    created_at: datetime
    status: Status
    app: str | None = None
    style: str | None = None
    raw_text: str | None = None
    final_text: str | None = None
    snippet: str | None = None
    languages: str | None = None
    asr_model: str | None = None
    llm_model: str | None = None
    audio_file: str | None = None
    asr_ms: int | None = None
    llm_ms: int | None = None
    total_ms: int | None = None
    error: str | None = None


_UPDATABLE = set(HistoryEntry.model_fields) - {"id", "created_at", "audio_file"}


class History:
    def __init__(self, data_dir: Path, save_audio: bool = True) -> None:
        self.data_dir = data_dir
        self.audio_dir = data_dir / "audio"
        self.save_audio = save_audio
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = data_dir / "history.db"
        with closing(self._connect()) as db:
            db.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.db_path)
        db.row_factory = sqlite3.Row
        return db

    def create(self, audio: Audio, app: str | None) -> int:
        with closing(self._connect()) as db, db:
            cur = db.execute(
                "INSERT INTO dictations (created_at, status, app) VALUES (?, 'pending', ?)",
                (datetime.now().astimezone().isoformat(), app),
            )
            entry_id = cur.lastrowid
            if self.save_audio:
                name = f"{entry_id}.{audio.format}"
                (self.audio_dir / name).write_bytes(audio.data)
                db.execute("UPDATE dictations SET audio_file = ? WHERE id = ?", (name, entry_id))
        return entry_id

    def update(self, entry_id: int, **fields: object) -> None:
        unknown = fields.keys() - _UPDATABLE
        if unknown:
            raise ValueError(f"not updatable: {', '.join(sorted(unknown))}")
        if not fields:
            return
        assignments = ", ".join(f"{k} = ?" for k in fields)
        with closing(self._connect()) as db, db:
            db.execute(
                f"UPDATE dictations SET {assignments} WHERE id = ?", (*fields.values(), entry_id)
            )

    def get(self, entry_id: int) -> HistoryEntry | None:
        with closing(self._connect()) as db:
            row = db.execute("SELECT * FROM dictations WHERE id = ?", (entry_id,)).fetchone()
        return HistoryEntry(**row) if row else None

    def list(
        self, limit: int = 50, offset: int = 0, query: str | None = None
    ) -> list[HistoryEntry]:
        sql = "SELECT * FROM dictations"
        args: list[object] = []
        if query:
            sql += " WHERE final_text LIKE ? OR raw_text LIKE ?"
            args += [f"%{query}%"] * 2
        sql += " ORDER BY id DESC LIMIT ? OFFSET ?"
        args += [limit, offset]
        with closing(self._connect()) as db:
            rows = db.execute(sql, args).fetchall()
        return [HistoryEntry(**row) for row in rows]

    def audio_path(self, entry_id: int) -> Path | None:
        entry = self.get(entry_id)
        if entry is None or entry.audio_file is None:
            return None
        path = self.audio_dir / entry.audio_file
        return path if path.is_file() else None

    def load_audio(self, entry_id: int) -> Audio | None:
        path = self.audio_path(entry_id)
        if path is None:
            return None
        return Audio(data=path.read_bytes(), format=path.suffix.lstrip("."))
