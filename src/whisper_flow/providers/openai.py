"""OpenAI (and OpenAI-compatible) provider.

Chat uses the Chat Completions endpoint rather than Responses so the same code
works against OpenAI-compatible vendors via `base_url`.
"""

from __future__ import annotations

import os

import openai
from openai import AsyncOpenAI

from whisper_flow.config import ASRConfig, LLMConfig, ProviderConfig
from whisper_flow.providers.base import (
    Audio,
    Message,
    ProviderError,
    Transcript,
    TranscriptionHints,
)


def make_client(name: str, cfg: ProviderConfig, **kwargs) -> AsyncOpenAI:
    api_key = os.environ.get(cfg.api_key_env)
    if not api_key:
        raise ProviderError(f"provider '{name}': environment variable {cfg.api_key_env} is not set")
    return AsyncOpenAI(
        api_key=api_key,
        base_url=cfg.base_url,
        timeout=cfg.timeout,
        max_retries=cfg.max_retries,
        **kwargs,
    )


class OpenAITranscriber:
    def __init__(self, client: AsyncOpenAI, cfg: ASRConfig) -> None:
        self._client = client
        self._cfg = cfg
        self.model = cfg.model

    async def transcribe(self, audio: Audio, hints: TranscriptionHints) -> Transcript:
        params: dict = {
            "model": self.model,
            "file": (audio.filename, audio.data, audio.mime_type),
        }
        if hints.prompt:
            params["prompt"] = hints.prompt
        if hints.languages:
            params["languages"] = list(hints.languages)
        if hints.keywords and self._cfg.send_keywords:
            params["keywords"] = list(hints.keywords)
        try:
            result = await self._client.audio.transcriptions.create(**params)
        except openai.APIError as e:
            raise ProviderError(f"transcription failed ({self.model}): {e}") from e
        langs = tuple(
            code
            for item in (getattr(result, "languages", None) or [])
            if (code := _lang_code(item))
        )
        return Transcript(text=result.text.strip(), languages=langs)


class OpenAIChat:
    def __init__(self, client: AsyncOpenAI, cfg: LLMConfig) -> None:
        self._client = client
        self._cfg = cfg
        self.model = cfg.model

    async def complete(self, messages: list[Message]) -> str:
        params: dict = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
        }
        if self._cfg.reasoning_effort is not None:
            params["reasoning_effort"] = self._cfg.reasoning_effort
        if self._cfg.temperature is not None:
            params["temperature"] = self._cfg.temperature
        try:
            response = await self._client.chat.completions.create(**params)
        except openai.APIError as e:
            raise ProviderError(f"chat completion failed ({self.model}): {e}") from e
        return (response.choices[0].message.content or "").strip()


def _lang_code(item: object) -> str | None:
    if isinstance(item, dict):
        return item.get("code")
    return getattr(item, "code", None)
