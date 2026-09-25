"""Provider registry: maps a provider `type` to the code that builds its clients.

To add a vendor, implement `Transcriber` and/or `ChatModel` (see base.py) in a
new module and register its factories in `_TYPES`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from whisper_flow.config import ASRConfig, Config, LLMConfig, ProviderConfig
from whisper_flow.providers import openai as openai_provider
from whisper_flow.providers.base import (
    Audio,
    ChatModel,
    Message,
    ProviderError,
    Transcriber,
    Transcript,
    TranscriptionHints,
)

__all__ = [
    "Audio",
    "ChatModel",
    "Message",
    "ProviderError",
    "Transcriber",
    "Transcript",
    "TranscriptionHints",
    "build_chat",
    "build_transcriber",
]


@dataclass(frozen=True)
class _ProviderType:
    transcriber: Callable[[str, ProviderConfig, ASRConfig], Transcriber] | None
    chat: Callable[[str, ProviderConfig, LLMConfig], ChatModel] | None


_TYPES: dict[str, _ProviderType] = {
    "openai": _ProviderType(
        transcriber=lambda name, p, asr: openai_provider.OpenAITranscriber(
            openai_provider.make_client(name, p), asr
        ),
        chat=lambda name, p, llm: openai_provider.OpenAIChat(
            openai_provider.make_client(name, p), llm
        ),
    ),
}


def _lookup(config: Config, name: str) -> tuple[ProviderConfig, _ProviderType]:
    provider = config.providers[name]
    impl = _TYPES.get(provider.type)
    if impl is None:
        known = ", ".join(sorted(_TYPES))
        raise ProviderError(
            f"provider '{name}' has unknown type '{provider.type}' (known: {known})"
        )
    return provider, impl


def build_transcriber(config: Config) -> Transcriber:
    name = config.asr.provider
    provider, impl = _lookup(config, name)
    if impl.transcriber is None:
        raise ProviderError(f"provider type '{provider.type}' does not support speech-to-text")
    return impl.transcriber(name, provider, config.asr)


def build_chat(config: Config) -> ChatModel:
    name = config.llm.provider
    provider, impl = _lookup(config, name)
    if impl.chat is None:
        raise ProviderError(f"provider type '{provider.type}' does not support chat")
    return impl.chat(name, provider, config.llm)
