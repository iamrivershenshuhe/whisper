"""Configuration loading.

Config is a TOML file. Lookup order: explicit path → $WHISPER_FLOW_CONFIG →
./config.toml → <user config dir>/config.toml → built-in defaults.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Literal

from platformdirs import user_config_path, user_data_path
from pydantic import BaseModel, Field, field_validator, model_validator

APP_NAME = "whisper-flow"


class ProviderConfig(BaseModel):
    """One configured vendor endpoint. `type` selects the implementation.

    Any OpenAI-compatible endpoint (Groq, OpenRouter, Ollama, ...) can reuse
    `type = "openai"` with a different `base_url` and `api_key_env`.
    """

    type: str = "openai"
    api_key_env: str = "OPENAI_API_KEY"
    base_url: str | None = None
    timeout: float = 30.0
    max_retries: int = 2


class ASRConfig(BaseModel):
    provider: str = "openai"
    model: str = "gpt-transcribe"
    # Language hints for the recognizer, e.g. ["zh", "en"] for mixed speech.
    languages: list[str] = Field(default_factory=lambda: ["zh", "en"])
    # Send the dictionary as the `keywords` parameter (gpt-transcribe only).
    send_keywords: bool = True


class LLMConfig(BaseModel):
    provider: str = "openai"
    model: str = "gpt-6-luna"
    # "" or None means "don't send the parameter" (for providers that reject it).
    reasoning_effort: str | None = "none"
    temperature: float | None = None

    @field_validator("reasoning_effort")
    @classmethod
    def _empty_is_unset(cls, v: str | None) -> str | None:
        return v or None


class CleanupConfig(BaseModel):
    enabled: bool = True
    # Output locale; drives script (繁/简) and punctuation rules.
    locale: str = "zh-TW"
    # Extra rules appended to the system prompt.
    instructions: str = ""


class Style(BaseModel):
    description: str


class AppRule(BaseModel):
    # Case-insensitive substrings matched against the app identifier the
    # client sends (process name, window title, or URL).
    match: list[str]
    style: str


class Snippet(BaseModel):
    triggers: list[str]
    text: str


class HistoryConfig(BaseModel):
    enabled: bool = True
    save_audio: bool = True
    data_dir: Path | None = None

    def resolved_dir(self) -> Path:
        return self.data_dir or user_data_path(APP_NAME, appauthor=False)


class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8765


DEFAULT_STYLES: dict[str, Style] = {
    "default": Style(
        description="Clear written text with natural punctuation. Keep the speaker's tone."
    ),
    "casual": Style(
        description=(
            "A chat message. Keep it conversational and short, as the speaker said it. "
            "No greetings or sign-offs that weren't spoken. A trailing period is optional."
        )
    ),
    "formal": Style(
        description=(
            "Professional writing for documents or email. Complete sentences, "
            "well-formed paragraphs, and bullet or numbered lists when the speaker "
            "enumerates items. Do not add content that was not spoken."
        )
    ),
}

DEFAULT_APP_RULES: list[AppRule] = [
    AppRule(
        match=["line", "messenger", "discord", "telegram", "whatsapp", "slack"],
        style="casual",
    ),
    AppRule(
        match=["notion", "docs.google.com", "winword", "gmail", "outlook", "mail.google.com"],
        style="formal",
    ),
]


class Config(BaseModel):
    providers: dict[str, ProviderConfig] = Field(default_factory=dict)
    asr: ASRConfig = Field(default_factory=ASRConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    cleanup: CleanupConfig = Field(default_factory=CleanupConfig)
    dictionary: list[str] = Field(default_factory=list)
    snippets: list[Snippet] = Field(default_factory=list)
    styles: dict[str, Style] = Field(default_factory=dict)
    apps: list[AppRule] = Field(default_factory=lambda: list(DEFAULT_APP_RULES))
    default_style: str = "default"
    history: HistoryConfig = Field(default_factory=HistoryConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)

    @model_validator(mode="after")
    def _merge_and_check(self) -> Config:
        # User providers and styles extend/override the built-ins rather than replacing them.
        self.providers = {"openai": ProviderConfig(), **self.providers}
        self.styles = {**DEFAULT_STYLES, **self.styles}
        for role, name in (("asr", self.asr.provider), ("llm", self.llm.provider)):
            if name not in self.providers:
                raise ValueError(f"{role}.provider '{name}' is not defined under [providers]")
        style_names = {self.default_style, *(r.style for r in self.apps)}
        missing = style_names - self.styles.keys()
        if missing:
            raise ValueError(f"unknown style(s): {', '.join(sorted(missing))}")
        return self


ConfigSource = Literal["explicit", "env", "cwd", "user", "defaults"]


def find_config_file(path: Path | None = None) -> tuple[Path | None, ConfigSource]:
    if path is not None:
        return path, "explicit"
    if env := os.environ.get("WHISPER_FLOW_CONFIG"):
        return Path(env), "env"
    cwd = Path.cwd() / "config.toml"
    if cwd.is_file():
        return cwd, "cwd"
    user = user_config_path(APP_NAME, appauthor=False) / "config.toml"
    if user.is_file():
        return user, "user"
    return None, "defaults"


def load_config(path: Path | None = None) -> Config:
    file, _ = find_config_file(path)
    if file is None:
        return Config()
    with file.open("rb") as f:
        return Config.model_validate(tomllib.load(f))
