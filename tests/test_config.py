from pathlib import Path

import pytest
from pydantic import ValidationError

from whisper_flow.config import Config, load_config


def test_defaults_are_valid():
    config = Config()
    assert config.asr.provider == "openai"
    assert config.providers["openai"].api_key_env == "OPENAI_API_KEY"


def test_load_toml_with_extra_provider(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(
        """
dictionary = ["Wispr Flow"]

[providers.groq]
type = "openai"
base_url = "https://api.groq.com/openai/v1"
api_key_env = "GROQ_API_KEY"

[asr]
provider = "groq"
model = "whisper-large-v3"
send_keywords = false

[[snippets]]
triggers = ["電子郵件"]
text = "me@example.com"
""",
        encoding="utf-8",
    )
    config = load_config(path)
    assert config.asr.provider == "groq"
    assert config.providers["groq"].base_url == "https://api.groq.com/openai/v1"
    assert config.llm.provider == "openai"  # default provider still present
    assert config.snippets[0].text == "me@example.com"


def test_unknown_provider_rejected():
    with pytest.raises(ValidationError, match="not defined"):
        Config.model_validate({"llm": {"provider": "nope"}})


def test_unknown_style_rejected():
    with pytest.raises(ValidationError, match="unknown style"):
        Config.model_validate({"apps": [{"match": ["x"], "style": "nope"}]})


def test_example_config_is_valid():
    config = load_config(Path(__file__).parent.parent / "config.example.toml")
    assert config.snippets
    assert config.styles["code"]


def test_empty_reasoning_effort_means_unset():
    assert Config.model_validate({"llm": {"reasoning_effort": ""}}).llm.reasoning_effort is None
