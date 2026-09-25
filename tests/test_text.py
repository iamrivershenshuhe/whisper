from whisper_flow.config import Config, Snippet
from whisper_flow.text import (
    cleanup_messages,
    match_snippet,
    normalize_utterance,
    resolve_style,
    transcription_hints,
)

SNIPPETS = [Snippet(triggers=["電子郵件", "My Email"], text="me@example.com")]


def test_normalize_drops_punctuation_space_and_case():
    assert normalize_utterance(" My  Email. ") == "myemail"
    assert normalize_utterance("電子郵件。") == "電子郵件"
    assert normalize_utterance("ＡＢＣ！") == "abc"  # full-width folds to ASCII


def test_snippet_matches_whole_utterance_only():
    assert match_snippet("電子郵件。", SNIPPETS) is SNIPPETS[0]
    assert match_snippet("my email", SNIPPETS) is SNIPPETS[0]
    assert match_snippet("我的電子郵件是什麼", SNIPPETS) is None
    assert match_snippet("", SNIPPETS) is None


def test_style_follows_app_rules():
    config = Config()
    assert resolve_style(config, "LINE.exe").name == "casual"
    assert resolve_style(config, "https://www.notion.so/page").name == "formal"
    assert resolve_style(config, "code.exe").name == "default"
    assert resolve_style(config, None).name == "default"


def test_user_rules_and_styles():
    config = Config.model_validate(
        {
            "styles": {"code": {"description": "Keep identifiers verbatim."}},
            "apps": [{"match": ["code.exe"], "style": "code"}],
        }
    )
    style = resolve_style(config, "Code.exe")
    assert style.name == "code"
    assert style.description == "Keep identifiers verbatim."
    assert "casual" in config.styles  # built-ins are still available


def test_transcription_hints_carry_locale_and_dictionary():
    config = Config(dictionary=["Claude", "uv"])
    hints = transcription_hints(config)
    assert "繁體中文" in hints.prompt
    assert "Claude" in hints.prompt
    assert hints.keywords == ("Claude", "uv")
    assert hints.languages == ("zh", "en")


def test_cleanup_messages():
    config = Config(dictionary=["Claude"], cleanup={"instructions": "Never use emoji."})
    style = resolve_style(config, "LINE")
    system, user = cleanup_messages(config, "嗯 你好", style, 'LINE"><x>')
    assert system.role == "system"
    assert "Traditional characters" in system.content
    assert "Claude" in system.content
    assert "casual" in system.content
    assert "Never use emoji." in system.content
    assert "<transcript>\n嗯 你好\n</transcript>" in user.content
    assert '<context app="LINEx" />' in user.content
