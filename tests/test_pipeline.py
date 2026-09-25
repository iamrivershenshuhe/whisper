import pytest
from conftest import FakeChat, FakeTranscriber

from whisper_flow.pipeline import Dictation, EntryNotFound
from whisper_flow.providers import ProviderError

pytestmark = pytest.mark.anyio


async def test_cleanup_path(dictation, chat, audio, history):
    result = await dictation.dictate(audio, app="LINE.exe")

    assert result.text == "明天下午六點開會。"
    assert result.raw_text == "嗯 明天下午五點 不對 六點開會"
    assert result.style == "casual"
    assert result.llm_model == "fake-llm"
    assert "<transcript>" in chat.calls[0][1].content

    entry = history.get(result.id)
    assert entry.status == "done"
    assert entry.final_text == result.text
    assert entry.app == "LINE.exe"
    assert history.load_audio(result.id) == audio


async def test_snippet_skips_llm(config, chat, history, audio):
    dictation = Dictation(config, FakeTranscriber("電子郵件。"), chat, history)
    result = await dictation.dictate(audio)

    assert result.text == "me@example.com"
    assert result.snippet == "電子郵件"
    assert result.llm_model is None
    assert chat.calls == []


async def test_silence_skips_llm(config, chat, history, audio):
    dictation = Dictation(config, FakeTranscriber(""), chat, history)
    result = await dictation.dictate(audio)
    assert result.text == ""
    assert chat.calls == []


async def test_llm_failure_falls_back_to_raw(config, transcriber, history, audio):
    dictation = Dictation(config, transcriber, FakeChat(fail=True), history)
    result = await dictation.dictate(audio)

    assert result.text == result.raw_text
    assert result.llm_model is None
    assert result.warnings == ["llm down"]
    assert history.get(result.id).error == "llm down"


async def test_asr_failure_keeps_audio_and_can_retry(config, chat, history, audio):
    transcriber = FakeTranscriber(fail=True)
    dictation = Dictation(config, transcriber, chat, history)

    with pytest.raises(ProviderError):
        await dictation.dictate(audio)

    [entry] = history.list()
    assert entry.status == "error"
    assert entry.error == "asr down"
    assert history.load_audio(entry.id) == audio

    transcriber.fail = False
    result = await dictation.retry(entry.id)
    assert result.id == entry.id
    assert history.get(entry.id).status == "done"


async def test_retry_unknown_entry(dictation):
    with pytest.raises(EntryNotFound):
        await dictation.retry(999)


async def test_works_without_history_or_cleanup(config, transcriber, audio):
    dictation = Dictation(config, transcriber, None, None)
    result = await dictation.dictate(audio)
    assert result.id is None
    assert result.text == transcriber.text
