import pytest

from whisper_flow.history import History


def test_create_update_list(history, audio):
    first = history.create(audio, "LINE")
    second = history.create(audio, None)
    history.update(first, status="done", final_text="明天見")
    history.update(second, status="done", final_text="see you")

    assert [e.id for e in history.list()] == [second, first]
    assert [e.id for e in history.list(query="明天")] == [first]
    assert history.get(first).app == "LINE"
    assert history.get(12345) is None


def test_audio_not_saved_when_disabled(tmp_path, audio):
    history = History(tmp_path, save_audio=False)
    entry_id = history.create(audio, None)
    assert history.load_audio(entry_id) is None


def test_update_rejects_unknown_columns(history, audio):
    entry_id = history.create(audio, None)
    with pytest.raises(ValueError):
        history.update(entry_id, audio_file="../../evil")
