import pytest
from conftest import FakeTranscriber
from fastapi.testclient import TestClient

from whisper_flow.pipeline import Dictation
from whisper_flow.server import create_app


@pytest.fixture
def client(dictation):
    return TestClient(create_app(dictation=dictation))


def test_health(client):
    assert client.get("/health").json() == {
        "status": "ok",
        "asr_model": "fake-asr",
        "llm_model": "fake-llm",
    }


def test_dictate_and_history_roundtrip(client, audio):
    r = client.post(
        "/v1/dictate",
        files={"audio": ("clip.wav", audio.data, "audio/wav")},
        data={"app": "Notion.exe"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["text"] == "明天下午六點開會。"
    assert body["style"] == "formal"

    entries = client.get("/v1/history").json()
    assert [e["id"] for e in entries] == [body["id"]]
    assert client.get(f"/v1/history/{body['id']}").json()["app"] == "Notion.exe"

    wav = client.get(f"/v1/history/{body['id']}/audio")
    assert wav.status_code == 200
    assert wav.headers["content-type"] == "audio/wav"
    assert wav.content == audio.data


def test_format_from_content_type(client, audio):
    r = client.post("/v1/dictate", files={"audio": ("blob", audio.data, "audio/webm;codecs=opus")})
    assert r.status_code == 200, r.text


def test_rejects_bad_input(client, audio):
    empty = client.post("/v1/dictate", files={"audio": ("clip.wav", b"", "audio/wav")})
    assert empty.status_code == 400
    bad = client.post(
        "/v1/dictate", files={"audio": ("clip.exe", b"MZ", "application/x-msdownload")}
    )
    assert bad.status_code == 415


def test_provider_error_is_502_and_retryable(config, chat, history, audio):
    transcriber = FakeTranscriber(fail=True)
    client = TestClient(create_app(dictation=Dictation(config, transcriber, chat, history)))

    r = client.post("/v1/dictate", files={"audio": ("clip.wav", audio.data, "audio/wav")})
    assert r.status_code == 502
    [entry] = client.get("/v1/history").json()
    assert entry["status"] == "error"

    transcriber.fail = False
    retried = client.post(f"/v1/history/{entry['id']}/retry")
    assert retried.status_code == 200
    assert retried.json()["id"] == entry["id"]


def test_missing_entries_are_404(client):
    assert client.get("/v1/history/999").status_code == 404
    assert client.get("/v1/history/999/audio").status_code == 404
    assert client.post("/v1/history/999/retry").status_code == 404
