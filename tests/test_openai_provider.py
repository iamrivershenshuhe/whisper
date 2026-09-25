"""Checks the exact requests sent to OpenAI, using a mock HTTP transport (no network)."""

import json

import httpx2
import pytest
from openai import AsyncOpenAI

from whisper_flow.config import ASRConfig, Config, LLMConfig, ProviderConfig
from whisper_flow.providers import Message, ProviderError, TranscriptionHints, build_chat
from whisper_flow.providers.openai import OpenAIChat, OpenAITranscriber, make_client

pytestmark = pytest.mark.anyio


def mock_client(handler) -> tuple[AsyncOpenAI, list[httpx2.Request]]:
    requests: list[httpx2.Request] = []

    def record(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return handler(request)

    client = AsyncOpenAI(
        api_key="test",
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(record)),
    )
    return client, requests


async def test_transcription_request(audio):
    client, requests = mock_client(
        lambda r: httpx2.Response(200, json={"text": " 你好 ", "languages": [{"code": "zh"}]})
    )
    transcriber = OpenAITranscriber(client, ASRConfig())
    hints = TranscriptionHints(prompt="繁體中文", keywords=("Claude",), languages=("zh", "en"))

    result = await transcriber.transcribe(audio, hints)

    assert result.text == "你好"
    assert result.languages == ("zh",)
    [req] = requests
    assert req.url.path.endswith("/audio/transcriptions")
    body = req.content.decode("utf-8", errors="replace")
    for expected in ("gpt-transcribe", "繁體中文", "Claude", "audio.wav"):
        assert expected in body


async def test_keywords_can_be_disabled(audio):
    client, requests = mock_client(lambda r: httpx2.Response(200, json={"text": "hi"}))
    transcriber = OpenAITranscriber(client, ASRConfig(model="whisper-1", send_keywords=False))

    await transcriber.transcribe(audio, TranscriptionHints(keywords=("Claude",)))

    body = requests[0].content.decode("utf-8", errors="replace")
    assert "Claude" not in body
    assert "whisper-1" in body


def chat_reply(text: str) -> dict:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": "gpt-6-luna",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": text},
            }
        ],
    }


async def test_chat_request():
    client, requests = mock_client(lambda r: httpx2.Response(200, json=chat_reply(" 好。 ")))
    chat = OpenAIChat(client, LLMConfig())

    assert await chat.complete([Message("system", "s"), Message("user", "u")]) == "好。"

    sent = json.loads(requests[0].content)
    assert sent["model"] == "gpt-6-luna"
    assert sent["reasoning_effort"] == "none"
    assert "temperature" not in sent
    assert sent["messages"] == [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
    ]


async def test_optional_params_omitted_for_compatible_vendors():
    client, requests = mock_client(lambda r: httpx2.Response(200, json=chat_reply("ok")))
    chat = OpenAIChat(client, LLMConfig(model="llama", reasoning_effort=None, temperature=0.2))

    await chat.complete([Message("user", "u")])

    sent = json.loads(requests[0].content)
    assert "reasoning_effort" not in sent
    assert sent["temperature"] == 0.2


async def test_api_errors_become_provider_errors():
    client, _ = mock_client(lambda r: httpx2.Response(401, json={"error": {"message": "bad key"}}))
    with pytest.raises(ProviderError, match="gpt-6-luna"):
        await OpenAIChat(client, LLMConfig()).complete([Message("user", "u")])


def test_missing_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ProviderError, match="OPENAI_API_KEY"):
        make_client("openai", ProviderConfig())
    with pytest.raises(ProviderError):
        build_chat(Config())
