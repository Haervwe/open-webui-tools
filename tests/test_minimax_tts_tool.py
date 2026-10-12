"""Offline request, response, and playback tests for the MiniMax speech tool."""

import asyncio
import base64
import os
import sys
from unittest.mock import AsyncMock, patch

import aiohttp
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

from minimax_tts_tool import DEFAULT_MODEL, SPEECH_ENDPOINTS, Tools


class FakeResponse:
    def __init__(self, payload, status=200):
        self.payload = payload
        self.status = status

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def json(self):
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


class FakeSession:
    def __init__(self, response):
        self.response = response
        self.requests = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    def post(self, url, **kwargs):
        self.requests.append((url, kwargs))
        return self.response


def success_payload(audio=b"ID3audio"):
    return {"base_resp": {"status_code": 0}, "data": {"status": 2, "audio": audio.hex()}}


@pytest.mark.asyncio
@pytest.mark.parametrize("region", ["global_en", "cn_zh"])
async def test_speech_request_and_audio_playback(region):
    tool = Tools()
    tool.valves.MINIMAX_API_KEY = "test-key"
    tool.valves.REGION = region
    emitter = AsyncMock()
    session = FakeSession(FakeResponse(success_payload()))

    with patch("minimax_tts_tool.aiohttp.ClientSession", return_value=session) as client:
        result = await tool.generate_speech("Hello", __event_emitter__=emitter)

    assert client.call_args.kwargs["headers"]["Authorization"] == "Bearer test-key"
    assert client.call_args.kwargs["timeout"].total == 180
    assert session.requests == [(SPEECH_ENDPOINTS[region], {
        "json": {
            "model": DEFAULT_MODEL,
            "text": "Hello",
            "stream": False,
            "output_format": "hex",
            "voice_setting": {"voice_id": "English_expressive_narrator", "speed": 1.0},
            "audio_setting": {"format": "mp3"},
        },
        "allow_redirects": False,
    })]
    response, context = result
    html = response.body.decode()
    assert "<audio controls" in html
    assert "data:audio/mpeg;base64," + base64.b64encode(b"ID3audio").decode() in html
    assert 'download="minimax-speech.mp3"' in html
    assert "test-key" not in html + context
    assert emitter.await_args_list[-1].args[0]["data"]["done"] is True


@pytest.mark.asyncio
async def test_voice_and_model_settings_are_forwarded():
    tool = Tools()
    tool.valves.MINIMAX_API_KEY = "test-key"
    tool.valves.MODEL = "configured-model"
    tool.valves.VOICE_ID = "configured-voice"
    tool.valves.SPEED = 1.5
    session = FakeSession(FakeResponse(success_payload()))
    with patch("minimax_tts_tool.aiohttp.ClientSession", return_value=session):
        await tool.generate_speech("Hello")
    payload = session.requests[0][1]["json"]
    assert payload["model"] == "configured-model"
    assert payload["voice_setting"] == {"voice_id": "configured-voice", "speed": 1.5}


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["", "   ", "a" * 10000])
async def test_invalid_text_does_not_send_request(text):
    tool = Tools()
    tool.valves.MINIMAX_API_KEY = "test-key"
    with patch("minimax_tts_tool.aiohttp.ClientSession") as client:
        result = await tool.generate_speech(text)
    assert "Text must" in result
    client.assert_not_called()


@pytest.mark.asyncio
async def test_missing_key_does_not_send_request():
    emitter = AsyncMock()
    with patch("minimax_tts_tool.aiohttp.ClientSession") as client:
        result = await Tools().generate_speech("Hello", __event_emitter__=emitter)
    assert "not configured" in result
    client.assert_not_called()
    assert emitter.await_args_list[-1].args[0]["data"]["done"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("payload,status,expected", [
    ({"base_resp": {"status_code": 1004, "status_msg": "sensitive request details"}}, 200, "rejected"),
    ({"data": {"status": 2, "audio": "abcd"}}, 200, "rejected"),
    ({"base_resp": None}, 200, "rejected"),
    ([], 200, "invalid response"),
    ({"base_resp": {"status_code": 0}, "data": None}, 200, "completed audio"),
    ({"base_resp": {"status_code": 0}, "data": {"status": 1, "audio": "abcd"}}, 200, "completed audio"),
    ({"base_resp": {"status_code": 0}, "data": {"status": 2}}, 200, "no audio"),
    ({"base_resp": {"status_code": 0}, "data": {"status": 2, "audio": "zz"}}, 200, "encoding"),
    (success_payload(b""), 200, "no audio"),
    (ValueError("sensitive request details"), 200, "invalid JSON"),
    ({}, 401, "HTTP 401"),
    ({}, 302, "HTTP 302"),
])
async def test_invalid_responses_never_produce_audio(payload, status, expected):
    tool = Tools()
    tool.valves.MINIMAX_API_KEY = "test-key"
    emitter = AsyncMock()
    session = FakeSession(FakeResponse(payload, status))
    with patch("minimax_tts_tool.aiohttp.ClientSession", return_value=session):
        result = await tool.generate_speech("Hello", __event_emitter__=emitter)
    assert isinstance(result, str) and expected in result
    assert "sensitive request details" not in result
    assert emitter.await_args_list[-1].args[0]["data"]["done"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("error,expected", [
    (asyncio.TimeoutError(), "timed out"),
    (aiohttp.ClientError("sensitive request details"), "connection"),
])
async def test_transport_errors_are_reported_without_details(error, expected):
    tool = Tools()
    tool.valves.MINIMAX_API_KEY = "test-key"
    with patch("minimax_tts_tool.aiohttp.ClientSession", side_effect=error):
        result = await tool.generate_speech("Hello")
    assert expected in result
    assert "sensitive request details" not in result
