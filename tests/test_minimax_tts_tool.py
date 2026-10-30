"""Offline request, response, and playback tests for the MiniMax speech tool."""

import asyncio
import os
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

# Other test modules may replace fastapi/starlette with MagicMock at collection
# time; this tool needs the real HTMLResponse/UploadFile/HTTPException.
for _name in [n for n in sys.modules if n.split(".")[0] in ("fastapi", "starlette")]:
    if isinstance(sys.modules[_name], MagicMock):
        del sys.modules[_name]

# Stub the Open WebUI modules the tool imports; tests patch the used symbols.
for _name in (
    "open_webui",
    "open_webui.models",
    "open_webui.models.chats",
    "open_webui.models.users",
    "open_webui.routers",
    "open_webui.routers.files",
):
    sys.modules.setdefault(_name, MagicMock())

from fastapi import HTTPException

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

import minimax_tts_tool
from minimax_tts_tool import DEFAULT_MODEL, SPEECH_ENDPOINTS, Tools, _script_json

FILE_ID = "0b8f8a52-7c1e-4a8e-9d0f-6a1d2b3c4d5e"
USER = {"id": "user-1", "name": "Tester"}
METADATA = {"chat_id": "chat-1", "message_id": "msg-1", "session_id": "sess-1"}
REQUEST = SimpleNamespace(app=None)


@pytest.fixture(autouse=True)
def owui(monkeypatch):
    """Mock Open WebUI storage, user lookup and chat-file linking."""
    mocks = SimpleNamespace(
        upload=AsyncMock(return_value=SimpleNamespace(id=FILE_ID)),
        get_user=AsyncMock(return_value=SimpleNamespace(id="user-1")),
        link=AsyncMock(return_value=None),
    )
    monkeypatch.setattr(minimax_tts_tool, "upload_file_handler", mocks.upload)
    monkeypatch.setattr(
        minimax_tts_tool, "Users", SimpleNamespace(get_user_by_id=mocks.get_user)
    )
    monkeypatch.setattr(
        minimax_tts_tool, "Chats", SimpleNamespace(insert_chat_files=mocks.link)
    )
    return mocks


def configured_tool(**valves):
    tool = Tools()
    tool.valves.MINIMAX_API_KEY = "test-key"
    for key, value in valves.items():
        setattr(tool.valves, key, value)
    return tool


async def generate(tool, text="Hello", payload=None, **kwargs):
    session = FakeSession(FakeResponse(payload or success_payload()))
    kwargs.setdefault("__user__", USER)
    kwargs.setdefault("__request__", REQUEST)
    kwargs.setdefault("__metadata__", METADATA)
    with patch("minimax_tts_tool.aiohttp.ClientSession", return_value=session):
        result = await tool.generate_speech(text, **kwargs)
    return result, session


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
async def test_speech_request_and_audio_playback(region, owui):
    tool = Tools()
    tool.valves.MINIMAX_API_KEY = "test-key"
    tool.valves.REGION = region
    emitter = AsyncMock()
    session = FakeSession(FakeResponse(success_payload()))

    with patch("minimax_tts_tool.aiohttp.ClientSession", return_value=session) as client:
        result = await tool.generate_speech(
            "Hello",
            __user__=USER,
            __request__=REQUEST,
            __metadata__=METADATA,
            __event_emitter__=emitter,
        )

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
    assert response.headers["content-disposition"] == "inline"
    html = response.body.decode()
    audio_url = f"/api/v1/files/{FILE_ID}/content"
    assert "<audio" in html
    assert f'"{audio_url}"' in html
    assert "data:audio" not in html and "base64" not in html
    assert "localStorage.getItem('token')" in html
    assert "minimax-speech.mp3" in html
    assert context == {
        "message": "Speech generated successfully and displayed in the audio player.",
        "audio_url": audio_url,
        "voice_id": "English_expressive_narrator",
        "model": DEFAULT_MODEL,
    }
    assert "test-key" not in html + str(context)
    assert emitter.await_args_list[-1].args[0]["data"]["done"] is True

    owui.get_user.assert_awaited_once_with("user-1")
    owui.upload.assert_awaited_once()
    args, kwargs = owui.upload.await_args
    assert args == (REQUEST,)
    assert kwargs["process"] is False
    assert kwargs["user"] is owui.get_user.return_value
    assert kwargs["metadata"] == METADATA
    upload = kwargs["file"]
    assert upload.filename.startswith("minimax_tts_") and upload.filename.endswith(".mp3")
    assert upload.content_type == "audio/mpeg"
    assert upload.file.getvalue() == b"ID3audio"
    owui.link.assert_awaited_once_with(
        chat_id="chat-1", message_id="msg-1", file_ids=[FILE_ID], user_id="user-1"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("metadata", [
    {"chat_id": "local:abc", "message_id": "msg-1"},
    {"chat_id": "temporary:abc", "message_id": "msg-1"},
    {"chat_id": "channel:abc", "message_id": "msg-1"},
    {"chat_id": "chat-1"},
    {},
    None,
])
async def test_unsaved_chats_are_not_linked(metadata, owui):
    result, _ = await generate(configured_tool(), __metadata__=metadata)
    assert isinstance(result, tuple)
    owui.link.assert_not_awaited()


@pytest.mark.asyncio
async def test_chat_link_failure_still_returns_player(owui):
    owui.link.side_effect = RuntimeError("db down")
    result, _ = await generate(configured_tool())
    assert isinstance(result, tuple)
    assert result[1]["audio_url"] == f"/api/v1/files/{FILE_ID}/content"


@pytest.mark.asyncio
@pytest.mark.parametrize("error,expected", [
    (HTTPException(status_code=413, detail="File too large sensitive"), "HTTP 413"),
    (RuntimeError("sensitive storage details"), "failed unexpectedly"),
])
async def test_storage_failure_returns_clean_error(error, expected, owui):
    owui.upload.side_effect = error
    emitter = AsyncMock()
    result, _ = await generate(configured_tool(), __event_emitter__=emitter)
    assert isinstance(result, str) and expected in result
    assert "sensitive" not in result and "test-key" not in result
    assert emitter.await_args_list[-1].args[0]["data"]["done"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("kwargs", [
    {"__request__": None},
    {"__user__": None},
    {"__user__": {}},
])
async def test_missing_owui_context_is_reported(kwargs, owui):
    result, _ = await generate(configured_tool(), **kwargs)
    assert isinstance(result, str) and "could not be saved" in result
    owui.upload.assert_not_awaited()


@pytest.mark.asyncio
async def test_unknown_user_is_reported(owui):
    owui.get_user.return_value = None
    result, _ = await generate(configured_tool())
    assert isinstance(result, str) and "user could not be resolved" in result
    owui.upload.assert_not_awaited()


@pytest.mark.asyncio
async def test_upload_without_file_id_is_reported(owui):
    owui.upload.return_value = None
    result, _ = await generate(configured_tool())
    assert isinstance(result, str) and "file ID" in result


@pytest.mark.asyncio
async def test_player_escapes_text_and_settings():
    text = "<script>alert(1)</script> & \"quotes\""
    tool = configured_tool(VOICE_ID="<b>voice</b>", MODEL="m</script>")
    result, _ = await generate(tool, text=text)
    html = result[0].body.decode()
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt; &amp; &quot;quotes&quot;" in html
    assert "<b>voice</b>" not in html and "m</script>" not in html


@pytest.mark.asyncio
async def test_long_text_is_truncated_in_player():
    text = "a" * 5000 + "TAIL"
    result, session = await generate(configured_tool(), text=text)
    assert session.requests[0][1]["json"]["text"] == text
    assert "TAIL" not in result[0].body.decode()


def test_script_json_escapes_closing_tags():
    assert "</" not in _script_json("/x</script><script>")


@pytest.mark.asyncio
@pytest.mark.parametrize("valves", [
    {"VOICE_ID": "personal-voice", "SPEED": 0.75},
    SimpleNamespace(VOICE_ID="personal-voice", SPEED=0.75),
])
async def test_user_valves_override_voice_and_speed(valves):
    user = {**USER, "valves": valves}
    result, session = await generate(configured_tool(), __user__=user)
    payload = session.requests[0][1]["json"]
    assert payload["voice_setting"] == {"voice_id": "personal-voice", "speed": 0.75}
    assert result[1]["voice_id"] == "personal-voice"


@pytest.mark.asyncio
async def test_empty_user_valves_fall_back_to_admin_settings():
    user = {**USER, "valves": Tools.UserValves()}
    _, session = await generate(configured_tool(SPEED=1.25), __user__=user)
    payload = session.requests[0][1]["json"]
    assert payload["voice_setting"] == {
        "voice_id": "English_expressive_narrator",
        "speed": 1.25,
    }


@pytest.mark.asyncio
async def test_rejection_reports_only_numeric_code():
    payload = {"base_resp": {"status_code": 1008, "status_msg": "sensitive balance info"}}
    result, _ = await generate(configured_tool(), payload=payload)
    assert result == "MiniMax rejected the speech request (code 1008); check your configuration."


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
