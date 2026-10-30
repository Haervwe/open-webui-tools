"""Offline tests for the MuAPI media tool.

Open WebUI internals (file storage, users, chats) are replaced with stubs so
these tests run without an Open WebUI install and without network access.
They use ``asyncio.run`` and need only pytest, aiohttp, fastapi and pydantic.
"""

import asyncio
import base64
import os
import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


def _install_open_webui_stubs() -> None:
    """Provide stand-ins for the Open WebUI modules the tool imports.

    Uses setdefault per module so it coexists with stubs installed by other
    test modules; every test below monkeypatches the attributes it relies on.
    """
    defaults = {
        "open_webui": {},
        "open_webui.models": {},
        "open_webui.models.chats": {
            "Chats": SimpleNamespace(insert_chat_files=AsyncMock())
        },
        "open_webui.models.files": {
            "Files": SimpleNamespace(get_file_by_id=AsyncMock(return_value=None))
        },
        "open_webui.models.users": {
            "Users": SimpleNamespace(get_user_by_id=AsyncMock(return_value=None))
        },
        "open_webui.routers": {},
        "open_webui.routers.files": {"upload_file_handler": AsyncMock()},
        "open_webui.storage": {},
        "open_webui.storage.provider": {
            "Storage": SimpleNamespace(get_file=lambda path: path)
        },
    }
    for name, attrs in defaults.items():
        module = sys.modules.setdefault(name, types.ModuleType(name))
        for attr, value in attrs.items():
            if isinstance(module, types.ModuleType) and not hasattr(module, attr):
                setattr(module, attr, value)


_install_open_webui_stubs()
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

import muapi_media_tool  # noqa: E402
from muapi_media_tool import (  # noqa: E402
    DEFAULT_API_BASE_URL,
    DEFAULT_IMAGE_MODEL,
    DEFAULT_IMAGE_TO_VIDEO_MODEL,
    DEFAULT_VIDEO_MODEL,
    Tools,
)

BASE = DEFAULT_API_BASE_URL
PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 200
MP4_BYTES = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 200


class FakeUploadFile:
    def __init__(self, file, filename, headers=None):
        self.file = file
        self.filename = filename
        self.content_type = (headers or {}).get("content-type")


class FakeContent:
    def __init__(self, body):
        self.body = body

    async def iter_chunked(self, size):
        for start in range(0, len(self.body), size):
            yield self.body[start : start + size]


class FakeResponse:
    def __init__(self, payload=None, status=200, body=None, headers=None):
        self.payload = payload
        self.status = status
        self.body = body or b""
        self.headers = headers or {}
        self.content = FakeContent(self.body)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    async def json(self):
        return self.payload

    async def text(self):
        return str(self.payload)


class FakeRouter:
    """Routes requests of every ClientSession by (method, url) and records them."""

    def __init__(self, routes):
        self.routes = {key: list(value) for key, value in routes.items()}
        self.requests = []
        self.session_headers = []

    def session(self, *args, **kwargs):
        return FakeSession(self, kwargs.get("headers") or {})

    def respond(self, method, url, session_headers, body):
        self.requests.append((method, url, body, dict(session_headers)))
        queue = self.routes.get((method, url))
        if not queue:
            raise AssertionError(f"Unexpected request {method} {url}")
        return queue.pop(0) if len(queue) > 1 else queue[0]


class FakeSession:
    def __init__(self, router, headers):
        self.router = router
        self.headers = headers

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    def post(self, url, json=None, data=None):
        return self.router.respond("POST", url, self.headers, json if json is not None else data)

    def get(self, url):
        return self.router.respond("GET", url, self.headers, None)


@pytest.fixture
def owui(monkeypatch):
    """Stub OWUI storage: every upload gets a sequential file id."""
    user = SimpleNamespace(id="user-1", role="user", email="u@example.test", name="U")
    uploads = []

    async def fake_upload(request, file, metadata, process, user):
        uploads.append(
            {
                "filename": file.filename,
                "content_type": file.content_type,
                "data": file.file.read(),
                "metadata": metadata,
                "process": process,
                "user": user,
            }
        )
        return SimpleNamespace(id=f"file-{len(uploads)}")

    insert_chat_files = AsyncMock()
    # Other test modules replace fastapi with a MagicMock; use a plain stand-in.
    monkeypatch.setattr(muapi_media_tool, "UploadFile", FakeUploadFile)
    monkeypatch.setattr(muapi_media_tool, "upload_file_handler", fake_upload)
    monkeypatch.setattr(
        muapi_media_tool,
        "Users",
        SimpleNamespace(get_user_by_id=AsyncMock(return_value=user)),
    )
    monkeypatch.setattr(
        muapi_media_tool, "Chats", SimpleNamespace(insert_chat_files=insert_chat_files)
    )
    monkeypatch.setattr(muapi_media_tool.asyncio, "sleep", AsyncMock())
    return SimpleNamespace(user=user, uploads=uploads, insert_chat_files=insert_chat_files)


def use_router(monkeypatch, routes):
    router = FakeRouter(routes)
    monkeypatch.setattr(muapi_media_tool.aiohttp, "ClientSession", router.session)
    return router


def call_kwargs():
    return {
        "__user__": {"id": "user-1"},
        "__request__": object(),
        "__metadata__": {"chat_id": "chat-1", "message_id": "msg-1"},
    }


def test_default_valves_use_current_muapi_endpoints():
    tool = Tools()

    assert tool.valves.API_BASE_URL == DEFAULT_API_BASE_URL
    assert tool.valves.IMAGE_MODEL == DEFAULT_IMAGE_MODEL
    assert tool.valves.VIDEO_MODEL == DEFAULT_VIDEO_MODEL
    assert tool.valves.IMAGE_TO_VIDEO_MODEL == DEFAULT_IMAGE_TO_VIDEO_MODEL
    assert tool.valves.ALLOW_PROVIDER_URL_FALLBACK is False


def test_user_valves_override_key_and_models():
    tool = Tools()
    tool.valves.MUAPI_API_KEY = "admin-key"
    user_valves = tool.UserValves(
        MUAPI_API_KEY="user-key",
        IMAGE_MODEL="custom-image-model",
    )

    config = tool._resolve_config({"valves": user_valves})

    assert config["api_key"] == "user-key"
    assert config["image_model"] == "custom-image-model"
    assert config["video_model"] == DEFAULT_VIDEO_MODEL


def test_generate_image_stores_output_and_returns_viewer(monkeypatch, owui):
    tool = Tools()
    tool.valves.MUAPI_API_KEY = "test-key"
    cdn = "https://cdn.example.test/image.png?sig=secret"
    router = use_router(
        monkeypatch,
        {
            ("POST", f"{BASE}/nano-banana"): [
                FakeResponse({"id": "image-request-1", "status": "queued"})
            ],
            ("GET", f"{BASE}/predictions/image-request-1/result"): [
                FakeResponse({"id": "image-request-1", "status": "processing"}),
                FakeResponse(
                    {
                        "id": "image-request-1",
                        "status": "completed",
                        "output": {"outputs": [cdn]},
                    }
                ),
            ],
            ("GET", cdn): [
                FakeResponse(body=PNG_BYTES, headers={"Content-Type": "image/png"})
            ],
        },
    )

    result = asyncio.run(
        tool.generate_image(
            "a <b>lighthouse</b>", aspect_ratio="16:9", **call_kwargs()
        )
    )

    assert isinstance(result, tuple) and len(result) == 2
    html_response, context = result
    body = html_response.body.decode("utf-8")
    assert html_response.headers["content-disposition"] == "inline"
    assert 'data-muapi-src="/api/v1/files/file-1/content"' in body
    assert "<img" in body
    assert "a &lt;b&gt;lighthouse&lt;/b&gt;" in body
    assert "/api/v1/files/file-1/content" in context
    for text in (body, context):
        assert "cdn.example.test" not in text
        assert "test-key" not in text

    assert router.requests[0][:3] == (
        "POST",
        f"{BASE}/nano-banana",
        {"prompt": "a <b>lighthouse</b>", "aspect_ratio": "16:9"},
    )
    assert router.requests[0][3]["x-api-key"] == "test-key"
    download = [r for r in router.requests if r[1] == cdn][0]
    assert "x-api-key" not in download[3]

    assert len(owui.uploads) == 1
    upload = owui.uploads[0]
    assert upload["data"] == PNG_BYTES
    assert upload["content_type"] == "image/png"
    assert upload["filename"].endswith(".png")
    assert upload["process"] is False
    assert upload["user"] is owui.user
    assert upload["metadata"] == {"chat_id": "chat-1", "message_id": "msg-1"}
    owui.insert_chat_files.assert_awaited_once_with(
        chat_id="chat-1", message_id="msg-1", file_ids=["file-1"], user_id="user-1"
    )


def test_generate_video_submits_options_and_embeds_stored_video(monkeypatch, owui):
    tool = Tools()
    tool.valves.MUAPI_API_KEY = "test-key"
    cdn = "https://cdn.example.test/video.mp4"
    router = use_router(
        monkeypatch,
        {
            ("POST", f"{BASE}/seedance-2.5-text-to-video"): [
                FakeResponse({"request_id": "video-request-1"})
            ],
            ("GET", f"{BASE}/predictions/video-request-1/result"): [
                FakeResponse({"status": "succeeded", "outputs": [cdn]})
            ],
            ("GET", cdn): [
                FakeResponse(
                    body=MP4_BYTES,
                    headers={"Content-Type": "application/octet-stream"},
                )
            ],
        },
    )
    kwargs = call_kwargs()
    kwargs["__metadata__"] = {"chat_id": "temporary:abc", "message_id": "msg-1"}

    result = asyncio.run(
        tool.generate_video(
            "a city at night",
            duration=8,
            aspect_ratio="16:9",
            seed=42,
            high_bitrate=True,
            **kwargs,
        )
    )

    assert isinstance(result, tuple) and len(result) == 2
    html_response, context = result
    body = html_response.body.decode("utf-8")
    assert "<video" in body
    assert 'data-muapi-src="/api/v1/files/file-1/content"' in body
    assert "authFetch" in body
    assert "cdn.example.test" not in body + context
    assert "/api/v1/files/file-1/content" in context
    assert router.requests[0][:3] == (
        "POST",
        f"{BASE}/seedance-2.5-text-to-video",
        {
            "prompt": "a city at night",
            "duration": 8,
            "aspect_ratio": "16:9",
            "seed": 42,
            "high_bitrate": True,
        },
    )
    assert owui.uploads[0]["content_type"] == "video/mp4"
    # Temporary chats are not linked to chat_file rows.
    assert owui.uploads[0]["metadata"] == {}
    owui.insert_chat_files.assert_not_awaited()


def test_generate_video_from_image_uses_attached_http_url(monkeypatch, owui):
    tool = Tools()
    tool.valves.MUAPI_API_KEY = "test-key"
    cdn = "https://cdn.example.test/i2v.mp4"
    source = "https://images.example.test/input.png"
    router = use_router(
        monkeypatch,
        {
            ("POST", f"{BASE}/seedance-2.5-image-to-video"): [
                FakeResponse({"id": "i2v-request-1"})
            ],
            ("GET", f"{BASE}/predictions/i2v-request-1/result"): [
                FakeResponse(
                    {
                        "status": "completed",
                        "output": {"outputs": [cdn]},
                    }
                )
            ],
            ("GET", cdn): [
                FakeResponse(body=MP4_BYTES, headers={"Content-Type": "video/mp4"})
            ],
        },
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "animate this"},
                {"type": "image_url", "image_url": {"url": source}},
            ],
        }
    ]

    result = asyncio.run(
        tool.generate_video_from_image(
            "make the ocean waves move",
            duration=5,
            __messages__=messages,
            **call_kwargs(),
        )
    )

    assert isinstance(result, tuple)
    assert "/api/v1/files/file-1/content" in result[1]
    assert router.requests[0][:3] == (
        "POST",
        f"{BASE}/seedance-2.5-image-to-video",
        {
            "prompt": "make the ocean waves move",
            "image_url": source,
            "duration": 5,
        },
    )


def test_image_to_video_uploads_data_uri_attachment(monkeypatch, owui):
    """OWUI 0.12.5 converts attached images to data: URIs before tools run."""
    tool = Tools()
    tool.valves.MUAPI_API_KEY = "test-key"
    hosted = "https://cdn.example.test/uploads/input.png"
    cdn = "https://cdn.example.test/i2v.mp4"
    router = use_router(
        monkeypatch,
        {
            ("POST", f"{BASE}/upload_file"): [FakeResponse({"url": hosted})],
            ("POST", f"{BASE}/seedance-2.5-image-to-video"): [
                FakeResponse({"id": "i2v-2"})
            ],
            ("GET", f"{BASE}/predictions/i2v-2/result"): [
                FakeResponse(
                    {
                        "status": "completed",
                        # MuAPI may echo the input image; it must not be treated as output.
                        "image_url": hosted,
                        "outputs": [cdn],
                    }
                )
            ],
            ("GET", cdn): [
                FakeResponse(body=MP4_BYTES, headers={"Content-Type": "video/mp4"})
            ],
        },
    )
    data_uri = "data:image/png;base64," + base64.b64encode(PNG_BYTES).decode()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "animate"},
                {"type": "image_url", "image_url": {"url": data_uri}},
            ],
        }
    ]

    result = asyncio.run(
        tool.generate_video_from_image(
            "slow pan", __messages__=messages, **call_kwargs()
        )
    )

    assert isinstance(result, tuple), result
    upload_request = router.requests[0]
    assert upload_request[1] == f"{BASE}/upload_file"
    assert upload_request[3]["x-api-key"] == "test-key"
    submit = router.requests[1]
    assert submit[2] == {"prompt": "slow pan", "image_url": hosted}
    assert len(owui.uploads) == 1  # only the video, not the echoed input
    assert "data:image" not in result[1]


def test_image_to_video_reads_owui_file_attachment(monkeypatch, owui, tmp_path):
    tool = Tools()
    tool.valves.MUAPI_API_KEY = "test-key"
    stored = tmp_path / "abc_input.png"
    stored.write_bytes(PNG_BYTES)
    file_model = SimpleNamespace(
        id="abc-123",
        user_id="user-1",
        filename="input.png",
        path=str(stored),
        meta={"content_type": "image/png"},
    )
    monkeypatch.setattr(
        muapi_media_tool,
        "Files",
        SimpleNamespace(get_file_by_id=AsyncMock(return_value=file_model)),
    )
    monkeypatch.setattr(
        muapi_media_tool, "Storage", SimpleNamespace(get_file=lambda path: path)
    )
    hosted = "https://cdn.example.test/uploads/input.png"
    router = use_router(
        monkeypatch,
        {
            ("POST", f"{BASE}/upload_file"): [FakeResponse({"data": {"url": hosted}})],
            ("POST", f"{BASE}/seedance-2.5-image-to-video"): [
                FakeResponse({"id": "i2v-3"})
            ],
            ("GET", f"{BASE}/predictions/i2v-3/result"): [
                FakeResponse({"status": "completed", "outputs": ["https://cdn.example.test/v.mp4"]})
            ],
            ("GET", "https://cdn.example.test/v.mp4"): [
                FakeResponse(body=MP4_BYTES, headers={"Content-Type": "video/mp4"})
            ],
        },
    )

    result = asyncio.run(
        tool.generate_video_from_image(
            "animate", image_url="/api/v1/files/abc-123/content", **call_kwargs()
        )
    )

    assert isinstance(result, tuple), result
    assert router.requests[0][1] == f"{BASE}/upload_file"
    assert router.requests[1][2]["image_url"] == hosted


def test_image_to_video_rejects_other_users_file(monkeypatch, owui):
    tool = Tools()
    tool.valves.MUAPI_API_KEY = "test-key"
    file_model = SimpleNamespace(
        id="abc", user_id="someone-else", filename="x.png", path="/nope", meta={}
    )
    monkeypatch.setattr(
        muapi_media_tool,
        "Files",
        SimpleNamespace(get_file_by_id=AsyncMock(return_value=file_model)),
    )
    stub_access = types.ModuleType("open_webui.utils.access_control.files")
    stub_access.has_access_to_file = AsyncMock(return_value=False)
    monkeypatch.setitem(sys.modules, "open_webui.utils.access_control.files", stub_access)
    router = use_router(monkeypatch, {})

    result = asyncio.run(
        tool.generate_video_from_image(
            "animate", image_url="/api/v1/files/abc/content", **call_kwargs()
        )
    )

    assert isinstance(result, str)
    assert "do not have access" in result
    assert router.requests == []


def test_storage_failure_returns_error_without_provider_url(monkeypatch, owui):
    tool = Tools()
    tool.valves.MUAPI_API_KEY = "test-key"
    cdn = "https://cdn.example.test/image.png"
    use_router(
        monkeypatch,
        {
            ("POST", f"{BASE}/nano-banana"): [FakeResponse({"id": "r1"})],
            ("GET", f"{BASE}/predictions/r1/result"): [
                FakeResponse({"status": "completed", "outputs": [cdn]})
            ],
            ("GET", cdn): [
                FakeResponse(body=PNG_BYTES, headers={"Content-Type": "image/png"})
            ],
        },
    )

    class TooLarge(Exception):
        detail = "File size exceeds the limit"

    monkeypatch.setattr(
        muapi_media_tool, "upload_file_handler", AsyncMock(side_effect=TooLarge())
    )

    result = asyncio.run(tool.generate_image("a lighthouse", **call_kwargs()))

    assert isinstance(result, str)
    assert "Saving the generated image to Open WebUI failed" in result
    assert "cdn.example.test" not in result

    tool.valves.ALLOW_PROVIDER_URL_FALLBACK = True
    fallback = asyncio.run(tool.generate_image("a lighthouse", **call_kwargs()))
    assert isinstance(fallback, tuple)
    assert cdn in fallback[1]
    # The viewer must only attach the OWUI token to same-origin /api/ paths,
    # never to provider URLs (even ones containing "/api/").
    body = fallback[0].body.decode()
    assert "url.indexOf('/api/') !== 0" in body
    assert "url.indexOf('/api/') === -1" not in body


def test_private_output_host_is_not_downloaded(monkeypatch, owui):
    tool = Tools()
    tool.valves.MUAPI_API_KEY = "test-key"
    router = use_router(
        monkeypatch,
        {
            ("POST", f"{BASE}/nano-banana"): [FakeResponse({"id": "r2"})],
            ("GET", f"{BASE}/predictions/r2/result"): [
                FakeResponse(
                    {"status": "completed", "outputs": ["https://127.0.0.1/secret.png"]}
                )
            ],
        },
    )

    result = asyncio.run(tool.generate_image("a lighthouse", **call_kwargs()))

    assert "private or local host" in result
    assert all("127.0.0.1" not in r[1] for r in router.requests)


def test_extract_output_urls_prefers_documented_outputs():
    payload = {
        "data": {
            "status": "completed",
            "image_url": "https://in.example.test/a.png",
            "outputs": ["https://cdn.example.test/v.mp4"],
        }
    }
    assert Tools._extract_output_urls(payload) == ["https://cdn.example.test/v.mp4"]
    assert Tools._extract_output_urls(
        {"output": {"video": "https://cdn.example.test/v2.mp4"}}
    ) == ["https://cdn.example.test/v2.mp4"]


def test_provider_error_detail_is_truncated(monkeypatch, owui):
    tool = Tools()
    tool.valves.MUAPI_API_KEY = "test-key"
    use_router(
        monkeypatch,
        {
            ("POST", f"{BASE}/nano-banana"): [
                FakeResponse({"message": "x" * 5000}, status=400)
            ],
        },
    )

    result = asyncio.run(tool.generate_image("a lighthouse", **call_kwargs()))

    assert result.startswith("MuAPI image generation failed: MuAPI request failed (HTTP 400)")
    assert len(result) < 400


def test_missing_key_and_invalid_image_are_reported_without_requests(monkeypatch, owui):
    tool = Tools()
    router = use_router(monkeypatch, {})

    missing_key = asyncio.run(tool.generate_image("a lighthouse"))
    assert "MUAPI_API_KEY" in missing_key

    tool.valves.MUAPI_API_KEY = "test-key"
    invalid_image = asyncio.run(
        tool.generate_video_from_image("animate this", image_url="ftp://example.test/a.png")
    )
    assert "absolute HTTP(S) URL" in invalid_image

    no_image = asyncio.run(tool.generate_video_from_image("animate this"))
    assert "Attach an image" in no_image
    assert router.requests == []
