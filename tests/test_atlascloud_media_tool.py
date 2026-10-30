"""Unit tests for the Atlas Cloud media tool."""

import os
import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest


def _install_open_webui_stubs() -> None:
    """Stand-ins for the Open WebUI modules the tool imports (setdefault per
    module so it coexists with stubs installed by other test modules)."""
    defaults = {
        "open_webui": {},
        "open_webui.models": {},
        "open_webui.models.chats": {"Chats": SimpleNamespace(insert_chat_files=AsyncMock())},
        "open_webui.models.files": {"Files": SimpleNamespace(get_file_by_id=AsyncMock(return_value=None))},
        "open_webui.models.users": {
            "Users": SimpleNamespace(get_user_by_id=AsyncMock(return_value=None)),
            "UserModel": type("UserModel", (), {}),
        },
        "open_webui.routers": {},
        "open_webui.routers.files": {"upload_file_handler": AsyncMock()},
        "open_webui.storage": {},
        "open_webui.storage.provider": {"Storage": SimpleNamespace(get_file=lambda path: path)},
    }
    for name, attrs in defaults.items():
        module = sys.modules.setdefault(name, types.ModuleType(name))
        for attr, value in attrs.items():
            if isinstance(module, types.ModuleType) and not hasattr(module, attr):
                setattr(module, attr, value)


_install_open_webui_stubs()


@pytest.fixture(autouse=True)
def fake_storage(monkeypatch):
    """Outputs are persisted to Open WebUI storage; fake that step here."""
    async def persist(self, outputs, *, kind, expected_type, prefix, request, user, metadata):
        return [{"id": f"f{i}", "url": f"/api/v1/files/f{i}/content", "name": f"{prefix}_{i}.bin",
                 "content_type": expected_type} for i, _ in enumerate(outputs)]
    monkeypatch.setattr(Tools, "_persist_outputs", persist)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

from atlascloud_media_tool import (
    DEFAULT_IMAGE_MODEL,
    DEFAULT_IMAGE_EDIT_MODEL,
    DEFAULT_VIDEO_MODEL,
    DEFAULT_IMAGE_TO_VIDEO_MODEL,
    DEFAULT_AUDIO_TO_VIDEO_MODEL,
    DEFAULT_IMAGE_TO_3D_MODEL,
    DEFAULT_TEXT_TO_3D_MODEL,
    MODEL_VIEWER_VERSION,
    AtlasCloudError,
    MediaStorageError,
    Tools,
    model_3d_viewer_embed,
    select_glb_url,
    select_preview_url,
)


class FakeResponse:
    def __init__(self, payload, status=200):
        self.payload = payload
        self.status = status

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    async def json(self):
        return self.payload

    async def text(self):
        return str(self.payload)


class FakeSession:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    def post(self, url, json=None, data=None):
        self.requests.append(("POST", url, json or data))
        return next(self.responses)

    def get(self, url):
        self.requests.append(("GET", url, None))
        return next(self.responses)


def test_default_valves_use_current_atlas_models():
    tool = Tools()

    assert tool.valves.API_BASE_URL == "https://api.atlascloud.ai/api/v1"
    assert tool.valves.IMAGE_MODEL == DEFAULT_IMAGE_MODEL
    assert tool.valves.IMAGE_EDIT_MODEL == DEFAULT_IMAGE_EDIT_MODEL
    assert tool.valves.VIDEO_MODEL == DEFAULT_VIDEO_MODEL
    assert tool.valves.IMAGE_TO_VIDEO_MODEL == DEFAULT_IMAGE_TO_VIDEO_MODEL
    assert tool.valves.AUDIO_TO_VIDEO_MODEL == DEFAULT_AUDIO_TO_VIDEO_MODEL


def test_user_valves_override():
    tool = Tools()
    tool.valves.ATLASCLOUD_API_KEY = "admin-key"
    tool.valves.IMAGE_MODEL = "admin-image-model"

    user_valves = tool.UserValves(
        ATLASCLOUD_API_KEY="user-key",
        IMAGE_MODEL="user-image-model",
    )
    user_dict = {"valves": user_valves}

    config = tool._resolve_config(user_dict)
    assert config["api_key"] == "user-key"
    assert config["image_model"] == "user-image-model"
    assert config["video_model"] == DEFAULT_VIDEO_MODEL


@pytest.mark.asyncio
async def test_missing_api_key_is_reported():
    tool = Tools()
    emitter = AsyncMock()

    result = await tool.generate_image("a lighthouse", __event_emitter__=emitter)

    assert "API key is not configured" in result
    assert emitter.await_args_list[-1].args[0]["data"]["done"] is True


@pytest.mark.asyncio
async def test_generate_image_submits_polls_and_formats_outputs():
    tool = Tools()
    tool.valves.ATLASCLOUD_API_KEY = "test-key"
    session = FakeSession(
        [
            FakeResponse(
                {"code": 200, "data": {"id": "prediction-1", "status": "starting"}}
            ),
            FakeResponse({"code": 200, "data": {"status": "processing"}}),
            FakeResponse(
                {
                    "code": 200,
                    "data": {
                        "status": "completed",
                        "outputs": ["https://cdn.example.test/image.jpg"],
                    },
                }
            ),
        ]
    )

    with patch(
        "atlascloud_media_tool.aiohttp.ClientSession", return_value=session
    ), patch("atlascloud_media_tool.asyncio.sleep", new=AsyncMock()):
        result = await tool.generate_image("a lighthouse")

    html_response, context = result
    assert "/api/v1/files/f0/content" in html_response.body.decode()
    assert "cdn.example.test" not in str(context) and "cdn.example.test" not in html_response.body.decode()
    assert session.requests[0] == (
        "POST",
        "https://api.atlascloud.ai/api/v1/model/generateImage",
        {
            "model": DEFAULT_IMAGE_MODEL,
            "prompt": "a lighthouse",
            "size": "2048*2048",
            "output_format": "jpeg",
        },
    )
    assert session.requests[1][1].endswith("/model/prediction/prediction-1")


@pytest.mark.asyncio
async def test_generate_video_submits_expected_options_and_returns_html_embed():
    tool = Tools()
    tool.valves.ATLASCLOUD_API_KEY = "test-key"
    session = FakeSession(
        [
            FakeResponse({"data": {"id": "prediction-video"}}),
            FakeResponse(
                {
                    "data": {
                        "status": "completed",
                        "outputs": ["https://cdn.example.test/video.mp4"],
                    }
                }
            ),
        ]
    )

    with patch("atlascloud_media_tool.aiohttp.ClientSession", return_value=session):
        result = await tool.generate_video(
            "a city at night",
            duration=8,
            resolution="1080p",
            ratio="16:9",
            generate_audio=False,
        )

    # With RETURN_HTML_EMBED=True (default), result is (HTMLResponse, context_str)
    assert isinstance(result, tuple) and len(result) == 2
    html_response, context = result
    html_body = html_response.body.decode("utf-8")
    assert 'id="videos"' in html_body
    assert "/api/v1/files/f0/content" in html_body
    assert "cdn.example.test" not in html_body and "cdn.example.test" not in str(context)
    assert session.requests[0] == (
        "POST",
        "https://api.atlascloud.ai/api/v1/model/generateVideo",
        {
            "model": DEFAULT_VIDEO_MODEL,
            "prompt": "a city at night",
            "duration": 8,
            "resolution": "1080p",
            "ratio": "16:9",
            "generate_audio": False,
        },
    )


@pytest.mark.asyncio
async def test_generate_video_from_image_submits_correct_payload():
    tool = Tools()
    tool.valves.ATLASCLOUD_API_KEY = "test-key"
    session = FakeSession(
        [
            FakeResponse({"data": {"id": "prediction-i2v"}}),
            FakeResponse(
                {
                    "data": {
                        "status": "completed",
                        "outputs": ["https://cdn.example.test/i2v.mp4"],
                    }
                }
            ),
        ]
    )

    with patch("atlascloud_media_tool.aiohttp.ClientSession", return_value=session):
        result = await tool.generate_video_from_image(
            "make the ocean waves move",
            image_url="https://cdn.example.test/source.jpg",
            duration=5,
        )

    assert isinstance(result, tuple)
    html_response, context = result
    assert context["files"] == ["/api/v1/files/f0/content"]
    assert session.requests[0] == (
        "POST",
        "https://api.atlascloud.ai/api/v1/model/generateVideo",
        {
            "model": DEFAULT_IMAGE_TO_VIDEO_MODEL,
            "prompt": "make the ocean waves move",
            "image_url": "https://cdn.example.test/source.jpg",
            "duration": 5,
            "resolution": "720p",
            "ratio": "adaptive",
            "generate_audio": True,
        },
    )


@pytest.mark.asyncio
async def test_edit_image_with_attached_message():
    tool = Tools()
    tool.valves.ATLASCLOUD_API_KEY = "test-key"
    session = FakeSession(
        [
            FakeResponse({"data": {"id": "prediction-edit"}}),
            FakeResponse(
                {
                    "data": {
                        "status": "completed",
                        "outputs": ["https://cdn.example.test/edited.jpg"],
                    }
                }
            ),
        ]
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "make it sunset"},
                {"type": "image_url", "image_url": {"url": "https://cdn.example.test/input.jpg"}},
            ],
        }
    ]

    with patch("atlascloud_media_tool.aiohttp.ClientSession", return_value=session):
        result = await tool.edit_image(
            "make it sunset",
            __messages__=messages,
        )

    assert result[1]["files"] == ["/api/v1/files/f0/content"]
    assert session.requests[0] == (
        "POST",
        "https://api.atlascloud.ai/api/v1/model/generateImage",
        {
            "model": DEFAULT_IMAGE_EDIT_MODEL,
            "prompt": "make it sunset",
            "image_url": "https://cdn.example.test/input.jpg",
            "size": "2048*2048",
            "output_format": "jpeg",
        },
    )


@pytest.mark.asyncio
async def test_failed_prediction_is_reported():
    tool = Tools()
    tool.valves.ATLASCLOUD_API_KEY = "test-key"
    session = FakeSession(
        [
            FakeResponse({"data": {"id": "prediction-2"}}),
            FakeResponse({"data": {"status": "failed", "error": "model unavailable"}}),
        ]
    )

    with patch(
        "atlascloud_media_tool.aiohttp.ClientSession", return_value=session
    ), patch("atlascloud_media_tool.asyncio.sleep", new=AsyncMock()):
        result = await tool.generate_image("a lighthouse")

    assert "Atlas Cloud image generation failed: Atlas Cloud generation failed: model unavailable" in result


# ---------------------------------------------------------------------------
# 3D model generation
# ---------------------------------------------------------------------------

GLB_BYTES = b"glTF\x02\x00\x00\x00" + b"\x00" * 64
PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
PROVIDER = "cdn.example.test"


def _completed_3d(status="succeeded"):
    return {
        "code": 200,
        "data": {
            "status": status,
            "outputs": [
                f"https://{PROVIDER}/out/model.glb?sig=abc",
                f"https://{PROVIDER}/out/preview.png?sig=abc",
            ],
            "files": [
                {"url": f"https://{PROVIDER}/out/preview.png?sig=abc", "type": "image",
                 "content_type": "image/png", "file_name": "preview.png"},
                {"url": f"https://{PROVIDER}/out/model.glb?sig=abc", "type": "model",
                 "content_type": "model/gltf-binary", "file_name": "model.glb"},
            ],
            "thumbnail": f"https://{PROVIDER}/out/thumb.webp?sig=abc",
        },
    }


class _Store3D:
    """Fakes download + Open WebUI storage for the 3D path."""

    def __init__(self, glb=GLB_BYTES, fail_save=False):
        self.glb = glb
        self.fail_save = fail_save
        self.fetched = []
        self.saved = []

    async def fetch(self, url, **kwargs):
        self.fetched.append(url)
        if ".glb" in url:
            return self.glb, "application/octet-stream"
        return PNG_BYTES, "image/png"

    async def save(self, data, **kwargs):
        if self.fail_save:
            raise MediaStorageError("Open WebUI rejected the file (HTTP 400).")
        file_id = "glb1" if kwargs["content_type"] == "model/gltf-binary" else "prev1"
        self.saved.append(kwargs)
        return {"id": file_id, "url": f"/api/v1/files/{file_id}/content", "name": kwargs.get("filename"),
                "content_type": kwargs["content_type"], "size": len(data), "linked": True, "fallback": False}


def _run_3d(tool, session, store, **kwargs):
    with patch("atlascloud_media_tool.aiohttp.ClientSession", return_value=session), \
            patch("atlascloud_media_tool.asyncio.sleep", new=AsyncMock()), \
            patch("atlascloud_media_tool.fetch_media_bytes", new=store.fetch), \
            patch("atlascloud_media_tool.save_media_bytes", new=store.save), \
            patch("atlascloud_media_tool.resolve_user", new=AsyncMock(return_value=SimpleNamespace(id="u1"))):
        return asyncio_run(tool.generate_3d_model(**kwargs))


def asyncio_run(coro):
    import asyncio

    return asyncio.run(coro)


def _tool_3d(**valves):
    tool = Tools()
    tool.valves.ATLASCLOUD_API_KEY = "test-key"
    for key, value in valves.items():
        setattr(tool.valves, key, value)
    return tool


def test_3d_default_valves():
    tool = Tools()
    assert tool.valves.TEXT_TO_3D_MODEL == DEFAULT_TEXT_TO_3D_MODEL == "tencent/hunyuan3d-rapid/text-to-3d"
    assert tool.valves.IMAGE_TO_3D_MODEL == DEFAULT_IMAGE_TO_3D_MODEL == "tencent/hunyuan3d-rapid/image-to-3d"
    assert tool.valves.ENABLE_PBR_3D is True
    assert tool.valves.FACE_LIMIT_3D is None
    assert tool.valves.MAX_WAIT_3D_SECONDS == 900.0


def test_text_to_3d_uses_text_model_and_stores_glb():
    tool = _tool_3d()
    store = _Store3D()
    emitter = AsyncMock()
    session = FakeSession([
        FakeResponse({"code": 200, "data": {"id": "p3d", "status": "processing"}}),
        FakeResponse({"code": 200, "data": {"status": "created"}}),
        FakeResponse({"code": 200, "data": {"status": "processing"}}),
        FakeResponse(_completed_3d("succeeded")),
    ])
    prompt = "A bronze helmet <script>alert(1)</script>"
    result = _run_3d(tool, session, store, prompt=prompt, __event_emitter__=emitter,
                     __messages__=[{"role": "user", "content": "make a 3d helmet"}])

    html_response, message = result
    assert html_response.headers["Content-Disposition"] == "inline"
    assert session.requests[0] == (
        "POST",
        "https://api.atlascloud.ai/api/v1/model/generateImage",
        {"model": DEFAULT_TEXT_TO_3D_MODEL, "prompt": prompt, "enable_pbr": True, "format": "GLB"},
    )
    assert session.requests[1][1].endswith("/model/prediction/p3d")
    # GLB picked from files[] (query string ignored), preview from thumbnail.
    assert store.fetched == [f"https://{PROVIDER}/out/model.glb?sig=abc", f"https://{PROVIDER}/out/thumb.webp?sig=abc"]
    glb_save = store.saved[0]
    assert glb_save["content_type"] == "model/gltf-binary"
    assert glb_save["filename"] == "a-bronze-helmet-script-alert-1-script.glb"
    assert store.saved[1]["content_type"] == "image/png"

    body = html_response.body.decode()
    assert PROVIDER not in body and PROVIDER not in message
    assert "/api/v1/files/glb1/content" in body and "/api/v1/files/prev1/content" in body
    assert "<script>alert(1)" not in body and "&lt;script&gt;alert(1)" in body
    assert f"@google/model-viewer@{MODEL_VIEWER_VERSION}/" in body
    assert "/api/v1/files/glb1/content" in message
    assert "Hunyuan3D Rapid" in message and "$0.02" in message and "shown" in message
    # Status updates deduplicated: created + processing once each.
    statuses = [c.args[0]["data"]["description"] for c in emitter.await_args_list]
    assert sum("status: processing" in s for s in statuses) == 1
    assert statuses[-1] == "Atlas Cloud 3D model generated"


def test_image_to_3d_hunyuan_uses_image_field_from_attachment():
    tool = _tool_3d()
    store = _Store3D()
    session = FakeSession([FakeResponse({"data": {"id": "p1"}}), FakeResponse(_completed_3d("completed"))])
    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": "turn this into 3d"},
            {"type": "image_url", "image_url": {"url": f"https://{PROVIDER}/in/chair.png"}},
        ]},
    ]
    result = _run_3d(tool, session, store, prompt="wooden chair", __messages__=messages)

    assert isinstance(result, tuple)
    assert session.requests[0][2] == {
        "model": DEFAULT_IMAGE_TO_3D_MODEL,
        "image": f"https://{PROVIDER}/in/chair.png",
        "enable_pbr": True,
        "format": "GLB",
    }


def test_image_to_3d_tripo_uses_image_url_field_and_clamps_faces():
    tool = _tool_3d(IMAGE_TO_3D_MODEL="tripo-h3.1/image-to-3d", ENABLE_PBR_3D=False, FACE_LIMIT_3D=5_000_000)
    store = _Store3D()
    session = FakeSession([FakeResponse({"data": {"id": "p1"}}), FakeResponse(_completed_3d())])
    result = _run_3d(tool, session, store, image_url=f"https://{PROVIDER}/in/robot.jpg")

    assert session.requests[0][2] == {
        "model": "tripo-h3.1/image-to-3d",
        "image_url": f"https://{PROVIDER}/in/robot.jpg",
        "texture": True,
        "pbr": False,
        "face_limit": 2_000_000,
    }
    assert "Tripo H3.1" in result[1] and "$0.10" in result[1]


def test_image_to_3d_from_files_param():
    tool = _tool_3d()
    store = _Store3D()
    session = FakeSession([FakeResponse({"data": {"id": "p1"}}), FakeResponse(_completed_3d())])
    files = [{"type": "file", "id": "doc1", "name": "notes.pdf", "meta": {"content_type": "application/pdf"}},
             {"type": "image", "url": f"https://{PROVIDER}/in/vase.webp"}]
    _run_3d(tool, session, store, prompt="vase", __files__=files,
            __messages__=[{"role": "user", "content": "3d this"}])
    assert session.requests[0][2]["image"] == f"https://{PROVIDER}/in/vase.webp"


def test_old_chat_image_does_not_switch_to_image_to_3d():
    tool = _tool_3d()
    store = _Store3D()
    session = FakeSession([FakeResponse({"data": {"id": "p1"}}), FakeResponse(_completed_3d())])
    messages = [
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"https://{PROVIDER}/old.png"}}]},
        {"role": "assistant", "content": "nice picture"},
        {"role": "user", "content": "now make a 3d model of a teapot"},
    ]
    _run_3d(tool, session, store, prompt="a teapot", __messages__=messages)
    assert session.requests[0][2]["model"] == DEFAULT_TEXT_TO_3D_MODEL


def test_user_valves_override_3d_model():
    tool = _tool_3d()
    store = _Store3D()
    session = FakeSession([FakeResponse({"data": {"id": "p1"}}), FakeResponse(_completed_3d())])
    user = {"id": "u1", "valves": tool.UserValves(TEXT_TO_3D_MODEL="meshy-v7/text-to-3d")}
    _run_3d(tool, session, store, prompt="a lamp", __user__=user)
    assert session.requests[0][2] == {
        "model": "meshy-v7/text-to-3d", "prompt": "a lamp", "mode": "full",
        "enable_pbr": True, "target_formats": ["glb"],
    }


def test_image_to_3d_meshy_uses_image_field_and_texturing():
    tool = _tool_3d(IMAGE_TO_3D_MODEL="meshy-v7/image-to-3d")
    session = FakeSession([FakeResponse({"data": {"id": "p1"}}), FakeResponse(_completed_3d())])
    result = _run_3d(tool, session, _Store3D(), image_url=f"https://{PROVIDER}/in/lamp.png")
    assert isinstance(result, tuple)
    assert session.requests[0][2] == {
        "model": "meshy-v7/image-to-3d",
        "image": f"https://{PROVIDER}/in/lamp.png",
        "should_texture": True,
        "enable_pbr": True,
        "target_formats": ["glb"],
    }


class NonJSONResponse(FakeResponse):
    async def json(self):
        raise ValueError("Expecting value")


class RaisingGet:
    """Context manager whose __aenter__ raises (connection reset / per-request timeout)."""

    def __init__(self, exc):
        self.exc = exc

    async def __aenter__(self):
        raise self.exc

    async def __aexit__(self, exc_type, exc, traceback):
        return False


def test_3d_poll_survives_transient_errors():
    import asyncio
    import aiohttp

    tool = _tool_3d()
    store = _Store3D()
    emitter = AsyncMock()
    session = FakeSession([
        FakeResponse({"code": 200, "data": {"id": "p3d", "status": "processing"}}),
        FakeResponse({"code": 200, "data": {"status": "processing"}}),
        FakeResponse({"message": "bad gateway"}, status=502),
        NonJSONResponse(None, status=200),
        RaisingGet(aiohttp.ClientConnectionError("connection reset")),
        RaisingGet(asyncio.TimeoutError()),
        FakeResponse({"code": 200, "data": {"status": "processing"}}),
        FakeResponse(_completed_3d("succeeded")),
    ])
    result = _run_3d(tool, session, store, prompt="a helmet", __event_emitter__=emitter)

    assert isinstance(result, tuple), result
    assert len(session.requests) == 8
    assert store.saved and store.saved[0]["content_type"] == "model/gltf-binary"
    statuses = [c.args[0]["data"]["description"] for c in emitter.await_args_list]
    assert sum("retrying" in s for s in statuses) == 4
    # Status re-announced once after recovery.
    assert sum("status: processing" in s for s in statuses) == 2


def test_3d_poll_gives_up_after_consecutive_transient_errors():
    tool = _tool_3d()
    session = FakeSession(
        [FakeResponse({"data": {"id": "p-abc"}})]
        + [FakeResponse({"message": "unavailable"}, status=503) for _ in range(5)]
    )
    result = _run_3d(tool, session, _Store3D(), prompt="a cube")
    assert len(session.requests) == 6
    assert "status checks kept failing for prediction p-abc" in result
    assert "HTTP 503" in result


def test_3d_poll_definitive_4xx_aborts_immediately():
    tool = _tool_3d()
    session = FakeSession([
        FakeResponse({"data": {"id": "p1"}}),
        FakeResponse({"message": "prediction not found"}, status=404),
    ])
    result = _run_3d(tool, session, _Store3D(), prompt="a cube")
    assert len(session.requests) == 2
    assert "request failed (404): prediction not found" in result


def test_unsupported_3d_model_is_rejected_without_requests():
    tool = _tool_3d(TEXT_TO_3D_MODEL="bytedance/seed3d/text-to-3d")
    session = FakeSession([])
    result = _run_3d(tool, session, _Store3D(), prompt="a cube")
    assert "Unsupported text-to-3D model" in result and session.requests == []

    # An image-to-3d id configured for text-to-3d is a mismatch too.
    tool = _tool_3d(TEXT_TO_3D_MODEL=DEFAULT_IMAGE_TO_3D_MODEL)
    assert "Unsupported" in _run_3d(tool, FakeSession([]), _Store3D(), prompt="a cube")


def test_text_to_3d_requires_prompt():
    result = _run_3d(_tool_3d(), FakeSession([]), _Store3D())
    assert "Provide a prompt" in result


def test_3d_failure_surfaces_provider_error():
    tool = _tool_3d()
    session = FakeSession([
        FakeResponse({"data": {"id": "p1"}}),
        FakeResponse({"data": {"status": "failed", "error": "mesh reconstruction failed"}}),
    ])
    result = _run_3d(tool, session, _Store3D(), prompt="a cube")
    assert result == ("Atlas Cloud 3D model generation failed: Atlas Cloud generation failed: "
                      "mesh reconstruction failed")


def test_3d_uses_dedicated_timeout():
    tool = _tool_3d(GENERATION_TIMEOUT_SECONDS=600.0, MAX_WAIT_3D_SECONDS=900.0)
    # monotonic() calls: start, loop check, elapsed, loop check, elapsed, loop check.
    ticks = [0.0, 0.0, 0.0, 700.0, 700.0, 950.0]
    fake_time = SimpleNamespace(monotonic=lambda: ticks.pop(0) if len(ticks) > 1 else ticks[0])
    session = FakeSession([
        FakeResponse({"data": {"id": "p1"}}),
        FakeResponse({"data": {"status": "processing"}}),
        FakeResponse({"data": {"status": "processing"}}),
    ])
    with patch("atlascloud_media_tool.time", fake_time):
        result = _run_3d(tool, session, _Store3D(), prompt="a cube")
    # Polled past the 600 s media timeout, stopped at the 900 s 3D timeout.
    assert len(session.requests) == 3
    assert result == "Atlas Cloud 3D model generation failed: Atlas Cloud generation timed out (prediction p1)."


def test_3d_without_glb_fails_clearly():
    tool = _tool_3d()
    store = _Store3D()
    session = FakeSession([
        FakeResponse({"data": {"id": "p1"}}),
        FakeResponse({"data": {"status": "completed", "outputs": [f"https://{PROVIDER}/a.obj", f"https://{PROVIDER}/b.png"]}}),
    ])
    result = _run_3d(tool, session, store, prompt="a cube")
    assert "returned no GLB" in result and store.fetched == []


def test_3d_non_glb_bytes_rejected():
    store = _Store3D(glb=b"<html>expired</html>")
    session = FakeSession([FakeResponse({"data": {"id": "p1"}}), FakeResponse(_completed_3d())])
    result = _run_3d(_tool_3d(), session, store, prompt="a cube")
    assert "not a valid GLB" in result and store.saved == []


def test_3d_storage_failure_is_clear_and_hides_provider_urls():
    store = _Store3D(fail_save=True)
    session = FakeSession([FakeResponse({"data": {"id": "p1"}}), FakeResponse(_completed_3d())])
    result = _run_3d(_tool_3d(), session, store, prompt="a cube")
    assert isinstance(result, str)
    assert result.startswith("Atlas Cloud generated the 3D model, but it could not be saved to Open WebUI")
    assert PROVIDER not in result


def test_3d_without_embed_returns_download_link():
    session = FakeSession([FakeResponse({"data": {"id": "p1"}}), FakeResponse(_completed_3d())])
    result = _run_3d(_tool_3d(RETURN_HTML_EMBED=False), session, _Store3D(), prompt="a cube")
    assert "/api/v1/files/glb1/content" in result and PROVIDER not in result


def test_select_glb_url_variants():
    # files[] by content type, query string on the URL
    assert select_glb_url({"files": [
        {"url": "https://x.test/f/123?token=1", "content_type": "model/gltf-binary"},
    ]}) == "https://x.test/f/123?token=1"
    # files[] by file_name
    assert select_glb_url({"files": [
        {"url": "https://x.test/p.png", "file_name": "p.png"},
        {"url": "https://x.test/blob/9", "file_name": "Mesh.GLB"},
    ]}) == "https://x.test/blob/9"
    # outputs fallback: first .glb path, query string ignored
    assert select_glb_url({"outputs": [
        "https://x.test/a.obj", "https://x.test/m.glb?X-Amz-Signature=z", "https://x.test/n.glb",
    ]}) == "https://x.test/m.glb?X-Amz-Signature=z"
    assert select_glb_url({"outputs": "https://x.test/single.glb"}) == "https://x.test/single.glb"
    with pytest.raises(AtlasCloudError, match="no GLB"):
        select_glb_url({"outputs": ["https://x.test/m.glb.zip", "https://x.test/p.png"], "files": []})


def test_select_preview_url_variants():
    glb = "https://x.test/m.glb"
    assert select_preview_url({"thumbnail": "https://x.test/t.webp", "outputs": [glb]}, exclude=glb) == "https://x.test/t.webp"
    assert select_preview_url({"files": [{"url": "https://x.test/r?s=1", "content_type": "image/jpeg"}]}) == "https://x.test/r?s=1"
    assert select_preview_url({"outputs": [glb, "https://x.test/a.png?q=1", "https://x.test/b.jpg"]}, exclude=glb) == "https://x.test/b.jpg"
    assert select_preview_url({"outputs": [glb]}, exclude=glb) is None


def test_model_3d_viewer_embed_escapes_and_has_fallbacks():
    body = model_3d_viewer_embed(
        "/api/v1/files/abc/content", "/api/v1/files/def/content",
        title='Robot "<b>"', prompt="</script><img src=x onerror=alert(1)>",
    )
    assert "</script><img" not in body
    assert "&lt;/script&gt;&lt;img" in body
    assert "<title>Robot &quot;&lt;b&gt;&quot;</title>" in body
    assert "camera-controls" in body and "auto-rotate" in body and "environment-image" in body
    assert "shadow-intensity" in body and "requestFullscreen" in body and "mv-bar" in body
    assert "iframe Sandbox Allow Same Origin" in body  # shared runtime fallback hint
    assert "sha256-" in body and "integrity" in body
    assert "iframe:height" in body
